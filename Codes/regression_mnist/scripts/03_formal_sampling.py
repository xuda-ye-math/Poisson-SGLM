#!/usr/bin/env python3
"""Run the 60 paired formal SGLD/SGLM sampling jobs used in the paper."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sglm_mnist.data import (  # noqa: E402
    atomic_save_json,
    atomic_save_npz,
    binary_labels,
    feature_matrix,
    load_mnist_training,
    training_arrays,
)
from sglm_mnist.observables import torch_pool  # noqa: E402
from sglm_mnist.samplers import (  # noqa: E402
    TorchLogisticMinibatchOracle,
    TorchPairedState,
    torch_paired_transition,
)
from sglm_mnist.seeds import make_generator  # noqa: E402
from sglm_mnist.settings import (  # noqa: E402
    BATCHES,
    BURN_IN,
    DATA_ROOT,
    FUNCTION_NAMES,
    MODEL_FILE,
    PREFIXES,
    FORMAL_SAMPLING_DIR,
    REPETITIONS,
    RETAINED,
    SNAPSHOT_COUNT,
    SNAPSHOT_INDICES,
    pair_file,
    pair_manifest,
)


def load_inputs(device_name: str) -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("run scripts/01_prepare.py first")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal sampling requires an explicit CUDA device")
    with np.load(MODEL_FILE, allow_pickle=False) as model:
        dataset = load_mnist_training(DATA_ROOT, download=False)
        images, targets = training_arrays(dataset)
        ids = model["posterior_train_ids"]
        digits = tuple(int(x) for x in model["digits"])
        features = feature_matrix(images, ids)
        labels = binary_labels(targets, ids, digits).astype(np.float64)
        result = {
            "features": features,
            "features_t": torch.as_tensor(features, dtype=torch.float64, device=device),
            "labels_t": torch.as_tensor(labels, dtype=torch.float64, device=device),
            "theta_map_t": torch.as_tensor(model["theta_map"], dtype=torch.float64, device=device),
            "eigenvalues_t": torch.as_tensor(model["eigenvalues"], dtype=torch.float64, device=device),
            "eigenvectors_t": torch.as_tensor(model["eigenvectors"], dtype=torch.float64, device=device),
            "L": float(model["L"][0]),
            "dimension": int(model["theta_map"].size),
            "device": device,
        }
    return result


def write_progress(batch: int, repetition: int, completed: int, started: float, status: str) -> None:
    total = BURN_IN + RETAINED
    elapsed = time.perf_counter() - started
    payload = {
        "status": status,
        "batch": batch,
        "repetition": repetition,
        "completed_transitions": completed,
        "total_transitions": total,
        "fraction": completed / total,
        "elapsed_seconds": elapsed,
        "eta_seconds": None if completed == 0 else elapsed * (total - completed) / completed,
    }
    atomic_save_json(FORMAL_SAMPLING_DIR / "progress" / f"batch_{batch}_rep_{repetition}.json", payload)


def run_pair(batch: int, repetition: int, device_name: str) -> None:
    output = pair_file(batch, repetition)
    manifest_path = pair_manifest(batch, repetition)
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"pair output already exists: batch={batch}, repetition={repetition}")
    values = load_inputs(device_name)
    device = values["device"]
    torch.set_num_threads(2)
    oracle = TorchLogisticMinibatchOracle(values["features_t"], values["labels_t"])
    step_size = 1.0 / values["L"]
    oracle_rng, oracle_seed = make_generator(
        "formal_sampling_oracle", "4_9", f"batch_{batch}", f"rep_{repetition}"
    )
    brownian_rng, brownian_seed = make_generator(
        "formal_sampling_brownian", "4_9", f"batch_{batch}", f"rep_{repetition}"
    )
    chi_initial = np.ascontiguousarray(brownian_rng.standard_normal(values["dimension"]))
    state = TorchPairedState(
        values["theta_map_t"].clone(),
        values["theta_map_t"].clone(),
        torch.as_tensor(chi_initial, dtype=torch.float64, device=device),
        0,
    )
    sums = torch.zeros((2, len(FUNCTION_NAMES)), dtype=torch.float64, device=device)
    prefix_sums = np.full((len(PREFIXES), 2, len(FUNCTION_NAMES)), np.nan)
    prediction_states = np.full((2, SNAPSHOT_COUNT, values["dimension"]), np.nan)
    saved = np.zeros(SNAPSHOT_COUNT, dtype=bool)
    snapshot_indices = np.asarray(SNAPSHOT_INDICES, dtype=np.int64)
    total = BURN_IN + RETAINED
    retained_count = 0
    started = time.perf_counter()
    last_progress = -1e30
    write_progress(batch, repetition, 0, started, "running")

    for k in range(total):
        if k >= BURN_IN:
            retained_index = k - BURN_IN
            states = torch.stack((state.ld, state.lm))
            observed = torch_pool(
                states,
                values["theta_map_t"],
                values["eigenvalues_t"],
                values["eigenvectors_t"],
            )
            sums += observed
            retained_count += 1
            if retained_count in PREFIXES:
                prefix_sums[PREFIXES.index(retained_count)] = sums.detach().cpu().numpy()
            slot = retained_index // (RETAINED // SNAPSHOT_COUNT)
            if slot < SNAPSHOT_COUNT and retained_index == snapshot_indices[slot]:
                prediction_states[:, slot] = states.detach().cpu().numpy()
                saved[slot] = True

        batch_ids = np.ascontiguousarray(
            oracle_rng.integers(0, values["features"].shape[0], size=batch, dtype=np.int64)
        )
        noise = np.ascontiguousarray(brownian_rng.standard_normal(values["dimension"]))
        batch_t = torch.as_tensor(batch_ids, dtype=torch.int64, device=device)
        noise_t = torch.as_tensor(noise, dtype=torch.float64, device=device)
        state = torch_paired_transition(
            state,
            oracle(state.ld, batch_t),
            oracle(state.lm, batch_t),
            noise_t,
            step_size,
        )
        if state.completed_transitions % 256 == 0:
            finite = torch.isfinite(state.ld).all() and torch.isfinite(state.lm).all()
            if not bool(finite):
                raise FloatingPointError("non-finite sampler state")
        now = time.perf_counter()
        if now - last_progress >= 30.0:
            write_progress(batch, repetition, state.completed_transitions, started, "running")
            last_progress = now

    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    if retained_count != RETAINED or not np.all(saved) or not np.all(np.isfinite(prefix_sums)):
        raise RuntimeError("incomplete retained averages or prediction snapshots")
    arrays = {
        "prefix_sums": prefix_sums,
        "prediction_states": prediction_states,
        "snapshot_indices": snapshot_indices,
        "final_states": torch.stack((state.ld, state.lm)).detach().cpu().numpy(),
    }
    atomic_save_npz(output, **arrays)
    manifest = {
        "status": "complete",
        "batch": batch,
        "repetition": repetition,
        "device": device_name,
        "step_size": step_size,
        "burn_in": BURN_IN,
        "retained": RETAINED,
        "prefixes": list(PREFIXES),
        "snapshot_indices": list(SNAPSHOT_INDICES),
        "methods": ["sgld", "sglm"],
        "function_names": list(FUNCTION_NAMES),
        "runtime_seconds": runtime,
        "seeds": {"minibatches": oracle_seed, "gaussians": brownian_seed},
    }
    atomic_save_json(manifest_path, manifest)
    write_progress(batch, repetition, total, started, "complete")
    print(json.dumps(manifest, indent=2))


def complete(batch: int, repetition: int) -> bool:
    path = pair_manifest(batch, repetition)
    if not path.exists() or not pair_file(batch, repetition).exists():
        return False
    return json.loads(path.read_text()).get("status") == "complete"


def launch(devices: list[str]) -> None:
    if not 1 <= len(devices) <= 4 or len(set(devices)) != len(devices):
        raise ValueError("provide one to four distinct CUDA devices")
    jobs = [(b, r) for r in range(REPETITIONS) for b in BATCHES if not complete(b, r)]
    active: list[dict] = []
    available = list(devices)
    log_dir = FORMAL_SAMPLING_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    while jobs or active:
        while jobs and available:
            batch, repetition = jobs.pop(0)
            device = available.pop(0)
            handle = (log_dir / f"batch_{batch}_rep_{repetition}.log").open("wb")
            cmd = [sys.executable, __file__, "pair", "--batch", str(batch), "--repetition", str(repetition), "--device", device]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active.append({"process": process, "batch": batch, "repetition": repetition, "device": device, "handle": handle})
            print(f"started batch={batch} repetition={repetition} on {device}", flush=True)
        time.sleep(2)
        keep = []
        for job in active:
            code = job["process"].poll()
            if code is None:
                keep.append(job)
                continue
            job["handle"].close()
            available.append(job["device"])
            if code:
                raise RuntimeError(f"pair failed: batch={job['batch']} repetition={job['repetition']}")
            print(f"finished batch={job['batch']} repetition={job['repetition']}", flush=True)
        active = keep
    check()


def check() -> None:
    missing = [(b, r) for b in BATCHES for r in range(REPETITIONS) if not complete(b, r)]
    if missing:
        raise RuntimeError(f"missing or incomplete pairs: {missing}")
    print(json.dumps({"status": "complete", "paired_runs": 60, "individual_trajectories": 120}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--batch", type=int, choices=BATCHES, required=True)
    pair.add_argument("--repetition", type=int, choices=range(REPETITIONS), required=True)
    pair.add_argument("--device", required=True)
    launcher = sub.add_parser("launch")
    launcher.add_argument("--devices", nargs="+", required=True)
    sub.add_parser("check")
    args = parser.parse_args()
    if args.command == "pair":
        run_pair(args.batch, args.repetition, args.device)
    elif args.command == "launch":
        launch(args.devices)
    else:
        check()


if __name__ == "__main__":
    main()
