#!/usr/bin/env python3
"""Run and summarize the four full-gradient NUTS reference chains."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pyro
import torch
from pyro.infer.mcmc import MCMC, NUTS

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
from sglm_mnist.reference import (  # noqa: E402
    ReferenceTransform,
    TorchTransformedLogisticPotential,
    torch_theta_from_z,
)
from sglm_mnist.seeds import derive_seed, make_generator  # noqa: E402
from sglm_mnist.settings import (  # noqa: E402
    DATA_ROOT,
    FUNCTION_NAMES,
    MODEL_FILE,
    REFERENCE_CHAINS,
    REFERENCE_DIR,
    REFERENCE_MAX_TREE_DEPTH,
    REFERENCE_RETAINED,
    REFERENCE_SUMMARY,
    REFERENCE_TARGET_ACCEPT,
    REFERENCE_WARMUP,
)


def chain_file(chain: int) -> Path:
    return REFERENCE_DIR / f"chain_{chain}.npz"


def chain_manifest(chain: int) -> Path:
    return REFERENCE_DIR / f"chain_{chain}.json"


def load_training(model: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_mnist_training(DATA_ROOT, download=False)
    images, targets = training_arrays(dataset)
    ids = model["posterior_train_ids"]
    digits = tuple(int(x) for x in model["digits"])
    return feature_matrix(images, ids), binary_labels(targets, ids, digits)


class Progress:
    def __init__(self, chain: int, potential: TorchTransformedLogisticPotential) -> None:
        self.chain = chain
        self.potential = potential
        self.started = time.perf_counter()
        self.last = -1e30
        self.warmup = 0
        self.samples = 0
        self.path = REFERENCE_DIR / "progress" / f"chain_{chain}.json"

    def write(self, phase: str, completed: int, total: int) -> None:
        if phase == "warmup":
            self.warmup = completed
        else:
            self.samples = completed
        done = self.warmup + self.samples
        full = REFERENCE_WARMUP + REFERENCE_RETAINED
        elapsed = time.perf_counter() - self.started
        payload = {
            "status": "running",
            "chain": self.chain,
            "phase": phase,
            "phase_completed": completed,
            "phase_total": total,
            "overall_fraction": done / full,
            "elapsed_seconds": elapsed,
            "eta_seconds": None if done == 0 else elapsed * (full - done) / done,
            "potential_evaluations": self.potential.evaluation_count,
        }
        atomic_save_json(self.path, payload)
        print(json.dumps(payload), flush=True)
        self.last = time.perf_counter()

    def hook(self, kernel, params, stage: str, iteration: int) -> None:
        del kernel, params
        phase = "warmup" if "warm" in stage.lower() else "sample"
        total = REFERENCE_WARMUP if phase == "warmup" else REFERENCE_RETAINED
        completed = iteration + 1
        now = time.perf_counter()
        if completed in (1, total) or now - self.last >= 30.0:
            self.write(phase, completed, total)


def run_chain(chain: int, device_name: str) -> None:
    output = chain_file(chain)
    manifest_path = chain_manifest(chain)
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"reference chain {chain} already exists")
    if not MODEL_FILE.exists():
        raise FileNotFoundError("run scripts/01_prepare.py first")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NUTS reference requires an explicit CUDA device")

    with np.load(MODEL_FILE, allow_pickle=False) as model:
        theta_map = np.ascontiguousarray(model["theta_map"])
        hessian = np.ascontiguousarray(model["hessian_map"])
        eigenvalues = np.ascontiguousarray(model["eigenvalues"])
        eigenvectors = np.ascontiguousarray(model["eigenvectors"])
        features, labels = load_training(model)
    transform = ReferenceTransform.from_hessian(theta_map, hessian)
    torch.set_num_threads(2)
    features_t = torch.as_tensor(features, dtype=torch.float64, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.float64, device=device)
    theta_map_t = torch.as_tensor(theta_map, dtype=torch.float64, device=device)
    cholesky_t = torch.as_tensor(transform.cholesky_lower, dtype=torch.float64, device=device)
    eigenvalues_t = torch.as_tensor(eigenvalues, dtype=torch.float64, device=device)
    eigenvectors_t = torch.as_tensor(eigenvectors, dtype=torch.float64, device=device)
    potential = TorchTransformedLogisticPotential(features_t, labels_t, theta_map_t, cholesky_t)

    initial_rng, initial_seed = make_generator(
        "stage9_formal_reference_initial_z", "4_9", f"chain_{chain}"
    )
    initial_z = np.ascontiguousarray(initial_rng.standard_normal(theta_map.size))
    nuts_seed, nuts_seed_record = derive_seed(
        "stage9_formal_reference_nuts", "4_9", f"chain_{chain}"
    )
    pyro.clear_param_store()
    pyro.set_rng_seed(nuts_seed % (2**32 - 1))
    recorder = Progress(chain, potential)
    kernel = NUTS(
        potential_fn=potential,
        target_accept_prob=REFERENCE_TARGET_ACCEPT,
        max_tree_depth=REFERENCE_MAX_TREE_DEPTH,
        full_mass=False,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        jit_compile=False,
    )
    mcmc = MCMC(
        kernel,
        num_samples=REFERENCE_RETAINED,
        warmup_steps=REFERENCE_WARMUP,
        initial_params={"z": torch.as_tensor(initial_z, dtype=torch.float64, device=device)},
        num_chains=1,
        hook_fn=recorder.hook,
        disable_progbar=True,
        disable_validation=False,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    mcmc.run()
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started

    z = mcmc.get_samples(group_by_chain=True)["z"][0]
    pool_values = np.empty((REFERENCE_RETAINED, len(FUNCTION_NAMES)), dtype=np.float64)
    for start in range(0, REFERENCE_RETAINED, 8192):
        stop = min(start + 8192, REFERENCE_RETAINED)
        theta = torch_theta_from_z(z[start:stop], theta_map_t, cholesky_t)
        pool_values[start:stop] = torch_pool(
            theta, theta_map_t, eigenvalues_t, eigenvectors_t
        ).detach().cpu().numpy()
    diagnostics = mcmc._diagnostics[0]
    divergences = [int(x) for x in diagnostics.get("divergences", [])]
    atomic_save_npz(output, pool_values=pool_values, initial_z=initial_z)
    manifest = {
        "status": "complete",
        "chain": chain,
        "device": device_name,
        "warmup": REFERENCE_WARMUP,
        "retained": REFERENCE_RETAINED,
        "target_acceptance": REFERENCE_TARGET_ACCEPT,
        "maximum_tree_depth": REFERENCE_MAX_TREE_DEPTH,
        "runtime_seconds": runtime,
        "divergence_count": len(divergences),
        "divergence_indices": divergences,
        "adapted_step_size": float(kernel.step_size),
        "potential_evaluations": potential.evaluation_count,
        "seeds": {"initial": initial_seed, "nuts": nuts_seed_record},
    }
    atomic_save_json(manifest_path, manifest)
    atomic_save_json(recorder.path, {**manifest, "overall_fraction": 1.0})
    print(json.dumps(manifest, indent=2))


def aggregate() -> None:
    if REFERENCE_SUMMARY.exists():
        raise FileExistsError(f"reference summary already exists: {REFERENCE_SUMMARY}")
    chains = []
    manifests = []
    for chain in range(REFERENCE_CHAINS):
        manifests.append(json.loads(chain_manifest(chain).read_text()))
        with np.load(chain_file(chain), allow_pickle=False) as archive:
            values = archive["pool_values"]
            if values.shape != (REFERENCE_RETAINED, len(FUNCTION_NAMES)):
                raise ValueError(f"bad reference shape for chain {chain}: {values.shape}")
            chains.append(values.copy())
    traces = np.stack(chains)
    idata = az.from_dict(
        posterior={name: traces[:, :, i] for i, name in enumerate(FUNCTION_NAMES)}
    )
    rhat = az.rhat(idata, method="rank")
    bulk = az.ess(idata, method="bulk")
    tail = az.ess(idata, method="tail")
    mcse = az.mcse(idata, method="mean")
    functions = {}
    for i, name in enumerate(FUNCTION_NAMES):
        functions[name] = {
            "mean": float(traces[:, :, i].mean()),
            "rhat": float(rhat[name].values),
            "ess_bulk": float(bulk[name].values),
            "ess_tail": float(tail[name].values),
            "mcse_mean": float(mcse[name].values),
        }
    divergence_count = sum(item["divergence_count"] for item in manifests)
    reliable = (
        divergence_count == 0
        and max(item["rhat"] for item in functions.values()) < 1.01
        and min(item["ess_bulk"] for item in functions.values()) > 1000
        and min(item["ess_tail"] for item in functions.values()) > 1000
    )
    summary = {
        "status": "reliable" if reliable else "diagnostics_failed",
        "chains": REFERENCE_CHAINS,
        "warmup_per_chain": REFERENCE_WARMUP,
        "retained_per_chain": REFERENCE_RETAINED,
        "retained_total": REFERENCE_CHAINS * REFERENCE_RETAINED,
        "divergence_count": divergence_count,
        "reference_means": {name: item["mean"] for name, item in functions.items()},
        "functions": functions,
        "summary": {
            "maximum_rhat": max(item["rhat"] for item in functions.values()),
            "minimum_bulk_ess": min(item["ess_bulk"] for item in functions.values()),
            "minimum_tail_ess": min(item["ess_tail"] for item in functions.values()),
        },
    }
    atomic_save_json(REFERENCE_SUMMARY, summary)
    print(json.dumps(summary, indent=2))
    if not reliable:
        raise RuntimeError("reference diagnostics failed")


def chain_complete(chain: int) -> bool:
    manifest_path = chain_manifest(chain)
    if not manifest_path.exists() or not chain_file(chain).exists():
        return False
    return json.loads(manifest_path.read_text()).get("status") == "complete"


def launch(devices: list[str]) -> None:
    if not 1 <= len(devices) <= 4 or len(set(devices)) != len(devices):
        raise ValueError("provide one to four distinct CUDA devices")
    if REFERENCE_SUMMARY.exists():
        print(f"reference summary already complete: {REFERENCE_SUMMARY}")
        return
    pending = [i for i in range(REFERENCE_CHAINS) if not chain_complete(i)]
    active: list[tuple[subprocess.Popen, int, object]] = []
    available = list(devices)
    log_dir = REFERENCE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    while pending or active:
        while pending and available:
            chain = pending.pop(0)
            device = available.pop(0)
            handle = (log_dir / f"chain_{chain}.log").open("wb")
            cmd = [sys.executable, __file__, "chain", "--chain", str(chain), "--device", device]
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            active.append((process, chain, (device, handle)))
            print(f"started reference chain {chain} on {device}", flush=True)
        time.sleep(2)
        keep = []
        for process, chain, resources in active:
            device, handle = resources
            code = process.poll()
            if code is None:
                keep.append((process, chain, resources))
            else:
                handle.close()
                available.append(device)
                if code:
                    raise RuntimeError(
                        f"reference chain {chain} failed; inspect outputs/reference/logs"
                    )
                print(f"finished reference chain {chain}", flush=True)
        active = keep
    aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    chain = sub.add_parser("chain")
    chain.add_argument("--chain", type=int, choices=range(REFERENCE_CHAINS), required=True)
    chain.add_argument("--device", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--devices", nargs="+", required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "chain":
        run_chain(args.chain, args.device)
    elif args.command == "launch":
        launch(args.devices)
    else:
        aggregate()


if __name__ == "__main__":
    main()
