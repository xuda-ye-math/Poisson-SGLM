#!/usr/bin/env python3
"""Compute MSE curves, paired intervals, and endpoint summaries."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sglm_mnist.data import atomic_save_json  # noqa: E402
from sglm_mnist.seeds import make_generator  # noqa: E402
from sglm_mnist.settings import (  # noqa: E402
    BATCHES,
    BOOTSTRAPS,
    FUNCTION_NAMES,
    METHODS,
    PREFIXES,
    REFERENCE_BOOTSTRAP_BLOCK,
    REFERENCE_CHAINS,
    REFERENCE_DIR,
    REFERENCE_RETAINED,
    REFERENCE_SUMMARY,
    REPETITIONS,
    RESULTS_DIR,
    pair_file,
    pair_manifest,
)

CSV_PATH = RESULTS_DIR / "mse_source.csv"
JSON_PATH = RESULTS_DIR / "mse_summary.json"
REPORT_PATH = RESULTS_DIR / "mse_summary.md"


def load_estimates() -> dict[int, np.ndarray]:
    estimates = {}
    for batch in BATCHES:
        values = np.empty((REPETITIONS, len(METHODS), len(PREFIXES), len(FUNCTION_NAMES)))
        for repetition in range(REPETITIONS):
            manifest = json.loads(pair_manifest(batch, repetition).read_text())
            if manifest.get("status") != "complete":
                raise RuntimeError(f"incomplete pair: batch={batch}, repetition={repetition}")
            with np.load(pair_file(batch, repetition), allow_pickle=False) as archive:
                sums = archive["prefix_sums"]
            expected_prefix = (len(PREFIXES), len(METHODS))
            if sums.shape[:2] != expected_prefix or sums.shape[2] < len(FUNCTION_NAMES):
                raise ValueError(f"invalid prefix sums: batch={batch}, repetition={repetition}")
            sums = sums[:, :, : len(FUNCTION_NAMES)]
            if not np.all(np.isfinite(sums)):
                raise ValueError(f"non-finite prefix sums: batch={batch}, repetition={repetition}")
            values[repetition] = np.transpose(
                sums / np.asarray(PREFIXES)[:, None, None], (1, 0, 2)
            )
        estimates[batch] = values
    return estimates


def load_reference_traces() -> np.ndarray:
    chains = []
    for chain in range(REFERENCE_CHAINS):
        with np.load(REFERENCE_DIR / f"chain_{chain}.npz", allow_pickle=False) as archive:
            trace = archive["pool_values"]
        expected = (REFERENCE_RETAINED, len(FUNCTION_NAMES))
        if trace.shape != expected or not np.all(np.isfinite(trace)):
            raise ValueError(f"invalid reference chain {chain}")
        chains.append(trace.copy())
    return np.stack(chains)


def circular_block_means(trace: np.ndarray, length: int) -> np.ndarray:
    extended = np.concatenate((trace, trace[: length - 1]))
    cumulative = np.empty((extended.shape[0] + 1, extended.shape[1]))
    cumulative[0] = 0.0
    np.cumsum(extended, axis=0, out=cumulative[1:])
    return (cumulative[length:] - cumulative[:-length]) / length


def reference_bootstrap(traces: np.ndarray, block_length: int) -> np.ndarray:
    chain_count, retained, function_count = traces.shape
    if retained % block_length:
        raise ValueError("reference block length must divide the retained length")
    blocks = retained // block_length
    rng, _ = make_generator(
        "formal_final_reference_moving_block_bootstrap", f"block_{block_length}"
    )
    output = np.zeros((BOOTSTRAPS, function_count))
    for chain in range(chain_count):
        block_values = circular_block_means(traces[chain], block_length)
        for start in range(0, BOOTSTRAPS, 16):
            stop = min(start + 16, BOOTSTRAPS)
            indices = rng.integers(0, retained, size=(stop - start, blocks), dtype=np.int64)
            output[start:stop] += block_values[indices].mean(axis=1) / chain_count
    return output


def paired_bootstrap(estimates: np.ndarray, reference: np.ndarray, batch: int) -> np.ndarray:
    rng, _ = make_generator("formal_final_paired_bootstrap", f"batch_{batch}")
    indices = rng.integers(0, REPETITIONS, size=(BOOTSTRAPS, REPETITIONS), dtype=np.int64)
    output = np.empty((BOOTSTRAPS, len(METHODS), len(PREFIXES), len(FUNCTION_NAMES)))
    for start in range(0, BOOTSTRAPS, 250):
        stop = min(start + 250, BOOTSTRAPS)
        error = estimates[indices[start:stop]] - reference[start:stop, None, None, None, :]
        output[start:stop] = np.mean(error * error, axis=1)
    return output


def main() -> None:
    reference_summary = json.loads(REFERENCE_SUMMARY.read_text())
    if reference_summary.get("status") != "reliable":
        raise RuntimeError("reference diagnostics have not passed")
    estimates = load_estimates()
    reference_mean = np.asarray(
        [reference_summary["reference_means"][name] for name in FUNCTION_NAMES]
    )
    traces = load_reference_traces()
    reference_boot = reference_bootstrap(traces, REFERENCE_BOOTSTRAP_BLOCK)
    reference_boot_sensitivity = reference_bootstrap(traces, 2 * REFERENCE_BOOTSTRAP_BLOCK)
    del traces

    rows = []
    endpoint = {}
    for batch in BATCHES:
        error = estimates[batch] - reference_mean[None, None, None, :]
        mse = np.mean(error * error, axis=0)
        boot = paired_bootstrap(estimates[batch], reference_boot, batch)
        boot_sensitivity = paired_bootstrap(estimates[batch], reference_boot_sensitivity, batch)
        mse_ci = np.quantile(boot, (0.025, 0.975), axis=0)
        delta = mse[0] - mse[1]
        delta_ci = np.quantile(boot[:, 0] - boot[:, 1], (0.025, 0.975), axis=0)
        delta_ci_sensitivity = np.quantile(
            boot_sensitivity[:, 0] - boot_sensitivity[:, 1], (0.025, 0.975), axis=0
        )
        for function_index, function in enumerate(FUNCTION_NAMES):
            for method_index, method in enumerate(METHODS):
                for prefix_index, prefix in enumerate(PREFIXES):
                    rows.append({
                        "function": function,
                        "batch": batch,
                        "method": method,
                        "sample_size": prefix,
                        "mse": mse[method_index, prefix_index, function_index],
                        "mse_bootstrap_lower": mse_ci[0, method_index, prefix_index, function_index],
                        "mse_bootstrap_upper": mse_ci[1, method_index, prefix_index, function_index],
                        "delta_sgld_minus_sglm": delta[prefix_index, function_index],
                        "delta_bootstrap_lower": delta_ci[0, prefix_index, function_index],
                        "delta_bootstrap_upper": delta_ci[1, prefix_index, function_index],
                        "delta_sensitivity_lower": delta_ci_sensitivity[0, prefix_index, function_index],
                        "delta_sensitivity_upper": delta_ci_sensitivity[1, prefix_index, function_index],
                    })

        endpoint_mse = mse[:, -1]
        endpoint_delta = endpoint_mse[0] - endpoint_mse[1]
        endpoint_boot = boot[:, :, -1]
        endpoint_delta_boot = endpoint_boot[:, 0] - endpoint_boot[:, 1]
        endpoint_delta_ci = np.quantile(endpoint_delta_boot, (0.025, 0.975), axis=0)
        gain = 1.0 - endpoint_mse[1] / endpoint_mse[0]
        gain_boot = 1.0 - endpoint_boot[:, 1] / endpoint_boot[:, 0]
        run_values = estimates[batch][:, :, -1]
        bias = run_values.mean(axis=0) - reference_mean
        variance = run_values.var(axis=0)
        endpoint[str(batch)] = {
            "sglm_lower_mse_count": int(np.sum(endpoint_delta > 0)),
            "pointwise_paired_interval_supported_count": int(np.sum(endpoint_delta_ci[0] > 0)),
            "median_mse_reduction": float(np.median(gain)),
            "median_mse_reduction_interval": np.quantile(
                np.median(gain_boot, axis=1), (0.025, 0.975)
            ).tolist(),
            "median_absolute_finite_time_bias_ratio": float(
                np.median(np.abs(bias[1]) / np.abs(bias[0]))
            ),
            "median_cross_repetition_variance_ratio": float(
                np.median(variance[1] / variance[0])
            ),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete",
        "design": {
            "batches": list(BATCHES),
            "repetitions": REPETITIONS,
            "sample_sizes": list(PREFIXES),
            "functions": list(FUNCTION_NAMES),
            "bootstrap_replicates": BOOTSTRAPS,
            "reference_block_length": REFERENCE_BOOTSTRAP_BLOCK,
            "reference_block_sensitivity": 2 * REFERENCE_BOOTSTRAP_BLOCK,
        },
        "endpoint_N_131072": endpoint,
    }
    atomic_save_json(JSON_PATH, payload)
    lines = [
        "# MSE endpoint summary", "",
        "| Batch | SGLM lower MSE | Supported pointwise intervals | Median reduction | Bias ratio | Variance ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in BATCHES:
        item = endpoint[str(batch)]
        lines.append(
            f"| {batch} | {item['sglm_lower_mse_count']}/16 | "
            f"{item['pointwise_paired_interval_supported_count']}/16 | "
            f"{100 * item['median_mse_reduction']:.1f}% | "
            f"{item['median_absolute_finite_time_bias_ratio']:.3f} | "
            f"{item['median_cross_repetition_variance_ratio']:.3f} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
