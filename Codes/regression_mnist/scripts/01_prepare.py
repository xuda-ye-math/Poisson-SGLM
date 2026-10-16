#!/usr/bin/env python3
"""Download MNIST, make the fixed split, and compute MAP/Hessian geometry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sglm_mnist.data import (  # noqa: E402
    atomic_save_json,
    atomic_save_npz,
    binary_labels,
    candidate_ids,
    feature_matrix,
    load_mnist_training,
    ndarray_sha256,
    training_arrays,
    validation_ids_for_digit,
)
from sglm_mnist.map_fit import curvature_at_map, fit_map  # noqa: E402
from sglm_mnist.observables import top_eigenpairs  # noqa: E402
from sglm_mnist.posterior import PRIOR_PRECISION  # noqa: E402
from sglm_mnist.settings import (  # noqa: E402
    DATA_ROOT,
    DIGITS,
    MODEL_FILE,
    PREPARE_MANIFEST,
    PREPARED_DIR,
    VALIDATION_PER_DIGIT,
    VALIDATION_SEED_BASE,
)

EXPECTED = {
    "posterior_ids_sha256": "73bace3269cfd77ed53e4fffb081b222a27b74418dbe58b6c8bbbdd7b6a44ac9",
    "validation_ids_sha256": "300abfd4908ce5aed3d6d3fef6aae0716becdd23df523b7c4977fefd72b3e85b",
    "features_sha256": "afe7b0a67f4db034e1c261379daeb23c185f6b5157ebae1fd315c7d0fefb82ad",
    "labels_sha256": "320fce62fec6ed2cd89ab82eb9c0b54eb66c33b05c387e22727e584f13244c2f",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace only the preparation outputs")
    args = parser.parse_args()
    existing = [path for path in (MODEL_FILE, PREPARE_MANIFEST) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"preparation output exists: {existing}; pass --force to rebuild")

    dataset = load_mnist_training(DATA_ROOT, download=True)
    images, targets = training_arrays(dataset)
    validation = {
        digit: validation_ids_for_digit(
            targets,
            digit,
            count=VALIDATION_PER_DIGIT,
            seed_base=VALIDATION_SEED_BASE,
        )
        for digit in DIGITS
    }
    posterior_ids, validation_ids = candidate_ids(targets, DIGITS, validation)
    features = feature_matrix(images, posterior_ids)
    labels = binary_labels(targets, posterior_ids, DIGITS)

    identities = {
        "posterior_ids_sha256": ndarray_sha256(posterior_ids),
        "validation_ids_sha256": ndarray_sha256(validation_ids),
        "features_sha256": ndarray_sha256(features),
        "labels_sha256": ndarray_sha256(labels),
    }
    if identities != EXPECTED:
        raise RuntimeError(f"MNIST split/preprocessing identity mismatch: {identities}")

    fit = fit_map(features, labels, precision=PRIOR_PRECISION, target_residual=1e-5)
    if not fit.success:
        raise RuntimeError(f"MAP fit failed: {fit.metadata()}")
    curvature = curvature_at_map(fit.theta, features, labels, precision=PRIOR_PRECISION)
    eigenvalues, eigenvectors = top_eigenpairs(curvature["hessian"], count=8)

    arrays = {
        "digits": np.asarray(DIGITS, dtype=np.int64),
        "posterior_train_ids": posterior_ids,
        "validation_ids": validation_ids,
        "theta_map": fit.theta,
        "hessian_map": curvature["hessian"],
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "gram_max_eigenvalue": np.asarray([curvature["gram_max_eigenvalue"]]),
        "L": np.asarray([curvature["L"]]),
    }
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(MODEL_FILE, **arrays)
    manifest = {
        "status": "complete",
        "digits": list(DIGITS),
        "binary_encoding": {"4": 0, "9": 1},
        "training_count": int(posterior_ids.size),
        "training_class_counts": {
            "4": int(np.sum(labels == 0)),
            "9": int(np.sum(labels == 1)),
        },
        "validation_count": int(validation_ids.size),
        "feature_shape": list(features.shape),
        "preprocessing": "flatten, divide pixels by 255, prepend intercept; no standardization or PCA",
        "prior_variance": 1e-3,
        "map": fit.metadata(),
        "L": float(curvature["L"]),
        "step_size": 1.0 / float(curvature["L"]),
        "top_hessian_eigenvalues": eigenvalues.tolist(),
        "identities": identities,
        "array_hashes": {name: ndarray_sha256(value) for name, value in arrays.items()},
    }
    atomic_save_json(PREPARE_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
