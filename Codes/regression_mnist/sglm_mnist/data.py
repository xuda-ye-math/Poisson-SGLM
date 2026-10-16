"""Deterministic MNIST data preparation for the SGLM/SGLD experiment.

This module intentionally exposes only the official MNIST training split.
The formal test split is loaded later by a separate prediction-stage module.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from torchvision.datasets import MNIST


ARRAY_HASH_FORMAT = "ndarray-v1:dtype+shape+c-order-bytes"


def load_mnist_training(root: Path, *, download: bool) -> MNIST:
    """Load only the official MNIST training split."""

    return MNIST(root=str(root), train=True, download=download)


def training_arrays(dataset: MNIST) -> Tuple[np.ndarray, np.ndarray]:
    """Return uint8 images and int64 targets using original MNIST indices."""

    images = dataset.data.detach().cpu().numpy()
    targets = dataset.targets.detach().cpu().numpy().astype(np.int64, copy=False)
    if images.shape != (60_000, 28, 28):
        raise ValueError(f"Unexpected MNIST training image shape: {images.shape}")
    if targets.shape != (60_000,):
        raise ValueError(f"Unexpected MNIST training target shape: {targets.shape}")
    if images.dtype != np.uint8:
        raise TypeError(f"Unexpected MNIST image dtype: {images.dtype}")
    return images, targets


def validation_ids_for_digit(
    targets: np.ndarray,
    digit: int,
    *,
    count: int,
    seed_base: int,
) -> np.ndarray:
    """Select validation IDs deterministically and return them sorted."""

    available = np.flatnonzero(targets == digit).astype(np.int64, copy=False)
    if available.size < count:
        raise ValueError(f"Digit {digit} has only {available.size} records")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed_base + digit))
    permutation = torch.randperm(available.size, generator=generator).numpy()
    selected = np.sort(available[permutation[:count]])
    if selected.size != count or np.unique(selected).size != count:
        raise AssertionError(f"Validation selection failed for digit {digit}")
    return np.ascontiguousarray(selected, dtype=np.int64)


def candidate_ids(
    targets: np.ndarray,
    digits: Tuple[int, int],
    validation_by_digit: Mapping[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted posterior-training and validation IDs for one pair."""

    low, high = digits
    if not low < high:
        raise ValueError(f"Digits must be strictly increasing, got {digits}")
    all_pair = np.flatnonzero((targets == low) | (targets == high)).astype(np.int64)
    validation = np.sort(
        np.concatenate((validation_by_digit[low], validation_by_digit[high]))
    ).astype(np.int64, copy=False)
    posterior = np.setdiff1d(all_pair, validation, assume_unique=True)
    verify_candidate_partition(targets, digits, posterior, validation)
    return (
        np.ascontiguousarray(posterior, dtype=np.int64),
        np.ascontiguousarray(validation, dtype=np.int64),
    )


def verify_candidate_partition(
    targets: np.ndarray,
    digits: Tuple[int, int],
    posterior_ids: np.ndarray,
    validation_ids: np.ndarray,
) -> None:
    """Raise if IDs do not exactly partition the pair's training records."""

    low, high = digits
    all_pair = np.flatnonzero((targets == low) | (targets == high)).astype(np.int64)
    posterior_ids = np.asarray(posterior_ids, dtype=np.int64)
    validation_ids = np.asarray(validation_ids, dtype=np.int64)
    if np.unique(posterior_ids).size != posterior_ids.size:
        raise AssertionError("Duplicate posterior-training IDs")
    if np.unique(validation_ids).size != validation_ids.size:
        raise AssertionError("Duplicate validation IDs")
    if np.intersect1d(posterior_ids, validation_ids).size:
        raise AssertionError("Posterior-training and validation IDs overlap")
    reconstructed = np.sort(np.concatenate((posterior_ids, validation_ids)))
    if not np.array_equal(reconstructed, all_pair):
        raise AssertionError("Candidate IDs do not partition the official training records")
    labels = targets[validation_ids]
    if int(np.sum(labels == low)) != 500 or int(np.sum(labels == high)) != 500:
        raise AssertionError("Validation split must contain 500 records per digit")


def binary_labels(targets: np.ndarray, ids: np.ndarray, digits: Tuple[int, int]) -> np.ndarray:
    """Encode the smaller digit as 0 and the larger digit as 1."""

    low, high = digits
    selected = targets[ids]
    if not np.all((selected == low) | (selected == high)):
        raise AssertionError("IDs include a label outside the candidate pair")
    return np.ascontiguousarray(selected == high, dtype=np.int64)


def feature_matrix(images: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Construct float64 [intercept, pixels/255] features."""

    selected = images[ids]
    features = np.empty((ids.size, 785), dtype=np.float64)
    features[:, 0] = 1.0
    features[:, 1:] = selected.reshape(ids.size, 784).astype(np.float64) / 255.0
    if not np.all(features[:, 0] == 1.0):
        raise AssertionError("Intercept construction failed")
    pixels = features[:, 1:]
    if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
        raise AssertionError("Normalized pixels are outside [0, 1]")
    return np.ascontiguousarray(features)


def ndarray_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes under a versioned convention."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"ndarray-v1\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_label_hashes(
    images: np.ndarray,
    targets: np.ndarray,
    ids: np.ndarray,
    digits: Tuple[int, int],
) -> Dict[str, Any]:
    features = feature_matrix(images, ids)
    labels = binary_labels(targets, ids, digits)
    return {
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_sha256": ndarray_sha256(features),
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "label_sha256": ndarray_sha256(labels),
    }


def raw_file_records(raw_dir: Path) -> Sequence[Dict[str, Any]]:
    records = []
    for path in sorted(candidate for candidate in raw_dir.rglob("*") if candidate.is_file()):
        records.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not records:
        raise FileNotFoundError(f"No raw MNIST files found under {raw_dir}")
    return records


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def class_counts(targets: np.ndarray, ids: Iterable[int], digits: Tuple[int, int]) -> Dict[str, int]:
    selected = targets[np.asarray(ids, dtype=np.int64)]
    return {str(digit): int(np.sum(selected == digit)) for digit in digits}

