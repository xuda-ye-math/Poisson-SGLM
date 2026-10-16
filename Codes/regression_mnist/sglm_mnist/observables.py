"""The 16 fixed MAP-Hessian observables used in the MNIST experiment."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


MODE_COUNT = 8
POOL_NAMES = tuple(
    [f"cos_z{j}" for j in range(1, 9)]
    + [f"bump1d_z{j}" for j in range(1, 5)]
    + ["bump2d_z1_z2", "bump2d_z3_z4", "bump2d_z5_z6", "bump2d_z7_z8"]
)
POOL_FAMILIES = {
    **{f"cos_z{j}": "cosine" for j in range(1, 9)},
    **{f"bump1d_z{j}": "gaussian_bump_1d" for j in range(1, 5)},
    **{f"bump2d_z{a}_z{b}": "gaussian_bump_2d" for a, b in ((1, 2), (3, 4), (5, 6), (7, 8))},
}
POOL_MODES = {
    **{f"cos_z{j}": (j,) for j in range(1, 9)},
    **{f"bump1d_z{j}": (j,) for j in range(1, 5)},
    **{f"bump2d_z{a}_z{b}": (a, b) for a, b in ((1, 2), (3, 4), (5, 6), (7, 8))},
}


def orient_eigenvector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).copy()
    index = int(np.argmax(np.abs(value)))
    if value[index] < 0.0:
        value *= -1.0
    return np.ascontiguousarray(value)


def top_eigenpairs(hessian: np.ndarray, count: int = MODE_COUNT) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(hessian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not 1 <= count <= matrix.shape[0]:
        raise ValueError("invalid Hessian or eigenpair count")
    values, vectors = np.linalg.eigh(matrix)
    eigenvalues = np.ascontiguousarray(values[-count:][::-1])
    if np.any(eigenvalues <= 0.0):
        raise ValueError("selected Hessian eigenvalues must be positive")
    eigenvectors = np.stack([orient_eigenvector(vectors[:, -(j + 1)]) for j in range(count)])
    return eigenvalues, np.ascontiguousarray(eigenvectors)


def numpy_pool(theta: np.ndarray, theta_map: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    delta = np.asarray(theta, dtype=np.float64) - np.asarray(theta_map, dtype=np.float64)
    z = (delta @ np.asarray(eigenvectors, dtype=np.float64).T) * np.sqrt(np.asarray(eigenvalues, dtype=np.float64))
    outputs = [np.cos(z[..., j]) for j in range(8)]
    outputs.extend(np.exp(-0.5 * z[..., j] ** 2) for j in range(4))
    outputs.extend(np.exp(-0.5 * (z[..., a] ** 2 + z[..., b] ** 2)) for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)))
    return np.ascontiguousarray(np.stack(outputs, axis=-1))


def torch_pool(theta: torch.Tensor, theta_map: torch.Tensor, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    z = ((theta - theta_map) @ eigenvectors.T) * torch.sqrt(eigenvalues)
    outputs = [torch.cos(z[..., j]) for j in range(8)]
    outputs.extend(torch.exp(-0.5 * z[..., j] ** 2) for j in range(4))
    outputs.extend(torch.exp(-0.5 * (z[..., a] ** 2 + z[..., b] ** 2)) for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)))
    return torch.stack(outputs, dim=-1)
