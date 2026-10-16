"""Hessian-whitened coordinates and Pyro potential for NUTS references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch
from scipy.linalg import solve_triangular

from .posterior import PRIOR_PRECISION, full_gradient, full_potential


@dataclass(frozen=True)
class ReferenceTransform:
    """The transform theta = theta_map + C^{-T} z for H*=C C^T."""

    theta_map: np.ndarray
    cholesky_lower: np.ndarray

    @classmethod
    def from_hessian(
        cls, theta_map: np.ndarray, hessian: np.ndarray
    ) -> "ReferenceTransform":
        theta_map = np.ascontiguousarray(theta_map, dtype=np.float64)
        hessian = np.asarray(hessian, dtype=np.float64)
        if hessian.shape != (theta_map.size, theta_map.size):
            raise ValueError("Hessian shape does not match theta_map")
        if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-10):
            raise ValueError("Hessian must be symmetric")
        cholesky = np.linalg.cholesky(hessian)
        return cls(theta_map, np.ascontiguousarray(cholesky))

    @property
    def dimension(self) -> int:
        return int(self.theta_map.size)

    def theta_from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.dimension,):
            raise ValueError("z has the wrong shape")
        displacement = solve_triangular(
            self.cholesky_lower.T, z, lower=False, check_finite=True
        )
        return np.ascontiguousarray(self.theta_map + displacement)

    def theta_from_z_batch(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError("z batch has the wrong shape")
        displacement_transpose = solve_triangular(
            self.cholesky_lower.T, z.T, lower=False, check_finite=True
        )
        return np.ascontiguousarray(self.theta_map[None, :] + displacement_transpose.T)

    def z_from_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.dimension,):
            raise ValueError("theta has the wrong shape")
        return np.ascontiguousarray(self.cholesky_lower.T @ (theta - self.theta_map))

    def transformed_potential(
        self,
        z: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        precision: float = PRIOR_PRECISION,
    ) -> float:
        """Return U(theta(z)); the constant log-Jacobian is omitted."""

        return full_potential(
            self.theta_from_z(z), features, labels, precision=precision
        )

    def transformed_gradient(
        self,
        z: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        precision: float = PRIOR_PRECISION,
    ) -> np.ndarray:
        gradient_theta = full_gradient(
            self.theta_from_z(z), features, labels, precision=precision
        )
        return np.ascontiguousarray(
            solve_triangular(
                self.cholesky_lower,
                gradient_theta,
                lower=True,
                check_finite=True,
            )
        )


def torch_theta_from_z(
    z: torch.Tensor, theta_map: torch.Tensor, cholesky_lower: torch.Tensor
) -> torch.Tensor:
    """Torch transform supporting either one vector or a leading sample axis."""

    if z.ndim == 1:
        displacement = torch.linalg.solve_triangular(
            cholesky_lower.T, z.unsqueeze(-1), upper=True
        ).squeeze(-1)
        return theta_map + displacement
    if z.ndim == 2:
        displacement = torch.linalg.solve_triangular(
            cholesky_lower.T, z.T, upper=True
        ).T
        return theta_map.unsqueeze(0) + displacement
    raise ValueError("z must be a vector or matrix")


class TorchTransformedLogisticPotential:
    """Callable Pyro potential with a full-data gradient-evaluation counter."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        theta_map: torch.Tensor,
        cholesky_lower: torch.Tensor,
        *,
        precision: float = PRIOR_PRECISION,
    ) -> None:
        if features.dtype != torch.float64 or labels.dtype != torch.float64:
            raise TypeError("reference features and labels must use torch.float64")
        if features.device != labels.device:
            raise ValueError("reference tensors must share a device")
        self.features = features
        self.labels = labels
        self.theta_map = theta_map
        self.cholesky_lower = cholesky_lower
        self.precision = float(precision)
        self.evaluation_count = 0

    def __call__(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(params) != {"z"}:
            raise ValueError("transformed reference expects exactly one site named z")
        self.evaluation_count += 1
        theta = torch_theta_from_z(
            params["z"], self.theta_map, self.cholesky_lower
        )
        logits = self.features @ theta
        return 0.5 * self.precision * torch.dot(theta, theta) + (
            torch.nn.functional.softplus(logits) - self.labels * logits
        ).sum()

    def tensor_identity(self) -> Dict[str, object]:
        return {
            "device": str(self.features.device),
            "dtype": str(self.features.dtype),
            "records": int(self.features.shape[0]),
            "dimension": int(self.features.shape[1]),
            "precision": self.precision,
        }

