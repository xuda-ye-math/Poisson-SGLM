"""Torch float64 kernels for production paired SGLD/SGLM runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class TorchPairedState:
    ld: torch.Tensor
    lm: torch.Tensor
    chi_prev: torch.Tensor
    completed_transitions: int = 0


class TorchLogisticMinibatchOracle:
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        precision: float = 1000.0,
    ) -> None:
        if features.ndim != 2 or labels.shape != (features.shape[0],):
            raise ValueError("features and labels have incompatible shapes")
        if features.dtype != torch.float64 or labels.dtype != torch.float64:
            raise TypeError("oracle tensors must be float64")
        if features.device != labels.device:
            raise ValueError("oracle tensors must share a device")
        self.features = features
        self.labels = labels
        self.precision = float(precision)
        self.n_records = int(features.shape[0])
        self.dimension = int(features.shape[1])

    def __call__(self, theta: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if theta.shape != (self.dimension,):
            raise ValueError("theta has the wrong dimension")
        if indices.ndim != 1 or indices.numel() == 0 or indices.dtype != torch.int64:
            raise ValueError("indices must be a nonempty int64 vector")
        batch_features = self.features.index_select(0, indices)
        batch_labels = self.labels.index_select(0, indices)
        residual = torch.sigmoid(batch_features @ theta) - batch_labels
        return self.precision * theta + (self.n_records / indices.numel()) * (
            batch_features.T @ residual
        )

    def full_potential(self, theta: torch.Tensor) -> torch.Tensor:
        logits = self.features @ theta
        return 0.5 * self.precision * torch.dot(theta, theta) + (
            torch.nn.functional.softplus(logits) - self.labels * logits
        ).sum()


def torch_paired_transition(
    state: TorchPairedState,
    gradient_ld: torch.Tensor,
    gradient_lm: torch.Tensor,
    chi_new: torch.Tensor,
    step_size: float,
) -> TorchPairedState:
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    coefficient_ld = torch.sqrt(
        torch.tensor(2.0 * step_size, dtype=state.ld.dtype, device=state.ld.device)
    )
    coefficient_lm = torch.sqrt(
        torch.tensor(step_size / 2.0, dtype=state.ld.dtype, device=state.ld.device)
    )
    ld = state.ld - step_size * gradient_ld + coefficient_ld * chi_new
    lm = state.lm - step_size * gradient_lm + coefficient_lm * (
        state.chi_prev + chi_new
    )
    return TorchPairedState(ld, lm, chi_new.clone(), state.completed_transitions + 1)


def torch_observables(
    theta: torch.Tensor,
    theta_map: torch.Tensor,
    eigenvector: torch.Tensor,
    kappa: float,
    selected_feature: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    f_c = torch.cos(
        torch.sqrt(torch.tensor(kappa, dtype=theta.dtype, device=theta.device))
        * torch.dot(eigenvector, theta - theta_map)
    )
    f_p = torch.sigmoid(torch.dot(selected_feature, theta))
    return f_c, f_p

