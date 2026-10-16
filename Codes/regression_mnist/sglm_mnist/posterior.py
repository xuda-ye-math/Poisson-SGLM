"""Bayesian logistic-regression posterior used by the MNIST experiment.

All public potential and derivative functions use the summed-likelihood
convention.  The Gaussian prior has covariance 0.001 I and precision 1000 I,
including the intercept coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import expit


PRIOR_VARIANCE = 0.001
PRIOR_PRECISION = 1.0 / PRIOR_VARIANCE


@dataclass(frozen=True)
class LogisticMinibatchOracle:
    """Validated-once production oracle for repeated sampler updates."""

    features: np.ndarray
    labels: np.ndarray
    precision: float = PRIOR_PRECISION

    def __post_init__(self) -> None:
        features, labels = _validate_data(self.features, self.labels)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        if not np.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be positive and finite")

    @property
    def n_records(self) -> int:
        return int(self.features.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])

    def __call__(self, theta: np.ndarray, indices: np.ndarray) -> np.ndarray:
        theta = _validate_theta(theta, self.dimension)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("indices must be a nonempty vector")
        if int(indices.min()) < 0 or int(indices.max()) >= self.n_records:
            raise IndexError("minibatch index outside the data range")
        batch_features = self.features[indices]
        residual = expit(batch_features @ theta) - self.labels[indices]
        scale = self.n_records / indices.size
        return np.ascontiguousarray(
            self.precision * theta + scale * (batch_features.T @ residual)
        )


def _validate_data(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have one entry per feature row")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain non-finite values")
    return np.ascontiguousarray(features), np.ascontiguousarray(labels)


def _validate_theta(theta: np.ndarray, dimension: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (dimension,):
        raise ValueError(f"theta must have shape ({dimension},), got {theta.shape}")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values")
    return theta


def logistic_losses(theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return per-record negative log likelihoods, excluding the prior."""

    features, labels = _validate_data(features, labels)
    theta = _validate_theta(theta, features.shape[1])
    logits = features @ theta
    return np.logaddexp(0.0, logits) - labels * logits


def full_potential(
    theta: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    precision: float = PRIOR_PRECISION,
) -> float:
    """Return U(theta) with the likelihood summed over observations."""

    theta = np.asarray(theta, dtype=np.float64)
    losses = logistic_losses(theta, features, labels)
    return float(0.5 * precision * np.dot(theta, theta) + np.sum(losses))


def full_gradient(
    theta: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    precision: float = PRIOR_PRECISION,
) -> np.ndarray:
    """Return the exact gradient of the summed potential."""

    features, labels = _validate_data(features, labels)
    theta = _validate_theta(theta, features.shape[1])
    residual = expit(features @ theta) - labels
    return np.ascontiguousarray(precision * theta + features.T @ residual)


def minibatch_gradient(
    theta: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    precision: float = PRIOR_PRECISION,
) -> np.ndarray:
    """Return the with-replacement unbiased gradient estimator.

    ``indices`` may contain duplicates.  The prior gradient is added exactly
    once, outside the likelihood rescaling.
    """

    features, labels = _validate_data(features, labels)
    theta = _validate_theta(theta, features.shape[1])
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("indices must be a nonempty vector")
    if int(indices.min()) < 0 or int(indices.max()) >= features.shape[0]:
        raise IndexError("minibatch index outside the data range")
    batch_features = features[indices]
    residual = expit(batch_features @ theta) - labels[indices]
    scale = features.shape[0] / indices.size
    return np.ascontiguousarray(
        precision * theta + scale * (batch_features.T @ residual)
    )


def hessian_vector_product(
    theta: np.ndarray,
    vector: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    precision: float = PRIOR_PRECISION,
) -> np.ndarray:
    """Return the exact Hessian-vector product of the summed potential."""

    features, labels = _validate_data(features, labels)
    theta = _validate_theta(theta, features.shape[1])
    vector = _validate_theta(vector, features.shape[1])
    probabilities = expit(features @ theta)
    weights = probabilities * (1.0 - probabilities)
    return np.ascontiguousarray(
        precision * vector + features.T @ (weights * (features @ vector))
    )


def full_hessian(
    theta: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    precision: float = PRIOR_PRECISION,
) -> np.ndarray:
    """Materialize the exact symmetric Hessian of the summed potential."""

    features, labels = _validate_data(features, labels)
    theta = _validate_theta(theta, features.shape[1])
    probabilities = expit(features @ theta)
    weights = probabilities * (1.0 - probabilities)
    hessian = (features.T * weights) @ features
    hessian.flat[:: hessian.shape[0] + 1] += precision
    return np.ascontiguousarray(0.5 * (hessian + hessian.T))
