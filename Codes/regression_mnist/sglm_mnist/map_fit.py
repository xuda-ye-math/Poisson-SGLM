"""MAP fitting and deterministic curvature calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from .posterior import PRIOR_PRECISION, full_gradient, full_hessian, full_potential


@dataclass(frozen=True)
class MapFitResult:
    theta: np.ndarray
    success: bool
    scipy_success: bool
    message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    full_potential: float
    full_gradient_inf_norm: float
    scaled_gradient_inf_norm: float
    target_full_gradient_inf_norm: float
    optimizer: str
    options: Dict[str, Any]

    def metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("theta")
        return payload


def fit_map(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    precision: float = PRIOR_PRECISION,
    target_residual: float = 1e-5,
    maxiter: int = 2000,
    maxcor: int = 50,
) -> MapFitResult:
    """Fit from zero with full-data L-BFGS-B on U/n.

    ``ftol=0`` prevents an objective-change stop from pre-empting the declared
    gradient criterion.  The reported residual is always the unscaled one.
    """

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n, dimension = features.shape
    initial = np.zeros(dimension, dtype=np.float64)

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        return (
            full_potential(theta, features, labels, precision=precision) / n,
            full_gradient(theta, features, labels, precision=precision) / n,
        )

    options: Dict[str, Any] = {
        "maxiter": int(maxiter),
        "maxcor": int(maxcor),
        "maxls": 100,
        "gtol": float(target_residual / n),
        "ftol": 0.0,
        "maxfun": int(maxiter * 110),
    }
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options=options,
    )
    theta = np.ascontiguousarray(result.x, dtype=np.float64)
    gradient = full_gradient(theta, features, labels, precision=precision)
    residual = float(np.linalg.norm(gradient, ord=np.inf))
    finite = bool(np.all(np.isfinite(theta)) and np.isfinite(result.fun) and np.isfinite(residual))
    return MapFitResult(
        theta=theta,
        success=finite and residual <= target_residual,
        scipy_success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        gradient_evaluations=int(result.njev),
        full_potential=full_potential(theta, features, labels, precision=precision),
        full_gradient_inf_norm=residual,
        scaled_gradient_inf_norm=residual / n,
        target_full_gradient_inf_norm=float(target_residual),
        optimizer="scipy.optimize.minimize:L-BFGS-B",
        options=options,
    )


def canonicalize_eigenvector_sign(vector: np.ndarray) -> Tuple[np.ndarray, int]:
    """Make the lowest-index largest-absolute coordinate positive."""

    vector = np.asarray(vector, dtype=np.float64).copy()
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector must be nonempty and one-dimensional")
    maximum = np.max(np.abs(vector))
    pivot = int(np.flatnonzero(np.abs(vector) == maximum)[0])
    if vector[pivot] < 0.0:
        vector *= -1.0
    return np.ascontiguousarray(vector), pivot


def extreme_eigenpairs(matrix: np.ndarray) -> Dict[str, Any]:
    """Return deterministic smallest/largest symmetric eigenpairs."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    dimension = matrix.shape[0]
    values, vectors = eigh(
        matrix,
        subset_by_index=(0, dimension - 1),
        driver="evr",
        check_finite=True,
        overwrite_a=False,
    )
    largest_vector, pivot = canonicalize_eigenvector_sign(vectors[:, -1])
    largest_value = float(values[-1])
    smallest_value = float(values[0])
    residual = float(
        np.linalg.norm(matrix @ largest_vector - largest_value * largest_vector)
        / (1.0 + abs(largest_value))
    )
    return {
        "smallest_value": smallest_value,
        "largest_value": largest_value,
        "largest_vector": largest_vector,
        "largest_vector_sign_pivot": pivot,
        "largest_residual_relative": residual,
        "condition_number": largest_value / smallest_value,
        "solver": "scipy.linalg.eigh",
        "driver": "evr",
        "subset": "full_spectrum_for_extremes",
    }


def curvature_at_map(
    theta_map: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    precision: float = PRIOR_PRECISION,
) -> Dict[str, Any]:
    """Compute H*, its extreme eigenvalues, and the global bound L."""

    hessian = full_hessian(theta_map, features, labels, precision=precision)
    hessian_eigen = extreme_eigenpairs(hessian)
    gram = np.ascontiguousarray(features.T @ features)
    gram_max = float(
        eigh(
            gram,
            subset_by_index=(gram.shape[0] - 1, gram.shape[0] - 1),
            driver="evr",
            check_finite=True,
            eigvals_only=True,
        )[0]
    )
    return {
        "hessian": hessian,
        "kappa": hessian_eigen["largest_value"],
        "v": hessian_eigen["largest_vector"],
        "hessian_min_eigenvalue": hessian_eigen["smallest_value"],
        "hessian_condition_number": hessian_eigen["condition_number"],
        "eigen_residual_relative": hessian_eigen["largest_residual_relative"],
        "eigenvector_sign_pivot": hessian_eigen["largest_vector_sign_pivot"],
        "gram_max_eigenvalue": gram_max,
        "L": float(precision + 0.25 * gram_max),
        "solver": hessian_eigen["solver"],
        "solver_driver": hessian_eigen["driver"],
    }

