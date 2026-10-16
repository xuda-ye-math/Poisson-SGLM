import numpy as np
import torch

from sglm_mnist.observables import torch_pool
from sglm_mnist.posterior import full_gradient, minibatch_gradient
from sglm_mnist.samplers import TorchPairedState, torch_paired_transition
from sglm_mnist.seeds import derive_seed
from sglm_mnist.settings import FUNCTION_NAMES, PREFIXES, SNAPSHOT_INDICES


def test_fixed_design_dimensions():
    assert len(FUNCTION_NAMES) == 16
    assert len(PREFIXES) == 16
    assert PREFIXES[0] == 8192 and PREFIXES[-1] == 131072
    assert len(SNAPSHOT_INDICES) == 256
    assert SNAPSHOT_INDICES[0] == 0 and SNAPSHOT_INDICES[-1] == 130560


def test_observables_equal_one_at_map():
    theta_map = torch.zeros(10, dtype=torch.float64)
    theta = torch.stack((theta_map, theta_map))
    eigenvalues = torch.arange(1, 9, dtype=torch.float64)
    eigenvectors = torch.eye(10, dtype=torch.float64)[:8]
    values = torch_pool(theta, theta_map, eigenvalues, eigenvectors)
    assert values.shape == (2, 16)
    torch.testing.assert_close(values, torch.ones_like(values))


def test_paired_transition_equations():
    ld = torch.tensor([1.0, -1.0], dtype=torch.float64)
    lm = torch.tensor([0.5, -0.5], dtype=torch.float64)
    previous = torch.tensor([0.2, -0.3], dtype=torch.float64)
    fresh = torch.tensor([-0.4, 0.7], dtype=torch.float64)
    grad_ld = torch.tensor([2.0, 3.0], dtype=torch.float64)
    grad_lm = torch.tensor([-1.0, 4.0], dtype=torch.float64)
    h = 0.01
    state = TorchPairedState(ld, lm, previous, 0)
    result = torch_paired_transition(state, grad_ld, grad_lm, fresh, h)
    torch.testing.assert_close(result.ld, ld - h * grad_ld + np.sqrt(2 * h) * fresh)
    torch.testing.assert_close(result.lm, lm - h * grad_lm + np.sqrt(h / 2) * (previous + fresh))
    assert result.completed_transitions == 1


def test_minibatch_equals_full_gradient_for_full_index_set():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 3))
    y = np.asarray([0, 1, 0, 1, 1, 0, 1])
    theta = rng.normal(size=3)
    indices = np.arange(7)
    np.testing.assert_allclose(
        minibatch_gradient(theta, x, y, indices),
        full_gradient(theta, x, y),
        rtol=1e-14,
        atol=1e-14,
    )


def test_seed_derivation_is_stable():
    seed_a, record_a = derive_seed("formal_sampling_oracle", "4_9", "batch_1024", "rep_0")
    seed_b, record_b = derive_seed("formal_sampling_oracle", "4_9", "batch_1024", "rep_0")
    assert seed_a == seed_b
    assert record_a["sha256"] == record_b["sha256"]
