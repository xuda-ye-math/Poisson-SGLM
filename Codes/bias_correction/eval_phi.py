"""Evaluate grad phi of the trained network (checkpoint of train_phi.py, or a snapshot) at the validation states
valid_h16.f32 of sglm_states.jl, which the training did not use.

    python -u eval_phi.py [noise=gaussian] [epoch]

Without epoch the final checkpoint is used and results/grad_phi_NOISE.f32 is written; with epoch the snapshot
results/snapshots_phi_NOISE/epoch_EEE.eqx is used and results/grad_phi_NOISE_eEEE.f32 is written. The file holds
grad phi(X_n) (n x d, row = sample) in the order of valid_h16.f32; phi_coefficients.jl reads it.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import tomllib
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
NOISE = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
EPOCH = int(sys.argv[2]) if len(sys.argv) > 2 else None
DATA = HERE / "results" / f"states_{NOISE}"
WIDTH, DEPTH = 64, 3                      # as in train_phi.py

meta = tomllib.loads((DATA / "meta.toml").read_text())
D, N = meta["d"], meta["n"]
model = eqx.nn.MLP(D, "scalar", WIDTH, DEPTH, activation=jnp.tanh, key=jax.random.PRNGKey(2026))
if EPOCH is None:
    src = HERE / "results" / f"checkpoint_phi_{NOISE}.eqx"
    m = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_array))
    model, _, _ = eqx.tree_deserialise_leaves(src, (model, m, m))
    out = HERE / "results" / f"grad_phi_{NOISE}.f32"
else:
    src = HERE / "results" / f"snapshots_phi_{NOISE}" / f"epoch_{EPOCH:03d}.eqx"
    model = eqx.tree_deserialise_leaves(src, model)
    out = HERE / "results" / f"grad_phi_{NOISE}_e{EPOCH:03d}.f32"

grad_batch = eqx.filter_jit(lambda mdl, X: jax.vmap(jax.grad(mdl))(X))
X = np.fromfile(DATA / "valid_h16.f32", dtype=np.float32).reshape(N, D)
G = np.concatenate([np.asarray(grad_batch(model, jnp.asarray(X[i:i + 2**18]))) for i in range(0, N, 2**18)])
G.astype(np.float32).tofile(out)
print(f"grad phi at {N} validation states (h = 1/16) from {src.name}: rms |grad phi| "
      f"{np.sqrt(np.mean(np.sum(G.astype(np.float64)**2, axis=1))):.5f}; wrote {out}", flush=True)
