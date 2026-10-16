"""Train phi_C from the extrapolated energy with JAX + Equinox, on the SGLM states written by sglm_states.jl.

    python -u train_phi.py [noise=gaussian] [epochs=100] [--resume]   > results/train_phi_NOISE.log

The energy 1/2 pi(|grad phi|^2) + 1/2 pi(psi_C phi) is replaced by
    E_h(phi) = 1/2 pi_h(|grad phi|^2) + (pi_{2h}(phi) - pi_h(phi)) / h,    h = 1/16,
with pi_h and pi_{2h} the empirical measures of the stored states at h = 1/16 and h = 1/8; it uses neither U, grad U nor
the covariance C. Every step takes a batch of 20 000 states from each set; every epoch shuffles both sets and goes
through them, so each state is used once per epoch. Adam (written out below) with a cosine learning rate from 1e-3 to
1e-5 over all steps; the network is an MLP 16 -> 64 x 3 -> 1 (tanh).
After every epoch the log gives, on the validation sets (independent chains), the Dirichlet term
D = 1/2 pi_h(|grad phi|^2), the extrapolation term L = (pi_{2h}(phi) - pi_h(phi)) / h, their sum, the ratio -L / (2D)
(1 at a minimizer along the scalings of phi) and the root mean square shift h (2D)^(1/2). A checkpoint (model, Adam
state, epoch) is written to results/checkpoint_phi_NOISE.eqx with .json after every epoch, --resume continues from it,
and a copy of the model is kept every 10 epochs in results/snapshots_phi_NOISE/.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # allocate GPU memory on demand, not 75% upfront

import json
import sys
import time
import tomllib
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
NOISE = ARGS[0] if len(ARGS) > 0 else "gaussian"
EPOCHS = int(ARGS[1]) if len(ARGS) > 1 else 100
RESUME = "--resume" in sys.argv
BATCH = 20_000
WIDTH, DEPTH = 64, 3
LR0, LR1 = 1e-3, 1e-5
B1, B2, EPS = 0.9, 0.999, 1e-8
REPORT = 30.0
SNAP_EVERY = 10
CHUNK = 2**18
DATA = HERE / "results" / f"states_{NOISE}"
CKPT = HERE / "results" / f"checkpoint_phi_{NOISE}.eqx"
CKPT_META = HERE / "results" / f"checkpoint_phi_{NOISE}.json"
SNAPS = HERE / "results" / f"snapshots_phi_{NOISE}"


def log(s):
    print(s, flush=True)


meta = tomllib.loads((DATA / "meta.toml").read_text())
D, N, H = meta["d"], meta["n"], float(meta["h"])
assert abs(float(meta["h_coarse"]) - 2 * H) < 1e-12


def load(name):
    return jnp.asarray(np.fromfile(DATA / name, dtype=np.float32).reshape(N, D))


XH, X2H = load("train_h16.f32"), load("train_h8.f32")
VH, V2H = load("valid_h16.f32"), load("valid_h8.f32")
assert N % BATCH == 0, "every state must lie in a batch"
NB = N // BATCH
TOTAL = EPOCHS * NB
log(f"noise = {NOISE}, d = {D}, {N} states at h = 1/{round(1 / H)} and {N} at h = 1/{round(1 / (2 * H))} (training), "
    f"as many of each for validation, batch {BATCH} + {BATCH}, {NB} steps per epoch, {EPOCHS} epochs, "
    f"MLP {D} -> {WIDTH} x {DEPTH} -> 1 (tanh), device {jax.devices()[0]}")


def dirichlet(model, X):
    g = jax.vmap(jax.grad(model))(X)
    return 0.5 * jnp.mean(jnp.sum(g * g, axis=1))


def energy(model, Xh, X2h):
    return dirichlet(model, Xh) + (jnp.mean(jax.vmap(model)(X2h)) - jnp.mean(jax.vmap(model)(Xh))) / H


@eqx.filter_jit
def train_step(model, m, v, t, Xh, X2h, lr):
    loss, grads = eqx.filter_value_and_grad(energy)(model, Xh, X2h)
    g = eqx.filter(grads, eqx.is_array)
    t = t + 1
    m = jax.tree.map(lambda a, b: B1 * a + (1 - B1) * b, m, g)
    v = jax.tree.map(lambda a, b: B2 * a + (1 - B2) * b * b, v, g)
    c1, c2 = 1 - B1 ** t, 1 - B2 ** t
    upd = jax.tree.map(lambda a, b: -lr * (a / c1) / (jnp.sqrt(b / c2) + EPS), m, v)
    return eqx.apply_updates(model, upd), m, v, t, loss


@eqx.filter_jit
def chunk_terms(model, X):
    """sum of 1/2 |grad phi|^2 and sum of phi over the chunk"""
    g = jax.vmap(jax.grad(model))(X)
    return 0.5 * jnp.sum(g * g), jnp.sum(jax.vmap(model)(X))


def validation(model):
    d_sum, p_h, p_2h = 0.0, 0.0, 0.0
    for i in range(0, N, CHUNK):
        a, b = chunk_terms(model, VH[i:i + CHUNK])
        d_sum, p_h = d_sum + float(a), p_h + float(b)
        p_2h += float(chunk_terms(model, V2H[i:i + CHUNK])[1])
    Dv = d_sum / N
    Lv = (p_2h - p_h) / N / H
    return Dv, Lv


def lr_at(step):
    return LR1 + (LR0 - LR1) * (1 + np.cos(np.pi * step / TOTAL)) / 2


def gpu_memory():
    st = jax.devices()[0].memory_stats() or {}
    return f"GPU in use {st.get('bytes_in_use', 0) / 2**30:.2f} GiB, peak {st.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB"


model = eqx.nn.MLP(D, "scalar", WIDTH, DEPTH, activation=jnp.tanh, key=jax.random.PRNGKey(2026))
m = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_array))
v = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_array))
t = jnp.array(0, dtype=jnp.int32)
start = 0
if RESUME and CKPT.exists():
    model, m, v = eqx.tree_deserialise_leaves(CKPT, (model, m, v))
    info = json.loads(CKPT_META.read_text())
    start, t = info["epochs_done"], jnp.array(info["t"], dtype=jnp.int32)
    log(f"resumed from {CKPT} after epoch {start} (step {info['t']})")
elif RESUME:
    log(f"no checkpoint at {CKPT}; training from the start")
SNAPS.mkdir(exist_ok=True)

key = jax.random.PRNGKey(7)
t_start = last = time.time()
for epoch in range(start, EPOCHS):
    t_ep = time.time()
    perm_h = jax.random.permutation(jax.random.fold_in(key, 2 * epoch), N)
    perm_2h = jax.random.permutation(jax.random.fold_in(key, 2 * epoch + 1), N)
    loss_sum = jnp.array(0.0)
    for b in range(NB):
        lr = jnp.array(lr_at(epoch * NB + b), dtype=jnp.float32)
        model, m, v, t, loss = train_step(model, m, v, t, XH[perm_h[b * BATCH:(b + 1) * BATCH]],
                                          X2H[perm_2h[b * BATCH:(b + 1) * BATCH]], lr)
        loss_sum = loss_sum + loss
        if time.time() - last > REPORT:
            last = time.time()
            log(f"    epoch {epoch + 1}: batch {b + 1}/{NB}, energy {float(loss):+.5f}, lr {float(lr):.1e}, "
                f"{last - t_start:.0f} s")
    mean_loss = float(loss_sum) / NB
    epoch_time = time.time() - t_ep
    eqx.tree_serialise_leaves(CKPT, (model, m, v))
    CKPT_META.write_text(json.dumps({"epochs_done": epoch + 1, "t": int(t), "epochs": EPOCHS, "batch": BATCH}))
    if (epoch + 1) % SNAP_EVERY == 0:
        eqx.tree_serialise_leaves(SNAPS / f"epoch_{epoch + 1:03d}.eqx", model)
    Dv, Lv = validation(model)
    log(f"  epoch {epoch + 1}/{EPOCHS}: {epoch_time:.1f} s, mean energy {mean_loss:+.5f}; validation: energy "
        f"{Dv + Lv:+.5f}, D {Dv:.5f}, L {Lv:+.5f}, -L/(2D) {-Lv / (2 * Dv):.4f}, rms shift {H * np.sqrt(2 * Dv):.5f}; "
        f"lr {lr_at((epoch + 1) * NB):.1e}; {gpu_memory()}")
