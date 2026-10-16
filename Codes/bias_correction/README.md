# Section 8.3: bias correction of SGLM in a 16D product of ring distributions

Code for Section 8.3 of the paper *Stochastic Gradient Leimkuhler–Matthews Integrator with Second Order
Discretization Error* (X. Ye, Y. Wang, G. Lin, C. X. Kong): the numerical bias of SGLM at $h = 2^{-4}$ and the errors
of its three corrections (the explicit coefficient, the Richardson–Romberg extrapolation and the shifting
correction with a trained function $\varphi_{\mathcal C}$), in the two tables of the section (Gaussian and
non-Gaussian gradient noise).

All calculations are written in Julia, except the training of $\varphi_{\mathcal C}$ and the evaluation of its
gradient, which use JAX; Python also writes the tables from saved results.

## Environment

- **Julia 1.13.1** (installed with juliaup). The Julia scripts use the standard library only and are run without a
  project, `julia -t 32 SCRIPT.jl`.
- **Python**: `environment.yml` pins the versions used for the reported results (Python 3.14.8, NumPy 2.4.6,
  JAX 0.11.2 with its CUDA 13 plugin, Equinox 0.13.8). The launchers call `python`, the interpreter of this
  environment.
- **Hardware of the reported runs.** 32 CPU threads; one NVIDIA GPU for the training (peak memory about 5 GiB).

## Setting

- **Target.** $\pi \propto e^{-U}$ on $\mathbb R^{16}$, a rotated product of $8$ rings,
  $U(x) = \sum_{i=1}^{8} u_i(|y^{(i)}|)$ with $y = Qx$, $y^{(i)} = (y_{2i-1}, y_{2i})$ and
  $u_i(r) = r^2/2 + \beta_i e^{-r^2/2}$, a ring of minima at $r = \sqrt{2 \log \beta_i}$. The orthogonal matrix $Q$
  and $\beta_i \in [2, 4]$ are drawn from a fixed seed and stored in `model_d16.toml` (`model.jl` writes and reads
  it); exact samples of $\pi$ are available, so every chain starts from one.
- **Gradient noises**, both with $\sigma = 1$ and the covariance $\mathcal C = \sigma^2 I_{16}$:
  Gaussian, $\nabla U(x) + \sigma \xi$ with $\xi \sim \mathcal N(0, I_{16})$; non-Gaussian,
  $\nabla U(x) + \sigma R(x) \zeta$, with $\zeta_k = E_k - 1$ for independent $E_k \sim \mathrm{Exp}(1)$ and $R(x)$
  the rotation by $x_{2i-1} + x_{2i}$ in each plane $(x_{2i-1}, x_{2i})$. For both,
  $\psi_{\mathcal C} = \sigma^2 (U - \pi(U))$, so the exact coefficient of the numerical bias of a test function $f$
  is $\frac12 \sigma^2 \mathrm{Cov}_\pi(f, U)$.
- **Test functions.** The twelve functions $f_1, \dots, f_{12}$ of the paper, defined in `model.jl`.
- **Step sizes.** $h = 1/16$ and $2h = 1/8$.

## Workflow

Every command is run in this folder; `results/` is ignored by git. For `NOISE` in `gaussian nongaussian`:

1. **SGLM runs, exact coefficients and exact means**:

   ```bash
   bash run_production.sh NOISE
   ```

   - `sglm_runs.jl`: at each step size, 32 chains started from exact samples of $\pi$, a burn-in of time $100$,
     then the time $2 \times 10^7$, with the twelve test functions averaged over blocks of time $100$ (about
     3 minutes). Writes `results/sglm_NOISE_h8.csv`, `results/sglm_NOISE_h16.csv`,
     `results/extrapolation_NOISE.csv` and `results/data_NOISE/`.
   - `exact_coefficients.jl`: $\frac12 \sigma^2 \mathrm{Cov}_\pi(f, U)$ from as many exact samples of $\pi$ as the
     run at $h = 1/16$ has states (several minutes); `results/exact_coefficients_NOISE.csv`. The coefficients are
     the same for both noises, and `table.py` reads the Gaussian file, so a second argument other than `exact`
     (for instance `bash run_production.sh nongaussian skip`) skips them.
   - `exact_means.jl` (run once, if `results/exact_means.csv` is missing): the means $\pi(f)$ from exact samples;
     $\pi$ does not depend on the noise.
   - `table.py`: the table, without the shifting correction until step 2 has run.
2. **Shifting correction**:

   ```bash
   bash run_phi.sh NOISE [epochs=100]
   ```

   - `sglm_states.jl`: stored SGLM states at $h = 1/16$ and $h = 1/8$, one per unit of time after a burn-in of time
     $100$, from 32 chains for training and 32 independent chains for validation, $525 000$ states per chain,
     that is $1.68 \times 10^7$ states per set (`results/states_NOISE/`).
   - `train_phi.py` (GPU, about 2 minutes for 100 epochs): $\varphi_{\mathcal C}$ minimizing the extrapolated
     energy on the training states (see below); checkpoints in `results/checkpoint_phi_NOISE.eqx` and
     `results/snapshots_phi_NOISE/`.
   - `eval_phi.py`: $\nabla \varphi_{\mathcal C}$ at the validation states at $h = 1/16$
     (`results/grad_phi_NOISE.f32`).
   - `phi_coefficients.jl`: the coefficient implied by the shift and the biases of the corrections on the
     validation states (`results/phi_NOISE.csv`).
3. **Table of the paper**:

   ```bash
   python table.py NOISE
   ```

   writes `tables/table_NOISE.csv`, with the columns SGLM, explicit, extrapolation and shift.

## The four estimates of $\pi(f)$

Each column of `tables/table_NOISE.csv` is an estimate of $\pi(f)$ minus the exact mean of
`results/exact_means.csv`:

| column | estimate | data |
|---|---|---|
| SGLM | $\pi_{1/16}(f)$ | `results/sglm_NOISE_h16.csv` |
| explicit | $\pi_{1/16}(f) - \frac1{16} \cdot \frac12 \pi(f \psi_{\mathcal C})$ | exact coefficient, `results/exact_coefficients_gaussian.csv` |
| extrapolation | $2\pi_{1/16}(f) - \pi_{1/8}(f)$ | `results/sglm_NOISE_h8.csv` and `_h16.csv` |
| shift | $\pi_{1/16}(f) - \frac1{16} c_\varphi(f)$, $c_\varphi(f) = -(\pi_h(f \circ T) - \pi_h(f))/h$, $T(x) = x + h \nabla \varphi_{\mathcal C}(x)$ | `results/phi_NOISE.csv` |

The exact means and the exact coefficients use exact samples of $\pi$, which the corrections themselves do not
use: they serve only as the reference. The column shift adds the change caused by the shift, computed on the
validation states, to the time average of the production run at $h = 1/16$.

## Training of $\varphi_{\mathcal C}$

`train_phi.py` replaces the energy $\frac12 \pi(|\nabla\varphi|^2) + \frac12 \pi(\psi_{\mathcal C}\varphi)$ by the
extrapolated energy

$$
\mathcal E_h(\varphi) = \frac12 \pi_h\bigl(|\nabla\varphi|^2\bigr) + \frac{\pi_{2h}(\varphi) - \pi_h(\varphi)}{h},
\qquad h = 1/16,
$$

with $\pi_h$ and $\pi_{2h}$ the empirical measures of the stored training states at $h = 1/16$ and $h = 1/8$; it uses
neither $U$, $\nabla U$ nor $\mathcal C$. The network is an MLP $16 \to 64 \times 3 \to 1$ with tanh activations.
Every step takes a batch of $20 000$ states from each set, and every epoch shuffles both sets and goes through
them once ($840$ steps per epoch); Adam, written out in the script, uses a cosine learning rate from $10^{-3}$ to
$10^{-5}$ over $100$ epochs. After every epoch the log `results/train_phi_NOISE.log` gives, on the validation
states, the Dirichlet term $D = \frac12 \pi_h(|\nabla\varphi|^2)$, the extrapolation term
$L = (\pi_{2h}(\varphi) - \pi_h(\varphi))/h$, their sum, the ratio $-L/(2D)$ (equal to $1$ at a minimizer along the
scalings of $\varphi$) and the root mean square shift $h (2D)^{1/2}$. `--resume` continues from the last
checkpoint.

`phi_coefficients.jl` writes, with the standard errors over the 32 chains of each set:

| quantity | definition |
|---|---|
| `data_train` | $(\pi_{2h}(f) - \pi_h(f))/h$ on the training states |
| `data_valid` | the same on the validation states |
| `implied` | $-(\pi_h(f \circ T) - \pi_h(f))/h$ on the validation states at $h$ |
| `bias_sglm` | $\pi_h(f) - \pi(f)$ on the validation states |
| `bias_richardson` | $2\pi_h(f) - \pi_{2h}(f) - \pi(f)$ on the validation states |
| `bias_shift` | $\pi_h(f \circ T) - \pi(f)$ on the validation states at $h$ |

## Files

- `model.jl`, `model_d16.toml`: the target, the noises, the exact sampler and the twelve test functions.
- `sglm_runs.jl`, `exact_coefficients.jl`, `exact_means.jl`, `run_production.sh`: step 1.
- `sglm_states.jl`, `train_phi.py`, `eval_phi.py`, `phi_coefficients.jl`, `run_phi.sh`: step 2.
- `table.py`, `tables/table_gaussian.csv`, `tables/table_nongaussian.csv`: step 3 and the tables of the paper.
