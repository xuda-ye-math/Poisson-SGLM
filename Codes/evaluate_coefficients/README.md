# Section 8.1: coefficients in a 4D Gaussian mixture distribution

Code for Section 8.1 of the paper *Stochastic Gradient Leimkuhler–Matthews Integrator with Second Order
Discretization Error* (X. Ye, Y. Wang, G. Lin, C. X. Kong): the evaluated and exact coefficients of the numerical
bias of SGLD and SGLM in the two figures of the section (Gaussian and non-Gaussian gradient noise), and the exact
coefficients of the Wasserstein-1 errors in its table.

All calculations are written in Julia; Python only reads saved results to make the tables and figures.

## Environment

- **Julia 1.13.1** (installed with juliaup). `Project.toml` and `Manifest.toml` pin CUDA.jl 6.4.2, which only
  `w1_coefficients.jl` needs (run with `--project=.`); the other Julia scripts use the standard library only.
- **Python**, for the tables and figures: `environment.yml` pins the versions used for the reported results
  (Python 3.14.8, NumPy 2.4.6, Matplotlib 3.11.2). Below, `python` is the interpreter of this environment, as in
  `w1_run.sh`.
- **Hardware of the reported runs.** 16 CPU threads for the simulations and the quadrature; one NVIDIA RTX 5090
  (about 8 GB of GPU memory on the finest grid) for the Kantorovich–Rubinstein norms.

## Workflow

Every command is run in this folder; `results/` is ignored by git.

1. **Model.** `model.toml` holds the mixture, the two gradient noises and the pool of test functions, and its
   `selected` list fixes the four test functions $f_1, \dots, f_4$ of the paper. It was written once by
   `julia make_model.jl`, which is kept as the record of how the numbers were drawn.
2. **Exact bias coefficients**, by quadrature (seconds):

   ```bash
   julia -t 16 coefficients.jl > results/coefficients.log 2>&1
   ```

   writes `results/coefficients.csv`, `results/pool_coefficients.csv`, `results/quadrature_convergence.csv` and
   `results/riesz_checks.csv`.
3. **Simulations.** For `SCHEME` in `SGLD SGLM`, `NOISE` in `gaussian nongaussian` and `HEXP` in `1 2 3 4 5`
   ($h = 2^{-\mathrm{HEXP}}$):

   ```bash
   julia -t 16 simulate.jl SCHEME NOISE HEXP out=results/runs_T5e7 T=5e7 block=1e6 seed=1
   ```

   Each run has 16 chains started from exact samples of $\pi$, a burn-in of time $200$, then the time
   $5 \times 10^7$ per chain; it takes from under a minute at $h = 2^{-1}$ to about 11 minutes at $h = 2^{-5}$.
   The block sums of all test functions of the pool are saved, so the selection of test functions can change
   without a rerun.
4. **Evaluated coefficients** from the saved block sums:

   ```bash
   python analyze.py results/runs_T5e7
   ```

   writes `results/runs_T5e7/estimates.csv`.
5. **Figures of the bias coefficients**:

   ```bash
   python plot_bias.py results/runs_T5e7 gaussian
   python plot_bias.py results/runs_T5e7 nongaussian
   ```

   write `figures/bias_gaussian.{pdf,png}` and `figures/bias_nongaussian.{pdf,png}`.
6. **Wasserstein-1 coefficients** (GPU):

   ```bash
   bash w1_run.sh test_gaussian SGLD_gaussian SGLM_gaussian SGLD_nongaussian SGLM_nongaussian
   ```

   runs `w1_coefficients.jl` on the grids $25^4$, $50^4$, $100^4$ for each case and then `w1_table.py`, which writes
   `tables/w1_coefficients_gaussian.csv` and `tables/w1_coefficients_nongaussian.csv`, the table of the paper.

The rest of this file explains how the coefficients are computed and checked.

## Setting

- **Target.** $\pi=\sum_{k=1}^5 w_k\,\mathcal N(\mu_k,\Sigma_k)$ on $\mathbb R^4$ and $U=-\log\pi$, with the
  weights $0.34,0.25,0.19,0.13,0.09$, the means on a regular simplex of edge $4.5$ (rotated and jittered),
  and covariances of random orientation with eigenvalues in $[0.6,1.6]$ (`make_model.jl` writes
  `model.toml`). $U$ has five local minima, one near each mean, and the largest eigenvalue of
  $\nabla^2U$ is at most $\max_k\lambda_{\max}(\Sigma_k^{-1})=1.61$.
- **Stochastic gradient.** $\nabla U(x)+\eta(x;\xi)$, with $\eta$ of mean $0$ and covariance $\mathcal C(x)$:
  - Gaussian noise: $\eta=\sigma_1\xi$, $\xi\sim\mathcal N(0,I_4)$, so $\mathcal C=\sigma_1^2I$, with
    $\sigma_1^2=1$;
  - non-Gaussian noise: $\eta=\sqrt{\gamma(U(x))}\,R(x)\,\zeta$, where the $\zeta_j=E_j-1$ with
    $E_j\sim\mathrm{Exp}(1)$ are i.i.d. (mean 0, variance 1, skewness 2), $R(x)$ is the rotation by
    $x_1+x_2$ in the $(x_1,x_2)$-plane and by $x_3-x_4$ in the $(x_3,x_4)$-plane, and
    $\gamma(u)=s^2\big(1+\kappa-\kappa e^{-(u-U_*)}\big)$ with $s^2=1$, $\kappa=1$,
    $U_*=\min_kU(\mu_k)$; so $\mathcal C=\gamma(U)\,I$, from about $I$ at the deepest well to $2I$ far from
    the wells.
    This noise is a vector added to the gradient, not the gradient of a random potential.
- **Test functions.** Four functions with closed-form Gaussian expectations, chosen from a pool of twelve
  in `model.toml`:

| | $f(x)$ | $\lVert\nabla f\rVert_\infty$ | $\pi(f)$ |
|---|---|---|---|
| $f_1$ | $a_1\cdot x$, $\lvert a_1\rvert=1$ | $1$ | $+0.027219$ |
| $f_2$ | $\cos(a_2\cdot x+\varphi_2)$, $\lvert a_2\rvert=0.6$ | $0.6$ | $+0.187557$ |
| $f_3$ | $\exp(-\lvert x-\mu_1\rvert^2/2)$ | $e^{-1/2}$ | $+0.068087$ |
| $f_4$ | $\exp(-\lvert x-\mu_3\rvert^2/2)$ | $e^{-1/2}$ | $+0.048621$ |

  They are taken from the pool among those whose SGLD and SGLM coefficients have the same sign, with
  $|\text{SGLD}|>|\text{SGLM}|\ge0.025$, for both noises.

## Part 1. Riesz representors and the coefficients of the numerical bias

### Coefficients

By Corollaries `cor: bias coefficient` and `cor: sgld coefficient` of the SGLM paper, with $\phi_0$ the
solution of the Poisson equation $-\mathcal L\phi_0=f-\pi(f)$,

$$
\lim_{h\to0}\frac{\pi_h(f)-\pi(f)}{h}=\frac12\,\pi\big(\nabla^2\phi_0:\mathcal C\big),\qquad
\lim_{h\to0}\frac{\bar\pi_h(f)-\pi(f)}{h}=\frac12\,\pi\big(\nabla^2\phi_0:(\nabla^2U+\mathcal C)\big)
$$

for SGLM and SGLD. The Riesz representor $\psi_{\mathcal A}$ of a matrix field $\mathcal A$ is the
function with $\pi(\psi_{\mathcal A})=0$ and
$\pi\big(\psi_{\mathcal A}\,(-\mathcal L\omega)\big)=\pi\big(\nabla^2\omega:\mathcal A\big)$ for every
test function $\omega$; with $\omega=\phi_0$ it turns both limits into integrals against $\pi$, without
$\phi_0$:

$$
c_{\mathrm{SGLM}}(f)=\frac12\,\pi\big(f\,\psi_{\mathcal C}\big),\qquad
c_{\mathrm{SGLD}}(f)=\frac12\,\pi\big(f\,(\psi_{\nabla^2U}+\psi_{\mathcal C})\big).
$$

### Explicit representors

Integration by parts gives $\pi(\psi\,(-\mathcal L\omega))=\pi(\nabla\psi\cdot\nabla\omega)$, so a
representor is found by writing $\pi(\nabla^2\omega:\mathcal A)$ as $\pi(\nabla\omega\cdot\nabla\psi)$.

- **Hessian.** As in (`eq: psi hessian`) of the paper,
  $\psi_{\nabla^2U}=\frac12|\nabla U|^2-\Delta U+\frac12\pi(\Delta U)$, the same for both noises.
- **Isotropic covariance.** For $\mathcal C=\gamma(U)\,I$,
  $\pi\big(\nabla^2\omega:\gamma(U)I\big)=\int\gamma(U)\,\Delta\omega\,e^{-U}dx
  =-\int\nabla\omega\cdot\nabla\big(\gamma(U)e^{-U}\big)dx
  =\pi\big(\nabla\omega\cdot(\gamma-\gamma')(U)\,\nabla U\big)$, and
  $(\gamma-\gamma')(U)\,\nabla U=\nabla\big(\Gamma(U)-\gamma(U)\big)$ with $\Gamma'=\gamma$. Hence
  $\psi_{\mathcal C}=\Gamma(U)-\gamma(U)-\pi\big(\Gamma(U)-\gamma(U)\big)$:
  - Gaussian noise, $\gamma=\sigma_1^2$: $\psi_{\mathcal C}=\sigma_1^2\,(U-\pi(U))$;
  - non-Gaussian noise, $\Gamma(u)=s^2\big((1+\kappa)u+\kappa e^{-(u-U_*)}\big)$:
    $\psi_{\mathcal C}=s^2\big((1+\kappa)(U-\pi(U))+2\kappa(E-\pi(E))\big)$ with $E=e^{-(U-U_*)}$.

  The isotropic form of the two covariances is chosen so that the representors are explicit; for a
  general $\mathcal C$ they solve an elliptic equation in $\mathbb R^4$.
- **Derivatives of $U$.** With the responsibilities $r_k(x)$ of the components and
  $g_k=\Sigma_k^{-1}(x-\mu_k)$: $\nabla U=\sum_kr_kg_k$,
  $\nabla^2U=\sum_kr_k\Sigma_k^{-1}-\big(\sum_kr_kg_kg_k^{\mathsf T}-\nabla U\nabla U^{\mathsf T}\big)$ and
  $\Delta U$ its trace, computed in one pass over the components with a running log-sum-exp (`model.jl`).

### Quadrature

Every $\pi$-integral is $\pi(g)=\sum_kw_k\,\mathbb E\,g(\mu_k+L_kz)$, $z\sim\mathcal N(0,I_4)$, $L_k$ the
Cholesky factor of $\Sigma_k$, by a tensor Gauss–Hermite rule with 28, 40, 52 and 64 nodes per
dimension. The coefficients are covariances, $c(f)=\frac12\big(\pi(f\psi)-\pi(f)\pi(\psi)\big)$, computed
from the pieces $U$, $\frac12|\nabla U|^2-\Delta U$ and $E$, so the constants of the representors are not
needed separately.

### Checks

- $\pi(f)$ by quadrature against the closed forms: $3\times10^{-14}$.
- Change of every coefficient from 52 to 64 nodes: at most $2.2\times10^{-6}$.
- The Riesz identity $\pi(\psi_{\mathcal A}(-\mathcal L\omega))=\pi(\nabla^2\omega:\mathcal A)$ for
  $\mathcal A\in\{\sigma_1^2I,\ \gamma(U)I,\ \nabla^2U\}$ and $\omega$ two cosines and a Gaussian bump: the
  two sides agree to about $2\times10^{-7}$.
- $\nabla U$ and $\nabla^2U$ against finite differences: about $10^{-8}$.

### Results

| | $f_1$ | $f_2$ | $f_3$ | $f_4$ |
|---|---|---|---|---|
| SGLD, Gaussian | $-0.225822$ | $-0.127316$ | $-0.071353$ | $-0.060780$ |
| SGLM, Gaussian | $-0.103323$ | $-0.097693$ | $-0.042093$ | $-0.024738$ |
| SGLD, non-Gaussian | $-0.290862$ | $-0.189981$ | $-0.095117$ | $-0.076107$ |
| SGLM, non-Gaussian | $-0.168364$ | $-0.160357$ | $-0.065857$ | $-0.040064$ |

The simulations (`simulate.jl`, `analyze.py`) measure $(\pi_h(f)-\pi(f))/h$ along the chains for
$h=2^{-1},\dots,2^{-5}$; the figures `figures/bias_gaussian.pdf` and `figures/bias_nongaussian.pdf` set them
against these coefficients.

### Files

- `make_model.jl`, `model.toml`: the mixture, the noises and the pool of test functions.
- `model.jl`: $U$ and its derivatives, the test functions and their Gaussian expectations, the noises.
- `coefficients.jl`: the quadrature, the checks and the coefficients; it writes
  `results/coefficients.csv` (the coefficients of the selected test functions), `results/pool_coefficients.csv` (the whole pool),
  `results/quadrature_convergence.csv`, `results/riesz_checks.csv` and `results/coefficients.log`.

## Part 2. Kantorovich–Rubinstein norms and the coefficients of the $\mathcal W_1$ error

### What is computed

By Corollary `cor: w1 limits` of the SGLM paper,

$$
\lim_{h\to0}\frac{\mathcal W_1(\pi_h,\pi)}{h}=\frac12\,\|\psi_{\mathcal C}\,\pi\|_{\mathrm{KR}},\qquad
\lim_{h\to0}\frac{\mathcal W_1(\bar\pi_h,\pi)}{h}=\frac12\,\|(\psi_{\nabla^2U}+\psi_{\mathcal C})\,\pi\|_{\mathrm{KR}}
$$

for SGLM and SGLD, with the representors of Part 1. For each noise and each scheme the coefficient is
therefore half the Kantorovich–Rubinstein norm of an explicit signed measure $\nu=\psi\,\pi$ on
$\mathbb R^4$ with $\nu(\mathbb R^4)=0$:

$$
\|\nu\|_{\mathrm{KR}}=\sup\Big\{\int v\,d\nu:\ v\ \text{1-Lipschitz}\Big\}
=\min\Big\{\int|F|\,dx:\ \operatorname{div}F=-\nu\Big\}.
$$

The first form is the dual (a potential $v$), the second the minimal-flow form (a flux $F$ that carries
the positive part of $\nu$ onto the negative part at least total length). The norm is not additive, so
$\psi_{\nabla^2U}$, although shared by the two SGLD measures, does not give a shared part of the
solution: each measure is a separate transport problem. In 4D there is no closed form, and the norm is
computed numerically. The $\pi$-means in the representors are computed as in Part 1 (Gauss–Hermite, 52
nodes per dimension).

### Discretization

- **Box and grid.** $[-L,L]^4$ with $L=9$ and $n$ cells per side, $\Delta=2L/n$. The mixture means lie
  within $\pm2.5$ and the component standard deviations below $1.2$, so the mass of $\pi$ outside the box
  is negligible.
- **Measure.** The cell averages $\nu_i$ of $\psi\pi$, by a 3-point Gauss–Legendre rule in each
  dimension (81 points per cell). Their sum, which differs from 0 by the quadrature error and the mass
  outside the box, is shifted to exactly $0$; the size of the shift is written to the log.
- **Unknowns.** A potential $v_i$ at every cell centre, and a flux $F_{i,k}$ on the face between cell $i$
  and cell $i+e_k$; the faces on the boundary of the box carry no flux.
- **Discrete problem.** With the forward differences $(Dv)_{i,k}=(v_{i+e_k}-v_i)/\Delta$,

$$
\max_v\ \Delta^4\sum_i v_i\,\nu_i\quad\text{s.t.}\quad |(Dv)_i|\le1\ \text{for every cell }i,
\qquad\Longleftrightarrow\qquad
\min_F\ \Delta^4\sum_i |F_i|\quad\text{s.t.}\quad D^{\mathsf T}F=\nu,
$$

  where $|(Dv)_i|$ and $|F_i|$ are the Euclidean norms of the four components at cell $i$. These are the
  two forms above on the grid: a discrete 1-Lipschitz potential and a discrete flow.

### Solver

The discrete problem is a saddle point $\min_v\max_F\ \langle Dv,F\rangle-\sum_i|F_i|-\langle v,\nu\rangle$,
solved by the primal–dual algorithm (PDHG, Chambolle–Pock) with diagonal step sizes:

$$
F\leftarrow\operatorname{shrink}_{\tau_F}\big(F+\tau_F\,D\bar v\big),\qquad
v^{\mathrm{new}}=v-\tau_v\,(D^{\mathsf T}F-\nu),\qquad
\bar v=2v^{\mathrm{new}}-v,
$$

where $\operatorname{shrink}_{\tau_F}(z)_i=z_i\max(0,1-\tau_F/|z_i|)$ at each cell. The steps
$\tau_v=\theta\Delta/8$ and $\tau_F=\Delta/(2\theta)$ satisfy the convergence condition for every
$\theta>0$ (each potential enters at most 8 differences, each flux 2). On the $32^4$ grid of the Gaussian test,
among $\theta\in\{1,3,10,30,100,300,1000\}$, $\theta=100$ gave the closest agreement between the lower
bound and the flux cost after 20,000 iterations (larger $\theta$ settles the flux faster and the potential
slower), and is the default.

Each grid is warm-started from the next coarser one ($n=25\to50\to100$) by multilinear interpolation of the
potential and of the fluxes.

### Stopping rule and reported value

Every 200 iterations the solver evaluates three quantities:

- the **lower bound** $\Delta^4\langle v,\nu\rangle/\max_i|(Dv)_i|$: the potential rescaled to satisfy the
  constraint, hence a valid lower bound on the discrete optimum at every iteration;
- the **flux cost** $\Delta^4\sum_i|F_i|$;
- the **residual** $\Delta^4\sum_i|\nu_i-(D^{\mathsf T}F)_i|$ of the flow constraint.

It stops when the lower bound and the flux cost agree within $10^{-4}$ (relative) and the residual is
below $10^{-4}\int|\nu|$. The value reported is the lower bound; the coefficient is half of it.

### Accuracy

The number reported is the discrete optimum on the grid; its distance to the continuous norm is the
discretization error, which decreases as $\Delta\to0$. It is measured on a test case whose norm is exact:
$\pi=\mathcal N(0,I_4)$ and $\psi=\frac12(|x|^2-4)$, for which $\|\psi\pi\|_{\mathrm{KR}}=\mathbb E|X|/2
=0.939986$ (the measure is radial, the 1-Lipschitz function $|x|$ attains the supremum, and the radial
flow attains the minimum). Results (`results/w1/w1_runs.csv`):

| grid | $\Delta$ | value | error |
|---|---|---|---|
| $25^4$ | 0.72 | 0.924328 | $-1.67\%$ |
| $50^4$ | 0.36 | 0.935160 | $-0.51\%$ |
| $100^4$ | 0.18 | 0.938618 | $-0.146\%$ |

The error falls by a factor of about 3.3–3.5 per halving of $\Delta$. Extrapolating the three values
(Aitken: $V_\infty=V_{100}+(V_{100}-V_{50})^2/\big((V_{50}-V_{25})-(V_{100}-V_{50})\big)$) gives 0.940238,
within $0.027\%$ of the exact value. The mixture measures have the same scale (component standard
deviations between 0.77 and 1.26); their refinements converge more slowly, with successive differences
shrinking by a factor 2.47 (SGLM) and 2.36 (SGLD) per halving instead of 3.13 for the test, so their
extrapolations carry a larger error, of order $0.1\%$ (SGLM) to $0.3\%$ (SGLD) judging from an earlier
refinement $16\to32\to64$ whose extrapolations differed from the present ones by $0.06\%$ and $0.35\%$.

A two-sided bound valid for the continuous problem was tried and dropped. The multilinear interpolant of
the discrete potential has Lipschitz constant close to 2, not 1: the constraint controls only forward
differences, the optimal potential bends, and it is arbitrary where $\nu$ is negligible. On the flow side,
the variation of $\nu$ inside the cells adds a term of order $\Delta^2\int|\nabla\nu|$, several percent on
the grids that can be afforded. Neither bound was useful.

### Results

| coefficient $\frac12\lVert\psi\pi\rVert_{\mathrm{KR}}$ | $25^4$ | $50^4$ | $100^4$ | extrapolated |
|---|---|---|---|---|
| SGLD, Gaussian | 0.956631 | 0.971481 | 0.977764 | 0.982373 |
| SGLM, Gaussian | 0.557038 | 0.561382 | 0.563140 | 0.564335 |
| SGLD, non-Gaussian | 1.333075 | 1.351083 | 1.358481 | 1.363641 |
| SGLM, non-Gaussian | 0.957707 | 0.965145 | 0.968130 | 0.970129 |

The final tables `tables/w1_coefficients_gaussian.csv` (Gaussian noise) and
`tables/w1_coefficients_nongaussian.csv` (non-Gaussian noise) give, for SGLM and SGLD, the extrapolated
coefficient and, for each test function $f$ of Part 1, the ratio
$\big(|c(f)|/\|\nabla f\|_\infty\big)\big/\big(\frac12\|\psi\pi\|_{\mathrm{KR}}\big)$, with $c(f)$ the bias
coefficient of Part 1. Since $f/\|\nabla f\|_\infty$ is 1-Lipschitz and the $\mathcal W_1$ coefficient is the
supremum of $|c(f)|$ over such $f$, each ratio is at most 1: it is the fraction of the worst-case bias that
$f$ detects.

### Cost

The iteration runs on the GPU (`w1_pdhg.jl`, CUDA.jl, one GPU thread per cell); the measure, the warm
start and the output are computed by the CPU threads. On the RTX 5090 an iteration takes about 11 ms on
$100^4=10^8$ cells and uses about 8 GB of GPU memory. On $100^4$ the Gaussian test needed 114,800
iterations (21 minutes); SGLD and SGLM needed 193,400 and 291,200 iterations with the Gaussian noise (36 and
54 minutes), and 394,400 and 297,800 with the non-Gaussian noise (74 and 55 minutes). The grids $25^4$ and
$50^4$ take seconds.

### Files and usage

```
julia --project=. -t 16 w1_coefficients.jl CASE 25,50,100 [theta=100] [maxit=200000] [tol=1e-4] [L=9]
CASE = test_gaussian | SGLD_gaussian | SGLM_gaussian | SGLD_nongaussian | SGLM_nongaussian
bash w1_run.sh [CASE ...]         # the cases on 25, 50, 100, then the summary
                                   # (default: test_gaussian SGLD_gaussian SGLM_gaussian)
```

`--project=.` loads the CUDA.jl environment of this folder (`Project.toml`, `Manifest.toml`).

- `tables/w1_coefficients_gaussian.csv` and `tables/w1_coefficients_nongaussian.csv`: the final tables only,
  written by `python w1_table.py`: one row for SGLM and one for SGLD, with the $\mathcal W_1$ coefficient
  and the four ratios described under Results. The folder `tables/` holds nothing else.
- `results/w1/w1_runs.csv`: one row per run and grid, with columns `case, n, L, delta, iterations,
  kr_norm, flux_cost, residual, coefficient, wall_seconds, date`.
- `results/w1/w1_extrapolation.csv`: per case (with the Gaussian test) and sequence of grids, the values
  on the three grids, the ratio of their differences, the extrapolation and the coefficients.
- `results/w1/w1_coefficients.log`: progress lines (every 30 s) and the final values; the console output
  of each run is in `results/w1/launch_CASE_*.out`.
- `results/w1/CASE_nN.f64`: the potential and the four flux arrays of the finest grid (Float64, $n^4$
  values each, in that order), for later checks without a rerun.
