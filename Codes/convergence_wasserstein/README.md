# Section 8.2: convergence of the Wasserstein-1 error in a radial distribution

Code for Section 8.2 of the paper *Stochastic Gradient Leimkuhler–Matthews Integrator with Second Order
Discretization Error* (X. Ye, Y. Wang, G. Lin, C. X. Kong): the evaluated coefficients
$\mathcal W_1(\pi_h,\pi)/h$ of SGLM and $\mathcal W_1(\bar\pi_h,\pi)/h$ of SGLD in dimension $d = 128$, the exact
coefficients $\frac12\|\psi\pi\|_{\mathrm{KR}}$ against the dimension $d = 2, 4, \dots, 256$, and the figure of
the section with these two panels. Everything is computed without sampling, by reducing the rotation-invariant
distributions to their radial parts.

All calculations are written in Julia; Python only reads saved results to make the figures.

## Environment

- **Julia 1.13.1** (installed with juliaup). `Project.toml` and `Manifest.toml` pin SpecialFunctions.jl 2.9.0 and
  FastGaussQuadrature.jl 1.3.0; every Julia script is run with `--project=.`.
- **Python**, for the figures: `environment.yml` pins the versions used for the reported results (Python 3.14.8,
  Matplotlib 3.11.2). Below, `python` is the interpreter of this environment.
- **Hardware of the reported runs.** 32 CPU threads; no GPU.

## Workflow

Every command is run in this folder; `results/` is ignored by git.

1. **Checks** of the two solvers, against closed forms and under refinement:

   ```bash
   julia --project=. -t 32 check_wasserstein.jl > results/check_wasserstein.log
   julia --project=. -t 32 check_coefficients.jl > results/check_coefficients.log
   ```

2. **Exact coefficients** $\frac12\|\psi\pi\|_{\mathrm{KR}}$ for $d = 2, 4, \dots, 256$, with $\sigma = 1$ and
   $\sigma = 2$:

   ```bash
   julia --project=. -t 32 coefficient_table.jl >> results/coefficient_table_sigma1.log
   julia --project=. -t 32 coefficient_table.jl 2 >> results/coefficient_table_sigma2.log
   ```

   write `results/coefficients_sigma1.csv` and `results/coefficients_sigma2.csv`; values already in a table are
   kept, and only the missing dimensions are computed.
3. **Evaluated coefficients** $\mathcal W_1/h$ in $d = 128$ for $h = 2^{-1}, \dots, 2^{-8}$ and
   $\sigma = 2, 1$:

   ```bash
   julia --project=. -t 32 wasserstein_table.jl > results/wasserstein_table_d128.log
   ```

   writes `results/wasserstein_d128.csv`, after a closed-form check and a refinement check in $d = 128$.
4. **Figures**:

   ```bash
   python plot_wasserstein.py
   python plot_coefficients.py
   ```

   write `figures/wasserstein.{pdf,png}` (left panel of the figure in the paper) and
   `figures/coefficients.{pdf,png}` (right panel).

The potential of the section is not three times differentiable at the origin, so it does not satisfy
Assumption 1 of the paper; the example serves to observe the convergence of the Wasserstein-1 error numerically
(see the refinement check below).

The rest of this file explains how the distances and the coefficients are computed and checked.

## Setting

- **Target.** $\pi\propto e^{-U}$ on $\mathbb R^d$, with $U(x)=u(|x|)$ and
  $u(r)=(r^2-r)\,\dfrac{r^2}{r^2+1/4}$. The potential grows quadratically ($u''\to2$) and is not convex:
  $u''$ ranges over $[-1.83,\,2.25]$, and $U$ has a ring of minima at $|x|\approx0.638$.
- **Stochastic gradient.** $\nabla U(x)+\sigma\xi$ with $\xi\sim\mathcal N(0,I_d)$, so
  $\mathcal C(x)=\sigma^2I_d$.
- **Schemes.** SGLD: $Z_{k+1}=Z_k-h(\nabla U(Z_k)+\sigma\xi_k)+\sqrt{2h}\,\chi_{k+1}$. SGLM in the Markov form
  (`eq: sglm markov` of the paper): $Y_{k+1}=Y_k-h(\nabla U(Y_k+a\chi_k)+\sigma\xi_k)+2a\chi_k$ with
  $a=\sqrt{h/2}$, and $Z_k=Y_k+a\chi_k$, whose stationary law is $\pi_h$.

## Reduction to one dimension

Both chains commute with rotations, and their invariant laws are unique, so $\bar\pi_h$ and $\pi_h$ are
rotation invariant. Between rotation-invariant laws $\mu,\nu$,
$$
\mathcal W_1(\mu,\nu)=\int_0^\infty\big|F_\mu(r)-F_\nu(r)\big|\,dr ,
$$
with $F$ the distribution function of $|X|$: the function $|x|$ is 1-Lipschitz, which gives $\ge$, and
the coupling $X=R\theta$, $Y=R'\theta$ with a common uniform direction $\theta$ and the monotone coupling
of the radii gives $\le$. The radial density of $\pi$ is $\propto r^{d-1}e^{-u(r)}$.

## Radial kernels

Write $\mathrm{nchi}_d(x;\lambda,s)$ for the density of $|\lambda e_1+s\eta|$, $\eta\sim\mathcal N(0,I_d)$:
$$
\mathrm{nchi}_d(x;\lambda,s)=\frac{x}{s^2}\Big(\frac x\lambda\Big)^{d/2-1}
e^{-(x^2+\lambda^2)/(2s^2)}\,I_{d/2-1}\Big(\frac{x\lambda}{s^2}\Big),
$$
the folded normal for $d=1$, computed in logarithms with the scaled Bessel function `besselix`.

- **SGLD.** The gradient noise merges with the Brownian increment: $Z_{k+1}=Z_k-h\nabla U(Z_k)+s\chi$,
  $s^2=2h+h^2\sigma^2$. Since $Z_k-h\nabla U(Z_k)=(r-hu'(r))\,Z_k/r$, the radius is a Markov chain with
  the kernel $\mathrm{nchi}_d(r';\,|r-hu'(r)|,\,s)$.
- **SGLM.** Given $Y_k=y$, $|y|=r$, put $W=y+a\chi_k\sim\mathcal N(y,a^2I)$ and $F(w)=2w-h\nabla U(w)$.
  Then $Y_{k+1}=V-h\sigma\xi_k$ with $V=F(W)-y$. The map $F$ is radial, $F(w)=f(|w|)\,w/|w|$ with
  $f(\omega)=2\omega-hu'(\omega)$, and it is a diffeomorphism when $f'>0$ and $f(\omega)/\omega>0$, which
  holds for $h<2/2.25$; the code accepts $0<h\le1/2$. Hence $V$ has the density
  $$
  p_V(v)=\frac{\varphi_a\big(F^{-1}(v+y)-y\big)}{f'(g)\,\big(f(g)/g\big)^{d-1}},\qquad g=f^{-1}(|v+y|),
  $$
  with $\varphi_a$ the density of $\mathcal N(0,a^2I_d)$. By symmetry about the axis of $y$, the density
  of $|V|$ at $t$ is the angular integral
  $q_r(t)=|S^{d-2}|\,t^{d-1}\int_0^\pi p_V(t,\theta)\sin^{d-2}\theta\,d\theta$ ($p_V(t)+p_V(-t)$ for
  $d=1$), and the radial kernel of $Y$ is $K(r,r')=\int q_r(t)\,\mathrm{nchi}_d(r';t,h\sigma)\,dt$.
  The radial density of $Z=Y+a\chi$ is $\int\rho_Y(r)\,\mathrm{nchi}_d(r';r,a)\,dr$.

## Discretization

- **Grid.** $[0,R]$ in panels with 12 Gauss-Legendre nodes each, of width $\frac12 s$ (SGLD) or
  $\frac12 a$ (SGLM), at most $0.1$. $R$ is where the radial density of $\pi$ has dropped by $e^{-50}$,
  widened by $\sqrt{(2+h\sigma^2)/(2(1-h))}$, the variance ratio of the Euler scheme for $u\sim r^2$; the
  computed densities are checked to be below $10^{-14}$ of their maximum on the last panel.
- **Angular integral (SGLM).** $|V-(F(y)-y)|\le\mathrm{Lip}(F)\,a\,(\sqrt d+10)$ outside an event of
  probability below $e^{-50}$, so $t$ ranges over that band and $\theta$ over the cap of the sphere
  $|v|=t$ that meets the ball; the cap is integrated by a composite rule of $4\times16$ Gauss-Legendre
  nodes. $f^{-1}$ is a table lookup followed by two Newton steps.
- **Smoothing of scale $h\sigma$ (SGLM).** $q_r$ is interpolated in each panel by its Lagrange basis,
  and each basis function is integrated against $\mathrm{nchi}_d(r';t,h\sigma)$ on sub-intervals of
  width $\frac12 h\sigma$ with 8 nodes each, so $K=QM$ with $Q_{ik}=q_{r_i}(r_k)$ and $M$ the smoothing
  matrix; this resolves $h\sigma$ without refining the grid.
- **Invariant density.** Nyström: $\rho(r_j)=\sum_iw_i\rho(r_i)K(r_i,r_j)$. The rows of $K$ are
  normalized to unit mass, which absorbs the mass lost beyond $R$ and the quadrature error of the mass;
  the discrete kernel then conserves mass, and $(I-A+\mathbf 1w^{\mathsf T})\rho=\mathbf 1$ with
  $A_{ji}=w_iK(r_i,r_j)$ gives $w^{\mathsf T}\rho=1$ and $(I-A)\rho=0$, every node's equation kept.
- **The integral of $|F_h-F|$.** On each panel the difference of the densities is interpolated by its
  Legendre series, integrated to the difference $G$ of the distribution functions, and $|G|$ is
  integrated exactly between the real roots of $G$ in the panel.

The parameters are fixed constants at the top of `wasserstein.jl`.

## Coefficients $\frac12\|\psi\pi\|_{\mathrm{KR}}$ (`coefficients.jl`)

By Corollary `cor: w1 limits`, $\mathcal W_1(\bar\pi_h,\pi)/h\to\frac12\|(\psi_{\nabla^2U}+\psi_{\mathcal C})\pi\|_{\mathrm{KR}}$
(`coefficient_sgld(d, σ)`) and $\mathcal W_1(\pi_h,\pi)/h\to\frac12\|\psi_{\mathcal C}\pi\|_{\mathrm{KR}}$
(`coefficient_sglm(d, σ)`). Both representors are explicit and radial:

- $\psi_{\mathcal C}=\sigma^2\big(U-\pi(U)\big)$, the first row of Table `tab: explicit representors` with a
  constant $a=\sigma^2$;
- $\psi_{\nabla^2U}=\frac12|\nabla U|^2-\Delta U+\frac12\pi(\Delta U)$ (`eq: psi hessian`), with
  $|\nabla U|=|u'(r)|$ and $\Delta U=u''(r)+(d-1)\,u'(r)/r$.

For a radial $\psi$ with $\pi(\psi)=0$, the supremum over 1-Lipschitz $f$ that defines the norm may be
taken over radial $f(x)=g(|x|)$, since averaging $f$ over rotations keeps its Lipschitz constant and
the value of $\pi(f\psi)$. With $\rho_\pi$ the radial density of $\pi$, the norm is therefore the
one-dimensional norm of $\psi\rho_\pi$ on $[0,\infty)$:
$$
\|\psi\pi\|_{\mathrm{KR}}=\sup_{\|g'\|_\infty\le1}\int_0^\infty g\,\psi\rho_\pi\,dr
=\int_0^\infty\Big|\int_0^r\psi(s)\rho_\pi(s)\,ds\Big|\,dr .
$$
`coefficients.jl` computes $\pi(U)$, $\pi(\Delta U)$ and this integral on $[0,R]$ with panels of width
$0.02$ and 12 Gauss-Legendre nodes each ($R$ as in the grid above, with $h=0$), the integral of
$|G|$, $G(r)=\int_0^r\psi\rho_\pi$, being computed as for $\mathcal W_1$ (Legendre interpolant, exact
between the roots of $G$).

## Checks (all in dimensions $d\ge2$)

### `check_wasserstein.jl` (log `results/check_wasserstein.log`)

1. **Closed form for $u(r)=r^2/2$.** The invariant laws are $\mathcal N(0,vI)$ with
   $v=(2+h\sigma^2)/(2-h)$ (SGLD) and $v=1+h\sigma^2/(2-h)$ (SGLM), so
   $\mathcal W_1=|\sqrt v-1|\,\mathbb E|X|$. For $d=2,4,10$, $h=2^{-1},2^{-3},2^{-5}$, $\sigma=0,1$, both
   functions agree with it to an absolute error of at most $1.3\times10^{-13}$
   ($\mathcal W_1(\pi_h,\pi)=0$ for LM).
2. **Refinement by the factor 2** (panel width, angular nodes, smoothing sub-intervals) for the ring
   potential, $\sigma=1$:

   | case | SGLD, relative change | SGLM, relative change |
   |---|---|---|
   | $d=2$, $h=2^{-4}$ | $4.5\times10^{-13}$ | $4.4\times10^{-8}$ |
   | $d=4$, $h=2^{-2}$ | $7.3\times10^{-15}$ | $3.2\times10^{-13}$ |
   | $d=4$, $h=2^{-6}$ | $1.7\times10^{-12}$ | $7.9\times10^{-12}$ |
   | $d=10$, $h=2^{-4}$ | $1.2\times10^{-13}$ | $1.3\times10^{-11}$ |

   The largest change, SGLM in $d=2$, comes from the potential: $u(r)\approx-4r^3$ near $0$, so $U$ is
   $C^2$ but not $C^3$ at the origin, and the density of $V$ is less smooth where $W=0$, that is at
   $V=-y$. The angular integral averages this point over the sphere $|v|=t$; the averaging smooths it
   more as $d$ grows. In $d=1$, which the checks exclude, it is a jump of the second derivative inside
   a panel, and SGLM changes by $4\times10^{-5}$ under the same refinement.
3. **Limits as $h\to0$**, from `coefficients.jl`, for $\sigma=1$:

   | $h$ | SGLD, $d=2$ | SGLM, $d=2$ | SGLD, $d=4$ | SGLM, $d=4$ |
   |---|---|---|---|---|
   | $2^{-1}$ | 0.6246185 | 0.2746264 | 1.0946281 | 0.4529792 |
   | $2^{-3}$ | 0.4580421 | 0.2084820 | 0.7795814 | 0.3365272 |
   | $2^{-5}$ | 0.4468128 | 0.2029584 | 0.7372566 | 0.3208681 |
   | $2^{-6}$ | 0.4451613 | 0.2028895 | 0.7311200 | 0.3186353 |
   | $2^{-7}$ | 0.4443006 | 0.2028839 | 0.7281155 | 0.3175505 |
   | limit | 0.4434078 | 0.2028891 | 0.7251479 | 0.3164850 |

   The table lists $\mathcal W_1/h$. For SGLD in $d=2,4$ and SGLM in $d=4$, its distance to the limit
   halves with $h$ at the smallest step sizes; for SGLM in $d=2$ it is within $5.2\times10^{-6}$ of
   the limit from $h=2^{-6}$ on.

### `check_coefficients.jl` (log `results/check_coefficients.log`)

1. **Closed form for $u(r)=r^2/2$.** $\psi_{\mathcal C}=\sigma^2(|x|^2-d)/2$,
   $\psi_{\nabla^2U}=(|x|^2-d)/2$, and $\|\frac12(|x|^2-d)\pi\|_{\mathrm{KR}}=\frac12\mathbb E|X|$, attained
   by $f(x)=|x|$ and the flux $\frac12x\pi(x)$, so the coefficients are $\sigma^2\mathbb E|X|/4$ (SGLM)
   and $(1+\sigma^2)\mathbb E|X|/4$ (SGLD). For $d=2,4,10$ and $\sigma=0.5,1,2$ the computed values agree
   to a relative error of at most $1.1\times10^{-14}$.
2. **Refinement by the factor 2** of the panel width for the ring potential, $d=2,4,10$,
   $\sigma=0.5,1,2$: relative changes of at most $1.2\times10^{-14}$.

## Usage and cost

```
julia --project=. -t 32
julia> include("coefficients.jl"); using .Wasserstein, .Coefficients
julia> w1_sgld(4, 2.0^-5, 1.0), w1_sglm(4, 2.0^-5, 1.0)
julia> coefficient_sgld(4, 1.0), coefficient_sglm(4, 1.0)

julia --project=. -t 32 check_wasserstein.jl > results/check_wasserstein.log
julia --project=. -t 32 check_coefficients.jl > results/check_coefficients.log
```

`coefficients.jl` includes `wasserstein.jl`. With 32 threads, one value of $\mathcal W_1$ in $d=4$ at
$h=2^{-7}$ takes about 0.1 s (SGLD) and 1.6 s (SGLM); the SGLM grid grows like $h^{-1/2}$ and its kernel
matrix like $h^{-1}$.

## Files

- `wasserstein.jl`: module `Wasserstein`, with `w1_sgld(d, h, σ)` and `w1_sglm(d, h, σ)` for the ring
  potential, and the same functions with a `Potential` as first argument.
- `coefficients.jl`: module `Coefficients`, with `coefficient_sgld(d, σ)` and `coefficient_sglm(d, σ)`.
- `check_wasserstein.jl`, `check_coefficients.jl`: the checks above.
- `Project.toml`, `Manifest.toml`: the Julia environment (SpecialFunctions, FastGaussQuadrature).
- `results/`: logs, ignored by git.
