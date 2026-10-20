# Stochastic gradient Leimkuhler–Matthews integrator with second order discretization error

[![arXiv](https://img.shields.io/badge/arXiv-2610.24781-b31b1b.svg)](https://arxiv.org/abs/2610.24781)
[![Paper: CC BY 4.0](https://img.shields.io/badge/Paper-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](Codes/LICENSE)
[![Julia 1.13.1](https://img.shields.io/badge/Julia-1.13.1-9558B2.svg)](https://julialang.org)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB.svg)](https://www.python.org)

The manuscript and the source code for the numerical experiments of the paper *Stochastic Gradient
Leimkuhler–Matthews Integrator with Second Order Discretization Error* by Xuda Ye, Yuliang Wang, Guang Lin and
Cindy Xiangrui Kong.

## The integrator

The overdamped Langevin dynamics

$$
dX_t = -\nabla U(X_t) dt + \sqrt{2} dB_t
$$

has the invariant distribution $`\pi(x) \propto e^{-U(x)}`$. In Bayesian inference on large data sets, $`\nabla U`$
is replaced by an unbiased stochastic gradient $`\nabla \hat U(x;\xi)`$ computed from a batch of the data. With
independent $`\chi_k \sim \mathcal N(0, I_d)`$ and step size $`h`$, stochastic gradient Langevin dynamics (SGLD) is

$$
Z_{k+1} = Z_k - h \nabla \hat U(Z_k;\xi_k) + \sqrt{2h} \chi_{k+1} ,
$$

and the stochastic gradient Leimkuhler–Matthews (SGLM) integrator, with the same cost of one stochastic gradient
per step, reuses each Gaussian vector in two consecutive steps:

$$
Z_{k+1} = Z_k - h \nabla \hat U(Z_k;\xi_k) + \sqrt{\frac h2} \bigl( \chi_k + \chi_{k+1} \bigr) .
$$

With the full gradient, SGLD is the Euler–Maruyama integrator, whose discretization error is of first order, and
SGLM is the Leimkuhler–Matthews integrator, whose discretization error for the invariant distribution is of second
order.

## Main results

- For a potential strongly convex outside a ball, the mean square error (MSE) of the time averages of SGLM over
  $`N`$ steps of size $`h`$ is $`\mathcal O\bigl(\frac{1}{Nh} + \Sigma^2 h^2 + h^4\bigr)`$, where $`\Sigma`$ bounds the
  variance of the gradient noise. The discretization error enters this bound as $`h^4`$ and the MSE bound of SGLD as
  $`h^2`$, while a larger batch reduces only the term $`\Sigma^2 h^2`$ in both bounds.
- In Bayesian logistic regression on MNIST, SGLM has a smaller estimated MSE than SGLD.
- The coefficient of $`h`$ in the numerical bias of SGLM is given explicitly by the covariance of the gradient noise
  and the continuous Poisson equation. The coefficient for SGLD contains in addition the Hessian of the potential,
  which comes from the discretization error.
- Through Riesz representors, the coefficients of the numerical bias determine the coefficients of the Wasserstein-1
  errors of SGLM and SGLD, and they lead to corrections of the numerical bias of SGLM: the explicit coefficient,
  Richardson–Romberg extrapolation, and a shifting correction of the samples trained from the samples alone.

## Earlier work

The analysis builds on the discrete Poisson equation of the earlier paper *Mean square error analysis of stochastic
Runge–Kutta integrators* (X. Ye), whose manuscript and code are in the repository
[Poisson-Overdamped](https://github.com/xuda-ye-math/Poisson-Overdamped).

## The paper

`Paper/` holds the manuscript:

- `main.pdf`, the compiled manuscript;
- `main.tex`, which includes `numerical.tex` (Sections 8.1 to 8.3), `mnist.tex` (Section 8.4) and `appendix.tex`
  (the proofs) by `\input`;
- `references.bib` and `main.bbl`, the bibliography compiled by BibTeX with the style `plainnat`;
- `figures/`, the figures of Section 8: `bias_gaussian.pdf` and `bias_nongaussian.pdf` are written by
  `Codes/evaluate_coefficients/plot_bias.py`, `wasserstein.pdf` and `coefficients.pdf` by
  `Codes/convergence_wasserstein/plot_wasserstein.py` and `plot_coefficients.py`, and `mnist_mse.pdf` by
  `Codes/regression_mnist/scripts/06_plot.py`; `digit_illustration.pdf` shows examples of the digits 4 and 9 of the
  MNIST data set.

## Repository layout

```text
Paper/                         The manuscript: main.pdf, its LaTeX source, bibliography and figures
Codes/
├── evaluate_coefficients/     Section 8.1  Evaluating the coefficients in a 4D Gaussian mixture distribution
├── convergence_wasserstein/   Section 8.2  Convergence of Wasserstein-1 error in a radial distribution
├── bias_correction/           Section 8.3  Bias correction of SGLM in a 16D product of ring distributions
└── regression_mnist/          Section 8.4  Bayesian logistic regression on MNIST
```

Each folder has its own `README.md`, with the setting, the commands in the order they are run, the files they
write and the checks of the computation, and its own environment file:

- Sections 8.1 to 8.3 compute in Julia 1.13.1 (`Project.toml` and `Manifest.toml` where packages are needed) and
  use Python 3.14 only for the tables and figures and, in Section 8.3, for training a small network with JAX;
  `environment.yml` lists the Python packages.
- Section 8.4 is a Python package with PyTorch and Pyro; `environment.yml` creates its conda environment.

The saved numerical results (simulation outputs, reference chains and network checkpoints) are not included in the
repository; the scripts of each folder regenerate them.
