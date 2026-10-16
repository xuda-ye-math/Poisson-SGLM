# Stochastic gradient Leimkuhler–Matthews integrator with second order discretization error

Source code for the numerical experiments of the paper *Stochastic Gradient Leimkuhler–Matthews Integrator with
Second Order Discretization Error* by Xuda Ye, Yuliang Wang, Guang Lin and Cindy Xiangrui Kong (in preparation).

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

- For a potential strongly convex outside a ball, the mean square error of the time averages of SGLM over $`N`$
  steps of size $`h`$ is $`\mathcal O\bigl(\frac{1}{Nh} + \Sigma^2 h^2 + h^4\bigr)`$, where $`\Sigma`$ bounds the
  variance of the gradient noise. The stochastic gradient error and the discretization error separate, so a larger
  batch reduces the error at a fixed step size.
- The coefficient of $`h`$ in the numerical bias of SGLM is explicit, in terms of the covariance of the gradient
  noise and the solution of the Poisson equation; that of SGLD contains in addition the Hessian of the potential,
  which no batch size removes.
- Through Riesz representors, these coefficients determine the Wasserstein-1 errors of SGLM and SGLD to first order
  in $`h`$, and they lead to corrections of the numerical bias of SGLM: the explicit coefficient, Richardson–Romberg
  extrapolation, and a shifting correction of the samples trained from the samples alone.

## Earlier work

The analysis builds on the discrete Poisson equation of the earlier paper *Mean square error analysis of stochastic
Runge–Kutta integrators* (X. Ye), whose manuscript and code are in the repository
[Poisson-Overdamped](https://github.com/xuda-ye-math/Poisson-Overdamped).

## Repository layout

```text
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
