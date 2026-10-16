# MNIST SGLD–SGLM experiment: reproduction code

This directory is a clean reproduction package for the MNIST experiment in the paper.

The fixed workflow is:

1. download MNIST and construct the 4-versus-9 split;
2. fit the Bayesian logistic-regression MAP and compute the Hessian geometry;
3. run four full-gradient NUTS reference chains;
4. run 20 paired formal SGLD/SGLM jobs at each of three mini-batch sizes;
5. compute MSE and paired bootstrap intervals;
6. generate the MSE figure used by the paper section.

## Fixed experiment

| Quantity | Value |
|---|---:|
| Digits | 4 mapped to 0; 9 mapped to 1 |
| Posterior training set | 10,791 images: 5,342 fours and 5,449 nines |
| Validation holdout | 500 images per digit from the official training split |
| Test set | official test split restricted to 4/9: 1,991 images |
| Features | intercept plus 784 row-major pixels divided by 255 |
| Parameter dimension | 785 |
| Prior | `N(0, 10^-3 I)` |
| Step size | `h = 1/L`, with `L = 1000 + lambda_max(X^T X)/4` |
| Expected `h` | `8.509918695515035e-6` |
| Mini-batch sizes | 1024, 2048, 4096 |
| Repetitions | 20 paired runs per mini-batch size for SGLM and SGLD |
| Burn-in | 16,384 transitions |
| Retained states | 131,072 per trajectory |
| MSE retained lengths | `N_j = 8192 j`, `j=1,...,16` |
| Prediction states | 256 states at indices `0, 512, ..., 130560` |
| NUTS reference | 4 chains; 16,384 warm-up and 524,288 retained states per chain |

All floating-point calculations use `float64`. Mini-batches are sampled uniformly with replacement. Within each paired repetition, SGLD and SGLM use the same mini-batches and the same fresh Gaussian variables. SGLM also carries its preceding Gaussian variable continuously through burn-in and retained sampling.

The 16 observables are eight directional cosines, four one-dimensional Gaussian bumps, and four two-dimensional Gaussian bumps in the leading MAP-Hessian coordinates.

## Environment

A CUDA-capable machine is required for NUTS and the formal repetitions. The launchers use at most four GPUs and do not use DDP.

```bash
cd ./regression_mnist
conda env create -f environment.yml
conda activate sglm-mnist
pytest -q
```

The environment file pins the versions used for the reported experiment. Exact bitwise equality is not guaranteed across CUDA libraries or GPU architectures, but the deterministic data identities, settings, qualitative conclusions, and reported summaries should agree within Monte Carlo variation.

## Run the experiment

### 1. Data, MAP, Hessian, and observables

```bash
python scripts/01_prepare.py
```

This command downloads MNIST into `data/`, recreates the fixed validation split, verifies the original index and feature hashes, fits the MAP with full-data L-BFGS-B, computes the full Hessian, extracts its eight leading eigenpairs, and saves:

- `outputs/prepared/model.npz`
- `outputs/prepared/manifest.json`

The expected training-index hash is `73bace...ac9`, `L` is approximately `117509.93585015424`, and the MAP full-gradient infinity norm should be below `1e-5`.

### 2. Four-chain NUTS reference

With four GPUs:

```bash
python scripts/02_reference.py launch \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

With fewer GPUs, provide one to three devices; the four chains are queued automatically. The launcher writes one log and one progress JSON file per chain and aggregates the chains when all four finish. To run or aggregate manually:

```bash
python scripts/02_reference.py chain --chain 0 --device cuda:0
python scripts/02_reference.py aggregate
```

The reference targets the full posterior in MAP-Hessian coordinates with full-data gradients and NUTS' Metropolis correction. The package stores the 16 observable traces rather than the full 785-dimensional parameter chains; this is sufficient to reproduce every reference average, observable diagnostic, MSE, interval, and figure in the paper section while greatly reducing disk use.

The completed reference stage has the following layout:

```text
outputs/reference/
├── chain_0.npz                 # 16 observable traces and the initial z
├── chain_0.json                # chain settings, seeds, runtime, and divergences
├── chain_1.npz
├── chain_1.json
├── chain_2.npz
├── chain_2.json
├── chain_3.npz
├── chain_3.json
├── reference.json              # pooled means and four-chain diagnostics
├── progress/chain_0.json       # one progress/status file per chain
└── logs/chain_0.log            # one launcher log per chain
```

Each `chain_<k>.npz` contains `pool_values` with shape `(524288, 16)` and
`initial_z` with shape `(785,)`. `reference.json` contains the pooled reference
means plus R-hat, bulk ESS, tail ESS, and mean MCSE for every observable. The
four chain NPZ files and `reference.json` are required by Step 4. The chain JSON
files are additionally required if `02_reference.py aggregate` is rerun;
progress files and logs are not required for downstream analysis.

A reference is accepted only when there are no retained divergences, every observable has rank-normalized split R-hat below 1.01, and every bulk and tail ESS exceeds 1,000. The reported run was much stronger: maximum R-hat about `1.000007`, minimum bulk ESS about `9.33e5`, and minimum tail ESS about `1.09e6`.

Before the formal SGLD/SGLM runs were started, reference precision was also checked for all 16 observables. Each reference-mean Monte Carlo standard error (MCSE) was required to be below 10% of the smallest RMSE observed at `N=131072` across SGLD/SGLM and the three batch sizes in the prior qualification runs. All 16 checks passed.

### 3. Formal SGLD/SGLM sampling

```bash
python scripts/03_formal_sampling.py launch \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

The launcher schedules 60 jobs—20 repetitions at each of the three batch sizes—over one to four distinct GPUs. A single job can be run directly:

```bash
python scripts/03_formal_sampling.py pair \
  --batch 1024 --repetition 0 --device cuda:0
```

Check completeness with:

```bash
python scripts/03_formal_sampling.py check
```

Each artifact stores the 16 cumulative observable sums needed for the MSE curves and the 256 parameter snapshots needed for prediction. Full parameter trajectories are intentionally not stored.

The completed formal-sampling stage has the following layout:

```text
outputs/formal_sampling/
├── batch_1024/
│   ├── rep_0.npz               # paired SGLD/SGLM numerical results
│   ├── rep_0.json              # settings, seeds, runtime, and completion status
│   └── ...                     # rep_1 through rep_19
├── batch_2048/
│   └── ...                     # rep_0 through rep_19
├── batch_4096/
│   └── ...                     # rep_0 through rep_19
├── progress/
│   └── batch_1024_rep_0.json   # one progress/status file per paired job
└── logs/
    └── batch_1024_rep_0.log    # one launcher log per paired job
```

There are 60 final NPZ files and 60 matching JSON manifests. Each NPZ contains
`prefix_sums` with shape `(16, 2, 16)`, `prediction_states` with shape
`(2, 256, 785)`, `snapshot_indices` with shape `(256,)`, and `final_states`
with shape `(2, 785)`; the method axis is ordered as SGLD then SGLM. Step 4
requires all 60 NPZ files and their JSON manifests. Progress files
and logs are not required for the analysis.

### 4. MSE and paired-bootstrap analysis

```bash
python scripts/04_analyze_mse.py
```

For each method, batch size, observable, and retained length `N`, the script computes the empirical MSE of the time averages over the first `N` retained states across the 20 trajectories against the pooled NUTS mean. It then uses 5,000 whole-pair bootstrap replicates. Reference uncertainty is propagated with a circular moving-block bootstrap applied independently within each NUTS chain; the main block length is 64 and the sensitivity block length is 128.

Outputs:

- `outputs/results/mse_source.csv`
- `outputs/results/mse_summary.json`
- `outputs/results/mse_summary.md`

### 5. Figures

```bash
python scripts/06_plot.py
```

This creates `outputs/results/figures/mnist_mse.{png,pdf}`. The figure uses a
base-2 logarithmic x-axis, linear panel-specific y-scales, and five y-ticks in
every panel.

## Expected paper-facing results

At `N=131072`, SGLM should have lower estimated MSE for all 16 observables at all three batch sizes. All 48 individual paired 95% bootstrap intervals for `MSE(SGLD) - MSE(SGLM)` should be positive.

| Batch | Median MSE reduction | Median absolute finite-time bias ratio | Median cross-repetition variance ratio |
|---:|---:|---:|---:|
| 1024 | 34.6% | 0.784 | 1.020 |
| 2048 | 46.5% | 0.677 | 1.018 |
| 4096 | 63.3% | 0.516 | 1.032 |

## Reproducibility and restart behavior

- Every random stream is derived with SHA-256 and NumPy `PCG64DXSM`; seed namespaces match the reported final experiment.
- Launchers never overwrite completed chain or repetition artifacts.
- Re-running a launcher skips completed repetitions/chains and schedules only missing ones.
- If a process fails after creating a partial final artifact, move that specific artifact and manifest out of `outputs/` before rerunning that job. Do not remove the entire output tree unless you intend to restart the experiment.
- Progress is written under `outputs/reference/progress/` and `outputs/formal_sampling/progress/`; subprocess logs are stored beside each experiment family.

## Code map

- `sglm_mnist/data.py`: deterministic MNIST split and features
- `sglm_mnist/posterior.py`: posterior potential, gradient, and Hessian
- `sglm_mnist/map_fit.py`: L-BFGS MAP and curvature calculation
- `sglm_mnist/observables.py`: the 16 Hessian-coordinate observables
- `sglm_mnist/reference.py`: MAP-Hessian transformation and NUTS potential
- `sglm_mnist/samplers.py`: float64 SGLD/SGLM transition kernels
- `sglm_mnist/seeds.py`: stable random-stream derivation
- `sglm_mnist/settings.py`: all fixed final-experiment constants
- `scripts/01_prepare.py` through `scripts/06_plot.py`: executable workflow in order
