"""Table of the numerical biases of SGLM at h = 1/16 and of its three corrections, from saved results only.

    python table.py [noise=gaussian]

Each column is an estimate of pi(f) minus the exact mean pi(f) of results/exact_means.csv (exact_means.jl, exact
samples only):
    SGLM            pi_{1/16}(f), no correction                 results/sglm_NOISE_h16.csv
    explicit        pi_{1/16}(f) - (1/16) 1/2 pi(f psi_C)       the exact coefficient 1/2 sigma^2 Cov_pi(f, U) of
                                                                results/exact_coefficients_gaussian.csv
                                                                (exact_coefficients.jl, exact samples only)
    extrapolation   2 pi_{1/16}(f) - pi_{1/8}(f)                results/sglm_NOISE_h8.csv and _h16.csv
    shift           pi_{1/16}(f) - (1/16) c_phi(f)              the implied coefficient
                    c_phi(f) = -(pi_h(f o T) - pi_h(f)) / h,    results/phi_NOISE.csv (phi_coefficients.jl);
                    T(x) = x + h grad phi_C(x)                  left out when it does not exist
Both noises of model.jl have the covariance C = sigma^2 I (the non-Gaussian one rotates independent centred
exponentials of unit variance), so psi_C = sigma^2 (U - pi(U)) and the exact coefficient are the same for both.
Prints the table in Markdown and writes tables/table_NOISE.csv.
"""

import csv
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOISE = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
RES = HERE / "results"


def column(name, key):
    with open(RES / name) as f:
        return {r["test"]: float(r[key]) for r in csv.DictReader(f)}


h = tomllib.loads((RES / f"data_{NOISE}" / "meta.toml").read_text())["h"]
pih = column(f"sglm_{NOISE}_h{round(1 / h)}.csv", "mean")
pi2h = column(f"sglm_{NOISE}_h{round(1 / (2 * h))}.csv", "mean")
exact = column("exact_coefficients_gaussian.csv", "exact")      # the same for both noises, see above
ref = column("exact_means.csv", "exact")
phi = column(f"phi_{NOISE}.csv", "implied") if (RES / f"phi_{NOISE}.csv").exists() else None

cols = ["SGLM", "explicit", "extrapolation"] + (["shift"] if phi is not None else [])
print("| test | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
TABLES = HERE / "tables"
TABLES.mkdir(exist_ok=True)
with open(TABLES / f"table_{NOISE}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test"] + cols)
    for k in pih:
        vals = [pih[k] - ref[k], pih[k] - h * exact[k] - ref[k], 2 * pih[k] - pi2h[k] - ref[k]]
        if phi is not None:
            vals.append(pih[k] - h * phi[k] - ref[k])
        print(f"| {k} | " + " | ".join(f"{v:+.5f}" for v in vals) + " |")
        w.writerow([k] + [f"{v:.10e}" for v in vals])
print(f"wrote {TABLES / f'table_{NOISE}.csv'}")
