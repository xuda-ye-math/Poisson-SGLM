"""Final W1 coefficients from the runs recorded in results/w1/w1_runs.csv (saved data only).

    python w1_table.py

For each case, the latest row of each grid, and for every sequence of grids n, 2n, 4n present the
three-grid (Aitken) extrapolation V3 + (V3 - V2)^2 / ((V2 - V1) - (V3 - V2)) of the norm, with V1, V2, V3
its values on the three grids; these go to results/w1/w1_extrapolation.csv (with the Gaussian test, whose
norm is exact). The final tables tables/w1_coefficients_gaussian.csv (Gaussian noise) and
tables/w1_coefficients_nongaussian.csv (non-Gaussian noise), each written once both schemes have been run,
have one row for SGLM and one for SGLD: the W1 coefficient 1/2 ||psi pi||_KR, extrapolated from the
finest sequence of grids, and
for each of the four test functions f of model.toml the ratio (|c(f)| / ||grad f||_inf) / (W1 coefficient),
with c(f) the exact bias coefficient of results/coefficients.csv. Each ratio is at most 1, since
f / ||grad f||_inf is 1-Lipschitz and the W1 coefficient is the supremum of |c(f)| over such f.
"""

import csv
import math
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXACT = {"test_gaussian": 0.939986}           # E|X|/2 for X ~ N(0, I_4)

rows = list(csv.DictReader(open(HERE / "results" / "w1" / "w1_runs.csv")))
latest = {}
for r in rows:                                 # rows are appended in time order
    latest[(r["case"], int(r["n"]))] = r

out = []
for case in dict.fromkeys(r["case"] for r in rows):
    v = {n: float(latest[(case, n)]["kr_norm"]) for (c, n) in latest if c == case}
    for n1 in sorted(v):
        n2, n3 = 2 * n1, 4 * n1
        if n2 not in v or n3 not in v:
            continue
        d1, d2 = v[n2] - v[n1], v[n3] - v[n2]
        ext = v[n3] + d2 * d2 / (d1 - d2)
        out.append({"case": case, "grids": f"{n1}-{n2}-{n3}", "kr_coarse": v[n1], "kr_middle": v[n2],
                    "kr_fine": v[n3], "ratio_of_differences": d1 / d2, "kr_extrapolated": ext,
                    "coefficient_fine": v[n3] / 2, "coefficient_extrapolated": ext / 2,
                    "kr_exact": EXACT.get(case, float("nan"))})

def write(path, recs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        for rec in recs:
            w.writerow({k: (f"{x:.6f}" if isinstance(x, float) else x) for k, x in rec.items()})


write(HERE / "results" / "w1" / "w1_extrapolation.csv", out)

model = tomllib.loads((HERE / "model.toml").read_text())
tests = {t["name"]: t for t in model["test"]}


def lipschitz(t):                              # ||grad f||_inf of the test functions of model.toml
    a = math.sqrt(sum(x * x for x in t["a"]))
    return a if t["type"] in ("linear", "cos") else math.exp(-0.5) / t["width"]


coefs = list(csv.DictReader(open(HERE / "results" / "coefficients.csv")))
for noise, name_out in (("gaussian", "w1_coefficients_gaussian.csv"), ("nongaussian", "w1_coefficients_nongaussian.csv")):
    bias = {(r["test"], r["scheme"]): float(r["coefficient"]) for r in coefs if r["noise"] == noise}
    final = []
    for scheme in ("SGLM", "SGLD"):
        recs = [r for r in out if r["case"] == f"{scheme}_{noise}"]
        if not recs:
            break
        w1 = max(recs, key=lambda r: int(r["grids"].split("-")[-1]))["coefficient_extrapolated"]
        row = {"scheme": scheme, "w1_coefficient": w1}
        for i, name in enumerate(model["selected"], 1):
            row[f"f{i}_ratio"] = abs(bias[(name, scheme)]) / lipschitz(tests[name]) / w1
        final.append(row)
    if len(final) < 2:
        continue
    write(HERE / "tables" / name_out, final)
    print(f"{noise} noise:")
    for rec in final:
        print(", ".join(f"{k} {x:.6f}" if isinstance(x, float) else f"{x}" for k, x in rec.items()))
