"""Bias and coefficient estimates from the saved block files of every run in a folder.

    python analyze.py results/runs_T5e7

For each run directory (SCHEME_NOISE_hexpJ, any number of step sizes), the blocks of all
chains give the time average of each test function, mean = sum of the block sums / number of
steps, the bias mean - pi(f), and its standard error from the spread of the block means
(batch means, one batch per block). The coefficient estimate is bias / h. The results go to
FOLDER/estimates.csv, and the table of the selected test functions is printed.
"""

import csv
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def theory():
    out = {}
    with open(HERE / "results" / "coefficients.csv") as f:
        for r in csv.DictReader(f):
            out[(r["test"], r["noise"], r["scheme"])] = float(r["coefficient"])
    return out


def load_run(d):
    meta = tomllib.loads((d / "run.toml").read_text())
    nf = len(meta["tests"])
    blocks = [np.fromfile(p, dtype="<f8").reshape(-1, 2 + nf) for p in sorted(d.glob("blocks_c*.f64"))]
    B = np.vstack(blocks)
    steps = B[:, 1]
    # runs with blocks_values = "sum" store sums; the earlier dry runs stored block means
    sums = B[:, 2:] if meta.get("blocks_values") == "sum" else B[:, 2:] * steps[:, None]
    means = sums / steps[:, None]
    mean = sums.sum(axis=0) / steps.sum()
    se = means.std(axis=0, ddof=1) / np.sqrt(len(means))
    return meta, mean, se, len(means), [len(b) for b in blocks]


def main(folder):
    folder = Path(folder)
    th = theory()
    rows = []
    for d in sorted(p for p in folder.iterdir() if (p / "run.toml").is_file() and list(p.glob("blocks_c*.f64"))):
        if "result" not in tomllib.loads((d / "run.toml").read_text()):
            print(f"skip {d.name}: not finished")
            continue
        meta, mean, se, nb, per_chain = load_run(d)
        h = meta["h"]
        t_chain = min(per_chain) * meta["t_block"]
        for i, name in enumerate(meta["tests"]):
            bias = mean[i] - meta["pi_f"][i]
            c = th.get((name, meta["noise"], meta["scheme"]), np.nan)
            rows.append(dict(scheme=meta["scheme"], noise=meta["noise"], hexp=meta["hexp"], h=h, test=name,
                             pi_f=meta["pi_f"][i], mean=mean[i], bias=bias, se=se[i], coef_est=bias / h,
                             coef_se=se[i] / h, coef_theory=c, ratio=bias / h / c, blocks=nb, t_chain=t_chain,
                             wall_seconds=meta["result"]["wall_seconds"]))
    with open(folder / "estimates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.16e}" if isinstance(v, float) else v) for k, v in r.items()})

    selected = tomllib.loads((HERE / "model.toml").read_text())["selected"]
    for key in sorted({(r["scheme"], r["noise"]) for r in rows}):
        sub = [r for r in rows if (r["scheme"], r["noise"]) == key]
        hexps = sorted({r["hexp"] for r in sub})
        print(f"\n{key[0]}, {key[1]} noise: coefficient estimate bias/h +- standard error (ratio to theory)")
        print(f"{'h':>7} {'T/chain':>9} " + " ".join(f"{s:>26}" for s in selected))
        for j in hexps:
            rs = {r["test"]: r for r in sub if r["hexp"] == j}
            r0 = rs[selected[0]]
            cells = [f"{rs[s]['coef_est']:+.4f}±{rs[s]['coef_se']:.4f} ({rs[s]['ratio']:.3f})" for s in selected]
            print(f"{'1/' + str(2 ** j):>7} {r0['t_chain']:9.2e} " + " ".join(f"{c:>26}" for c in cells))
        print(f"{'theory':>7} {'':>9} " + " ".join(f"{th[(s, key[1], key[0])]:>+26.4f}" for s in selected))
    print(f"\nwrote {folder / 'estimates.csv'}")


if __name__ == "__main__":
    main(sys.argv[1])
