"""Figure of W_1 / h against h in dimension d = 128, with the coefficients 1/2 ||psi pi||_KR as dashed lines,
from saved data only.

    python plot_wasserstein.py

Reads results/wasserstein_d128.csv (written by wasserstein_table.jl) and the column d = 128 of
results/coefficients_sigma2.csv and results/coefficients_sigma1.csv (written by coefficient_table.jl), and writes
figures/wasserstein.pdf and figures/wasserstein.png. Colours as in plot_coefficients.py.
"""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 12, "axes.titlesize": 15,
    "legend.fontsize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "mathtext.fontset": "cm", "font.family": "serif",
})

HERE = Path(__file__).resolve().parent
D = 128
# colours and markers of plot_coefficients.py: sigma = 2 red / blue, sigma = 1 close to red / blue
STYLE = {"SGLD": ("#4C72B0", "o"), "SGLM": ("#C44E52", "v")}
COLOR_SIGMA1 = {"SGLD": "#64B5CD", "SGLM": "#DD8452"}
DPI = 400


def coefficient(sigma, scheme):
    with open(HERE / "results" / f"coefficients_sigma{sigma}.csv") as f:
        rows = list(csv.reader(f))
    col = rows[0].index(str(D))
    return next(float(r[col]) for r in rows[1:] if r[0] == scheme)


def main():
    with open(HERE / "results" / f"wasserstein_d{D}.csv") as f:
        rows = list(csv.reader(f))
    hs = [float(v) for v in rows[0][2:]]
    curves = {(r[0], r[1]): [float(v) for v in r[2:]] for r in rows[1:]}

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    handles = {}
    for sigma in ("2", "1"):
        for scheme in ("SGLM", "SGLD"):
            col, mk = STYLE[scheme]
            col = COLOR_SIGMA1[scheme] if sigma == "1" else col
            y = curves[(scheme, f"{float(sigma):.1f}")]
            ax.plot(hs, y, color=col, marker=mk, markersize=4, linewidth=1.5)
            ax.axhline(coefficient(sigma, scheme), color=col, linestyle="--", linewidth=1.2)
            handles[(scheme, sigma)] = Line2D([], [], color=col, marker=mk, markersize=4, linewidth=1.5,
                                              label=rf"{scheme}, $\sigma = {float(sigma):.1f}$")
    coef = Line2D([], [], color="0.3", linestyle="--", linewidth=1.2, label=r"$\frac{1}{2}\,\|\psi\,\pi\|_{\mathrm{KR}}$")
    blank = Line2D([], [], linestyle="none", label=" ")

    ax.set_xscale("log", base=2)
    ax.set_xticks(hs)
    ax.set_xticklabels([rf"$2^{{-{round(-math.log2(h))}}}$" for h in hs])
    ax.minorticks_off()
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("Wasserstein-1 coefficients")
    ax.set_ylim(0.0, 17.5)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    # 3 x 2 legend, filled by columns: rows sigma = 2.0, sigma = 1.0, coefficient; columns SGLM, SGLD
    ax.legend(handles=[handles[("SGLM", "2")], handles[("SGLM", "1")], coef,
                       handles[("SGLD", "2")], handles[("SGLD", "1")], blank],
              loc="upper left", ncol=2, fontsize=10, frameon=False)

    out = HERE / "figures" / "wasserstein"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=DPI)
    fig.savefig(out.with_suffix(".png"), dpi=DPI)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
