"""Figure of the coefficients 1/2 ||psi pi||_KR of SGLM and SGLD against the dimension d, from saved data only.

    python plot_coefficients.py

Reads results/coefficients_sigma2.csv and results/coefficients_sigma1.csv (written by coefficient_table.jl)
and draws sigma = 2 and sigma = 1 together, in solid lines of different colours, in figures/coefficients.pdf
and figures/coefficients.png.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 12, "axes.titlesize": 15,
    "legend.fontsize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "mathtext.fontset": "cm", "font.family": "serif",
})

HERE = Path(__file__).resolve().parent
# colours and markers of evaluate_coefficients/plot_bias.py: SGLD blue, SGLM red
STYLE = {"SGLD": ("#4C72B0", "o"), "SGLM": ("#C44E52", "v")}
# sigma = 1: a colour close to red for SGLM and close to blue for SGLD
COLOR_SIGMA1 = {"SGLD": "#64B5CD", "SGLM": "#DD8452"}
DPI = 400


def read_table(sigma):
    with open(HERE / "results" / f"coefficients_sigma{sigma}.csv") as f:
        rows = list(csv.reader(f))
    return [int(v) for v in rows[0][1:]], {row[0]: [float(v) for v in row[1:]] for row in rows[1:]}


def save(fig, name):
    out = HERE / "figures" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=DPI)
    fig.savefig(out.with_suffix(".png"), dpi=DPI)
    print(f"wrote {out}.pdf and {out}.png")


def main():
    """sigma = 2 in the colours of STYLE, sigma = 1 in COLOR_SIGMA1; legend at the top left: sigma = 2 then
    sigma = 1, SGLM before SGLD"""
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    for sigma in ("2", "1"):
        dims, values = read_table(sigma)
        for scheme in ("SGLM", "SGLD"):
            col, mk = STYLE[scheme]
            col = COLOR_SIGMA1[scheme] if sigma == "1" else col
            ax.plot(dims, values[scheme], color=col, marker=mk, markersize=4, markevery=8,
                    linewidth=1.5, label=rf"{scheme}, $\sigma = {float(sigma):.1f}$")
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$\frac{1}{2}\,\|\psi\,\pi\|_{\mathrm{KR}}$")
    ax.set_xlim(0, dims[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    save(fig, "coefficients")


if __name__ == "__main__":
    main()
