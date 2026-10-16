"""1 x 4 figure of the bias coefficients for one noise, from saved data only.

    python plot_bias.py results/runs_T5e7 gaussian

Reads FOLDER/estimates.csv (written by analyze.py from the saved block sums) and the order of
the test functions from model.toml. Each panel is one test function: for SGLD and SGLM the
estimate (pi_h(f) - pi(f)) / h against h, solid with markers and error bars of two standard
errors, and the exact coefficient as a dashed line of the same colour. Writes
figures/bias_NOISE.pdf and figures/bias_NOISE.png.
"""

import csv
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 13, "axes.titlesize": 15,
    "legend.fontsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "mathtext.fontset": "cm", "font.family": "serif",
})

HERE = Path(__file__).resolve().parent
# SGLD in the colour and marker of Euler-Maruyama in the Poisson figures
# (SRK-Codes/Convergence/plot_style.py), SGLM in red
STYLE = {"SGLD": ("#4C72B0", "o"), "SGLM": ("#C44E52", "v")}
DPI = 400
TITLES = [
    r"$f_1(x) = a_1 \cdot x$",
    r"$f_2(x) = \cos(a_2 \cdot x + \varphi_2)$",
    r"$f_3(x) = \exp(-|x - \vartheta_1|^2/2)$",
    r"$f_4(x) = \exp(-|x - \vartheta_3|^2/2)$",
]


def main(folder, noise):
    folder = Path(folder)
    selected = tomllib.loads((HERE / "model.toml").read_text())["selected"]
    with open(folder / "estimates.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["noise"] == noise]

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.3), constrained_layout=True)
    # the constrained layout ignores the width of the titles; the right margin keeps the last one inside
    fig.get_layout_engine().set(rect=(0, 0, 0.985, 1))
    for ax, name, title in zip(axes, selected, TITLES):
        # SGLD and SGLM have coefficients of the same sign for every selected f; the y-axis runs
        # from 0 at the bottom to the largest absolute value, with that sign, at the top (an
        # inverted axis for a negative coefficient), so SGLD lies above SGLM in every panel
        sign = 1.0 if float(next(r for r in rows if r["test"] == name)["coef_theory"]) > 0 else -1.0
        top = 0.0
        for scheme, (col, mk) in STYLE.items():
            rs = sorted((r for r in rows if r["scheme"] == scheme and r["test"] == name), key=lambda r: float(r["h"]))
            if not rs:
                continue
            h = [float(r["h"]) for r in rs]
            y = [float(r["coef_est"]) for r in rs]
            e = [2 * float(r["coef_se"]) for r in rs]
            ax.errorbar(h, y, yerr=e, color=col, marker=mk, markersize=5, linewidth=1.5, capsize=2.5, elinewidth=1.0)
            ax.axhline(float(rs[0]["coef_theory"]), color=col, linestyle="--", linewidth=1.2)
            top = max(top, max(sign * a + b for a, b in zip(y, e)), sign * float(rs[0]["coef_theory"]))
        ax.set_ylim(0, sign * 1.1 * top)
        hs = sorted({float(r["h"]) for r in rows})
        ax.set_xscale("log", base=2)
        ax.set_xticks(hs)
        ax.set_xticklabels([rf"$2^{{-{round(-__import__('math').log2(x))}}}$" for x in hs])
        ax.minorticks_off()
        ax.set_xlabel(r"$h$")
        ax.set_title(title)
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("bias coefficients")

    handles = []
    for scheme in ("SGLM", "SGLD"):                                  # legend order: SGLM first
        col, mk = STYLE[scheme]
        handles.append(Line2D([], [], color=col, marker=mk, markersize=5, linewidth=1.5, label=f"{scheme}, evaluated"))
        handles.append(Line2D([], [], color=col, linestyle="--", linewidth=1.2, label=f"{scheme}, exact"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)

    out = HERE / "figures" / f"bias_{noise}"
    fig.savefig(out.with_suffix(".pdf"), dpi=DPI)
    fig.savefig(out.with_suffix(".png"), dpi=DPI)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
