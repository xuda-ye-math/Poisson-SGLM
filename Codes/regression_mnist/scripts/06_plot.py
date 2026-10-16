#!/usr/bin/env python3
"""Generate the paper MSE figure."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sglm_mnist.settings import (  # noqa: E402
    BATCHES,
    FIGURE_DIR,
    FIGURE_FUNCTIONS,
    FIGURE_TITLES,
    METHODS,
    PREFIXES,
    RESULTS_DIR,
)

METHOD_STYLE = {"sgld": ("#4C72B0", "o"), "sglm": ("#C44E52", "v")}
BATCH_STYLE = {1024: "-", 2048: "--", 4096: ":"}
Y_TICKS = {
    "cos_z1": (0.00, 0.04, 0.08, 0.12, 0.16),
    "cos_z4": (0.0, 0.0004, 0.0008, 0.0012, 0.0016),
    "bump1d_z2": (0.0, 0.0004, 0.0008, 0.0012, 0.0016),
    "bump1d_z3": (0.0, 0.0002, 0.0004, 0.0006, 0.0008),
    "bump2d_z5_z6": (0.0, 0.00015, 0.00030, 0.00045, 0.00060),
    "bump2d_z7_z8": (0.0, 0.0002, 0.0004, 0.0006, 0.0008),
}


def load_curves() -> dict[tuple[str, int, str], list[float]]:
    with (RESULTS_DIR / "mse_source.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    curves = {}
    for function in FIGURE_FUNCTIONS:
        for batch in BATCHES:
            for method in METHODS:
                selected = sorted(
                    (
                        row for row in rows
                        if row["function"] == function
                        and int(row["batch"]) == batch
                        and row["method"] == method
                    ),
                    key=lambda row: int(row["sample_size"]),
                )
                if len(selected) != len(PREFIXES):
                    raise ValueError(f"missing MSE curve: {function}/{batch}/{method}")
                curves[(function, batch, method)] = [float(row["mse"]) for row in selected]
    return curves


def legend_handles() -> list[Line2D]:
    handles = [
        Line2D([], [], color=METHOD_STYLE[method][0], marker=METHOD_STYLE[method][1],
               linewidth=1.5, markersize=5, label=method.upper())
        for method in ("sglm", "sgld")
    ]
    handles += [
        Line2D([], [], color="0.25", linestyle=BATCH_STYLE[batch],
               linewidth=1.5, label=rf"$P={batch}$")
        for batch in BATCHES
    ]
    return handles


def mse_figure() -> None:
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 13, "axes.titlesize": 15,
        "legend.fontsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "mathtext.fontset": "cm", "font.family": "serif",
    })
    curves = load_curves()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.35), constrained_layout=True)
    for axis, function, title in zip(axes.flat, FIGURE_FUNCTIONS, FIGURE_TITLES):
        for batch in BATCHES:
            for method in METHODS:
                color, marker = METHOD_STYLE[method]
                axis.plot(
                    PREFIXES, curves[(function, batch, method)],
                    color=color, marker=marker, linestyle=BATCH_STYLE[batch],
                    markersize=4, linewidth=1.5,
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks((8192, 16384, 32768, 65536, 131072))
        axis.set_xticklabels((r"$2^{13}$", r"$2^{14}$", r"$2^{15}$", r"$2^{16}$", r"$2^{17}$"))
        ticks = Y_TICKS[function]
        axis.set_yticks(ticks)
        axis.set_ylim(ticks[0], ticks[-1])
        if function != "cos_z1":
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            axis.yaxis.set_major_formatter(formatter)
        axis.minorticks_off()
        axis.set_title(title)
        axis.grid(True, color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)
    for axis in axes[1]:
        axis.set_xlabel(r"retained length $N$")
    for axis in axes[:, 0]:
        axis.set_ylabel("MSE")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5, frameon=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "mnist_mse.pdf", dpi=400)
    fig.savefig(FIGURE_DIR / "mnist_mse.png", dpi=400)
    plt.close(fig)


def main() -> None:
    mse_figure()
    print(f"figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
