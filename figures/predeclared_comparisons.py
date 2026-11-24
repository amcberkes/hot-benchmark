"""The six comparisons fixed in advance (default Fairbanks office policy on six targets), as
estimates with Bonferroni-adjusted intervals, in the thesis style. Inputs are calculated from the ten trained seeds."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- thesis style (hexes as defined in main.tex) -----------------------------
NAVY, TEAL, AMBER, GREY = "#2B3A67", "#1F8A80", "#C98A1B", "#8A9099"
INK, MUTED, GRID = "#1C2233", "#5B6170", "#E5E7EB"
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],  # thesis font (newpx)
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "axes.labelcolor": INK,
        "pdf.fonttype": 42,
    }
)
TEXT_WIDTH_IN = 5.6

# target, axes changed, energy (estimate, low, high) in kWh/m2, discomfort (estimate, low, high) in degC^2
import csv
from pathlib import Path

labels = [
    ("Office, Fairbanks", "envelope changed"),
    ("Office, Fairbanks", "occupancy changed"),
    ("Office, Seattle", "climate changed"),
    ("Retail, Fairbanks", "building type changed"),
    ("Retail, Seattle", "building type and climate changed"),
    ("Retail, Seattle", "all four axes changed"),
]
with Path("figure-A.2.csv").open() as stream:
    ROWS = [
        (
            *label,
            tuple(float(row[k]) for k in ("energy", "lower_E", "upper_E")),
            tuple(float(row[k]) for k in ("discomfort", "lower_C", "upper_C")),
        )
        for label, row in zip(labels, csv.DictReader(stream), strict=True)
    ]


def colour(lo, hi):
    """Teal: lower than the historical controller; amber: higher; grey: interval includes zero."""
    return TEAL if hi < 0 else AMBER if lo > 0 else GREY


def minus(text):
    return text.replace("-", "−")


def panel(ax, idx, xlim, ticks, title, digits):
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    for i, row in enumerate(ROWS):
        est, lo, hi = row[idx]
        c = colour(lo, hi)
        ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(est, i, "o", ms=4.6, color=c, mec="white", mew=0.6, zorder=3)
        ax.text(
            1.04,
            i - 0.13,
            minus(f"{est:+.{digits}f}"),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
        )
        ax.text(
            1.04,
            i + 0.22,
            minus(f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.3,
            color=MUTED,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(len(ROWS) - 0.45, -0.55)
    ax.set_xticks(ticks, [minus(f"{t:g}") for t in ticks])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, loc="left", pad=6)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)


fig = plt.figure(figsize=(TEXT_WIDTH_IN, 3.2))
ax_e = fig.add_axes([0.335, 0.155, 0.215, 0.755])
ax_c = fig.add_axes([0.685, 0.155, 0.165, 0.755])
panel(ax_e, 2, (-3.3, 1.3), [-3, -2, -1, 0, 1], "(a) Energy difference (kWh/m²)", 2)
panel(ax_c, 3, (-0.7, 0.12), [-0.6, -0.3, 0], "(b) Discomfort difference (°C²)", 4)
for i, (target, change, *_) in enumerate(ROWS):
    ax_e.text(
        -1.52, i - 0.13, target, transform=ax_e.get_yaxis_transform(), ha="left", va="center", fontsize=8.5
    )
    ax_e.text(
        -1.52,
        i + 0.22,
        change,
        transform=ax_e.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=7,
        color=MUTED,
    )
fig.savefig("predeclared_comparisons.pdf")
fig.savefig("predeclared_comparisons.png", dpi=220)
