"""The reserved transfer matrix (trained policies, of ten, that meet both criteria) as a shaded
matrix in the thesis style. Counts come from the final assessment records."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from transfer_figure_style import apply_style

# ---- thesis style (hexes as defined in main.tex) -----------------------------
NAVY, INK, MUTED = "#2B3A67", "#1C2233", "#5B6170"
TEXT_WIDTH_IN = 5.6
SCALE = LinearSegmentedColormap.from_list("navy", ["#F3F4F8", NAVY])

BLOCKS = ["Office, Fairbanks", "Office, Seattle", "Retail, Fairbanks", "Retail, Seattle"]
VARIANTS = ["default", "envelope", "occupancy", "occupancy + envelope"]
OWN = {(0, 0), (1, 1), (2, 2), (4, 3), (8, 4), (12, 5)}


def plot_transfer_matrix(counts, output_dir):
    apply_style(8.5, INK)
    W, H, GAP = 1.0, 0.62, 0.22
    col_x, x = [], 0.0
    for j in range(6):
        if j in (3, 4, 5):
            x += GAP                      # gaps between source groups
        col_x.append(x); x += W
    row_y, y = [], 0.0
    for i in range(16):
        if i and i % 4 == 0:
            y += GAP                      # gaps between target blocks
        row_y.append(y); y += H

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.55))
    for i, row in enumerate(counts):
        for j, v in enumerate(row):
            ax.add_patch(Rectangle((col_x[j], row_y[i]), W - 0.04, H - 0.04, fc=SCALE(v / 10), ec="none"))
            ax.text(col_x[j] + (W - 0.04) / 2, row_y[i] + (H - 0.04) / 2, str(v), ha="center", va="center",
                    fontsize=8.5, color="white" if v >= 5 else (MUTED if v == 0 else INK))
            if (i, j) in OWN:
                ax.add_patch(Rectangle((col_x[j], row_y[i]), W - 0.04, H - 0.04, fc="none", ec=INK, lw=1.3))

    for i in range(16):                                   # row labels
        ax.text(-0.15, row_y[i] + H / 2, VARIANTS[i % 4], ha="right", va="center", fontsize=8, color=INK)
    for b, name in enumerate(BLOCKS):
        ax.text(-2.75, row_y[4 * b] + 2 * H - 0.02, name, ha="right", va="center", fontsize=8.5, fontweight="bold")
    # column headers: building and climate on the first line, variant on the second
    GROUPS = [("Office, Fairbanks", 0, 2), ("Office,\nSeattle", 3, 3), ("Retail,\nFairbanks", 4, 4), ("Retail,\nSeattle", 5, 5)]
    for name, j0, j1 in GROUPS:
        xm = (col_x[j0] + col_x[j1] + W - 0.04) / 2
        ax.text(xm, -0.62, name, ha="center", va="bottom", fontsize=8.5, fontweight="bold", linespacing=1.1)
        ax.plot([col_x[j0], col_x[j1] + W - 0.04], [-0.52, -0.52], color=MUTED, lw=0.5)
    for j, name in enumerate(["default", "envelope", "occupancy", "default", "default", "default"]):
        ax.text(col_x[j] + (W - 0.04) / 2, -0.12, name, ha="center", va="bottom", fontsize=7.2)
    ax.text((col_x[0] + col_x[5] + W) / 2, -1.55, "Source", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.text(-2.75, -0.12, "Target", ha="right", va="bottom", fontsize=8.5, color=MUTED)

    ax.set_xlim(-4.75, col_x[5] + W); ax.set_ylim(row_y[15] + H, -1.95)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(output_dir / "figure-3.6.pdf")
    plt.close(fig)
