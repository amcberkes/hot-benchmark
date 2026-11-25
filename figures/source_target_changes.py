"""Change in HVAC energy and in occupied discomfort for every source and target of the transfer
matrix, in the thesis style. Values are calculated from trained-policy records and use the appendix precision:
ten-seed means relative to each target's historical controller, reserved calendar."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from transfer_figure_style import apply_style

# ---- thesis style (hexes as defined in main.tex) -----------------------------
NAVY, TEAL, AMBER, INK, MUTED = "#2B3A67", "#1F8A80", "#C98A1B", "#1C2233", "#5B6170"
TEXT_WIDTH_IN = 5.6
# thesis diverging scale: teal = lower than the historical controller, amber = higher
DIVERGING = LinearSegmentedColormap.from_list("better_worse", [TEAL, "#F3F4F8", AMBER])

BLOCKS = ["Office, Fairbanks", "Office, Seattle", "Retail, Fairbanks", "Retail, Seattle"]
VARIANTS = ["default", "envelope", "occupancy", "occ. + env."]
OWN = {(0, 0), (1, 1), (2, 2), (4, 3), (8, 4), (12, 5)}   # policy evaluated on the task it was trained on
# rows: 16 targets (block x variant); columns: sources OF default, OF envelope, OF occupancy, OS, RF, RS


def plot_source_target_changes(energy, discomfort, output_dir):
    apply_style(8, INK)
    W, H, GAP, BLOCKGAP = 1.0, 0.72, 0.18, 0.55
    col_x, x = [], 0.0
    for j in range(6):
        if j in (3, 4, 5):
            x += GAP
        col_x.append(x); x += W
    PANEL_W = x
    row_y, y = [], 0.0
    for i in range(16):
        if i % 4 == 0:
            y += BLOCKGAP                 # room for the block name above each block
        row_y.append(y); y += H

    def panel(ax, x0, values, limit, title, decimals):
        ax.text(x0, -2.05, title, ha="left", va="bottom", fontsize=8.5, fontweight="bold")
        groups = [("Office, Fairbanks", 0, 2), ("Office\nSeattle", 3, 3), ("Retail\nFairb.", 4, 4), ("Retail\nSeattle", 5, 5)]
        for name, j0, j1 in groups:
            xm = x0 + (col_x[j0] + col_x[j1] + W - 0.05) / 2
            ax.text(xm, -0.62, name, ha="center", va="bottom", fontsize=7, linespacing=1.05)
            ax.plot([x0 + col_x[j0], x0 + col_x[j1] + W - 0.05], [-0.5, -0.5], color=MUTED, lw=0.5)
        for j, name in enumerate(["def.", "env.", "occ."]):
            ax.text(x0 + col_x[j] + (W - 0.05) / 2, -0.08, name, ha="center", va="bottom", fontsize=6.8, color=MUTED)
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                t = max(-1.0, min(1.0, v / limit))
                ax.add_patch(Rectangle((x0 + col_x[j], row_y[i]), W - 0.05, H - 0.05, fc=DIVERGING(0.5 + t / 2), ec="none"))
                label = f"{v:+.{decimals}f}".replace("-", "−")
                ax.text(x0 + col_x[j] + (W - 0.05) / 2, row_y[i] + (H - 0.05) / 2, label, ha="center", va="center",
                        fontsize=6.6, color="white" if abs(t) > 0.62 else INK)
                if (i, j) in OWN:
                    ax.add_patch(Rectangle((x0 + col_x[j], row_y[i]), W - 0.05, H - 0.05, fc="none", ec=INK, lw=1.1))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 5.05))
    LABEL_W, PANEL_GAP = 2.05, 0.75
    xa, xb = LABEL_W, LABEL_W + PANEL_W + PANEL_GAP
    panel(ax, xa, energy, 15.0, "(a) Energy change (%)", 1)
    panel(ax, xb, discomfort, 100.0, "(b) Discomfort change (%)", 0)
    for i in range(16):
        ax.text(LABEL_W - 0.15, row_y[i] + H / 2, VARIANTS[i % 4], ha="right", va="center", fontsize=7.5)
    for b, name in enumerate(BLOCKS):
        ax.text(0.0, row_y[4 * b] - 0.14, name, ha="left", va="bottom", fontsize=8, fontweight="bold")
    ax.text(0.0, -0.08, "Target", ha="left", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(xa + PANEL_W / 2, -1.45, "Source", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(xb + PANEL_W / 2, -1.45, "Source", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    ax.set_xlim(-0.05, xb + PANEL_W); ax.set_ylim(row_y[15] + H, -2.5)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(output_dir / "figure-3.7.pdf")
    plt.close(fig)
