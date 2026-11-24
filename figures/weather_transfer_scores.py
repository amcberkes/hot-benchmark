"""Weather-transfer scores of the HOT paper (transfer efficiency and robustness) as two slope charts,
in the thesis palette. Values read from the published figures of Berkes et al."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- thesis style (same block as the other figures) -----------------------
NAVY, TEAL, AMBER, GREY = "#2B3A67", "#1F8A80", "#C98A1B", "#8A9099"   # hexes as defined in main.tex
INK, MUTED, GRID = "#222222", "#6B7280", "#E5E7EB"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],   # thesis font (newpx)
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42,
})
TEXT_WIDTH_IN = 5.6

# controller: (colour, C1, C2) -- learned controller navy, rule-based baseline grey,
# amber for the one to notice (the static controller, which collapses in the second city)
EFFICIENCY = {"PPO": (NAVY, 97, 94), "Seasonal": (GREY, 83, 82), "Static": (AMBER, 64, 19)}
ROBUSTNESS = {"PPO": (NAVY, 86, 76), "Seasonal": (GREY, 70, 69), "Static": (AMBER, 58, 22)}
CONDITIONS = ["Real weather,\nsame city (C1)", "Typical weather,\nsecond city (C2)"]

def panel(ax, data, ylabel, letter):
    for name, (colour, c1, c2) in data.items():
        ax.plot([0, 1], [c1, c2], color=colour, lw=1.8, marker="o", ms=5.5, mec="white", mew=0.9,
                solid_capstyle="round", zorder=3, clip_on=False)
        ax.text(-0.07, c1, f"{c1}", ha="right", va="center", fontsize=8, color=INK)
        ax.text(1.07, c2, f"{c2}", ha="left", va="center", fontsize=8, color=INK)
        ax.text(1.27, c2, name, ha="left", va="center", fontsize=8, color=INK, fontweight="normal")
    ax.set_xlim(-0.32, 1.95); ax.set_ylim(0, 100)
    ax.set_xticks([0, 1], CONDITIONS)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.32, 1.2)
    ax.tick_params(axis="x", length=0, pad=5); ax.tick_params(axis="y", length=0, pad=4)
    ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")

fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.75))
panel(a, EFFICIENCY, "Transfer efficiency (%)", "(a)")
panel(b, ROBUSTNESS, "Transfer robustness score", "(b)")
fig.tight_layout(w_pad=1.5)
fig.savefig("weather_transfer_scores.pdf"); fig.savefig("weather_transfer_scores.png", dpi=220)
