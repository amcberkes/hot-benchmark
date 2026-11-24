"""Envelope U-values by ASHRAE climate zone: range plot in the thesis palette."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- thesis style (shared by all figures) ---------------------------------
NAVY, TEAL, AMBER = "#2B3A67", "#1F8A80", "#C98A1B"   # hexes as defined in main.tex
INK, MUTED, GRID = "#222222", "#6B7280", "#E5E7EB"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],   # thesis font (newpx)
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42,
})
TEXT_WIDTH_IN = 5.6   # set to \textwidth of the thesis, in inches

# ---- data: zone, min, mean, max (W/m2K) -----------------------------------
DATA = [
    ("0A", 0.191, 0.969, 2.730), ("0B", 0.191, 0.966, 2.730),
    ("1A", 0.191, 0.969, 2.730), ("1B", 0.191, 0.965, 2.730),
    ("2A", 0.164, 0.739, 2.480), ("2B", 0.164, 0.730, 2.480),
    ("3A", 0.161, 0.715, 2.480), ("3B", 0.161, 0.711, 2.480), ("3C", 0.161, 0.715, 2.480),
    ("4A", 0.133, 0.656, 2.410), ("4B", 0.133, 0.656, 2.410), ("4C", 0.133, 0.671, 2.410),
    ("5A", 0.127, 0.632, 2.390), ("5B", 0.127, 0.632, 2.390), ("5C", 0.127, 0.632, 2.390),
    ("6A", 0.123, 0.615, 2.390), ("6B", 0.123, 0.615, 2.390),
    ("7",  0.113, 0.594, 2.390), ("8",  0.096, 0.559, 2.350),
]

fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.3))
ax.set_xscale("log")

# rows, with a small gap between zone numbers (0, 1, 2, ...)
y, ys, prev = 0.0, [], None
for z, *_ in DATA:
    if prev is not None and z[0] != prev:
        y += 0.45
    ys.append(y); prev = z[0]; y += 1.0

for (z, lo, mean, hi), yy in zip(DATA, ys):
    ax.plot([lo, hi], [yy, yy], color=NAVY, lw=1.1, alpha=0.45, solid_capstyle="butt", zorder=2)
    ax.plot([lo, hi], [yy, yy], ls="none", marker="|", ms=5, mew=1.1, color=NAVY, zorder=3)
    ax.plot(mean, yy, "o", ms=4.6, color=NAVY, mec="white", mew=0.8, zorder=4)
    ax.text(lo / 1.07, yy, f"{lo:.2f}", ha="right", va="center", fontsize=7, color=MUTED)
    ax.text(mean * 1.09, yy - 0.02, f"{mean:.2f}", ha="left", va="center", fontsize=7, color=INK,
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=3)
    ax.text(hi * 1.07, yy, f"{hi:.2f}", ha="left", va="center", fontsize=7, color=MUTED)

# column headers instead of a legend
top = ys[0] - 1.15
for x, s, ha, c in [(0.191 / 1.07, "min", "right", MUTED), (0.969 * 1.09, "mean", "left", INK),
                    (2.730 * 1.07, "max", "left", MUTED)]:
    ax.text(x, top, s, ha=ha, va="center", fontsize=7.5, color=c, style="italic")

ax.set_yticks(ys, [z for z, *_ in DATA])
ax.set_ylim(ys[-1] + 0.8, top - 0.6)
ax.set_xlim(0.062, 4.3)
ax.set_xticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
ax.minorticks_off()
ax.set_xlabel("Envelope U-value (W/m²·K), log scale")
ax.set_ylabel("ASHRAE climate zone")
ax.annotate("better insulated", xy=(0.0, -0.125), xycoords="axes fraction", ha="left", va="top",
            fontsize=7, color=MUTED)
ax.annotate("worse insulated", xy=(1.0, -0.125), xycoords="axes fraction", ha="right", va="top",
            fontsize=7, color=MUTED)
ax.text(-0.105, 0.0, "coldest", transform=ax.transAxes, ha="center", va="top", fontsize=7, color=MUTED)
ax.text(-0.105, 1.0, "hottest", transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=MUTED)

ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0, pad=6)
ax.tick_params(axis="x", length=3, width=0.6)

fig.tight_layout()
fig.savefig("envelope_uvalues.pdf")
fig.savefig("envelope_uvalues.png", dpi=220)
