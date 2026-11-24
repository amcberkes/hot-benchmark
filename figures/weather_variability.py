"""Real versus typical weather variability by climate zone, in the thesis palette.
Positions and classes are read from the published figure of Berkes et al. (accurate to roughly
+-50 degree days); replace DATA with the paper's own values where available."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- thesis style (same block as the other figures; hexes as defined in main.tex) ----
NAVY, TEAL, AMBER, GREY = "#2B3A67", "#1F8A80", "#C98A1B", "#A3A8B0"
INK, MUTED, GRID = "#1C2233", "#5B6170", "#E5E7EB"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],   # thesis font (newpx)
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42,
})
TEXT_WIDTH_IN = 5.6

# zone: (HDD, CDD, class, label dx, dy, ha)   class: 3 much more, 2 clearly more, 1 more,
#                                                    0 similar, -1 less variable than typical weather
DATA = {
 "0A": (70, 6354, 3, 11, 5, "left"),    "0B": (111, 6605, 1, 0, 8, "center"),
 "1A": (198, 5309, 1, -7, -3, "right"), "1B": (384, 5714, 3, 12, 0, "left"),
 "2A": (610, 4402, 1, -7, 0, "right"),  "2B": (900, 4382, 1, 7, 0, "left"),
 "3A": (1739, 2954, 0, 7, -2, "left"),  "3B": (1528, 3289, -1, 7, 3, "left"),
 "4A": (2794, 2011, 1, 7, -1, "left"),  "4B": (2533, 2282, -1, 6, 5, "left"),
 "4C": (2782, 1079, 0, -7, 0, "right"),
 "5A": (3757, 1431, 1, 7, -3, "left"),  "5B": (3624, 1722, 1, 0, 8, "center"),
 "5C": (3386, 628, 1, -7, -2, "right"),
 "6A": (4714, 1383, 1, 5, 6, "left"),   "6B": (4472, 1221, 1, 0, -9, "center"),
 "7":  (5714, 1088, 1, 0, 8, "center"), "8":  (7288, 648, 2, -1, 10, "center"),
}
STYLE = {3: (AMBER, 15), 2: (AMBER, 8.5), 1: (AMBER, 5.5), 0: (GREY, 5.5), -1: (NAVY, 5.5)}

fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.7))
for zone, (hdd, cdd, cls, dx, dy, ha) in sorted(DATA.items(), key=lambda kv: -kv[1][2]):
    colour, size = STYLE[cls]
    ax.plot(hdd, cdd, "o", ms=size, color=colour, mec="white", mew=0.9, alpha=0.92, zorder=3 + (3 - cls),
            clip_on=False)
    ax.annotate(zone, (hdd, cdd), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=8, color=INK, zorder=9)

# key, no box
kx, ky, step = 0.535, 0.95, 0.082
ax.text(kx - 0.015, ky, "Real weather, compared with typical:", transform=ax.transAxes, va="center",
        fontsize=9.5, color=INK)
rows = [(AMBER, "more variable"), (GREY, "about the same"), (NAVY, "less variable")]
for i, (c, t) in enumerate(rows, start=1):
    ax.plot(kx, ky - i * step, "o", ms=7.5, color=c, mec="white", mew=0.9, transform=ax.transAxes, clip_on=False)
    ax.text(kx + 0.032, ky - i * step, t, transform=ax.transAxes, va="center", fontsize=9.5, color=INK)

ax.set_xlim(-450, 7700); ax.set_ylim(0, 7200)
ax.set_xticks([0, 2000, 4000, 6000]); ax.set_yticks([0, 2000, 4000, 6000])
ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_xlabel("Heating degree days"); ax.set_ylabel("Cooling degree days")
ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0, pad=4)
fig.tight_layout()
fig.savefig("weather_variability.pdf"); fig.savefig("weather_variability.png", dpi=220)
