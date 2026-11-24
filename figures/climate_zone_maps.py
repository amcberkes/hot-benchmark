"""Climate-zone locations of the HOT dataset (thesis palette): writes climate_zones_world.pdf (main text)
and climate_zones_north_america.pdf (appendix).
Usage: python3 climate_zone_map.py <dir with Natural Earth geojson>"""
import json, sys, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

D = sys.argv[1] if len(sys.argv) > 1 else "."

# ---- thesis style (same block as the other figures) -----------------------
NAVY = "#2B3A67"                                   # hex as defined in main.tex
INK, MUTED, LAND, WATER = "#222222", "#6B7280", "#E3E6EA", "#FFFFFF"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],   # thesis font (newpx)
    "font.size": 7, "text.color": INK, "pdf.fonttype": 42,
})
TEXT_WIDTH_IN = 5.6

# ---- data: zone -> base city, further locations (name, lat, lon) ----------
from pathlib import Path as FilePath
Z = json.loads((FilePath(__file__).resolve().parents[1] / "data/locations.json").read_text())

# ---- projections -----------------------------------------------------------
def equal_earth(lon, lat):
    A1, A2, A3, A4, M = 1.340264, -0.081106, 0.000893, 0.003796, math.sqrt(3) / 2
    lam, phi = np.radians(lon), np.radians(lat)
    t = np.arcsin(M * np.sin(phi)); t2 = t * t; t6 = t2 ** 3
    x = lam * np.cos(t) / (M * (A1 + 3 * A2 * t2 + t6 * (7 * A3 + 9 * A4 * t2)))
    y = t * (A1 + A2 * t2 + t6 * (A3 + A4 * t2))
    return x, y

def albers(lon, lat, p1=30.0, p2=55.0, lon0=-103.0, lat0=40.0):
    p1, p2, l0, f0 = map(math.radians, (p1, p2, lon0, lat0))
    n = (math.sin(p1) + math.sin(p2)) / 2
    C = math.cos(p1) ** 2 + 2 * n * math.sin(p1)
    rho0 = math.sqrt(C - 2 * n * math.sin(f0)) / n
    lam, phi = np.radians(lon), np.radians(lat)
    rho = np.sqrt(np.maximum(C - 2 * n * np.sin(phi), 0)) / n
    th = n * (lam - l0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)

def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        yield poly

def add_polys(ax, feats, proj, keep=lambda lon, lat: True, **kw):
    verts, codes = [], []
    for f in feats:
        for poly in rings(f["geometry"]):
            a0 = np.asarray(poly[0], float)
            if not keep(a0[:, 0].mean(), a0[:, 1].mean()):
                continue
            for ring in poly:
                a = np.asarray(ring, float)
                x, y = proj(a[:, 0], np.clip(a[:, 1], -89.9, 89.9))
                verts += list(zip(x, y))
                codes += [Path.MOVETO] + [Path.LINETO] * (len(a) - 2) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), **kw))

def load(name):
    return json.load(open(f"{D}/{name}.geojson"))["features"]

# climate scale: thermal zone 0 (hottest) -> 8 (coldest). Warm half darkens towards the hot end,
# cool half darkens towards the cold end, so neighbouring zones differ in hue and in lightness.
ZONE_COLOURS = ["#7F2416", "#C4471C", "#E88A2A", "#F2C040", "#F8F5E6", "#86CBC2", "#3A93AA", "#295C96", "#14264F"]
MID = "#9AA0A8"
def zone_colour(z):
    return ZONE_COLOURS[int(z[0])]

def dots(ax, proj, pts, colour, base):
    if not pts:
        return
    x, y = proj(np.array([p[2] for p in pts]), np.array([p[1] for p in pts]))
    if base:   # base city: larger, dark outline
        ax.plot(x, y, "o", ms=5.0, color=colour, mec=INK, mew=0.8, zorder=6, ls="none")
    else:      # further location: smaller, no outline
        ax.plot(x, y, "o", ms=3.8, color=colour, mec="#4B5563", mew=0.35, zorder=5, ls="none")

def key(ax, x, y, dy=0.055, ndy=0.036):
    """Marker key and climate ramp, in axes coordinates, no box."""
    t = ax.transAxes
    ax.plot(x, y, "o", ms=5.0, color=MID, mec=INK, mew=0.8, transform=t, clip_on=False)
    ax.text(x + 0.018, y, "base city (real recorded weather)", transform=t, va="center", fontsize=6.5, color=MUTED)
    ax.plot(x, y - dy, "o", ms=3.8, color=MID, mec="#4B5563", mew=0.35, transform=t, clip_on=False)
    ax.text(x + 0.018, y - dy, "further location in the same zone", transform=t, va="center", fontsize=6.5, color=MUTED)
    step = 0.021
    for i in range(9):
        ax.plot(x + i * step, y - 2.15 * dy, "o", ms=4.4, color=ZONE_COLOURS[i], mec="#4B5563", mew=0.35, transform=t, clip_on=False)
        ax.text(x + i * step, y - 2.15 * dy - ndy, str(i), transform=t, ha="center", va="center", fontsize=6, color=MUTED)
    ax.text(x + 9 * step, y - 2.15 * dy, "zone number: 0 hottest, 8 coldest", transform=t, va="center",
            fontsize=6.5, color=MUTED)

def label(ax, proj, lat, lon, text, dx, dy, ha="left", color=INK, weight="normal", size=7):
    x, y = proj(np.array([lon]), np.array([lat]))
    ax.annotate(text, (x[0], y[0]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=size, color=color, fontweight=weight, zorder=7)

inNA = lambda p: -170 < p[2] < -50 and p[1] > 15

# ---- Figure 1: world (main text) -------------------------------------------
_x, _ = equal_earth(np.array([-180.0, 180.0]), np.array([0.0, 0.0]))
_, _y = equal_earth(np.array([0.0, 0.0]), np.array([-57.0, 84.0]))
WLIM = (_x[0], _x[1], _y[0], _y[1])
fig = plt.figure(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * (WLIM[3] - WLIM[2]) / (WLIM[1] - WLIM[0])))
axw = fig.add_axes([0, 0, 1, 1])
world = [f for f in load("ne_110m_admin_0_countries") if f["properties"]["ADMIN"] != "Antarctica"]
add_polys(axw, world, equal_earth, fc=LAND, ec="white", lw=0.35, zorder=1)
for z, (base, more) in Z.items():
    dots(axw, equal_earth, more, zone_colour(z), False)
for z, (base, more) in Z.items():
    dots(axw, equal_earth, [base], zone_colour(z), True)

W = {  # zone -> (lat, lon of anchor, text, dx, dy, ha)
 "0A": [(10.82, 106.63, "0A Ho Chi Minh City", 5, -6, "left"), (6.4, 2.4, "0A", 0, -8, "center")],
 "0B": [(25.20, 55.27, "0B Dubai", 2, -9, "center")],
 "1A": [(-17.4, 178.4, "1A", -7, 0, "right")],
 "1B": [(26.91, 75.79, "1B Jaipur", 7, 2, "left")],
 "2A": [(-25.9, -56.4, "2A", 9, 0, "left")],
 "2B": [(30.6, 30.7, "2B", -7, -4, "right")],
 "3A": [(31.2, 120.2, "3A", 8, -4, "left")],
 "3B": [(-28.9, 24.3, "3B", 8, 0, "left")],
 "3C": [(-34.2, 116.5, "3C", -8, 0, "right")],
 "4A": [(37.1, 127.4, "4A", 8, 2, "left")],
 "4B": [(41.8, 52.0, "4B", 7, 2, "left")],
 "4C": [(41.8, -6.0, "4C", -8, 0, "right")],
 "5A": [(50.8, 14.8, "5A", 0, 8, "center")],
 "7":  [(65.7, 24.4, "7", 8, 0, "left")],
 "8":  [(50.6, 109.3, "8", 9, 2, "left")],
}
for z, items in W.items():
    for lat, lon, text, dx, dy, ha in items:
        label(axw, equal_earth, lat, lon, text, dx, dy, ha)
key(axw, 0.045, 0.235)
axw.set_xlim(WLIM[0], WLIM[1]); axw.set_ylim(WLIM[2], WLIM[3])
axw.set_aspect("equal"); axw.axis("off")
fig.savefig("climate_zones_world.pdf"); fig.savefig("climate_zones_world.png", dpi=220)

# ---- Figure 2: North America (appendix) ------------------------------------
_pts = [p for b, m in Z.values() for p in [b] + m if inNA(p)]
_px, _py = albers(np.array([p[2] for p in _pts]), np.array([p[1] for p in _pts]))
NLIM = (_px.min() - 0.115, _px.max() + 0.14, _py.min() - 0.025, _py.max() + 0.03)
hn, STRIP = TEXT_WIDTH_IN * (NLIM[3] - NLIM[2]) / (NLIM[1] - NLIM[0]), 0.62
fig = plt.figure(figsize=(TEXT_WIDTH_IN, hn + STRIP))
axn = fig.add_axes([0, STRIP / (hn + STRIP), 1, hn / (hn + STRIP)])
axk = fig.add_axes([0, 0, 1, STRIP / (hn + STRIP)]); axk.axis("off"); axk.set_xlim(0, 1); axk.set_ylim(0, 1)
near = lambda lon, lat: -179 < lon < -40 and 5 < lat < 85
add_polys(axn, load("ne_50m_admin_0_countries"), albers, keep=near, fc=LAND, ec="white", lw=0.5, zorder=1)
states = [f for f in load("ne_50m_admin_1_states_provinces_lakes")
          if f["properties"].get("adm0_a3") in ("USA", "CAN")]
add_polys(axn, states, albers, keep=near, fc="none", ec="white", lw=0.35, zorder=2)
lakes = [f for f in load("ne_50m_lakes") if (f["properties"].get("scalerank") or 9) <= 1]
add_polys(axn, lakes, albers, keep=near, fc=WATER, ec="none", zorder=3)
for z, (base, more) in Z.items():
    dots(axn, albers, [p for p in more if inNA(p)], zone_colour(z), False)
for z, (base, more) in Z.items():
    dots(axn, albers, [p for p in [base] if inNA(p)], zone_colour(z), True)

B = {  # base-city label offsets: zone -> (dx, dy, ha)
 "1A": (6, -2, "left"), "2A": (-6, -3, "right"), "2B": (-3, -8, "center"), "3A": (6, 3, "left"),
 "3B": (5, -7, "left"), "3C": (-6, -2, "right"), "4A": (6, -3, "left"), "4B": (6, 4, "left"),
 "4C": (7, -5, "left"), "5A": (2, -8, "center"), "5B": (7, 0, "left"), "5C": (-7, -3, "right"),
 "6A": (6, -6, "left"), "6B": (6, 5, "left"), "7": (0, 9, "center"), "8": (7, 0, "left"),
}
for z, (dx, dy, ha) in B.items():
    name, lat, lon = Z[z][0]
    label(axn, albers, lat, lon, f"{z} {name}", dx, dy, ha)
for lat, lon, z, dx, dy, ha in [(42.5, -114.5, "5B", -3, -8, "center"), (49.9, -125.3, "5C", -7, 3, "right"),
                                (45.9, -73.2, "6A", 9, 3, "left"), (43.7, -99.5, "6B", 0, -9, "center")]:
    label(axn, albers, lat, lon, z, dx, dy, ha, MUTED)
key(axk, 0.03, 0.80, dy=0.25, ndy=0.17)
axn.set_xlim(NLIM[0], NLIM[1]); axn.set_ylim(NLIM[2], NLIM[3])
axn.set_aspect("equal"); axn.axis("off")
fig.savefig("climate_zones_north_america.pdf"); fig.savefig("climate_zones_north_america.png", dpi=220)
