"""Typography shared by the transfer plots."""

from pathlib import Path
import shutil
import subprocess

import matplotlib as mpl
from matplotlib import font_manager


def apply_style(font_size, text_color):
    family = "TeX Gyre Pagella"
    # TeX Live fonts are not always registered with the operating system.
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich:
        lookup = subprocess.run([kpsewhich, "texgyrepagella-regular.otf"], capture_output=True, text=True)
        if lookup.returncode == 0 and lookup.stdout.strip():
            for font in Path(lookup.stdout.strip()).parent.glob("texgyrepagella-*.otf"):
                font_manager.fontManager.addfont(str(font))
    try:
        for weight in ("normal", "bold"):
            properties = font_manager.FontProperties(family=[family], weight=weight)
            font_manager.findfont(properties, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            "Install the regular and bold TeX Gyre Pagella fonts to reproduce the figures."
        ) from exc
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family],
            "font.size": font_size,
            "text.color": text_color,
            "pdf.fonttype": 42,
        }
    )
