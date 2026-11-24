"""Shared typography for the two thesis transfer figures."""
from pathlib import Path
import shutil
import subprocess

import matplotlib as mpl
from matplotlib import font_manager


def apply_style(font_size, text_color):
    # A TeX Live font may be visible to LaTeX but absent from Matplotlib's system-font index.
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich:
        lookup = subprocess.run([kpsewhich, "texgyrepagella-regular.otf"], capture_output=True, text=True)
        if lookup.returncode == 0 and lookup.stdout.strip():
            for font in Path(lookup.stdout.strip()).parent.glob("texgyrepagella-*.otf"):
                font_manager.fontManager.addfont(str(font))
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],
        "font.size": font_size,
        "text.color": text_color,
        "pdf.fonttype": 42,
    })
