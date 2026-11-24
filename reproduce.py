"""Rebuild the HOT tables, figures and numerical summaries for Chapter 3 and Appendix A."""

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

from analysis import dataset, matrix

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "figures"


def figures(out, maps):
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PLOTS))
    from maps import map_layers
    from transfer_figure_style import apply_style
    from transfer_matrix import plot_transfer_matrix
    from source_target_changes import plot_source_target_changes

    apply_style(8, "#1C2233")
    if maps is None:
        maps = out / "natural-earth"
        map_layers(maps, download=True)
    else:
        map_layers(maps, download=False)
    with tempfile.TemporaryDirectory() as work:
        subprocess.run(
            ["pdflatex", "-halt-on-error", "-interaction=batchmode", str(PLOTS / "hot_overview.tex")],
            cwd=work,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        shutil.copyfile(Path(work) / "hot_overview.pdf", out / "figure-3.1.pdf")
    os.chdir(out)
    for script in (
        "envelope_uvalues",
        "climate_zone_maps",
        "weather_transfer_scores",
        "weather_variability",
        "predeclared_comparisons",
    ):
        sys.argv = [script, str(maps)]
        with plt.rc_context():
            runpy.run_path(str(PLOTS / f"{script}.py"), run_name="__main__")
        plt.close("all")
    rows = {(r["source"], r["target"]): r for r in matrix.read_csv(out / "transfer-matrix.csv")}

    def grid(field, convert):
        return [[convert(rows[s, t][field]) for s in matrix.SOURCES] for t in matrix.TARGETS]

    plot_transfer_matrix(grid("joint_of_10", int), out)
    # The chapter heatmap labels use values rounded first to two decimal places.
    plot_source_target_changes(
        grid("energy_pct", lambda x: round(float(x), 2)),
        grid("discomfort_pct", lambda x: round(float(x), 2)),
        out,
    )
    names = (
        "envelope_uvalues",
        "climate_zones_world",
        "weather_transfer_scores",
        "weather_variability",
        "transfer_matrix",
        "source_target_changes",
        "climate_zones_north_america",
        "predeclared_comparisons",
    )
    for label, name in zip(("3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "A.1", "A.2"), names):
        (out / f"{name}.pdf").replace(out / f"figure-{label}.pdf")
        (out / f"{name}.png").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--measurements",
        type=Path,
        help="New full matrix from simulation.collect, using the same task design",
    )
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--maps", type=Path, help="Directory containing the four pinned Natural Earth layers")
    args = parser.parse_args()
    out = args.out.resolve()
    if (
        out == ROOT
        or ROOT in out.parents
        and out.relative_to(ROOT).parts[0]
        in ("data", "analysis", "simulation", "figures", "building_gen", ".git")
    ):
        parser.error("Choose a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    summary = {**matrix.reproduce(ROOT / "data", out, args.measurements), **dataset.reproduce(ROOT, out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not args.tables_only:
        figures(out, args.maps.resolve() if args.maps else None)
    print(f"{summary['test_joint']}/960 test deployments passed both criteria. Results in {out}")


if __name__ == "__main__":
    main()
