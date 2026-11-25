"""Rebuild the empirical HOT results from the stored measurements."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from analysis import dataset, matrix

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "figures"


def figures(out):
    sys.path.insert(0, str(PLOTS))
    from predeclared_comparisons import plot_predeclared
    from transfer_figure_style import apply_style
    from transfer_matrix import plot_transfer_matrix
    from source_target_changes import plot_source_target_changes

    apply_style(8, "#1C2233")
    plot_predeclared(out / "figure-A.2.csv", out / "figure-A.2.pdf")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--measurements",
        type=Path,
        help="New full matrix from simulation.collect, using the same task design",
    )
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument(
        "--details", action="store_true", help="Also save the numerical summary and four intermediate CSVs"
    )
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
    measurements = args.measurements.resolve() if args.measurements else None
    os.environ.setdefault("MPLBACKEND", "Agg")
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        work = Path(tmp)
        os.environ.setdefault("MPLCONFIGDIR", str(work / ".matplotlib"))
        summary = matrix.reproduce(ROOT / "data", work, measurements)
        if not args.tables_only:
            figures(work)
        if args.details:
            summary.update(dataset.diagnostics(ROOT / "data", work))
            (work / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        for file in work.iterdir():
            if file.is_file() and (args.details or file.match("table-*.csv") or file.match("figure-*.pdf")):
                file.replace(out / file.name)
    print(f"{summary['test_joint']}/960 test deployments passed both criteria. Results in {out}")


if __name__ == "__main__":
    main()
