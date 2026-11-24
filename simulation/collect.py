"""Collect a complete rerun into the same seed and historical-control tables as the stored experiment."""

import argparse
import json
from pathlib import Path

from analysis.matrix import SOURCES, TARGETS, write_csv
from .audit import task_name
from building_gen.benchmark.config import load_benchmark_suite
from .fetch import HERE, digest


def collect(work, out):
    from tempfile import TemporaryDirectory

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    rows, native = [], []
    for calendar in ("validation", "test"):
        suite_path = HERE / "config" / f"{calendar}.json"
        suite = load_benchmark_suite(suite_path)
        for t in TARGETS:
            task_id = suite.task_map[task_name(t)].task_id
            windows = {
                w.name for w in suite.experiment_map["native_" + task_name(t)].evaluation_protocol.windows
            }

            def totals(path, status):
                audit = json.loads(path.read_text())
                if (
                    audit["task_id"] != task_id
                    or audit["status"] != status
                    or audit["experiment"] != "native_" + task_name(t)
                    or set(audit["counts"]) != windows
                    or set(audit["counts"].values()) != {2016}
                ):
                    raise ValueError(f"Incomplete or mislabelled audit at {path}")
                return audit, [audit["totals"][k] for k in ("energy", "squared", "people")]

            _, v = totals(work / calendar / "historical" / t / "audit.json", "qualified_native_execution")
            native.append([calendar, t, *v])
            for s in SOURCES:
                for k in range(10):
                    a, v = totals(
                        work / calendar / s / f"seed{k}" / t / "audit.json", "qualified_controlled_execution"
                    )
                    if (
                        a["seed"],
                        a["step"],
                        a["source_context"],
                        a["target_context"],
                        a["suite_sha256"],
                    ) != (k, 483840, task_name(s), task_name(t), digest(suite_path)):
                        raise ValueError(f"Incorrect source, checkpoint or suite for {s}, {t}, seed {k}")
                    rows.append([calendar, s, t, k, *v])
    fields = ["energy_kwh_m2", "squared_exposure_person_c2_h", "person_hours"]
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "measurements"
        stage.mkdir()
        write_csv(stage / "outcomes.csv", ["calendar", "source", "target", "seed", *fields], rows)
        write_csv(stage / "historical.csv", ["calendar", "target", *fields], native)
        stage.rename(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    collect(args.work, args.out)
