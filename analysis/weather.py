"""Recompute the weather descriptors from the 192 pinned public EPWs."""

import argparse
import csv
import hashlib
import math
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REVISION = "06e84189e6c1e196f689dee26acd49533aa5f3af"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="EPW cache; missing files are downloaded from BuildingBench/HOT"
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    with (Path(__file__).resolve().parents[1] / "data/weather.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        dest = args.directory / Path(row["path"]).name
        if dest.exists():
            body = dest.read_bytes()
        else:
            url = f'https://huggingface.co/datasets/BuildingBench/HOT/resolve/{REVISION}/{quote(row["path"])}'
            with urlopen(url, timeout=120) as response:
                body = response.read()
        if hashlib.sha256(body).hexdigest() != row["sha256"]:
            raise ValueError(f"EPW checksum mismatch: {dest}")
        if not dest.exists():
            dest.write_bytes(body)
        series = [
            float(line.split(",")[6]) for line in body.decode("latin1").splitlines()[8:] if line.strip()
        ]
        # Match the published computation: exclude missing values, then group consecutive valid hours.
        valid = [x for x in series if -100 < x < 60]
        daily = [sum(valid[i : i + 24]) / 24 for i in range(0, len(valid) - 23, 24)]
        values = (
            sum(max(0, 18.3 - x) for x in daily),
            sum(max(0, x - 18.3) for x in daily),
            max(valid),
            min(valid),
        )
        if not all(
            math.isclose(x, float(row[k]), abs_tol=1e-9)
            for k, x in zip(("hdd", "cdd", "tmax", "tmin"), values)
        ):
            raise ValueError(f"Weather descriptor mismatch: {dest.name}")
    print(f"{len(rows)} EPWs verified and remeasured")


if __name__ == "__main__":
    main()
