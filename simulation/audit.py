"""Remeasure a historical evaluation directly from its compressed simulator archive."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

from building_gen.comfort import ComfortSchedule
from .fetch import HERE, digest, download

DATA = HERE.parent / "data"


def task_name(code):
    return ("office" if code[0] == "O" else "retail") + f'_c{int(code[1]=="S")}_o{code[3]}_t{code[4]}'


def measure(stream, calendar, target, source, seed):
    suite = json.loads((HERE / "config" / f"{calendar}.json").read_text())
    task = next(t for t in suite["tasks"] if t["name"] == task_name(target))
    zones = task["action_profile"]["controlled_entities"]
    schedule = ComfortSchedule.from_mapping(task["reward"]["parameters"]["comfort_schedule"])
    start = 1 if calendar == "validation" else 15
    counts = Counter()
    energy = people = squared = 0.0
    hashed = hashlib.sha256()

    def lines():
        for line in stream:
            hashed.update(line)
            yield line.decode("utf-8")

    for row in csv.DictReader(lines()):
        clock = json.loads(row["decision_clock_json"])
        month = int(row["window_start_month"])
        window = f"month{month:02}" + ("_days15_21" if calendar == "test" else "")
        i = counts[window]
        date = dt.date(2017, month, start) + dt.timedelta(days=i // 288)
        if (
            row["window"] != window
            or int(row["seed"]) != seed
            or i >= 2016
            or [clock[k] for k in ("month", "day_of_month", "hour", "minutes", "episode_step_index")]
            != [month, date.day, (i // 12) % 24, 5 * (i % 12 + 1), i]
        ):
            raise ValueError("Calendar, seed or interval mismatch")
        counts[window] += 1
        area = float(row["controlled_floor_area_m2"])
        energy += (
            float(row["controlled_hvac_energy_electricity_kwh"])
            + float(row["controlled_hvac_energy_natural_gas_kwh"])
        ) / area
        segments = json.loads(row["reward_components_json"])["comfort_segments"]
        interval_people = interval_squared = 0.0
        for s in segments:
            midpoint = (s["start_minute"] + s["end_minute"]) / 2
            band = schedule.band_at(s["zone_id"], weekday=s["weekday"], minute=midpoint, month=month)
            if (s["minimum_c"], s["maximum_c"]) != (band.minimum_c, band.maximum_c):
                raise ValueError("Recorded comfort band differs from the target schedule")
            weight = s["occupant_count"] * (s["end_minute"] - s["start_minute"]) / 60
            interval_people += weight
            interval_squared += weight * band.violation_c(s["temperature_c"]) ** 2
        if not math.isclose(
            interval_people, float(row["occupied_person_hours"]), abs_tol=1e-8
        ) or not math.isclose(
            interval_squared,
            float(row["occupied_comfort_exposure_person_celsius_squared_hours_per_m2"]) * area,
            abs_tol=1e-8,
        ):
            raise ValueError("Mismatch in per-interval occupied exposure")
        squared += interval_squared
        people += interval_people
        if source != "historical":
            requested = json.loads(row["submitted_action_physical_json"])
            applied = json.loads(row["actuator_readback_physical_json"])
            if (
                row["actuator_readback_status"] != "available"
                or row["actuator_readback_phase"] != "end_zone_timestep"
                or [v["zone"] for v in requested] != zones
                or [v["zone"] for v in applied] != zones
            ):
                raise ValueError("Missing setpoint readbacks")
            for a, b in zip(requested, applied):
                for name in ("heating_setpoint_c", "cooling_setpoint_c"):
                    if not math.isclose(a[name], b[name], rel_tol=0, abs_tol=1e-6):
                        raise ValueError("Setpoint readback mismatch")
    if len(counts) != 12 or set(counts.values()) != {2016}:
        raise ValueError(f"Expected twelve complete weeks: {counts}")
    return hashed.hexdigest(), [energy, squared, people]


def audit(calendar, source, target, seed, archive):
    spec = json.loads((HERE / "archive-locations.json").read_text())
    with (HERE / "archives.csv").open() as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if (r["calendar"], r["source"], r["target"], int(r["seed"])) == (calendar, source, target, seed)
        ]
    if len(rows) != 1:
        raise ValueError("No pinned archive for this case. See simulation/README.md for coverage.")
    row = rows[0]
    if archive is None:
        if not row["directory"]:
            raise ValueError("Provide --archive with SHA-256 " + row["sha256"])
        path = spec["directories"][int(row["directory"])] + "/" + row["filename"]
        archive = download(spec, path, row["sha256"])
    elif digest(archive) != row["sha256"]:
        raise ValueError("Archive SHA-256 mismatch")
    # Earlier native archives also contain a warm-up trial. Identify the scored trajectory by its bytes.
    matches = []
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".steps.csv"):
                stream = tar.extractfile(member)
                hashed = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hashed.update(block)
                if hashed.hexdigest() == row["trajectory_sha256"]:
                    matches.append(member.name)
    if len(matches) != 1:
        raise ValueError("Expected one trajectory matching the registered checksum")
    trajectory = matches[0]
    diag_name = trajectory.removesuffix(".steps.csv") + ".energyplus-diagnostics.json"
    totals = diagnostic = None
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            if member.isfile() and member.name == trajectory:
                actual, totals = measure(tar.extractfile(member), calendar, target, source, seed)
                if actual != row["trajectory_sha256"]:
                    raise ValueError("Trajectory SHA-256 mismatch")
            if member.isfile() and member.name == diag_name:
                diagnostic = json.load(tar.extractfile(member))["episodes"]
    if totals is None or diagnostic is None or len(diagnostic) != 12:
        raise ValueError("Missing simulator evidence")
    for episode in diagnostic:
        d = episode["diagnostics"]
        if (
            not d["completed_successfully"]
            or d["severe_error_count"]
            or d["fatal_error_count"]
            or not episode["executed_model_sha256_matches"]
        ):
            raise ValueError("Simulator diagnostics failed")
    with (DATA / ("historical.csv" if source == "historical" else "outcomes.csv")).open() as f:
        (expected,) = [
            r
            for r in csv.DictReader(f)
            if r["calendar"] == calendar
            and r["target"] == target
            and r.get("source", "historical") == source
            and int(r.get("seed", 0)) == seed
        ]
    for key, value in zip(("energy_kwh_m2", "squared_exposure_person_c2_h", "person_hours"), totals):
        if not math.isclose(value, float(expected[key]), rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f"Archived total differs for {key}")
    return dict(
        calendar=calendar,
        source=source,
        target=target,
        seed=seed,
        intervals=24192,
        energy_kwh_m2=totals[0],
        occupied_mse_c2=totals[1] / totals[2],
        trajectory_sha256=actual,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("calendar", choices=("validation", "test"))
    p.add_argument("source", help="Source code, or historical for the target reference")
    p.add_argument("target")
    p.add_argument("seed", type=int, choices=range(10))
    p.add_argument("--archive", type=Path, help="Use a local archive of the registered SHA-256")
    args = p.parse_args()
    print(json.dumps(audit(**vars(args)), indent=2))
