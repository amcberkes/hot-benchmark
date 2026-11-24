"""Audit all twelve monthly native weeks of one factorial task."""

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from building_gen.benchmark.config import load_benchmark_suite
from building_gen.benchmark.reporting import _artifact_payload
from building_gen.benchmark.checkpoint_selection import weekly_validation_outcomes
from building_gen.comfort import ComfortSchedule
from building_gen.weekly_calibration import _verified_trajectory_rows
from building_gen.weekly_reward import occupied_comfort_exposures
from simulation.intervals import verify_segment_bands


def audit(
    suite_path,
    experiment_name,
    results,
    *,
    algorithm="native_incumbent",
    seed=0,
    expected_windows=12,
    declared_algorithm="native_incumbent",
    result_path=None,
):
    suite = load_benchmark_suite(suite_path)
    experiment = suite.experiment_map[experiment_name]
    task = suite.task_map[experiment.evaluation_tasks[0]]
    protocol = experiment.evaluation_protocol
    if (
        declared_algorithm not in ("native_incumbent", "scheduled_thermostat")
        or experiment.algorithm_id != declared_algorithm
        or (declared_algorithm == "native_incumbent" and experiment.parameter_mapping)
        or len(protocol.windows) != expected_windows
    ):
        raise ValueError("Require twelve declared historical-controller weeks")
    if result_path is None:
        (path,) = results.glob("*.result.json")
    else:
        path = result_path
    result = json.loads(path.read_text())
    run = result["run"]
    if (
        result["status"] != "succeeded"
        or result["failure"] is not None
        or run["evaluation_task_id"] != task.task_id
        or run["algorithm_id"] != algorithm
        or run["seed"] != seed
        or result["provenance"]["dirty"]
    ):
        raise ValueError("Native result identity or status mismatch")
    episodes = json.loads(
        _artifact_payload(result, path, role="energyplus_diagnostics", media_type="application/json").payload
    )["episodes"]
    if len(episodes) != expected_windows or {(e["window"], e["episode_index"]) for e in episodes} != {
        (w.name, 0) for w in protocol.windows
    }:
        raise ValueError("Incomplete simulator coverage")
    for e in episodes:
        d = e["diagnostics"]
        if (
            not d["completed_successfully"]
            or d["severe_error_count"]
            or d["fatal_error_count"]
            or e["executed_model_sha256_matches"] is not True
        ):
            raise ValueError("Unacceptable simulator diagnostics")
    (artifact,) = [a for a in result["artifacts"] if a["role"] == "step_trajectory"]
    schedule = ComfortSchedule.from_mapping(task.reward.parameter_mapping["comfort_schedule"])
    counts = {w.name: 0 for w in protocol.windows}
    windows = {w.name: w for w in protocol.windows}
    rows = []
    frequency = task.episode.timesteps_per_hour
    for row in _verified_trajectory_rows(artifact, path):
        w = windows[row["window"]]
        index = counts[w.name]
        date = dt.date(2017, w.start_month, w.start_day) + dt.timedelta(days=index // (24 * frequency))
        expected = dict(
            calendar_year=2017,
            month=date.month,
            day_of_month=date.day,
            day_of_week=(date.weekday() + 1) % 7 + 1,
            day_of_year=date.timetuple().tm_yday,
            hour=(index // frequency) % 24,
            minutes=(index % frequency + 1) * 60 // frequency,
            timestep_index=index % frequency + 1,
            timesteps_per_hour=frequency,
            episode_step_index=index,
            calendar_year_semantics="typical_weather_year",
        )
        if json.loads(row["decision_clock_json"]) != expected:
            raise ValueError("Decision clock mismatch")
        segments = json.loads(row["reward_components_json"])["comfort_segments"]
        verify_segment_bands(segments, schedule, date.month)
        people = math.fsum(s["occupant_count"] * (s["end_minute"] - s["start_minute"]) / 60 for s in segments)
        squared = occupied_comfort_exposures(segments)["mse"]
        for actual, recorded in [
            (people, float(row["occupied_person_hours"])),
            (
                squared,
                float(row["occupied_comfort_exposure_person_celsius_squared_hours_per_m2"])
                * float(row["controlled_floor_area_m2"]),
            ),
        ]:
            if not math.isclose(actual, recorded, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("Independent occupied arithmetic mismatch")
        counts[w.name] += 1
        rows.append({k: v for k, v in row.items() if not k.endswith("_json")})
    if set(counts.values()) != {168 * frequency}:
        raise ValueError("Missing monthly steps")
    outcomes = weekly_validation_outcomes(rows, task_id=task.task_id, protocol=protocol)
    totals = dict(
        energy=math.fsum(v.hvac_energy_kwh_per_m2 for v in outcomes.values()),
        people=math.fsum(v.occupied_person_hours for v in outcomes.values()),
        squared=math.fsum(v.comfort_exposure_person_celsius_squared_hours for v in outcomes.values()),
    )
    metrics = {m["name"]: m["value"] for m in result["metrics"]}
    for name, expected in [
        ("completed_windows", expected_windows),
        ("evaluated_hours", 168 * expected_windows),
        ("controlled_hvac_energy_intensity_kwh_per_m2", totals["energy"]),
        ("occupied_person_hours", totals["people"]),
        ("occupied_comfort_exposure_person_celsius_squared_hours", totals["squared"]),
    ]:
        if not math.isclose(metrics[name], expected, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("Recorded metric mismatch")
    return dict(
        schema="hot-sixteen-native-qualification/1",
        status=(
            "qualified_controlled_execution"
            if algorithm != "native_incumbent"
            else (
                "qualified_native_execution" if expected_windows == 12 else "diagnostic_native_windows_passed"
            )
        ),
        task_id=task.task_id,
        experiment=experiment_name,
        source_revision=result["provenance"]["revision"],
        trajectory_sha256=artifact["sha256"],
        counts=counts,
        totals=totals,
        occupied_mse_c2=totals["squared"] / totals["people"],
        scope=f"{expected_windows} independent monthly weeks; complete clocks and original occupied scoring. Zero severe/fatal errors. Execution qualification only; no learned performance or full-year claim.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "experiment", "results", "output"):
        p.add_argument("--" + name, type=Path if name != "experiment" else str, required=True)
    p.add_argument("--expected-windows", type=int, choices=(1, 3, 12, 53), default=12)
    a = p.parse_args()
    result = audit(a.suite, a.experiment, a.results, expected_windows=a.expected_windows)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
