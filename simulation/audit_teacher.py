"""Audit the declared source-local teacher, physical trajectories and weekly occupancy."""

import argparse
import json
import math
from pathlib import Path

from building_gen.benchmark.config import load_benchmark_suite
from building_gen.benchmark.registry import default_algorithm_registry
from building_gen.weekly_calibration import _verified_trajectory_rows
from building_gen.weekly_reward import WeeklyTrainingReward
from simulation.intervals import actuator_readbacks
from simulation.audit_reference import audit
from simulation.export_teacher import export


def report(suite_path, source, data, results, examples):
    suite = load_benchmark_suite(suite_path)
    task = suite.task_map[source]
    calibrated = WeeklyTrainingReward(task, data)
    ids = {t.name: t.task_id for t in suite.tasks}
    version = default_algorithm_registry().get("scheduled_thermostat").version
    expected = {e.experiment_id(task_ids=ids, algorithm_version=version): e for e in suite.experiments}
    if len(expected) != 2:
        raise ValueError("Require two seasonal teacher experiments")
    seen = set()
    checks = {}
    for path in sorted(results.glob("*.result.json")):
        result = json.loads(path.read_text())
        identity = result["run"]["experiment_id"]
        if identity not in expected or identity in seen or result["run"]["algorithm_version"] != version:
            raise ValueError("Unexpected or duplicate teacher experiment")
        seen.add(identity)
        experiment = expected[identity]
        checked = audit(
            suite_path,
            experiment.name,
            results,
            algorithm="scheduled_thermostat",
            declared_algorithm="scheduled_thermostat",
            expected_windows=len(experiment.evaluation_protocol.windows),
            result_path=path,
        )
        (artifact,) = [a for a in result["artifacts"] if a["role"] == "step_trajectory"]
        count, error = actuator_readbacks(
            _verified_trajectory_rows(artifact, path), task.action_profile.controlled_entities
        )
        for window in experiment.evaluation_protocol.windows:
            key = (window.start_month, window.start_day, window.end_month, window.end_day)
            role, calibration = calibrated.weeks[key]
            if role != "training":
                raise ValueError("Teacher examples must use training weeks")
            # Physical audit already reconciles each interval. Also establish
            # that the teacher did not change the catalog's exogenous occupancy.
            people = math.fsum(
                float(row["occupied_person_hours"])
                for row in _verified_trajectory_rows(artifact, path)
                if row["window"] == window.name
            )
            if not math.isclose(people, calibration.occupied_person_hours, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("Teacher occupancy differs from training calibration")
        checks[experiment.name] = dict(
            physical=checked, actuator_intervals_checked=count, maximum_readback_error_c=error
        )
    if seen != set(expected):
        raise ValueError("Missing teacher experiment")
    exported = export(results, suite_path, examples, source)
    return dict(
        status="audited_training_demonstrations",
        source=source,
        task_id=task.task_id,
        checks=checks,
        teacher_export=exported,
        scope="Four declared source-local training weeks. Authenticated experiments, complete physical clocks/metrics, zero severe/fatal errors, exact zone actuator readbacks and exogenous occupancy checks. No fitted policy or deployment-performance claim.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("suite", "data", "results", "examples", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    checked = report(args.suite, args.source, args.data, args.results, args.examples)
    with args.output.open("x") as stream:
        json.dump(checked, stream, indent=2, allow_nan=False)
        stream.write("\n")
