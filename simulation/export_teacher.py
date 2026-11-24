"""Verify scheduled training rollouts and export pre-action supervised examples."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from building_gen.benchmark.config import benchmark_suite_from_mapping


def export(root, suite_path, destination, task_name=None, teacher="scheduled_thermostat"):
    if teacher not in ("scheduled_thermostat", "native_incumbent"):
        raise ValueError("unsupported demonstration teacher")
    if teacher == "native_incumbent" and task_name is None:
        raise ValueError("native demonstrations require an explicit named task")
    suite = benchmark_suite_from_mapping(json.loads(suite_path.read_text()))
    tasks = {t.task_id: t for t in suite.tasks if task_name is None or t.name == task_name}
    if not tasks:
        raise ValueError("requested demonstration task not found")
    if task_name is not None and (
        len(tasks) != 1 or next(iter(tasks.values())).episode.episode_length_hours != 168
    ):
        raise ValueError("named demonstrations require one weekly task")
    declared = {}
    for task_id, task in tasks.items():
        windows = [
            w
            for e in suite.experiments
            if e.algorithm_id == teacher and task.name in e.evaluation_tasks
            for w in e.evaluation_protocol.windows
        ]
        by_month = {w.start_month: w for w in windows}
        if (
            len(windows) != 4
            or set(by_month) != {2, 5, 8, 11}
            or any(
                w.start_month != w.end_month or w.duration_hours_for_calendar_year(task.calendar_year) != 168
                for w in windows
            )
        ):
            raise ValueError("suite must declare four unique weekly demonstration windows")
        declared[task_id] = by_month
    data = {key: {} for key in tasks}
    receipts = []
    files = sorted(root.glob("*.result.json"))
    if task_name is not None:
        files = [p for p in files if json.loads(p.read_text())["run"]["evaluation_task_id"] in tasks]
    if len(files) != (4 if task_name is None else 2):
        raise ValueError("expected four successful task-season results")
    for path in files:
        result = json.loads(path.read_text())
        if result["status"] != "succeeded" or result["run"]["algorithm_id"] != teacher:
            raise ValueError("unsuccessful or wrong-controller result")
        task_id = result["run"]["evaluation_task_id"]
        task = tasks[task_id]
        week_steps = task.episode.max_steps if task_name is not None else 1008
        timesteps_per_hour = task.episode.timesteps_per_hour if task_name is not None else 6
        artifacts = {}
        for artifact in result["artifacts"]:
            raw = (root / artifact["uri"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
                raise ValueError("artifact checksum mismatch")
            artifacts[artifact["role"]] = raw
        diagnostics = json.loads(artifacts["energyplus_diagnostics"])
        for episode in diagnostics["episodes"]:
            d = episode["diagnostics"]
            if not d["completed_successfully"] or d["severe_error_count"] or d["fatal_error_count"]:
                raise ValueError("failed simulation diagnostics")
        for row in csv.DictReader(io.StringIO(artifacts["step_trajectory"].decode())):
            month, step = int(row["window_start_month"]), int(row["step_index"])
            window = declared[task_id].get(month)
            if window is None or (
                int(row["window_end_month"]),
                int(row["window_start_day"]),
                int(row["window_end_day"]),
            ) != (window.end_month, window.start_day, window.end_day):
                raise ValueError("unexpected demonstration window")
            if (
                row["task_id"] != task_id
                or int(row["expected_steps"]) != week_steps
                or int(row["timesteps_per_hour"]) != timesteps_per_hour
            ):
                raise ValueError("trajectory contract mismatch")
            x = np.asarray(json.loads(row["policy_observation_normalized_json"]), dtype=np.float32)
            if teacher == "native_incumbent":
                from building_gen.benchmark.native_demonstrations import encode_native_targets

                expected_native = dict(
                    schema_version="hot-native-schedule-control/1.0.0",
                    thermostat_overrides="released",
                    submitted_policy_setpoints=False,
                    actuator_readback_available=False,
                )
                if (
                    row["control_mode"] != "native_schedule"
                    or json.loads(row["native_control_json"]) != expected_native
                    or json.loads(row["submitted_action_physical_json"]) != []
                    or json.loads(row["proposed_action_normalized_json"]) != []
                ):
                    raise ValueError("demonstrations must use unmodified native controls")
                y = np.asarray(
                    encode_native_targets(
                        json.loads(row["native_thermostat_targets_json"]),
                        task,
                        json.loads(row["decision_clock_json"]),
                    ),
                    dtype=np.float32,
                )
            else:
                if not float(row["actuator_readback_max_abs_error_c"]) < 1e-6:
                    raise ValueError("actuator readback mismatch")
                y = np.asarray(json.loads(row["proposed_action_normalized_json"]), dtype=np.float32)
            if x.shape != tuple(task.observation_space.shape) or y.shape != (
                tuple(task.action_space.shape) if task_name is not None else (10,)
            ):
                raise ValueError("example shape mismatch")
            if not all(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() for v in (x, y)):
                raise ValueError("invalid normalized example")
            key = (month, step)
            if key in data[task_id]:
                raise ValueError("duplicate demonstration step")
            data[task_id][key] = (x, y)
        receipts.append(
            dict(
                result=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                artifacts=result["artifacts"],
            )
        )
    if any(
        set(rows)
        != {
            (month, step)
            for month in (2, 5, 8, 11)
            for step in range(tasks[task_id].episode.max_steps if task_name is not None else 1008)
        }
        for task_id, rows in data.items()
    ):
        raise ValueError("incomplete week coverage")
    destination.mkdir(exist_ok=False, parents=True)
    outputs = []
    for task_id, rows in data.items():
        keys = sorted(rows)
        path = destination / (tasks[task_id].name + ".npz")
        np.savez_compressed(
            path,
            observations=np.stack([rows[k][0] for k in keys]),
            actions=np.stack([rows[k][1] for k in keys]),
            month=np.asarray([k[0] for k in keys]),
            step=np.asarray([k[1] for k in keys]),
        )
        outputs.append(
            dict(
                file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                task_id=task_id,
                examples=len(keys),
                observation_dimension=tasks[task_id].observation_space.shape[0],
            )
        )
    receipt = dict(
        status="passed",
        suite_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        sources=receipts,
        outputs=outputs,
        scope="Training-only demonstrations. Pre-action observations, commanded normalized targets, verified actuator agreement. No policy fit or performance claim.",
    )
    if teacher == "native_incumbent":
        receipt.update(
            teacher=teacher,
            label_schema="hot-native-thermostat-targets/1",
            projection="none",
            target_simulator_transitions=sum(r["examples"] for r in outputs),
            scope="Training-only native-controller demonstrations. Pre-action observations paired with completed-interval thermostat outputs, exactly encoded without clipping. No actuator command, fit or performance claim.",
        )
    (destination / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("suite", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--task", help="Export one named weekly task, with its declared zone count and timestep.")
    p.add_argument(
        "--teacher", choices=["scheduled_thermostat", "native_incumbent"], default="scheduled_thermostat"
    )
    a = p.parse_args()
    export(a.root, a.suite, a.destination, a.task, a.teacher)
