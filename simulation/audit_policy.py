"""Audit a frozen source/target cell with exact checkpoint, clocks and applied actions."""

import argparse
import json
from pathlib import Path

from building_gen.benchmark._canonical import content_id
from building_gen.benchmark.config import load_benchmark_suite
from building_gen.benchmark.reporting import _artifact_payload
from building_gen.weekly_calibration import _verified_trajectory_rows
from simulation.audit_reference import audit
from simulation.intervals import actuator_readbacks
from simulation.fetch import digest as sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "checkpoint", "results", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source", "target", "checkpoint-sha256"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=range(10), required=True)
    p.add_argument("--step", type=int, choices=(0, 96768, 193536, 290304, 387072, 483840), required=True)
    a = p.parse_args()
    suite = load_benchmark_suite(a.suite)
    source = suite.task_map["source_" + a.source]
    target = suite.task_map[a.target]
    receipt = json.loads((a.checkpoint / "training.json").read_text())
    if (
        sha(a.checkpoint / "model.zip") != a.checkpoint_sha256
        or receipt["artifacts"]["model.zip"] != a.checkpoint_sha256
        or receipt["task_signature"]["task_id"] != source.task_id
        or receipt["seed"] != a.seed
        or receipt["completed_timesteps"] != a.step
        or receipt["parameters"]["policy_architecture"] != "shared_zone_v1"
    ):
        raise ValueError("Different source policy/checkpoint/seed")
    checked = audit(a.suite, "native_" + a.target, a.results, algorithm="sb3_ppo", seed=a.seed)
    (path,) = a.results.glob("*.result.json")
    raw = json.loads(path.read_text())
    if raw["run"]["train_task_ids"] != [source.task_id] or raw["run"]["parent_training_run_id"] != content_id(
        "HOT SB3 engineering training", receipt, prefix="training"
    ):
        raise ValueError("Different evaluation training lineage")
    policy = json.loads(
        _artifact_payload(raw, path, role="evaluation_evidence", media_type="application/json").payload
    )["policy"]
    if policy["checkpoint_sha256"] != a.checkpoint_sha256 or policy["training_step"] != a.step:
        raise ValueError("Different evaluated weights")
    (artifact,) = [v for v in raw["artifacts"] if v["role"] == "step_trajectory"]
    count, error = actuator_readbacks(
        _verified_trajectory_rows(artifact, path), target.action_profile.controlled_entities
    )
    checked.update(
        source_context=a.source,
        source_task_id=source.task_id,
        target_context=a.target,
        seed=a.seed,
        step=a.step,
        checkpoint_sha256=a.checkpoint_sha256,
        checkpoint_receipt_sha256=sha(a.checkpoint / "training.json"),
        suite_sha256=sha(a.suite),
        result_sha256=sha(path),
        actuator_intervals_checked=count,
        maximum_readback_error_c=error,
        scope="One fixed source/target/seed/snapshot cell, independent physical metrics and clocks, zero severe/fatal simulator errors and exact setpoint readbacks. Not a cohort, selector conclusion or convergence finding.",
    )
    with a.output.open("x") as stream:
        json.dump(checked, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(dict(source=a.source, target=a.target, seed=a.seed, step=a.step, status=checked["status"]))
    )


if __name__ == "__main__":
    main()
