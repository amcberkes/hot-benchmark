"""Qualify a newly trained source policy using the original physical and calendar checks."""

import subprocess

from .fetch import HERE, digest as sha
import json


def load(path):
    return json.loads(path.read_text())


def record(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)


def require_hash(path, expected):
    check(sha(path) == expected, f"Checksum differs: {path}")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(work, inputs, source, seed, short):
    from building_gen.benchmark._canonical import canonical_value
    from building_gen.benchmark.config import load_benchmark_suite
    from building_gen.benchmark.training_diagnostics_audit import audit_training_diagnostics
    from building_gen.wrapper.rl_baselines import load_agent

    setup = HERE / "config" / short
    task = load_benchmark_suite(setup / "training-suite.json").task_map[source]
    base = work / source
    training = base / "ppo" / f"seed{seed}"
    actor = base / "actors" / f"seed{seed}"
    examples = base / "teacher" / "examples"
    logs = base / "seed-runs" / f"seed{seed}"
    receipt = load(training / "training.json")
    fit = load(actor / "imitation.json")
    fit_parameters = HERE / "config" / ("fit-office.json" if short.startswith("O") else "ppo.json")
    check(
        (
            fit["status"],
            fit["seed"],
            fit["observation"],
            fit["epochs"],
            fit["batch_size"],
            fit["learning_rate"],
            fit["fit_examples"],
            fit["withheld_examples"],
        )
        == ("complete", seed, 1, 200, 256, 0.001, 6912, 1152),
        "Supervised initialization recipe differs",
    )
    require_hash(fit_parameters, fit["parameter_sha256"])
    check(
        fit["script_sha256"] == "e27b494acd6ded3b25c368d16268f5f6f991c8b55c2eb65b40fa31ea89ef574b",
        "Supervised actor fitter differs from the original producer",
    )
    require_hash(examples / "audit.json", fit["demonstration_audit_sha256"])
    check(fit["data"]["file"] == source + ".npz", "Supervised initialization used a different source")
    require_hash(examples / fit["data"]["file"], fit["data"]["sha256"])
    require_hash(actor / "actor_initialization.zip", fit["model_sha256"])
    check(
        (
            receipt["status"],
            receipt["completed_timesteps"],
            receipt["requested_timesteps"],
            receipt["optimizer_updates"],
            receipt["seed"],
            receipt["task_signature"]["task_id"],
        )
        == ("complete", 483840, 483840, 200, seed, task.task_id),
        "PPO source, budget or seed differs",
    )
    check(receipt["parameters"] == load(HERE / "config/ppo.json"), "PPO parameters differ")
    initialization = receipt["actor_initialization"]
    check(
        (
            initialization["kind"],
            initialization["seed"],
            initialization["source_task_id"],
            initialization["model_sha256"],
            initialization["environment_transitions"],
        )
        == ("supervised_actor_only", seed, task.task_id, fit["model_sha256"], 0),
        "PPO did not start from the fresh seed-specific supervised actor",
    )
    require_hash(actor / "imitation.json", initialization["receipt_sha256"])
    check(
        all(initialization[k] == "fresh" for k in ("critic", "optimizer", "policy_variance")),
        "Unexpected critic, optimizer or variance initialization",
    )
    expected_pool = sorted(load(HERE / "config/training-weeks.json"))
    calendar = receipt["continuation"]["calendar"]
    partition = calendar["calendar_partition"]
    check(
        sorted(partition["eligible_windows"]) == expected_pool
        and (
            calendar["schedule_seed"],
            partition["seed"],
            calendar["task_id"],
            calendar["next_episode_index"],
            partition["episode_length_hours"],
        )
        == (seed, seed, task.task_id, 240, 168),
        "Training date pool or episode count differs",
    )
    require_hash(training / "model.zip", receipt["artifacts"]["model.zip"])
    agent = load_agent(
        training,
        None,
        task_signature=dict(task_id=task.task_id, definition=canonical_value(task.identity_payload())),
    )
    check(
        (agent.num_timesteps, agent._n_updates, agent.seed) == (483840, 200, seed),
        "PPO model counters differ",
    )
    snapshots = receipt["validation_checkpoints"]
    check(
        [r["training_step"] for r in snapshots] == list(range(0, 483841, 96768)), "Fixed checkpoints differ"
    )
    for row in snapshots:
        directory = (training / row["directory"]).resolve()
        check(training.resolve() in directory.parents, "Checkpoint lies outside training output")
        require_hash(directory / "training.json", row["training_receipt_sha256"])
        artifacts = load(directory / "training.json")["artifacts"]
        check("model.zip" in artifacts, "Missing checkpoint weights")
        for filename, expected in artifacts.items():
            path = (directory / filename).resolve()
            check(directory in path.parents, "Artifact lies outside checkpoint")
            require_hash(path, expected)
    rollouts = load(training / "rollout_diagnostics.json")["records"]
    check(
        [r["training_step"] for r in rollouts] == list(range(24192, 483841, 24192)), "PPO rollout log differs"
    )
    diagnostic = audit_training_diagnostics(
        base / "ppo-work" / f"seed{seed}",
        building_path=inputs / task.building.relative_path,
        expected_building_sha256=task.building.sha256,
        weather_path=inputs / task.weather.relative_path,
        expected_weather_sha256=task.weather.sha256,
        expected_energyplus_version="24.1.0-9d7789a3ac",
        expected_complete_count=241,
    )
    physical_receipt = logs / "physical-audit.json"
    record(physical_receipt, diagnostic.to_dict())
    check(
        diagnostic.engineering_integrity_checks_passed, f"Physical training audit failed: {physical_receipt}"
    )
    record(
        training / "qualification.json",
        dict(
            status="qualified_fresh_training",
            source=source,
            seed=seed,
            task_id=task.task_id,
            producer=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
            training_receipt_sha256=sha(training / "training.json"),
            model_sha256=receipt["artifacts"]["model.zip"],
            initializer_sha256=fit["model_sha256"],
            imitation_receipt_sha256=sha(actor / "imitation.json"),
            physical_audit_sha256=sha(physical_receipt),
            checkpoint_steps=list(range(0, 483841, 96768)),
            rollout_updates=20,
            training_episodes=240,
            completed_simulator_episodes=241,
        ),
    )
