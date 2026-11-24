"""Collect teacher examples, train PPO, and evaluate fixed-final checkpoints on both calendars."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from analysis.matrix import SOURCES, TARGETS
from .audit import task_name
from .fetch import HERE, assets, digest

CONFIG = HERE / "config"
ROOT = HERE.parent
STEPS = 483840


def load(path):
    return json.loads(path.read_text())


def launch(module, args, plan):
    command = [sys.executable, "-B", "-m", module, *map(str, args)]
    print(shlex.join(command), flush=True)
    if not plan:
        subprocess.run(command, cwd=ROOT, check=True)


def fresh(path, plan):
    if not plan:
        path.mkdir(parents=True, exist_ok=False)


def train(stage, short, seed, data, work, plan):
    name = task_name(short)
    setup = CONFIG / short
    base = work / "training" / name
    teacher = base / "teacher"
    suite = setup / "training-suite.json"
    if stage == "teacher":
        fresh(teacher, plan)
        demos = setup / "demonstrations-suite.json"
        launch(
            "building_gen.cli",
            [
                "benchmark",
                "run-baselines",
                "--suite",
                demos,
                "--experiment",
                "winter_training_demonstrations",
                "--experiment",
                "summer_training_demonstrations",
                "--data-root",
                data,
                "--work-dir",
                teacher / "work",
                "--output-dir",
                teacher / "evaluation",
            ],
            plan,
        )
        launch(
            "simulation.audit_teacher",
            [
                "--suite",
                demos,
                "--source",
                name,
                "--data",
                data,
                "--results",
                teacher / "evaluation",
                "--examples",
                teacher / "examples",
                "--output",
                teacher / "audit.json",
            ],
            plan,
        )
        return
    logs = base / "seed-runs" / f"seed{seed}"
    fresh(logs, plan)
    actor = base / "actors" / f"seed{seed}"
    ppo = CONFIG / "ppo.json"
    fit = CONFIG / "fit-office.json" if short.startswith("O") else ppo
    if not plan:
        proof = load(teacher / "audit.json")
        if (
            proof["source"] != name
            or proof["status"] != "audited_training_demonstrations"
            or proof["teacher_export"] != load(teacher / "examples/audit.json")
        ):
            raise ValueError("Teacher examples failed verification")
    launch(
        "simulation.fit_actor",
        [
            teacher / "examples",
            fit,
            actor,
            "--observation",
            1,
            "--seed",
            seed,
            "--epochs",
            200,
            "--suite",
            suite,
            "--task",
            name,
        ],
        plan,
    )
    args = [
        "train-baseline",
        "--suite",
        suite,
        "--task",
        name,
        "--algorithm",
        "ppo",
        "--parameters",
        ppo,
        "--actor-initialization",
        actor,
        "--seed",
        seed,
        "--steps",
        STEPS,
        "--checkpoint-interval",
        96768,
        "--data-root",
        data,
        "--work-dir",
        base / "ppo-work" / f"seed{seed}",
        "--output-dir",
        base / "ppo" / f"seed{seed}",
    ]
    for start, end in ((1, 7), (15, 21)):
        for month in range(1, 13):
            args.extend(["--reserve-window", f"{month:02}-{start:02}:{month:02}-{end:02}"])
    launch("building_gen.cli", args, plan)
    if not plan:
        from .verify_training import verify

        verify(work / "training", data, name, seed, short)


def evaluate(stage, cal, source, target, seed, data, work, plan):
    suite = CONFIG / f"{cal}.json"
    t = task_name(target)
    case = work / cal / ("historical" if stage == "native" else f"{source}/seed{seed}") / target
    native = work / cal / "historical" / target / "audit.json"
    common = [
        "--suite",
        suite,
        "--data-root",
        data,
        "--work-dir",
        case / "work",
        "--output-dir",
        case / "evaluation",
    ]
    evidence = ["--suite", suite, "--results", case / "evaluation", "--output", case / "audit.json"]
    if stage == "native":
        fresh(case, plan)
        launch(
            "building_gen.cli", ["benchmark", "run-baselines", "--experiment", "native_" + t, *common], plan
        )
        launch("simulation.audit_reference", ["--experiment", "native_" + t, *evidence], plan)
        return
    s = task_name(source)
    ppo = work / "training" / s / "ppo" / f"seed{seed}"
    if plan:
        checkpoint = ppo / "<final-checkpoint>"
        sha = "<model-sha256>"
    else:
        ref = load(native)
        if ref["experiment"] != "native_" + t or ref["status"] != "qualified_native_execution":
            raise ValueError("Missing historical reference")
        proof = load(ppo / "qualification.json")
        if (proof["source"], proof["seed"], proof["training_receipt_sha256"]) != (
            s,
            seed,
            digest(ppo / "training.json"),
        ):
            raise ValueError("Training receipt identity mismatch")
        (record,) = [
            r for r in load(ppo / "training.json")["validation_checkpoints"] if r["training_step"] == STEPS
        ]
        checkpoint = (ppo / record["directory"]).resolve()
        if (
            ppo not in checkpoint.parents
            or digest(checkpoint / "training.json") != record["training_receipt_sha256"]
        ):
            raise ValueError("Fixed checkpoint differs from training receipt")
        sha = digest(checkpoint / "model.zip")
        if sha != load(checkpoint / "training.json")["artifacts"]["model.zip"]:
            raise ValueError("Final model checksum mismatch")
    fresh(case, plan)
    launch(
        "building_gen.cli",
        [
            "evaluate-baseline",
            "--source-task",
            "source_" + s,
            "--target-task",
            t,
            "--protocol-from-experiment",
            "native_" + t,
            "--run-dir",
            checkpoint,
            *common,
        ],
        plan,
    )
    launch(
        "simulation.audit_policy",
        [
            "--source",
            s,
            "--target",
            t,
            "--seed",
            seed,
            "--step",
            STEPS,
            "--checkpoint",
            checkpoint,
            "--checkpoint-sha256",
            sha,
            *evidence,
        ],
        plan,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("teacher", "train", "native", "policies", "all"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--target", choices=TARGETS)
    p.add_argument("--seed", type=int, choices=range(10))
    p.add_argument("--calendar", choices=("test", "validation"))
    p.add_argument(
        "--plan", action="store_true", help="Print the selected commands without starting EnergyPlus"
    )
    a = p.parse_args()
    data, work = a.data.resolve(), a.work.resolve()
    if (
        a.stage == "native"
        and (a.source or a.seed is not None)
        or a.stage == "teacher"
        and a.seed is not None
    ):
        p.error("This stage does not use a policy seed or, for native references, a source")
    if a.stage in ("teacher", "train") and (a.target or a.calendar):
        p.error("Targets and evaluation calendars apply to native and policy evaluations")
    if work == data or data in work.parents or work in data.parents or work == ROOT or ROOT in work.parents:
        p.error("Choose separate input and output directories with outputs outside the checkout")
    if not a.plan:
        assets(data, check=True)
        binary = Path(os.environ["ENERGYPLUS_PATH"]) / "energyplus"
        version = subprocess.check_output([binary, "--version"], text=True)
        if "24.1.0-9d7789a3ac" not in version:
            raise ValueError(f"Expected EnergyPlus 24.1.0-9d7789a3ac, got {version.strip()}")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
            raise ValueError(
                "Commit experiment changes before running: the physical audits record checkout identity"
            )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    stages = ("teacher", "train", "native", "policies") if a.stage == "all" else [a.stage]
    sources = [a.source] if a.source else SOURCES
    seeds = [a.seed] if a.seed is not None else range(10)
    for stage in stages:
        if stage in ("teacher", "train"):
            for s in sources:
                for k in [None] if stage == "teacher" else seeds:
                    train(stage, s, k, data, work, a.plan)
        else:
            for c in [a.calendar] if a.calendar else ("validation", "test"):
                for t in [a.target] if a.target else TARGETS:
                    for s in [None] if stage == "native" else sources:
                        for k in [None] if stage == "native" else seeds:
                            evaluate(stage, c, s, t, k, data, work, a.plan)


if __name__ == "__main__":
    main()
