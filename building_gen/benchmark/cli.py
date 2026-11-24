"""Evaluate a built-in controller on a HOT suite."""

from __future__ import annotations
import argparse
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from building_gen.benchmark.config import load_benchmark_suite
from building_gen.benchmark.energyplus_diagnostics import (
    DIAGNOSTICS_MODES,
    DiagnosticsGate,
    load_diagnostics_allowlist,
)
from building_gen.benchmark.environment import create_benchmark_environment
from building_gen.benchmark.provenance import (
    SCIENTIFIC_DEPENDENCIES,
    SCIENTIFIC_REPOSITORY,
    SoftwareProvenance,
    resolve_experiment_runs,
    validate_scientific_software_provenance,
)
from building_gen.benchmark.registry import AlgorithmKind, default_algorithm_registry
from building_gen.benchmark.specs import BenchmarkSuite, ExperimentSpec, TaskSpec

EnvironmentBuilder = Callable[..., Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser(
        "run-baselines",
        help="Evaluate one or more deterministic built-in policies.",
        allow_abbrev=False,
    )
    _add_suite_argument(baseline_parser)
    _add_execution_paths(baseline_parser)
    baseline_parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Built-in experiment name (repeatable); defaults to all enabled built-ins.",
    )
    baseline_parser.add_argument("--overwrite", action="store_true")
    _add_diagnostics_arguments(baseline_parser)
    _add_provenance_arguments(baseline_parser)
    baseline_parser.set_defaults(handler=_command_run_baselines)
    return parser


def _add_suite_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", type=Path, required=True, help="Benchmark suite YAML.")


def _add_execution_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root against which suite asset paths are resolved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Persistent artifact directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Disposable EnergyPlus work directory.",
    )


def _add_provenance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repository",
        default=os.environ.get("HOT_PROVENANCE_REPOSITORY", SCIENTIFIC_REPOSITORY),
    )
    parser.add_argument("--revision", help="Git revision; auto-detected from the checkout by default.")
    parser.add_argument(
        "--energyplus-version",
        default=os.environ.get("HOT_ENERGYPLUS_VERSION", "24.1.0"),
    )
    parser.add_argument(
        "--container-digest",
        default=os.environ.get("HOT_CONTAINER_DIGEST"),
        help=(
            "Exact sha256 image identity. The Slurm launcher computes the SIF digest and rejects a supplied "
            "value that differs before scientific execution."
        ),
    )


def _add_diagnostics_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--diagnostics-mode",
        choices=sorted(DIAGNOSTICS_MODES),
        default="engineering",
        help="Engineering records diagnostics best-effort; scientific fails closed.",
    )
    parser.add_argument(
        "--diagnostics-allowlist",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON policy scoped to one exact building SHA256, weather SHA256, and EnergyPlus version; "
            "repeat for multiple evaluation buildings."
        ),
    )


def _diagnostics_gate(args: argparse.Namespace) -> DiagnosticsGate:
    return DiagnosticsGate(
        mode=getattr(args, "diagnostics_mode", "engineering"),
        allowlists=tuple(
            load_diagnostics_allowlist(path) for path in getattr(args, "diagnostics_allowlist", ())
        ),
    )


def _experiment(suite: BenchmarkSuite, name: str) -> ExperimentSpec:
    try:
        return suite.experiment_map[name]
    except KeyError as error:
        available = ", ".join(sorted(suite.experiment_map))
        raise ValueError(f"unknown experiment {name!r}; available: {available}") from error


def _selected_experiments(suite: BenchmarkSuite, names: Sequence[str]) -> tuple[ExperimentSpec, ...]:
    if names:
        if len(set(names)) != len(names):
            raise ValueError("--experiment cannot contain duplicates")
        return tuple(_experiment(suite, name) for name in names)
    return suite.experiments


def _task_by_id(suite: BenchmarkSuite, task_id: str | None) -> TaskSpec:
    matches = [task for task in suite.tasks if task.task_id == task_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one suite task for task_id {task_id!r}; found {len(matches)}")
    return matches[0]


def _git_value(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalized_repository(value: str) -> str:
    """Normalize common Git remote spellings to a host/path identity."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("repository must be a non-empty string")
    candidate = value.strip()
    if "://" in candidate:
        parsed = urlsplit(candidate)
        if not parsed.hostname or not parsed.path:
            raise ValueError(f"repository URL is malformed: {value!r}")
        host = parsed.hostname.lower()
        path = parsed.path
    elif candidate.startswith("git@") and ":" in candidate:
        authority, path = candidate.split(":", 1)
        host = authority.split("@", 1)[1].lower()
    else:
        pieces = candidate.split("/", 1)
        if len(pieces) != 2:
            raise ValueError(f"repository identity is malformed: {value!r}")
        host, path = pieces[0].lower(), pieces[1]
    normalized_path = path.strip("/")
    if normalized_path.endswith(".git"):
        normalized_path = normalized_path[:-4]
    if not normalized_path or "/" not in normalized_path:
        raise ValueError(f"repository identity is malformed: {value!r}")
    return f"{host}/{normalized_path}"


def _normalized_remote_repository(value: str) -> str:
    """Normalize only authenticated-looking network Git remote spellings."""

    candidate = value.strip() if isinstance(value, str) else ""
    if candidate.startswith("git@") and ":" in candidate:
        return _normalized_repository(candidate)
    if "://" not in candidate:
        raise ValueError(f"Git remote is not a supported network URL: {value!r}")
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"https", "ssh"}:
        raise ValueError(f"Git remote uses an unsupported scheme: {value!r}")
    return _normalized_repository(candidate)


def _scientific_checkout_repository(repository_root: Path) -> str:
    """Authenticate the canonical repository against every configured remote."""

    remote_names = tuple(line for line in _git_value(repository_root, "remote").splitlines() if line)
    remote_urls: list[str] = []
    for remote_name in remote_names:
        remote_urls.extend(
            line
            for line in _git_value(repository_root, "remote", "get-url", "--all", remote_name).splitlines()
            if line
        )
    normalized_urls: set[str] = set()
    for remote_url in remote_urls:
        try:
            normalized_urls.add(_normalized_remote_repository(remote_url).casefold())
        except ValueError:
            continue
    if SCIENTIFIC_REPOSITORY.casefold() not in normalized_urls:
        raise ValueError(
            "scientific training checkout has no authenticated HOT Git remote; "
            f"configured URLs were {remote_urls!r}"
        )
    return SCIENTIFIC_REPOSITORY


def _git_hidden_index_entries(repository_root: Path) -> tuple[str, ...]:
    """Return paths hidden by assume-unchanged, skip-worktree, or other index flags."""

    records = _git_value(repository_root, "ls-files", "-v", "-z").split("\0")
    return tuple(record[2:] for record in records if record and not record.startswith("H "))


def _git_ignored_import_artifacts(repository_root: Path) -> tuple[str, ...]:
    """Find ignored Python/native modules that could shadow authenticated code."""

    records = _git_value(
        repository_root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
        "--",
        ":(glob)*.py",
        ":(glob)*.pyc",
        ":(glob)*.pyo",
        ":(glob)*.so",
        ":(glob)*.dylib",
        ":(glob)**/*.py",
        ":(glob)**/*.pyc",
        ":(glob)**/*.pyo",
        ":(glob)**/*.so",
        ":(glob)**/*.dylib",
    ).split("\0")
    return tuple(record for record in records if record)


def _git_checkout_dirty(repository_root: Path) -> bool:
    """Fail closed on all untracked files and index/import hiding mechanisms."""

    status = _git_value(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    return bool(
        status or _git_hidden_index_entries(repository_root) or _git_ignored_import_artifacts(repository_root)
    )


def _software_provenance(args: argparse.Namespace) -> SoftwareProvenance:
    try:
        revision = args.revision or _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD")
        dirty = bool(
            _git_value(REPOSITORY_ROOT, "status", "--porcelain") or _git_hidden_index_entries(REPOSITORY_ROOT)
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ValueError(
            "Cannot collect Git provenance for the code checkout. Run from a Git checkout "
            "with Git installed and readable metadata; --revision alone does not replace "
            "the required checkout status check. Validate this setup before submitting a cluster job."
        ) from error
    return SoftwareProvenance(
        repository=args.repository,
        revision=revision,
        dirty=dirty,
        python_version=platform.python_version(),
        energyplus_version=args.energyplus_version,
        container_digest=args.container_digest,
    )


def _scientific_software_provenance(args: argparse.Namespace) -> tuple[SoftwareProvenance, str]:
    """Resolve exact, clean runtime provenance before an expensive simulation."""

    if _normalized_repository(args.repository).casefold() != SCIENTIFIC_REPOSITORY.casefold():
        raise ValueError(f"scientific training requires --repository {SCIENTIFIC_REPOSITORY!r}")
    if _git_checkout_dirty(REPOSITORY_ROOT):
        raise ValueError("Scientific execution requires a clean source checkout")
    checkout_repository = _scientific_checkout_repository(REPOSITORY_ROOT)
    checkout_revision = _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD").lower()
    if args.revision is not None and args.revision.lower() != checkout_revision:
        raise ValueError(f"--revision {args.revision!r} does not equal checkout HEAD {checkout_revision!r}")
    dependencies: dict[str, str] = {}
    for distribution in SCIENTIFIC_DEPENDENCIES:
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"scientific training requires installed dependency {distribution!r}") from error
        if not installed.strip():
            raise ValueError(f"installed dependency {distribution!r} has no version")
        dependencies[distribution] = installed
    provenance = replace(
        _software_provenance(args),
        repository=checkout_repository,
        revision=checkout_revision,
        dependencies=dependencies,
    )
    validate_scientific_software_provenance(
        provenance,
        expected_revision=checkout_revision,
        context="scientific execution",
    )
    return provenance, checkout_revision


def _execution_software_provenance(
    args: argparse.Namespace,
    diagnostics_gate: DiagnosticsGate,
) -> SoftwareProvenance:
    """Resolve strict provenance exactly when an evaluation is scientific."""

    if diagnostics_gate.scientific:
        provenance, _checkout_revision = _scientific_software_provenance(args)
        return provenance
    return _software_provenance(args)


def _rollout_environment_factory(
    *,
    data_root: Path,
    work_dir: Path,
    environment_builder: EnvironmentBuilder = create_benchmark_environment,
    simulator_factory: Callable[..., Any] | None = None,
) -> Callable[[Any], Any]:
    """Build an evaluator factory with a collision-free directory per episode."""

    def factory(context: Any) -> Any:
        episode_work_dir = (
            work_dir / context.run.run_id / context.window.name / f"episode-{context.episode_index:04d}"
        )
        return environment_builder(
            context.task,
            context.window,
            data_root=data_root,
            work_dir=episode_work_dir,
            simulator_factory=simulator_factory,
        )

    return factory


def run_builtin_experiment(
    *,
    suite: BenchmarkSuite,
    experiment: ExperimentSpec,
    data_root: Path,
    output_dir: Path,
    work_dir: Path,
    provenance: SoftwareProvenance,
    overwrite: bool = False,
    environment_builder: EnvironmentBuilder = create_benchmark_environment,
    simulator_factory: Callable[..., Any] | None = None,
    diagnostics_gate: DiagnosticsGate | None = None,
) -> tuple[Path, ...]:
    """Evaluate one declared built-in policy; injectable for simulator-free tests."""

    from building_gen.benchmark.evaluation import (
        ashrae_seasonal_policy,
        comfort_band_policy,
        evaluate_run,
        static_midpoint_policy,
    )

    registry = default_algorithm_registry()
    descriptor = registry.get(experiment.algorithm_id)
    if descriptor.kind is not AlgorithmKind.BUILTIN_POLICY:
        raise ValueError(f"experiment {experiment.name!r} is not a built-in policy")
    if experiment.evaluation_protocol is None:
        raise ValueError(f"experiment {experiment.name!r} has no evaluation_protocol")
    if experiment.algorithm_id == "native_incumbent":
        from building_gen.benchmark.evaluation import native_incumbent_policy

        if experiment.parameter_mapping:
            raise ValueError("native_incumbent does not accept experiment parameters")
        if diagnostics_gate is not None and diagnostics_gate.mode != "engineering":
            raise ValueError("native incumbent requires engineering diagnostics mode")
        policy_factory = native_incumbent_policy
        for name in experiment.evaluation_tasks:
            policy_factory(suite.task_map[name])
    elif experiment.algorithm_id == "occupancy_reactive_thermostat":
        from building_gen.benchmark.evaluation import EvaluationPolicy, PolicyProvenance
        from building_gen.benchmark.environment import _expected_observation_names
        from building_gen.benchmark.policies import occupancy_reactive_action

        parameters = experiment.parameter_mapping
        expected = {
            "occupied_heating_c",
            "occupied_cooling_c",
            "unoccupied_heating_c",
            "unoccupied_cooling_c",
        }
        if set(parameters) != expected:
            raise ValueError("occupancy_reactive_thermostat requires both setpoint pairs")

        def policy_factory(task):
            names = _expected_observation_names(task)
            indices = [
                names.index(f"{zone}_occupant_count") for zone in task.action_profile.controlled_entities
            ]
            occupancy_reactive_action(task, occupied=[False] * len(indices), **parameters)

            def policy(observation, context):
                import math

                counts = [float(observation[index]) for index in indices]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in counts):
                    raise ValueError("invalid normalized occupancy input")
                return occupancy_reactive_action(task, occupied=[value > 0 for value in counts], **parameters)

            return EvaluationPolicy(policy, PolicyProvenance("built_in", "occupancy_reactive_thermostat"))

        for name in experiment.evaluation_tasks:
            policy_factory(suite.task_map[name])
    elif experiment.algorithm_id in ("scheduled_thermostat", "holiday_aware_thermostat"):
        from building_gen.benchmark.evaluation import EvaluationPolicy, PolicyProvenance
        from building_gen.benchmark.policies import scheduled_thermostat_action

        parameters = experiment.parameter_mapping
        expected = {
            "occupied_weekdays",
            "occupied_start_minute",
            "occupied_end_minute",
            "occupied_heating_c",
            "occupied_cooling_c",
            "unoccupied_heating_c",
            "unoccupied_cooling_c",
        }
        if set(parameters) != expected:
            raise ValueError("scheduled_thermostat requires a weekly interval and both setpoint pairs")

        def policy_factory(task):
            scheduled_thermostat_action(task, day_of_week=1, minute_of_day=0, **parameters)
            holiday_aware = experiment.algorithm_id == "holiday_aware_thermostat"
            if holiday_aware:
                from building_gen.benchmark.environment import _expected_observation_names

                names = _expected_observation_names(task)
                special_indices = [names.index(f"special_day_type_{i}") for i in range(6)]
                occupancy_indices = [
                    names.index(f"{z}_occupant_count") for z in task.action_profile.controlled_entities
                ]

            def policy(observation, context):
                clock = context.clock
                minute = 60 * (clock["hour"] + (clock["timestep_index"] - 1) / clock["timesteps_per_hour"])
                if holiday_aware:
                    import math
                    from building_gen.benchmark.policies import holiday_aware_thermostat_action

                    flags = [float(observation[i]) for i in special_indices]
                    counts = [float(observation[i]) for i in occupancy_indices]
                    if (
                        any(v not in (0.0, 1.0) for v in flags)
                        or sum(flags) != 1
                        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in counts)
                    ):
                        raise ValueError("invalid holiday/occupancy observation")
                    return holiday_aware_thermostat_action(
                        task,
                        special_day_type=flags.index(1.0),
                        occupied=[v > 0 for v in counts],
                        day_of_week=clock["day_of_week"],
                        minute_of_day=minute,
                        **parameters,
                    )
                return scheduled_thermostat_action(
                    task, day_of_week=clock["day_of_week"], minute_of_day=minute, **parameters
                )

            return EvaluationPolicy(policy, PolicyProvenance("built_in", experiment.algorithm_id))

        for name in experiment.evaluation_tasks:
            policy_factory(suite.task_map[name])
    elif experiment.algorithm_id == "seasonal_thermostat":
        from building_gen.benchmark.evaluation import EvaluationPolicy, PolicyProvenance
        from building_gen.benchmark.policies import seasonal_thermostat_action

        parameters = experiment.parameter_mapping
        expected = {
            "winter_months",
            "winter_heating_c",
            "winter_cooling_c",
            "summer_heating_c",
            "summer_cooling_c",
        }
        if set(parameters) != expected:
            raise ValueError("seasonal_thermostat requires winter_months and both seasonal setpoint pairs")

        def policy_factory(task):
            actions = {
                month: seasonal_thermostat_action(task, month=month, **parameters) for month in range(1, 13)
            }

            def policy(observation, context):
                month = context.clock.get("month")
                if isinstance(month, bool) or not isinstance(month, int) or month not in actions:
                    raise ValueError("decision clock month must be an integer from 1 through 12")
                return actions[month]

            return EvaluationPolicy(policy, PolicyProvenance("built_in", "seasonal_thermostat"))

        for name in experiment.evaluation_tasks:
            policy_factory(suite.task_map[name])
    elif experiment.algorithm_id == "fixed_thermostat":
        from building_gen.benchmark.evaluation import EvaluationPolicy, PolicyProvenance
        from building_gen.benchmark.policies import fixed_thermostat_action

        parameters = experiment.parameter_mapping
        if set(parameters) != {"heating_c", "cooling_c"}:
            raise ValueError("fixed_thermostat requires exactly heating_c and cooling_c")

        def policy_factory(task):
            action = fixed_thermostat_action(task, **parameters)
            return EvaluationPolicy(
                lambda observation, context: action, PolicyProvenance("built_in", "fixed_thermostat")
            )

        for name in experiment.evaluation_tasks:
            policy_factory(suite.task_map[name])
    elif experiment.algorithm_id == "comfort_band_20_24":
        if experiment.parameters:
            raise ValueError("comfort_band_20_24 does not accept experiment parameters")
        policy_factory = comfort_band_policy
    elif experiment.algorithm_id == "static_midpoint":
        if experiment.parameter_mapping:
            raise ValueError("static_midpoint does not accept experiment parameters")
        policy_factory = static_midpoint_policy
    elif experiment.algorithm_id == "ashrae_seasonal_setpoints":
        expected = {
            "winter_months": [11, 12, 1, 2, 3],
            "winter_heating_c": 20.0,
            "winter_cooling_c": 23.0,
            "summer_heating_c": 22.0,
            "summer_cooling_c": 26.0,
        }
        if experiment.parameter_mapping != expected:
            raise ValueError(
                "ashrae_seasonal_setpoints implementation supports only the pinned legacy HOT parameters"
            )
        policy_factory = ashrae_seasonal_policy
    else:
        raise ValueError(f"no built-in policy adapter for {experiment.algorithm_id!r}")

    runs = resolve_experiment_runs(
        experiment,
        suite.task_map,
        registry,
        require_available=True,
        require_checksums=True,
    )
    environment_factory = _rollout_environment_factory(
        data_root=data_root,
        work_dir=work_dir,
        environment_builder=environment_builder,
        simulator_factory=simulator_factory,
    )
    result_paths = []
    for run in runs:
        if run.phase != "evaluate":
            raise AssertionError("a built-in experiment unexpectedly resolved a training run")
        task = _task_by_id(suite, run.evaluation_task_id)
        output = evaluate_run(
            run=run,
            task=task,
            protocol=experiment.evaluation_protocol,
            environment_factory=environment_factory,
            policy=policy_factory(task),
            provenance=provenance,
            output_dir=output_dir,
            overwrite=overwrite,
            diagnostics_gate=diagnostics_gate,
        )
        result_paths.append(output.result_path)
    return tuple(result_paths)


def _command_run_baselines(args: argparse.Namespace) -> int:
    suite = load_benchmark_suite(args.suite)
    registry = default_algorithm_registry()
    if args.experiment:
        selected = _selected_experiments(suite, args.experiment)
    else:
        selected = tuple(
            experiment
            for experiment in suite.experiments
            if experiment.enabled
            and registry.get(experiment.algorithm_id).kind is AlgorithmKind.BUILTIN_POLICY
        )
    if not selected:
        raise ValueError("no built-in baseline experiments selected")
    diagnostics_gate = _diagnostics_gate(args)
    provenance = _execution_software_provenance(args, diagnostics_gate)
    written = []
    for experiment in selected:
        written.extend(
            run_builtin_experiment(
                suite=suite,
                experiment=experiment,
                data_root=args.data_root,
                output_dir=args.output_dir,
                work_dir=args.work_dir,
                provenance=provenance,
                overwrite=args.overwrite,
                diagnostics_gate=diagnostics_gate,
            )
        )
    for path in written:
        print(f"Wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
