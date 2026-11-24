"""Validation and first-report generation for HOT rollout artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from building_gen.accounting import CONTROLLED_HVAC_METER_BY_CARRIER, EnergyCarrier
from building_gen.benchmark.energyplus_diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    DiagnosticsAllowlist,
    diagnostics_allowlist_from_mapping,
    energyplus_diagnostics_from_mapping,
    evaluate_scientific_diagnostics,
)
from building_gen.benchmark.evaluation import (
    EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    LEGACY_EVALUATION_EVIDENCE_SCHEMA_VERSION,
    STEP_SCHEMA_VERSION,
    SUBMITTED_ACTION_SCHEMA_VERSION,
    EvaluationError,
    PolicyProvenance,
    _atomic_write_text,
    _submitted_action_evidence,
)
from building_gen.benchmark.policies import (
    ashrae_seasonal_action,
    static_midpoint_action,
)
from building_gen.benchmark.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    ResolvedRun,
    SoftwareProvenance,
    validate_scientific_software_provenance,
)
from building_gen.benchmark.specs import (
    CalendarWindowSpec,
    EvaluationProtocolSpec,
    TaskSpec,
)
from building_gen.benchmark.training_bundle import (
    TRAINING_BUNDLE_SCHEMA_VERSION,
    ExpectedTrainingContract,
    ScientificTrainingBundleError,
    final_checkpoint_selection_rule,
    validate_training_bundle_receipt,
)
from building_gen.benchmark.warmup_preconditioning import (
    EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION,
    WARMUP_PRECONDITIONING_FILENAME,
    WarmupPreconditioningEvidenceError,
    expected_warmup_clock_anchor,
    validate_warmup_preconditioning_payload,
)
from building_gen.reward_foundation import (
    HOT_OCCUPIED_ENERGY_COMFORT_V2,
    OccupiedEnergyComfortV2Components,
    OccupiedEnergyComfortV2Config,
)
from building_gen.scheduled_reward import (
    HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3, ScheduledOccupiedRewardConfig,
)
from building_gen.wrapper.simulation_clock import SimulationClock, advance_zone_timestep


@dataclass(frozen=True)
class ReportOutput:
    """Files written by :func:`generate_report`."""

    report_path: Path
    summary_path: Path
    result_run_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RunIdentity:
    run_id: str
    experiment_id: str
    algorithm_id: str
    algorithm_version: str
    seed: int
    task_id: str
    parent_training_run_id: str | None


@dataclass(frozen=True)
class _ArtifactPayload:
    """One checksum-authenticated, portable sibling artifact."""

    uri: str
    path: Path
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class ValidatedEvaluationBundle:
    """A publication evaluator bundle authenticated against its declared plan."""

    result_path: Path
    result_sha256: str
    result: Mapping[str, Any]
    run: ResolvedRun
    provenance: SoftwareProvenance
    task: TaskSpec
    protocol: EvaluationProtocolSpec
    trajectory: _ArtifactPayload
    diagnostics_artifact: _ArtifactPayload
    diagnostics_policy_artifact: _ArtifactPayload
    evidence_artifact: _ArtifactPayload
    warmup_preconditioning_artifact: _ArtifactPayload
    policy_checkpoint_artifact: _ArtifactPayload | None
    training_bundle_artifact: _ArtifactPayload | None
    rows: tuple[Mapping[str, str], ...]
    diagnostics: Mapping[str, Any]
    evidence: Mapping[str, Any]
    warmup_preconditioning: Mapping[str, Any]
    window_metrics: Mapping[str, Mapping[str, float]]


@dataclass
class _Accumulator:
    reward: float = 0.0
    interval_hours: float = 0.0
    legacy_zone_hours: float = 0.0
    legacy_degree_hours: float = 0.0
    legacy_all_comfort_hours: float = 0.0
    occupied_exposure_intensity: float = 0.0
    occupied_exposure: float = 0.0
    occupied_person_hours: float = 0.0
    controlled_floor_area_m2: float | None = None
    has_occupied_metrics: bool | None = None
    energy: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.energy is None:
            self.energy = {}

    def add(self, row: Mapping[str, str], carriers: Sequence[str]) -> None:
        interval = _float(row["interval_hours"], field="interval_hours", positive=True)
        self.reward += _float(row["reward"], field="reward")
        self.interval_hours += interval
        self.legacy_zone_hours += _float(
            row["legacy_24h_zone_hours"],
            field="legacy_24h_zone_hours",
            positive=True,
        )
        self.legacy_degree_hours += _float(
            row["legacy_24h_comfort_violation_degree_hours"],
            field="legacy_24h_comfort_violation_degree_hours",
            nonnegative=True,
        )
        all_comfort = _int(
            row["legacy_24h_all_zones_comfort"],
            field="legacy_24h_all_zones_comfort",
        )
        if all_comfort not in {0, 1}:
            raise EvaluationError("legacy_24h_all_zones_comfort must be 0 or 1")
        self.legacy_all_comfort_hours += interval * all_comfort

        occupied_fields = (
            "occupied_comfort_exposure_person_celsius_squared_hours_per_m2",
            "occupied_person_hours",
            "controlled_floor_area_m2",
        )
        present = tuple(row[field] != "" for field in occupied_fields)
        if len(set(present)) != 1:
            raise EvaluationError("occupied comfort fields must be all present or all empty")
        row_has_occupied = present[0]
        if self.has_occupied_metrics is None:
            self.has_occupied_metrics = row_has_occupied
        elif self.has_occupied_metrics != row_has_occupied:
            raise EvaluationError("occupied comfort availability changed within a rollout")
        if row_has_occupied:
            intensity = _float(
                row[occupied_fields[0]],
                field=occupied_fields[0],
                nonnegative=True,
            )
            person_hours = _float(
                row[occupied_fields[1]],
                field=occupied_fields[1],
                nonnegative=True,
            )
            area = _float(row[occupied_fields[2]], field=occupied_fields[2], positive=True)
            if self.controlled_floor_area_m2 is None:
                self.controlled_floor_area_m2 = area
            elif not math.isclose(area, self.controlled_floor_area_m2, rel_tol=0.0, abs_tol=1e-9):
                raise EvaluationError("controlled floor area changed within a rollout")
            self.occupied_exposure_intensity += intensity
            self.occupied_exposure += intensity * area
            self.occupied_person_hours += person_hours
        assert self.energy is not None
        for carrier in carriers:
            self.energy[carrier] = self.energy.get(carrier, 0.0) + _float(
                row[f"controlled_hvac_energy_{carrier}_kwh"],
                field=f"controlled_hvac_energy_{carrier}_kwh",
                nonnegative=True,
            )

    def metrics(self, *, divisor: float = 1.0) -> dict[str, float]:
        if divisor <= 0.0 or self.interval_hours <= 0.0 or self.legacy_zone_hours <= 0.0:
            raise EvaluationError("cannot summarize empty rollout evidence")
        assert self.energy is not None
        result = {
            "total_reward": self.reward / divisor,
            "evaluated_hours": self.interval_hours / divisor,
            "legacy_24h_evaluated_zone_hours": self.legacy_zone_hours / divisor,
            "legacy_24h_comfort_violation_degree_hours": self.legacy_degree_hours / divisor,
            "legacy_24h_mean_comfort_violation_c": self.legacy_degree_hours / self.legacy_zone_hours,
            "legacy_24h_all_zones_comfort_fraction": (self.legacy_all_comfort_hours / self.interval_hours),
            "controlled_hvac_energy_kwh": sum(self.energy.values()) / divisor,
        }
        result.update(
            {f"controlled_hvac_energy_{carrier}_kwh": value / divisor for carrier, value in self.energy.items()}
        )
        if self.has_occupied_metrics:
            assert self.controlled_floor_area_m2 is not None
            result.update(
                {
                    "controlled_hvac_energy_intensity_kwh_per_m2": (
                        sum(self.energy.values()) / divisor / self.controlled_floor_area_m2
                    ),
                    "occupied_comfort_exposure_person_celsius_squared_hours": (self.occupied_exposure / divisor),
                    "occupied_comfort_exposure_person_celsius_squared_hours_per_m2": (
                        self.occupied_exposure_intensity / divisor
                    ),
                    "occupied_person_hours": self.occupied_person_hours / divisor,
                }
            )
            if self.occupied_person_hours > 0.0:
                result["occupied_mean_squared_comfort_violation_celsius_squared"] = (
                    self.occupied_exposure / self.occupied_person_hours
                )
        return result


_METRIC_METADATA = {
    "total_reward": ("1", "maximize"),
    "evaluated_hours": ("h", "neutral"),
    "legacy_24h_evaluated_zone_hours": ("zone h", "neutral"),
    "legacy_24h_comfort_violation_degree_hours": ("degC h", "minimize"),
    "legacy_24h_mean_comfort_violation_c": ("degC", "minimize"),
    "legacy_24h_all_zones_comfort_fraction": ("1", "maximize"),
    "controlled_hvac_energy_kwh": ("kWh", "minimize"),
    "controlled_hvac_energy_intensity_kwh_per_m2": ("kWh/m2", "minimize"),
    "occupied_comfort_exposure_person_celsius_squared_hours": ("person degC2 h", "minimize"),
    "occupied_comfort_exposure_person_celsius_squared_hours_per_m2": (
        "person degC2 h/m2",
        "minimize",
    ),
    "occupied_person_hours": ("person h", "neutral"),
    "occupied_mean_squared_comfort_violation_celsius_squared": ("degC2", "minimize"),
}


def generate_report(
    result_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    task_specs: Mapping[str, TaskSpec] | Iterable[TaskSpec] | None = None,
    expected_runs: Iterable[ResolvedRun] | None = None,
    evaluation_protocols: Mapping[str, EvaluationProtocolSpec] | None = None,
    expected_training_contracts: Mapping[str, ExpectedTrainingContract] | None = None,
    title: str = "HOT single-building control benchmark",
    task_labels: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> ReportOutput:
    """Validate run artifacts and write ``summary.csv`` and ``report.md``.

    Results are pooled across seeds only when their experiment, algorithm
    version, and evaluation task match.  The function fails rather than
    silently combining duplicate seeds, different configurations, partial
    calendar windows, or non-finite evidence.
    """

    paths = tuple(Path(path) for path in result_paths)
    if not paths:
        raise EvaluationError("at least one result artifact is required")
    tasks_by_id = _validate_task_specs(task_specs)
    expected_by_id, protocols_by_experiment = _validate_expected_matrix(
        expected_runs,
        evaluation_protocols,
    )
    contracts_by_training_run_id = _validate_expected_training_contracts(
        expected_training_contracts,
        expected_by_id.values(),
    )
    identities: list[_RunIdentity] = []
    evidence_by_run_id: dict[str, Mapping[str, Any]] = {}
    provenance_by_run_id: dict[str, SoftwareProvenance] = {}
    run_window_metrics: list[tuple[_RunIdentity, str, dict[str, float]]] = []
    window_sets: dict[tuple[str, str], set[frozenset[str]]] = defaultdict(set)
    seen_run_ids: set[str] = set()
    seen_result_paths: set[Path] = set()

    for result_path in paths:
        canonical_path = result_path.resolve()
        if canonical_path in seen_result_paths:
            raise EvaluationError(f"duplicate result path: {result_path}")
        seen_result_paths.add(canonical_path)
        try:
            result = _read_json(result_path)
            run_id = _string(result.get("run_id"), field="run_id")
            expected_run = expected_by_id[run_id]
        except KeyError as error:
            raise EvaluationError(f"unexpected evaluation result run_id {run_id!r}") from error
        assert expected_run.evaluation_task_id is not None
        try:
            task = tasks_by_id[expected_run.evaluation_task_id]
        except KeyError as error:
            raise EvaluationError(
                f"publication report has no TaskSpec for evaluation task {expected_run.evaluation_task_id!r}"
            ) from error
        source_task = None
        if expected_run.parent_training_run_id is not None:
            if len(expected_run.train_task_ids) != 1:
                raise EvaluationError("publication learned evaluation requires exactly one trusted source task")
            source_task_id = expected_run.train_task_ids[0]
            try:
                source_task = tasks_by_id[source_task_id]
            except KeyError as error:
                raise EvaluationError(
                    f"publication report has no TaskSpec for source task {source_task_id!r}"
                ) from error
        bundle = validate_evaluation_bundle(
            result_path,
            task=task,
            expected_run=expected_run,
            protocol=protocols_by_experiment[expected_run.experiment_id],
            source_task=source_task,
            expected_training_contract=(
                None
                if expected_run.parent_training_run_id is None
                else contracts_by_training_run_id[expected_run.parent_training_run_id]
            ),
        )
        identity = _identity_from_run(bundle.run)
        if identity.run_id in seen_run_ids:
            raise EvaluationError(f"duplicate run_id {identity.run_id!r}")
        seen_run_ids.add(identity.run_id)
        evidence_by_run_id[identity.run_id] = bundle.evidence
        provenance_by_run_id[identity.run_id] = bundle.provenance
        metrics = bundle.window_metrics
        identities.append(identity)
        run_window_metrics.extend((identity, window, values) for window, values in metrics.items())
        window_sets[(identity.experiment_id, identity.task_id)].add(frozenset(metrics) - {"all_windows"})

    missing_run_ids = set(expected_by_id) - seen_run_ids
    if missing_run_ids:
        raise EvaluationError(f"missing expected evaluation result run_id(s): {sorted(missing_run_ids)!r}")
    _reject_mixed_configs_or_duplicate_seeds(
        identities,
        evidence_by_run_id,
        provenance_by_run_id,
    )
    inconsistent_windows = {key: signatures for key, signatures in window_sets.items() if len(signatures) != 1}
    if inconsistent_windows:
        raise EvaluationError("evaluation seeds contain different calendar-window sets: " f"{inconsistent_windows!r}")
    labels = _validate_task_labels(task_labels)
    summary_rows = _summarize_seeds(run_window_metrics, task_labels=labels)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.csv"
    report_path = destination / "report.md"
    if not overwrite:
        existing = [path for path in (summary_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite report artifact(s): " + ", ".join(map(str, existing)))
    summary_text = _summary_csv(summary_rows)
    report_text = _markdown_report(title, summary_rows, len(identities))
    _atomic_write_text(summary_path, summary_text)
    _atomic_write_text(report_path, report_text)
    return ReportOutput(report_path, summary_path, tuple(sorted(seen_run_ids)))


def validate_evaluation_bundle(
    result_path: str | Path,
    *,
    task: TaskSpec,
    expected_run: ResolvedRun,
    protocol: EvaluationProtocolSpec,
    source_task: TaskSpec | None = None,
    expected_training_contract: ExpectedTrainingContract | None = None,
) -> ValidatedEvaluationBundle:
    """Authenticate one complete evaluator result against its exact plan.

    This is the single publication gate used by both report generation and
    scientific reward-scale derivation.  It deliberately accepts neither an
    inferred run matrix nor a trajectory-only surrogate.
    """

    if not isinstance(task, TaskSpec):
        raise EvaluationError("evaluation bundle validation requires a TaskSpec")
    if not isinstance(expected_run, ResolvedRun) or expected_run.phase != "evaluate":
        raise EvaluationError("evaluation bundle validation requires one expected evaluation ResolvedRun")
    if not isinstance(protocol, EvaluationProtocolSpec):
        raise EvaluationError("evaluation bundle validation requires an EvaluationProtocolSpec")
    if not protocol.deterministic_actions:
        raise EvaluationError("publication evaluation protocol requires deterministic_actions=true")
    if expected_run.evaluation_task_id != task.task_id:
        raise EvaluationError("expected ResolvedRun evaluation task does not match the supplied TaskSpec")
    expected_training_run = _expected_parent_training_run(expected_run, source_task)
    if expected_training_run is None:
        if expected_training_contract is not None:
            raise EvaluationError("built-in evaluation cannot supply a training contract")
    elif not isinstance(expected_training_contract, ExpectedTrainingContract):
        raise EvaluationError("learned evaluation requires its independently derived training contract")

    path = Path(result_path)
    result = _read_json(path)
    _validate_finite_json(result, field=str(path), allow_strings=True, allow_none=True)
    identity = _result_identity(result)
    if identity.run_id != expected_run.run_id:
        raise EvaluationError(
            f"evaluation result run_id {identity.run_id!r} does not match explicitly expected "
            f"ResolvedRun {expected_run.run_id!r}"
        )
    provenance = _result_provenance(result)
    try:
        validate_scientific_software_provenance(
            provenance,
            context="publication evaluation",
        )
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"invalid scientific software provenance: {error}") from error
    policy_checkpoint_artifact = _policy_checkpoint_artifact(result, path, identity)
    training_bundle_artifact, training_bundle_reference = _training_bundle_artifact(
        result,
        path,
        identity,
        expected_training_run=expected_training_run,
        source_task=source_task,
        provenance=provenance,
        policy_checkpoint_artifact=policy_checkpoint_artifact,
        expected_training_contract=expected_training_contract,
    )
    policy_reference, diagnostics_policy = _validate_diagnostics_policy_artifact(
        result,
        path,
        identity,
        task,
        provenance,
    )
    diagnostics = _validate_diagnostics_artifact(
        result,
        path,
        identity,
        task,
        provenance,
        policy_reference,
        diagnostics_policy,
    )
    warmup_preconditioning_reference, warmup_preconditioning = _validate_warmup_preconditioning_artifact(
        result,
        path,
        identity,
        task,
        protocol,
        diagnostics,
    )
    evidence = _validate_evaluation_evidence_artifact(
        result,
        path,
        identity,
        task,
        policy_reference,
        warmup_preconditioning_reference,
        policy_checkpoint_artifact,
        training_bundle_reference,
        expected_training_contract,
    )
    trajectory = _trajectory_artifact(result, path)
    try:
        trajectory_text = trajectory.payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise EvaluationError(f"trajectory {trajectory.path} is not valid UTF-8") from error
    reader = csv.DictReader(io.StringIO(trajectory_text))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise EvaluationError(f"trajectory {trajectory.path} must have one unique CSV column per field")
    rows = tuple(reader)
    if not rows:
        raise EvaluationError(f"trajectory {trajectory.path} contains no rows")
    metrics = _validate_and_summarize_rows(
        rows,
        result,
        identity,
        task,
        evidence,
        diagnostics,
        protocol,
    )
    diagnostics_artifact = _artifact_payload(
        result,
        path,
        role="energyplus_diagnostics",
        media_type="application/json",
    )
    diagnostics_policy_artifact = _artifact_payload(
        result,
        path,
        role="diagnostics_acceptance_policy",
        media_type="application/json",
    )
    evidence_artifact = _artifact_payload(
        result,
        path,
        role="evaluation_evidence",
        media_type="application/json",
    )
    warmup_preconditioning_artifact = _artifact_payload(
        result,
        path,
        role="warmup_preconditioning_evidence",
        media_type="application/json",
    )
    try:
        result_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvaluationError(f"cannot read result artifact {path}: {error}") from error
    return ValidatedEvaluationBundle(
        result_path=path.resolve(),
        result_sha256=result_sha256,
        result=result,
        run=expected_run,
        provenance=provenance,
        task=task,
        protocol=protocol,
        trajectory=trajectory,
        diagnostics_artifact=diagnostics_artifact,
        diagnostics_policy_artifact=diagnostics_policy_artifact,
        evidence_artifact=evidence_artifact,
        warmup_preconditioning_artifact=warmup_preconditioning_artifact,
        policy_checkpoint_artifact=policy_checkpoint_artifact,
        training_bundle_artifact=training_bundle_artifact,
        rows=rows,
        diagnostics=diagnostics,
        evidence=evidence,
        warmup_preconditioning=warmup_preconditioning,
        window_metrics=metrics,
    )


def _identity_from_run(run: ResolvedRun) -> _RunIdentity:
    assert run.evaluation_task_id is not None
    return _RunIdentity(
        run.run_id,
        run.experiment_id,
        run.algorithm_id,
        run.algorithm_version,
        run.seed,
        run.evaluation_task_id,
        run.parent_training_run_id,
    )


def _expected_parent_training_run(
    expected_run: ResolvedRun,
    source_task: TaskSpec | None,
) -> ResolvedRun | None:
    """Reconstruct a learned run's parent solely from the trusted report plan."""

    if expected_run.parent_training_run_id is None:
        if source_task is not None:
            raise EvaluationError("built-in evaluation cannot supply a source training TaskSpec")
        return None
    if not isinstance(source_task, TaskSpec):
        raise EvaluationError("learned evaluation bundle validation requires its trusted source TaskSpec")
    if expected_run.train_task_ids != (source_task.task_id,):
        raise EvaluationError("trusted evaluation run does not identify the supplied source TaskSpec")
    try:
        training_run = ResolvedRun(
            experiment_id=expected_run.experiment_id,
            algorithm_id=expected_run.algorithm_id,
            algorithm_version=expected_run.algorithm_version,
            phase="train",
            seed=expected_run.seed,
            train_task_ids=expected_run.train_task_ids,
        )
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"cannot reconstruct trusted parent training run: {error}") from error
    if training_run.run_id != expected_run.parent_training_run_id:
        raise EvaluationError(
            "trusted evaluation matrix parent_training_run_id does not match its reconstructed training run"
        )
    return training_run


def _validate_expected_matrix(
    expected_runs: Iterable[ResolvedRun] | None,
    evaluation_protocols: Mapping[str, EvaluationProtocolSpec] | None,
) -> tuple[dict[str, ResolvedRun], dict[str, EvaluationProtocolSpec]]:
    if expected_runs is None:
        raise EvaluationError("publication reports require explicit expected ResolvedRun evaluation runs")
    runs = tuple(expected_runs)
    if not runs:
        raise EvaluationError("expected_runs cannot be empty")
    if any(not isinstance(run, ResolvedRun) or run.phase != "evaluate" for run in runs):
        raise EvaluationError("expected_runs must contain only evaluation ResolvedRun instances")
    expected_by_id = {run.run_id: run for run in runs}
    if len(expected_by_id) != len(runs):
        raise EvaluationError("expected_runs cannot contain duplicate evaluation runs")
    if evaluation_protocols is None:
        raise EvaluationError("publication reports require declared EvaluationProtocolSpecs")
    protocols = dict(evaluation_protocols)
    expected_experiment_ids = {run.experiment_id for run in runs}
    if set(protocols) != expected_experiment_ids:
        raise EvaluationError(
            "evaluation_protocols must exactly cover expected experiment IDs: "
            f"missing={sorted(expected_experiment_ids - set(protocols))}, "
            f"unknown={sorted(set(protocols) - expected_experiment_ids)}"
        )
    if any(not isinstance(protocol, EvaluationProtocolSpec) for protocol in protocols.values()):
        raise EvaluationError("evaluation_protocols values must be EvaluationProtocolSpec instances")
    if any(not protocol.deterministic_actions for protocol in protocols.values()):
        raise EvaluationError("publication evaluation protocols require deterministic_actions=true")
    return expected_by_id, protocols


def _validate_expected_training_contracts(
    contracts: Mapping[str, ExpectedTrainingContract] | None,
    expected_runs: Iterable[ResolvedRun],
) -> dict[str, ExpectedTrainingContract]:
    parent_ids = {run.parent_training_run_id for run in expected_runs if run.parent_training_run_id is not None}
    supplied = {} if contracts is None else dict(contracts)
    if set(supplied) != parent_ids:
        raise EvaluationError(
            "expected_training_contracts must exactly cover trusted learned parent runs: "
            f"missing={sorted(parent_ids - set(supplied))}, "
            f"unknown={sorted(set(supplied) - parent_ids)}"
        )
    if any(not isinstance(contract, ExpectedTrainingContract) for contract in supplied.values()):
        raise EvaluationError("expected_training_contracts values must be ExpectedTrainingContract instances")
    return supplied


def discover_results(results_dir: str | Path) -> tuple[Path, ...]:
    """Return sorted evaluator result artifacts from one directory."""

    directory = Path(results_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = tuple(sorted(directory.glob("*.result.json")))
    if not paths:
        raise FileNotFoundError(f"no *.result.json artifacts found in {directory}")
    return paths


def _validate_task_specs(
    task_specs: Mapping[str, TaskSpec] | Iterable[TaskSpec] | None,
) -> dict[str, TaskSpec]:
    if task_specs is None:
        raise EvaluationError("publication reports require task_specs for every evaluation task")
    if isinstance(task_specs, Mapping):
        items = tuple(task_specs.items())
        if not items:
            raise EvaluationError("task_specs cannot be empty")
        output: dict[str, TaskSpec] = {}
        for task_id, task in items:
            if not isinstance(task, TaskSpec):
                raise EvaluationError("task_specs values must be TaskSpec instances")
            if task_id != task.task_id:
                raise EvaluationError(f"task_specs key {task_id!r} does not match TaskSpec identity {task.task_id!r}")
            output[task_id] = task
        return output

    output = {}
    for task in task_specs:
        if not isinstance(task, TaskSpec):
            raise EvaluationError("task_specs must contain only TaskSpec instances")
        if task.task_id in output:
            raise EvaluationError(f"duplicate TaskSpec identity {task.task_id!r}")
        output[task.task_id] = task
    if not output:
        raise EvaluationError("task_specs cannot be empty")
    return output


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read result artifact {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError(f"result artifact {path} must contain a JSON object")
    return value


def _result_identity(result: Mapping[str, Any]) -> _RunIdentity:
    expected_result_fields = {
        "schema_version",
        "run_id",
        "run",
        "status",
        "provenance",
        "metrics",
        "artifacts",
        "failure",
    }
    if set(result) != expected_result_fields:
        raise EvaluationError("RunResult does not match the supported result schema")
    if result.get("schema_version") != "hot-control-result/1.0.0":
        raise EvaluationError("unsupported RunResult schema_version")
    if result.get("status") != "succeeded" or result.get("failure") is not None:
        raise EvaluationError("reports accept only succeeded RunResult artifacts")
    artifacts = result.get("artifacts")
    required_artifact_roles = {
        "step_trajectory",
        "energyplus_diagnostics",
        "diagnostics_acceptance_policy",
        "evaluation_evidence",
        "warmup_preconditioning_evidence",
    }
    allowed_artifact_roles = required_artifact_roles | {
        "policy_checkpoint",
        "training_bundle",
    }
    if not isinstance(artifacts, list) or any(not isinstance(item, Mapping) for item in artifacts):
        raise EvaluationError("RunResult artifacts must be a list of JSON objects")
    artifact_roles = [item.get("role") for item in artifacts]
    if any(not isinstance(role, str) for role in artifact_roles):
        raise EvaluationError("RunResult artifacts do not exactly match the current scientific evaluator bundle")
    for role in sorted(required_artifact_roles):
        if artifact_roles.count(role) != 1:
            raise EvaluationError(f"RunResult must contain exactly one {role} artifact")
    unknown_roles = set(artifact_roles) - allowed_artifact_roles
    if unknown_roles:
        raise EvaluationError(
            "RunResult contains artifacts outside the current scientific evaluator bundle: "
            f"{sorted(unknown_roles)!r}"
        )
    run = result.get("run")
    if not isinstance(run, Mapping):
        raise EvaluationError("RunResult must describe an evaluation run")
    expected_run_fields = {
        "schema_version",
        "experiment_id",
        "algorithm_id",
        "algorithm_version",
        "phase",
        "seed",
        "train_task_ids",
        "evaluation_task_id",
        "parent_training_run_id",
    }
    if set(run) != expected_run_fields:
        raise EvaluationError(
            "RunResult run payload does not match the ResolvedRun identity schema: "
            f"missing={sorted(expected_run_fields - set(run))}, "
            f"unknown={sorted(set(run) - expected_run_fields)}"
        )
    if run.get("schema_version") != RUN_SCHEMA_VERSION or run.get("phase") != "evaluate":
        raise EvaluationError("RunResult must describe a supported evaluation run")
    raw_train_task_ids = run.get("train_task_ids")
    if not isinstance(raw_train_task_ids, list) or any(
        not isinstance(value, str) or not value for value in raw_train_task_ids
    ):
        raise EvaluationError("run train_task_ids must be a JSON list of non-empty strings")
    if len(set(raw_train_task_ids)) != len(raw_train_task_ids):
        raise EvaluationError("run train_task_ids cannot contain duplicates")
    parent_training_run_id = run.get("parent_training_run_id")
    if parent_training_run_id is not None:
        parent_training_run_id = _string(
            parent_training_run_id,
            field="parent_training_run_id",
        )
    try:
        resolved = ResolvedRun(
            experiment_id=_string(run.get("experiment_id"), field="experiment_id"),
            algorithm_id=_string(run.get("algorithm_id"), field="algorithm_id"),
            algorithm_version=_string(run.get("algorithm_version"), field="algorithm_version"),
            phase="evaluate",
            seed=run.get("seed"),
            train_task_ids=tuple(raw_train_task_ids),
            evaluation_task_id=_string(run.get("evaluation_task_id"), field="evaluation_task_id"),
            parent_training_run_id=parent_training_run_id,
            schema_version=RUN_SCHEMA_VERSION,
        )
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"invalid embedded ResolvedRun: {error}") from error
    policy_checkpoint_count = artifact_roles.count("policy_checkpoint")
    training_bundle_count = artifact_roles.count("training_bundle")
    if resolved.parent_training_run_id is None:
        if policy_checkpoint_count or training_bundle_count:
            raise EvaluationError("built-in RunResult cannot contain policy_checkpoint or training_bundle artifacts")
    elif policy_checkpoint_count != 1 or training_bundle_count != 1:
        raise EvaluationError(
            "learned RunResult must contain exactly one policy_checkpoint and one training_bundle artifact"
        )
    run_id = _string(result.get("run_id"), field="run_id")
    if run_id != resolved.run_id:
        raise EvaluationError("RunResult run_id does not authenticate its embedded ResolvedRun")
    assert resolved.evaluation_task_id is not None
    return _RunIdentity(
        run_id,
        resolved.experiment_id,
        resolved.algorithm_id,
        resolved.algorithm_version,
        resolved.seed,
        resolved.evaluation_task_id,
        resolved.parent_training_run_id,
    )


def _result_provenance(result: Mapping[str, Any]) -> SoftwareProvenance:
    raw = result.get("provenance")
    expected = {
        "schema_version",
        "repository",
        "revision",
        "dirty",
        "python_version",
        "energyplus_version",
        "container_digest",
        "dependencies",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise EvaluationError("RunResult provenance does not match the supported schema")
    if raw.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise EvaluationError("unsupported software provenance schema_version")
    if not isinstance(raw.get("dirty"), bool):
        raise EvaluationError("software provenance dirty must be a boolean")
    dependencies = raw.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise EvaluationError("software provenance dependencies must be a JSON object")
    try:
        return SoftwareProvenance(
            repository=raw.get("repository"),
            revision=raw.get("revision"),
            dirty=raw.get("dirty"),
            python_version=raw.get("python_version"),
            energyplus_version=raw.get("energyplus_version"),
            container_digest=raw.get("container_digest"),
            dependencies=dependencies,
            schema_version=PROVENANCE_SCHEMA_VERSION,
        )
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"invalid software provenance: {error}") from error


def _artifact_payload(
    result: Mapping[str, Any],
    result_path: Path,
    *,
    role: str,
    media_type: str,
) -> _ArtifactPayload:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvaluationError("RunResult artifacts must be a list")
    if any(not isinstance(item, Mapping) for item in artifacts):
        raise EvaluationError("RunResult artifacts must contain only JSON objects")
    matches = [item for item in artifacts if item.get("role") == role]
    if len(matches) != 1:
        raise EvaluationError(f"RunResult must contain exactly one {role} artifact")
    artifact = matches[0]
    if set(artifact) != {"role", "uri", "media_type", "sha256"}:
        raise EvaluationError(f"{role} artifact does not match the supported artifact schema")
    if artifact.get("media_type") != media_type:
        raise EvaluationError(f"{role} must use {media_type}")
    uri = _portable_sibling_uri(artifact.get("uri"), field=f"{role} uri")
    path = result_path.parent / uri
    if path.is_symlink():
        raise EvaluationError(f"{role} artifact must not be a symbolic link: {path}")
    if not path.is_file():
        raise EvaluationError(f"{role} artifact does not exist: {path}")
    digest = _sha256(artifact.get("sha256"), field=f"{role} sha256")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvaluationError(f"cannot read {role} artifact {path}: {error}") from error
    if hashlib.sha256(payload).hexdigest() != digest:
        raise EvaluationError(f"{role} checksum mismatch for {path}")
    return _ArtifactPayload(uri, path.resolve(), digest, payload)


def _trajectory_artifact(result: Mapping[str, Any], result_path: Path) -> _ArtifactPayload:
    return _artifact_payload(
        result,
        result_path,
        role="step_trajectory",
        media_type="text/csv",
    )


def _policy_checkpoint_artifact(
    result: Mapping[str, Any],
    result_path: Path,
    identity: _RunIdentity,
) -> _ArtifactPayload | None:
    if identity.parent_training_run_id is None:
        return None
    return _artifact_payload(
        result,
        result_path,
        role="policy_checkpoint",
        media_type="application/octet-stream",
    )


def _training_bundle_artifact(
    result: Mapping[str, Any],
    result_path: Path,
    identity: _RunIdentity,
    *,
    expected_training_run: ResolvedRun | None,
    source_task: TaskSpec | None,
    provenance: SoftwareProvenance,
    policy_checkpoint_artifact: _ArtifactPayload | None,
    expected_training_contract: ExpectedTrainingContract | None,
) -> tuple[_ArtifactPayload | None, dict[str, str] | None]:
    """Authenticate the portable training-manifest receipt for a learned run."""

    if identity.parent_training_run_id is None:
        if expected_training_run is not None or source_task is not None:
            raise EvaluationError("built-in evaluation has unexpected parent-training inputs")
        return None, None
    if expected_training_run is None or source_task is None:
        raise EvaluationError("learned evaluation lacks a trusted reconstructed training run")
    if not isinstance(expected_training_contract, ExpectedTrainingContract):
        raise EvaluationError("learned evaluation lacks its independently derived training contract")
    if policy_checkpoint_artifact is None:
        raise EvaluationError("learned evaluation lacks its authenticated policy checkpoint")
    artifact = _artifact_payload(
        result,
        result_path,
        role="training_bundle",
        media_type="application/json",
    )
    try:
        receipt = validate_training_bundle_receipt(
            artifact.payload,
            expected_run=expected_training_run,
            task=source_task,
            expected_provenance=provenance,
            expected_revision=provenance.revision,
            expected_checkpoint_sha256=policy_checkpoint_artifact.sha256,
            expected_training_contract=expected_training_contract,
        )
    except ScientificTrainingBundleError as error:
        raise EvaluationError(f"invalid linked scientific training bundle: {error}") from error
    if receipt.bundle_sha256 != artifact.sha256:
        raise EvaluationError("training_bundle receipt digest is inconsistent with its RunResult artifact")
    return (
        artifact,
        {
            "uri": artifact.uri,
            "sha256": artifact.sha256,
            "schema_version": TRAINING_BUNDLE_SCHEMA_VERSION,
        },
    )


def _validate_diagnostics_policy_artifact(
    result: Mapping[str, Any],
    result_path: Path,
    identity: _RunIdentity,
    task: TaskSpec,
    provenance: SoftwareProvenance,
) -> tuple[dict[str, str], DiagnosticsAllowlist]:
    artifact = _artifact_payload(
        result,
        result_path,
        role="diagnostics_acceptance_policy",
        media_type="application/json",
    )
    try:
        text = artifact.payload.decode("utf-8", errors="strict")
        raw = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read diagnostics acceptance policy {artifact.path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise EvaluationError("diagnostics acceptance policy must contain a JSON object")
    try:
        policy = diagnostics_allowlist_from_mapping(raw)
    except ValueError as error:
        raise EvaluationError(f"invalid diagnostics acceptance policy: {error}") from error
    if text != policy.canonical_json() or artifact.sha256 != policy.sha256:
        raise EvaluationError("diagnostics acceptance policy is not the exact canonical policy bytes")
    expected_scope = _diagnostics_scope(task, provenance)
    policy_scope = {
        "building_asset_sha256": policy.asset_sha256,
        "weather_asset_sha256": policy.weather_sha256,
        "energyplus_version": policy.energyplus_version,
    }
    if policy_scope != expected_scope:
        raise EvaluationError(
            "diagnostics acceptance policy scope does not match the TaskSpec and RunResult provenance"
        )
    return (
        {
            "uri": artifact.uri,
            "sha256": artifact.sha256,
            "schema_version": policy.schema_version,
        },
        policy,
    )


def _validate_diagnostics_artifact(
    result: Mapping[str, Any],
    result_path: Path,
    identity: _RunIdentity,
    task: TaskSpec,
    provenance: SoftwareProvenance,
    policy_reference: Mapping[str, str],
    diagnostics_policy: DiagnosticsAllowlist,
) -> Mapping[str, Any]:
    """Resolve, authenticate, and independently re-evaluate diagnostics."""

    artifact = _artifact_payload(
        result,
        result_path,
        role="energyplus_diagnostics",
        media_type="application/json",
    )
    try:
        manifest = json.loads(artifact.payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read diagnostics artifact {artifact.path}: {error}") from error
    expected_fields = {
        "schema_version",
        "run_id",
        "mode",
        "scope",
        "diagnostics_acceptance_policy",
        "episodes",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise EvaluationError("evaluation diagnostics manifest does not match the supported schema")
    if manifest.get("schema_version") != EVALUATION_DIAGNOSTICS_SCHEMA_VERSION:
        raise EvaluationError("unsupported evaluation diagnostics schema_version")
    if manifest.get("run_id") != identity.run_id:
        raise EvaluationError("energyplus_diagnostics run_id does not match its RunResult")
    if manifest.get("mode") != "scientific":
        raise EvaluationError("publication reports require scientific diagnostics mode")
    if manifest.get("scope") != _diagnostics_scope(task, provenance):
        raise EvaluationError("evaluation diagnostics scope does not match the TaskSpec and RunResult provenance")
    if manifest.get("diagnostics_acceptance_policy") != policy_reference:
        raise EvaluationError("evaluation diagnostics reference a different acceptance policy")

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise EvaluationError("evaluation diagnostics must contain completed episodes")
    episode_keys: set[tuple[str, int]] = set()
    expected_episode_fields = {
        "window",
        "episode_index",
        "status",
        "acceptance",
        "expected_executed_model_sha256",
        "executed_model_sha256_matches",
        "diagnostics",
    }
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping) or set(episode) != expected_episode_fields:
            raise EvaluationError(f"evaluation diagnostics episode {index} does not match the accepted schema")
        window = episode.get("window")
        episode_index = episode.get("episode_index")
        if (
            not isinstance(window, str)
            or not window
            or isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise EvaluationError(f"evaluation diagnostics episode {index} has invalid identity")
        key = (window, episode_index)
        if key in episode_keys:
            raise EvaluationError(f"duplicate evaluation diagnostics episode {key!r}")
        episode_keys.add(key)
        acceptance = episode.get("acceptance")
        if (
            episode.get("status") != "accepted"
            or not isinstance(acceptance, Mapping)
            or set(acceptance) != {"accepted", "violations"}
            or acceptance.get("accepted") is not True
            or acceptance.get("violations") != []
        ):
            raise EvaluationError("every evaluation diagnostics episode must be accepted")
        expected_model_sha = _sha256(
            episode.get("expected_executed_model_sha256"),
            field=f"diagnostics episode {key!r} expected executed-model sha256",
        )
        if episode.get("executed_model_sha256_matches") is not True:
            raise EvaluationError(f"diagnostics episode {key!r} does not match the executed model")
        embedded = episode.get("diagnostics")
        if not isinstance(embedded, Mapping):
            raise EvaluationError(f"diagnostics episode {key!r} lacks embedded diagnostics")
        try:
            parsed = energyplus_diagnostics_from_mapping(embedded)
        except ValueError as error:
            raise EvaluationError(f"diagnostics episode {key!r} is malformed: {error}") from error
        if parsed.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
            raise EvaluationError(f"diagnostics episode {key!r} uses an unsupported schema")
        executed_model_sha = _sha256(
            embedded.get("executed_model_sha256"),
            field=f"diagnostics episode {key!r} executed-model sha256",
        )
        if executed_model_sha != expected_model_sha:
            raise EvaluationError(f"diagnostics episode {key!r} executed-model identity is inconsistent")
        decision = evaluate_scientific_diagnostics(
            parsed,
            asset_sha256=task.building.sha256 or "",
            weather_sha256=task.weather.sha256 or "",
            allowlist=diagnostics_policy,
        )
        if not decision.accepted:
            raise EvaluationError(
                f"diagnostics episode {key!r} fails its persisted acceptance policy: " + "; ".join(decision.violations)
            )
        assert decision.canonical_diagnostics_json is not None
        evaluated_core = json.loads(decision.canonical_diagnostics_json.decode("utf-8"))
        expected_embedded_fields = set(evaluated_core) | {"executed_model_sha256"}
        if set(embedded) != expected_embedded_fields:
            raise EvaluationError(f"diagnostics episode {key!r} contains unsupported or missing fields")
        persisted_core = dict(embedded)
        persisted_core.pop("executed_model_sha256")
        if persisted_core != evaluated_core:
            raise EvaluationError(f"diagnostics episode {key!r} does not equal its scientifically evaluated snapshot")
    return manifest


def _diagnostics_scope(
    task: TaskSpec,
    provenance: SoftwareProvenance,
) -> dict[str, str]:
    if not task.publication_ready:
        raise EvaluationError("publication reports require checksum-pinned building and weather assets")
    assert task.building.sha256 is not None and task.weather.sha256 is not None
    return {
        "building_asset_sha256": task.building.sha256,
        "weather_asset_sha256": task.weather.sha256,
        "energyplus_version": provenance.energyplus_version,
    }


def _validate_warmup_preconditioning_artifact(
    result: Mapping[str, Any],
    result_path: Path,
    identity: _RunIdentity,
    task: TaskSpec,
    protocol: EvaluationProtocolSpec,
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, str], Mapping[str, Any]]:
    """Authenticate every episode's identity-bearing warmup intervention."""

    artifact = _artifact_payload(
        result,
        result_path,
        role="warmup_preconditioning_evidence",
        media_type="application/json",
    )
    try:
        manifest = json.loads(artifact.payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read warmup preconditioning evidence {artifact.path}: {error}") from error
    expected_fields = {
        "schema_version",
        "run_id",
        "task",
        "contract",
        "expected_episode_count",
        "record_count",
        "records",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise EvaluationError("evaluation warmup preconditioning evidence does not match the supported schema")
    if manifest.get("schema_version") != EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION:
        raise EvaluationError("unsupported evaluation warmup preconditioning schema_version")
    if manifest.get("run_id") != identity.run_id:
        raise EvaluationError("warmup preconditioning evidence run_id does not match its RunResult")
    expected_task = {
        "task_name": task.name,
        "task_id": task.task_id,
        "building_sha256": task.building.sha256,
        "weather_sha256": task.weather.sha256,
    }
    if manifest.get("task") != expected_task:
        raise EvaluationError("warmup preconditioning evidence task identity does not match TaskSpec")
    expected_contract = {
        "enabled": task.warmup_preconditioning.enabled,
        "payload": task.warmup_preconditioning.identity_payload(),
        "sha256": task.warmup_preconditioning.sha256,
    }
    if manifest.get("contract") != expected_contract:
        raise EvaluationError("warmup preconditioning evidence contract does not match TaskSpec")

    expected_episode_keys = [
        (window.name, episode_index)
        for window in protocol.windows
        for episode_index in range(protocol.episodes_per_window)
    ]
    expected_count = len(expected_episode_keys)
    if manifest.get("expected_episode_count") != expected_count or manifest.get("record_count") != expected_count:
        raise EvaluationError("warmup preconditioning evidence episode counts are incomplete")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise EvaluationError("warmup preconditioning evidence records are incomplete")
    diagnostic_models: dict[tuple[str, int], str] = {}
    for episode in diagnostics.get("episodes", []):
        if not isinstance(episode, Mapping):
            raise EvaluationError("warmup preconditioning cannot cross-check malformed diagnostics")
        key = (episode.get("window"), episode.get("episode_index"))
        embedded = episode.get("diagnostics")
        if not isinstance(embedded, Mapping):
            raise EvaluationError("warmup preconditioning cannot cross-check missing diagnostics")
        diagnostic_models[key] = _sha256(
            embedded.get("executed_model_sha256"),
            field=f"diagnostics episode {key!r} executed-model sha256",
        )
    if set(diagnostic_models) != set(expected_episode_keys):
        raise EvaluationError("warmup preconditioning episodes do not match diagnostic episodes")

    for index, (record, expected_key) in enumerate(zip(records, expected_episode_keys, strict=True)):
        if not isinstance(record, Mapping):
            raise EvaluationError(f"warmup preconditioning record {index} must be an object")
        if (record.get("window"), record.get("episode_index")) != expected_key:
            raise EvaluationError("warmup preconditioning records are missing, duplicated, or out of canonical order")
        if record.get("contract_sha256") != task.warmup_preconditioning.sha256:
            raise EvaluationError(f"warmup preconditioning record {expected_key!r} contract SHA256 is inconsistent")
        if not task.warmup_preconditioning.enabled:
            if set(record) != {
                "window",
                "episode_index",
                "status",
                "contract_sha256",
                "evidence",
            }:
                raise EvaluationError(f"disabled warmup preconditioning record {expected_key!r} has an invalid schema")
            if record.get("status") != "disabled" or record.get("evidence") is not None:
                raise EvaluationError(f"disabled warmup preconditioning record {expected_key!r} claims an intervention")
            continue

        if set(record) != {
            "window",
            "episode_index",
            "status",
            "contract_sha256",
            "executed_model_sha256",
            "evidence",
        }:
            raise EvaluationError(f"enabled warmup preconditioning record {expected_key!r} has an invalid schema")
        if record.get("status") != "accepted":
            raise EvaluationError(f"enabled warmup preconditioning record {expected_key!r} was not accepted")
        executed_model_sha256 = _sha256(
            record.get("executed_model_sha256"),
            field=f"warmup preconditioning record {expected_key!r} executed-model sha256",
        )
        if executed_model_sha256 != diagnostic_models[expected_key]:
            raise EvaluationError(
                f"warmup preconditioning record {expected_key!r} identifies a different executed model"
            )
        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "source_path",
            "source_file_sha256",
            "canonical_payload_sha256",
            "payload",
        }:
            raise EvaluationError(f"warmup preconditioning record {expected_key!r} evidence has an invalid schema")
        if evidence.get("source_path") != (f"energyplus/{WARMUP_PRECONDITIONING_FILENAME}"):
            raise EvaluationError(f"warmup preconditioning record {expected_key!r} source path is not canonical")
        _sha256(
            evidence.get("source_file_sha256"),
            field=f"warmup preconditioning record {expected_key!r} source-file sha256",
        )
        payload = evidence.get("payload")
        if not isinstance(payload, Mapping):
            raise EvaluationError(f"warmup preconditioning record {expected_key!r} payload must be an object")
        canonical_payload_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if evidence.get("canonical_payload_sha256") != canonical_payload_sha256:
            raise EvaluationError(f"warmup preconditioning record {expected_key!r} canonical payload checksum mismatch")
        window = next(window for window in protocol.windows if window.name == expected_key[0])
        try:
            validate_warmup_preconditioning_payload(
                payload,
                task=task,
                window=window,
                executed_model_sha256=executed_model_sha256,
            )
        except WarmupPreconditioningEvidenceError as error:
            raise EvaluationError(f"warmup preconditioning record {expected_key!r} is invalid: {error}") from error

    return (
        {
            "uri": artifact.uri,
            "sha256": artifact.sha256,
            "schema_version": EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION,
        },
        manifest,
    )


def _validate_evaluation_evidence_artifact(
    result: Mapping[str, Any],
    result_path: Path,
    identity: _RunIdentity,
    task: TaskSpec,
    policy_reference: Mapping[str, str],
    warmup_preconditioning_reference: Mapping[str, str],
    policy_checkpoint_artifact: _ArtifactPayload | None,
    training_bundle_reference: Mapping[str, str] | None,
    expected_training_contract: ExpectedTrainingContract | None,
) -> Mapping[str, Any]:
    """Authenticate metric/action semantics and learned-policy provenance."""

    artifact = _artifact_payload(
        result,
        result_path,
        role="evaluation_evidence",
        media_type="application/json",
    )
    try:
        manifest = json.loads(artifact.payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read evaluation evidence artifact {artifact.path}: {error}") from error
    base_manifest_fields = {
        "schema_version",
        "run_id",
        "trajectory_schema_version",
        "diagnostics_acceptance_policy",
        "warmup_preconditioning_evidence",
        "policy",
        "energy",
        "comfort",
        "observations",
        "actions",
    }
    if not isinstance(manifest, Mapping):
        raise EvaluationError("evaluation evidence manifest does not match the supported schema")
    evidence_schema = manifest.get("schema_version")
    if evidence_schema == LEGACY_EVALUATION_EVIDENCE_SCHEMA_VERSION:
        if identity.parent_training_run_id is not None or training_bundle_reference is not None:
            raise EvaluationError("learned publication evidence requires the current training-bundle schema")
        expected_manifest_fields = base_manifest_fields
    elif evidence_schema == EVALUATION_EVIDENCE_SCHEMA_VERSION:
        if identity.parent_training_run_id is None or training_bundle_reference is None:
            raise EvaluationError("current evaluation evidence schema is reserved for linked learned policies")
        expected_manifest_fields = base_manifest_fields | {"training_bundle"}
    else:
        if set(manifest) not in (
            base_manifest_fields,
            base_manifest_fields | {"training_bundle"},
        ):
            raise EvaluationError("evaluation evidence manifest does not match the supported schema")
        raise EvaluationError("unsupported evaluation evidence schema_version")
    if set(manifest) != expected_manifest_fields:
        raise EvaluationError("evaluation evidence manifest does not match the supported schema")
    if manifest.get("run_id") != identity.run_id:
        raise EvaluationError("evaluation evidence run_id does not match its RunResult")
    if manifest.get("trajectory_schema_version") != STEP_SCHEMA_VERSION:
        raise EvaluationError("evaluation evidence references an unsupported trajectory schema")
    if manifest.get("diagnostics_acceptance_policy") != policy_reference:
        raise EvaluationError("evaluation evidence references a different diagnostics acceptance policy")
    if manifest.get("warmup_preconditioning_evidence") != warmup_preconditioning_reference:
        raise EvaluationError("evaluation evidence references different warmup preconditioning evidence")
    if evidence_schema == EVALUATION_EVIDENCE_SCHEMA_VERSION and (
        manifest.get("training_bundle") != training_bundle_reference
    ):
        raise EvaluationError("evaluation evidence references a different scientific training bundle")

    raw_policy = manifest.get("policy")
    policy_fields = {
        "kind",
        "policy_id",
        "parent_training_run_id",
        "checkpoint_sha256",
        "training_step",
        "selection_rule",
    }
    if not isinstance(raw_policy, Mapping) or set(raw_policy) != policy_fields:
        raise EvaluationError("evaluation evidence policy does not match the supported schema")
    try:
        policy = PolicyProvenance(
            kind=raw_policy.get("kind"),
            policy_id=raw_policy.get("policy_id"),
            parent_training_run_id=raw_policy.get("parent_training_run_id"),
            checkpoint_sha256=raw_policy.get("checkpoint_sha256"),
            training_step=raw_policy.get("training_step"),
            selection_rule=raw_policy.get("selection_rule"),
        )
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"invalid evaluation policy evidence: {error}") from error
    if dict(raw_policy) != policy.to_dict() or policy.policy_id != identity.algorithm_id:
        raise EvaluationError("evaluation evidence policy does not match the run algorithm")
    if identity.parent_training_run_id is None:
        if policy.kind != "built_in":
            raise EvaluationError("a run without parent training must identify a built-in policy")
        if policy_checkpoint_artifact is not None:
            raise EvaluationError("built-in policy evidence cannot attach a policy_checkpoint artifact")
        if training_bundle_reference is not None:
            raise EvaluationError("built-in policy evidence cannot attach a training_bundle artifact")
    else:
        if policy.kind != "learned_checkpoint":
            raise EvaluationError("a learned run must identify checkpoint policy evidence")
        if policy.parent_training_run_id != identity.parent_training_run_id:
            raise EvaluationError("policy evidence parent training run does not match RunResult")
        if policy_checkpoint_artifact is None:
            raise EvaluationError("learned policy evidence requires an authenticated policy_checkpoint artifact")
        if policy.checkpoint_sha256 != policy_checkpoint_artifact.sha256:
            raise EvaluationError("policy_checkpoint digest does not match evaluation policy evidence")
        if training_bundle_reference is None:
            raise EvaluationError("learned policy evidence requires an authenticated training_bundle artifact")
        if not isinstance(expected_training_contract, ExpectedTrainingContract):
            raise EvaluationError("learned policy evidence lacks its independently derived training contract")
        if policy.training_step != expected_training_contract.completed_transitions:
            raise EvaluationError("learned policy training_step does not match the training_bundle contract")
        if policy.selection_rule != final_checkpoint_selection_rule(expected_training_contract.completed_transitions):
            raise EvaluationError("learned policy selection_rule does not identify the sealed final checkpoint")

    energy = manifest.get("energy")
    energy_fields = {
        "metric_prefix",
        "scope",
        "unit",
        "interval_source",
        "carrier_meter_names",
        "excluded",
        "controlled_floor_area_m2",
        "floor_area_source",
    }
    if not isinstance(energy, Mapping) or set(energy) != energy_fields:
        raise EvaluationError("evaluation evidence energy semantics do not match the supported schema")
    expected_energy = {
        "metric_prefix": "controlled_hvac_energy",
        "scope": "controlled HVAC site energy only",
        "unit": "kWh",
        "interval_source": "EnergyPlus Zone Timestep meter values converted once from joules",
        "excluded": "non-HVAC building loads, source energy, and carbon emissions",
    }
    if any(energy.get(field) != value for field, value in expected_energy.items()):
        raise EvaluationError("evaluation evidence does not declare the exact controlled-HVAC energy semantics")
    meter_names = energy.get("carrier_meter_names")
    expected_meter_names = {
        carrier.value: meter_name for carrier, meter_name in CONTROLLED_HVAC_METER_BY_CARRIER.items()
    }
    if task.component_energy_meters is not None:
        expected_meter_names = dict(task.component_energy_meters)
        if not isinstance(meter_names, Mapping) or dict(meter_names) != expected_meter_names:
            raise EvaluationError('evaluation evidence must match every task-declared component meter')
    if not isinstance(meter_names, Mapping):
        raise EvaluationError("evaluation evidence carrier meter names must be a canonical controlled-HVAC mapping")
    actual_meter_names = dict(meter_names)
    if "electricity" not in actual_meter_names:
        raise EvaluationError("evaluation evidence must report the required controlled-HVAC electricity meter")
    if set(actual_meter_names) - set(expected_meter_names) or any(
        actual_meter_names[carrier] != expected_meter_names[carrier] for carrier in actual_meter_names
    ):
        raise EvaluationError(
            "evaluation evidence carrier meter names do not exactly match a canonical " "controlled-HVAC meter subset"
        )

    reward_v2 = task.reward.reward_id in (HOT_OCCUPIED_ENERGY_COMFORT_V2, HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3)
    expected_floor_source = f"{task.reward.reward_id} reward components" if reward_v2 else None
    if energy.get("floor_area_source") != expected_floor_source:
        raise EvaluationError("evaluation evidence floor-area source does not match the TaskSpec reward")
    area = energy.get("controlled_floor_area_m2")
    if reward_v2:
        _finite_real(area, field="evaluation evidence controlled_floor_area_m2", positive=True)
    elif area is not None:
        raise EvaluationError("non-v2 evaluation evidence cannot declare a controlled floor area")

    comfort = manifest.get("comfort")
    expected_comfort = {
        "primary_scope": ("occupied, person-weighted squared temperature-band violation" if reward_v2 else None),
        "primary_source": expected_floor_source,
        "legacy_diagnostics": (
            "24-hour unweighted zone-temperature violations are retained only in metrics prefixed legacy_24h_"
        ),
    }
    if task.reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3:
        schedule = ScheduledOccupiedRewardConfig.from_mapping(task.reward.parameter_mapping).comfort_schedule
        expected_comfort.update(scheduled_comfort=schedule.to_mapping(),
                                legacy_reference_band_c=[schedule.default.minimum_c, schedule.default.maximum_c])
    if not isinstance(comfort, Mapping) or dict(comfort) != expected_comfort:
        raise EvaluationError("evaluation evidence comfort semantics do not match the TaskSpec reward")

    observations = manifest.get("observations")
    expected_observations = {
        "policy_observation_normalized_json": ("normalized state presented to the policy before the recorded action"),
        "next_raw_observation_json": "physical simulator state returned after the submitted action",
        "policy_observation_encoding": task.observation_space.encoding,
    }
    if not isinstance(observations, Mapping) or dict(observations) != expected_observations:
        raise EvaluationError("evaluation evidence observation semantics do not match the TaskSpec")

    actions = manifest.get("actions")
    entities = task.action_profile.controlled_entities
    channels = task.action_profile.channels
    expected_actions = {
        "proposal": "normalized policy output before environment transformation",
        "proposal_channel_order": [f"{entity}:{channel.name}" for entity in entities for channel in channels],
        "proposal_units": [channel.unit for _entity in entities for channel in channels],
        "transform_id": task.action_profile.transform_id,
        "submitted_schema_version": SUBMITTED_ACTION_SCHEMA_VERSION,
        "submitted_channel_order": [
            f"{entity}:{name}" for entity in entities for name in ("heating_setpoint_c", "cooling_setpoint_c")
        ],
        "submitted_units": ["degC"] * (2 * len(entities)),
        "submitted_semantics": "setpoints transformed by the environment and submitted to EnergyPlus",
        "actuator_readback": "unavailable",
    }
    if isinstance(actions, Mapping) and actions.get('actuator_readback') == 'availability_and_values_recorded_per_step':
        expected_actions['actuator_readback'] = 'availability_and_values_recorded_per_step'
    if not isinstance(actions, Mapping) or dict(actions) != expected_actions:
        raise EvaluationError("evaluation evidence action semantics do not match the TaskSpec")
    return manifest


_CLOCK_FIELDS = (
    "calendar_year",
    "month",
    "day_of_month",
    "hour",
    "minutes",
    "day_of_week",
    "day_of_year",
    "timestep_index",
    "timesteps_per_hour",
    "episode_step_index",
    "calendar_year_semantics",
)


def _validate_and_summarize_rows(
    rows: Sequence[Mapping[str, str]],
    result: Mapping[str, Any],
    identity: _RunIdentity,
    task: TaskSpec,
    evidence: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    protocol: EvaluationProtocolSpec,
) -> dict[str, dict[str, float]]:
    required = {
        "schema_version",
        "run_id",
        "experiment_id",
        "algorithm_id",
        "algorithm_version",
        "seed",
        "task_id",
        "window",
        "window_start_month",
        "window_start_day",
        "window_end_month",
        "window_end_day",
        "episode_index",
        "episodes_per_window",
        "step_index",
        "expected_steps",
        "timesteps_per_hour",
        "interval_hours",
        "reward",
        "reward_components_json",
        "legacy_24h_comfort_violation_degree_hours",
        "legacy_24h_zone_hours",
        "legacy_24h_all_zones_comfort",
        "occupied_comfort_exposure_person_celsius_squared_hours_per_m2",
        "occupied_person_hours",
        "controlled_floor_area_m2",
        "terminated",
        "proposed_action_normalized_json",
        "proposed_action_channel_names_json",
        "proposed_action_units_json",
        "action_transform_id",
        "submitted_action_schema_version",
        "submitted_action_physical_json",
        "submitted_action_channel_names_json",
        "submitted_action_units_json",
        "actuator_readback_status",
        "actuator_readback_reason",
        "decision_clock_json",
        "policy_observation_normalized_json",
        "next_raw_observation_json",
    }
    required |= {f"decision_clock_{field}" for field in _CLOCK_FIELDS}
    required |= {f"clock_{field}" for field in _CLOCK_FIELDS}
    carrier_fields = {f"controlled_hvac_energy_{carrier.value}_kwh" for carrier in EnergyCarrier}
    required |= carrier_fields
    first_fields = set(rows[0])
    readback_fields = {'actuator_readback_phase', 'actuator_readback_physical_json',
                       'actuator_readback_max_abs_error_c'}
    if first_fields & readback_fields:
        required |= readback_fields
    if required != first_fields:
        raise EvaluationError(
            "trajectory columns do not exactly match the current evaluator schema: "
            f"missing={sorted(required - first_fields)}, unknown={sorted(first_fields - required, key=str)}"
        )
    if any(set(row) != first_fields for row in rows):
        raise EvaluationError("trajectory rows have inconsistent columns")

    episodes: dict[tuple[str, int], list[Mapping[str, str]]] = defaultdict(list)
    seen_step_keys: set[tuple[str, int, int]] = set()
    window_config: dict[str, tuple[int, ...]] = {}
    clocks_by_step: dict[tuple[str, int, int], tuple[SimulationClock, SimulationClock]] = {}
    reported_carriers: set[str] | None = None
    raw_observation_keys: set[str] | None = None
    actions = evidence["actions"]
    expected_proposed_names = actions.get("proposal_channel_order")
    expected_proposed_units = actions.get("proposal_units")
    expected_submitted_names = actions.get("submitted_channel_order")
    expected_submitted_units = actions.get("submitted_units")
    expected_transform = actions.get("transform_id")
    expected_submitted_schema = actions.get("submitted_schema_version")
    protocol_windows = {window.name: window for window in protocol.windows}
    for row in rows:
        if row["schema_version"] != STEP_SCHEMA_VERSION:
            raise EvaluationError("trajectory contains an unsupported step schema")
        expected_identity = {
            "run_id": identity.run_id,
            "experiment_id": identity.experiment_id,
            "algorithm_id": identity.algorithm_id,
            "algorithm_version": identity.algorithm_version,
            "seed": str(identity.seed),
            "task_id": identity.task_id,
        }
        for field, expected in expected_identity.items():
            if row[field] != expected:
                raise EvaluationError(f"trajectory {field} does not match its RunResult")
        window = _string(row["window"], field="window")
        episode = _int(row["episode_index"], field="episode_index", nonnegative=True)
        step = _int(row["step_index"], field="step_index", nonnegative=True)
        key = (window, episode, step)
        if key in seen_step_keys:
            raise EvaluationError(f"duplicate trajectory step {key!r}")
        seen_step_keys.add(key)
        episodes[key[:2]].append(row)
        config = tuple(
            _int(row[field], field=field, positive=True)
            for field in (
                "window_start_month",
                "window_start_day",
                "window_end_month",
                "window_end_day",
                "episodes_per_window",
                "expected_steps",
                "timesteps_per_hour",
            )
        )
        if window in window_config and window_config[window] != config:
            raise EvaluationError(f"mixed calendar configuration within window {window!r}")
        window_config[window] = config
        try:
            declared_window = protocol_windows[window]
        except KeyError as error:
            raise EvaluationError(f"trajectory contains undeclared evaluation window {window!r}") from error
        expected_window_config = (
            declared_window.start_month,
            declared_window.start_day,
            declared_window.end_month,
            declared_window.end_day,
            protocol.episodes_per_window,
        )
        if config[:5] != expected_window_config:
            raise EvaluationError(f"trajectory window {window!r} does not match its EvaluationProtocolSpec")
        try:
            declared_duration_hours = declared_window.duration_hours_for_calendar_year(task.calendar_year)
        except ValueError as error:
            raise EvaluationError(
                f"declared evaluation window {window!r} is invalid for the TaskSpec calendar year"
            ) from error
        if config[5] != declared_duration_hours * task.episode.timesteps_per_hour:
            raise EvaluationError(f"trajectory window {window!r} expected_steps does not match its declared protocol")
        if config[-1] != task.episode.timesteps_per_hour:
            raise EvaluationError("trajectory timesteps_per_hour does not match the TaskSpec")
        if not math.isclose(
            _float(row["interval_hours"], field="interval_hours", positive=True),
            1.0 / task.episode.timesteps_per_hour,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise EvaluationError("trajectory interval_hours does not match the TaskSpec")
        current_reported = {
            carrier.value for carrier in EnergyCarrier if row[f"controlled_hvac_energy_{carrier.value}_kwh"] != ""
        }
        if not current_reported:
            raise EvaluationError("a trajectory row reports no energy carrier")
        if reported_carriers is None:
            reported_carriers = current_reported
        elif current_reported != reported_carriers:
            raise EvaluationError("reported energy carriers change within a trajectory")
        proposed_action = _json_real_list(
            row["proposed_action_normalized_json"],
            field="proposed_action_normalized_json",
        )
        _validate_proposed_action(proposed_action, task=task)
        for field, expected in (
            ("proposed_action_channel_names_json", expected_proposed_names),
            ("proposed_action_units_json", expected_proposed_units),
            ("submitted_action_channel_names_json", expected_submitted_names),
            ("submitted_action_units_json", expected_submitted_units),
        ):
            _validate_json_numeric(row[field], field=field, allow_strings=True)
            if json.loads(row[field]) != expected:
                raise EvaluationError(f"trajectory {field} does not match evaluation evidence")
        if row["action_transform_id"] != expected_transform:
            raise EvaluationError("trajectory action transform does not match evaluation evidence")
        if row["submitted_action_schema_version"] != expected_submitted_schema:
            raise EvaluationError("trajectory submitted-action schema does not match evaluation evidence")
        submitted_action = _json_value(
            row["submitted_action_physical_json"],
            field="submitted_action_physical_json",
            allow_strings=True,
        )
        _validate_submitted_action(
            submitted_action,
            proposed_action=proposed_action,
            task=task,
        )
        if row['actuator_readback_status'] == 'available':
            if not readback_fields <= set(row):
                raise EvaluationError('trajectory actuator read-back is missing evidence columns')
            checked = _submitted_action_evidence(
                {'submitted_action_schema_version': row['submitted_action_schema_version'],
                 'submitted_action_physical': submitted_action,
                 'actuator_readback': {'available': True,
                     'phase': row['actuator_readback_phase'],
                     'records': _json_value(row['actuator_readback_physical_json'],
                                           field='actuator_readback_physical_json', allow_strings=True)}},
                task=task, proposed_action=tuple(proposed_action), where='trajectory')
            error = _float(row['actuator_readback_max_abs_error_c'], field='actuator_readback_max_abs_error_c')
            if error != checked['actuator_readback_max_abs_error_c'] or row['actuator_readback_reason']:
                raise EvaluationError('trajectory actuator read-back error or reason is inconsistent')
        elif row['actuator_readback_status'] == 'unavailable':
            _string(row['actuator_readback_reason'], field='actuator_readback_reason')
            if readback_fields <= set(row) and (
                    row['actuator_readback_phase'] or row['actuator_readback_physical_json'] != '[]'
                    or row['actuator_readback_max_abs_error_c']):
                raise EvaluationError('unavailable actuator read-back must not contain observed values')
        else:
            raise EvaluationError('unknown actuator read-back status')
        decision_clock = _clock_from_json_and_columns(
            row["decision_clock_json"],
            row,
            prefix="decision_clock_",
            task=task,
            step=step,
        )
        try:
            expected_current_clock = advance_zone_timestep(decision_clock)
        except ValueError as error:
            raise EvaluationError(f"cannot advance trajectory decision clock: {error}") from error
        allow_next_year_boundary = (
            step == config[5] - 1
            and _int(row["terminated"], field="terminated") == 1
            and task.calendar_year is not None
            and decision_clock.calendar_year == task.calendar_year
            and (
                decision_clock.month,
                decision_clock.day_of_month,
                decision_clock.hour,
                decision_clock.timestep_index,
            )
            == (12, 31, 23, task.episode.timesteps_per_hour)
        )
        current_clock = _clock_from_columns(
            row,
            prefix="clock_",
            task=task,
            expected_episode_step=step + 1,
            allow_next_year_boundary=allow_next_year_boundary,
        )
        if current_clock != expected_current_clock:
            raise EvaluationError("trajectory next-state clock is not one timestep after its decision clock")
        clocks_by_step[key] = (decision_clock, current_clock)
        if evidence["policy"]["kind"] == "built_in":
            _validate_built_in_policy_action(
                proposed_action,
                algorithm_id=identity.algorithm_id,
                task=task,
                month=decision_clock.month,
            )
        policy_observation = _json_real_list(
            row["policy_observation_normalized_json"],
            field="policy_observation_normalized_json",
        )
        _validate_policy_observation(policy_observation, task=task)
        raw_observation = _json_value(
            row["next_raw_observation_json"],
            field="next_raw_observation_json",
        )
        if not isinstance(raw_observation, Mapping) or not raw_observation:
            raise EvaluationError("next_raw_observation_json must contain a non-empty JSON object")
        current_raw_keys = set(raw_observation)
        if any(not isinstance(name, str) or not name for name in current_raw_keys):
            raise EvaluationError("next_raw_observation_json keys must be non-empty strings")
        for name, value in raw_observation.items():
            _finite_real(value, field=f"next_raw_observation_json[{name!r}]")
        if not any(name.endswith("_temperature") for name in current_raw_keys):
            raise EvaluationError("next_raw_observation_json contains no temperature fields")
        if raw_observation_keys is None:
            raw_observation_keys = current_raw_keys
        elif raw_observation_keys != current_raw_keys:
            raise EvaluationError("raw observation fields change within a trajectory")
        _validate_legacy_comfort_row(
            row,
            raw_observation=raw_observation,
            task=task,
        )
        _validate_reward_row(row, task=task, reported_carriers=current_reported)

    assert reported_carriers is not None
    if set(window_config) != set(protocol_windows):
        raise EvaluationError(
            "trajectory windows do not exactly match the EvaluationProtocolSpec: "
            f"missing={sorted(set(protocol_windows) - set(window_config))}, "
            f"unknown={sorted(set(window_config) - set(protocol_windows))}"
        )
    evidence_carriers = set(evidence["energy"].get("carrier_meter_names", {}))
    if reported_carriers != evidence_carriers:
        raise EvaluationError("trajectory energy carriers do not match evaluation evidence")
    diagnostic_episode_keys = {(episode["window"], episode["episode_index"]) for episode in diagnostics["episodes"]}
    if diagnostic_episode_keys != set(episodes):
        raise EvaluationError("evaluation diagnostics episodes do not match trajectory episodes")
    for window, config in window_config.items():
        episodes_expected = config[4]
        actual_indices = {episode for observed_window, episode in episodes if observed_window == window}
        if actual_indices != set(range(episodes_expected)):
            raise EvaluationError(f"incomplete episode set for window {window!r}")

    episode_accumulators: dict[tuple[str, int], _Accumulator] = {}
    for (window, episode), episode_rows in episodes.items():
        _, _, _, _, _, expected_steps, tph = window_config[window]
        episode_rows = sorted(
            episode_rows,
            key=lambda row: _int(row["step_index"], field="step_index"),
        )
        if len(episode_rows) != expected_steps:
            raise EvaluationError(
                f"incomplete window {window!r} episode {episode}: " f"{len(episode_rows)} of {expected_steps} rows"
            )
        expected_clock_sequence = _expected_clock_sequence(
            protocol_windows[window],
            tph,
            calendar_year=task.calendar_year,
        )
        if expected_steps != len(expected_clock_sequence):
            raise EvaluationError(f"trajectory expected_steps does not match calendar window {window!r}")
        observed_steps = {_int(row["step_index"], field="step_index") for row in episode_rows}
        if observed_steps != set(range(expected_steps)):
            raise EvaluationError(f"missing or extra step indices in {window!r} episode {episode}")
        observed_clock_sequence = []
        previous_current: SimulationClock | None = None
        if task.warmup_preconditioning.enabled:
            first_decision_clock = clocks_by_step[(window, episode, 0)][0]
            expected_day_of_week, expected_day_of_year = expected_warmup_clock_anchor(
                contract=task.warmup_preconditioning,
                window=protocol_windows[window],
                calendar_year=task.calendar_year,
            )
            if first_decision_clock.day_of_week != expected_day_of_week:
                raise EvaluationError(
                    "trajectory first decision day_of_week does not match the identity-bearing EPW calendar anchor"
                )
            if first_decision_clock.day_of_year != expected_day_of_year:
                raise EvaluationError("trajectory first decision day_of_year does not match the calendar window")
        for row in episode_rows:
            step = _int(row["step_index"], field="step_index")
            decision_clock, current_clock = clocks_by_step[(window, episode, step)]
            if previous_current is not None and decision_clock != previous_current:
                raise EvaluationError(f"trajectory clocks are discontinuous in {window!r} episode {episode}")
            previous_current = current_clock
            observed_clock_sequence.append(
                (
                    decision_clock.month,
                    decision_clock.day_of_month,
                    decision_clock.hour,
                    decision_clock.timestep_index,
                )
            )
        if observed_clock_sequence != expected_clock_sequence:
            raise EvaluationError(f"invalid controlled-calendar sequence in {window!r} episode {episode}")
        terminal_values = [_int(row["terminated"], field="terminated") for row in episode_rows]
        if any(value not in {0, 1} for value in terminal_values):
            raise EvaluationError("terminated must be 0 or 1")
        terminal_steps = [
            _int(row["step_index"], field="step_index")
            for row, terminal in zip(episode_rows, terminal_values)
            if terminal == 1
        ]
        if terminal_steps != [expected_steps - 1]:
            raise EvaluationError(f"invalid terminal marker in {window!r} episode {episode}")
        accumulator = _Accumulator()
        for row in episode_rows:
            accumulator.add(row, sorted(reported_carriers))
        evidence_area = evidence["energy"].get("controlled_floor_area_m2")
        if accumulator.has_occupied_metrics and not math.isclose(
            accumulator.controlled_floor_area_m2 or 0.0,
            _finite_real(
                evidence_area,
                field="evaluation evidence controlled_floor_area_m2",
                positive=True,
            ),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise EvaluationError("trajectory controlled floor area does not match evaluation evidence")
        episode_accumulators[(window, episode)] = accumulator

    output: dict[str, dict[str, float]] = {}
    all_windows = _Accumulator()
    for window in sorted(window_config):
        selected = [value for (name, _), value in episode_accumulators.items() if name == window]
        combined = _combine(selected)
        output[window] = combined.metrics(divisor=float(len(selected)))
        _add_average(all_windows, combined, divisor=float(len(selected)))
    output["all_windows"] = all_windows.metrics()
    _validate_result_metrics(
        result,
        episode_accumulators=tuple(episode_accumulators.values()),
        window_count=len(window_config),
        episode_count=len(episode_accumulators),
        control_steps=len(rows),
    )
    return output


def _json_value(text: str, *, field: str, allow_strings: bool = False) -> Any:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{field} is not valid JSON") from error
    _validate_finite_json(value, field=field, allow_strings=allow_strings)
    return value


def _json_real_list(text: str, *, field: str) -> list[float]:
    value = _json_value(text, field=field)
    if not isinstance(value, list):
        raise EvaluationError(f"{field} must contain a JSON list")
    return [_finite_real(item, field=f"{field}[{index}]") for index, item in enumerate(value)]


def _validate_built_in_policy_action(
    values: Sequence[float],
    *,
    algorithm_id: str,
    task: TaskSpec,
    month: int,
) -> None:
    try:
        if algorithm_id == "static_midpoint":
            expected = static_midpoint_action(task)
        elif algorithm_id == "ashrae_seasonal_setpoints":
            expected = ashrae_seasonal_action(task, month=month)
        else:
            raise EvaluationError(
                f"publication reporting cannot authenticate built-in policy semantics for {algorithm_id!r}"
            )
    except ValueError as error:
        raise EvaluationError(f"cannot reconstruct built-in policy {algorithm_id!r}: {error}") from error
    if len(values) != len(expected) or any(
        not math.isclose(observed, reference, rel_tol=0.0, abs_tol=1e-12)
        for observed, reference in zip(values, expected)
    ):
        raise EvaluationError(f"trajectory proposed action does not match built-in policy {algorithm_id!r} semantics")


def _validate_legacy_comfort_row(
    row: Mapping[str, str],
    *,
    raw_observation: Mapping[str, Any],
    task: TaskSpec,
) -> None:
    parameters = task.reward.parameter_mapping
    if task.reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3:
        band = ScheduledOccupiedRewardConfig.from_mapping(parameters).comfort_schedule.default
        parameters = {"comfort_min_c": band.minimum_c, "comfort_max_c": band.maximum_c}
    try:
        comfort_min = _finite_real(parameters["comfort_min_c"], field="TaskSpec comfort_min_c")
        comfort_max = _finite_real(parameters["comfort_max_c"], field="TaskSpec comfort_max_c")
    except KeyError as error:
        raise EvaluationError("TaskSpec reward must declare comfort_min_c and comfort_max_c") from error
    if comfort_min >= comfort_max:
        raise EvaluationError("TaskSpec comfort_min_c must be smaller than comfort_max_c")
    temperature_keys = tuple(f"{entity}_temperature" for entity in task.action_profile.controlled_entities)
    missing = [key for key in temperature_keys if key not in raw_observation]
    if missing:
        raise EvaluationError(
            "next_raw_observation_json is missing controlled-zone temperature field(s): " f"{missing!r}"
        )
    interval = _float(row["interval_hours"], field="interval_hours", positive=True)
    violations = [
        max(
            comfort_min - _finite_real(raw_observation[key], field=f"raw observation {key!r}"),
            _finite_real(raw_observation[key], field=f"raw observation {key!r}") - comfort_max,
            0.0,
        )
        for key in temperature_keys
    ]
    expected_degree_hours = math.fsum(violations) * interval
    expected_zone_hours = len(temperature_keys) * interval
    observed_degree_hours = _float(
        row["legacy_24h_comfort_violation_degree_hours"],
        field="legacy_24h_comfort_violation_degree_hours",
        nonnegative=True,
    )
    observed_zone_hours = _float(
        row["legacy_24h_zone_hours"],
        field="legacy_24h_zone_hours",
        positive=True,
    )
    observed_all_comfort = _int(
        row["legacy_24h_all_zones_comfort"],
        field="legacy_24h_all_zones_comfort",
    )
    expected_all_comfort = int(all(value == 0.0 for value in violations))
    if not math.isclose(observed_degree_hours, expected_degree_hours, rel_tol=0.0, abs_tol=1e-12):
        raise EvaluationError("legacy comfort degree-hours do not match controlled-zone raw temperatures")
    if not math.isclose(observed_zone_hours, expected_zone_hours, rel_tol=0.0, abs_tol=1e-12):
        raise EvaluationError("legacy zone-hours do not match the TaskSpec controlled zones")
    if observed_all_comfort != expected_all_comfort:
        raise EvaluationError("legacy all-zones-comfort flag does not match controlled-zone raw temperatures")


def _validate_proposed_action(values: Sequence[float], *, task: TaskSpec) -> None:
    space = task.action_space
    if len(values) != space.flat_size:
        raise EvaluationError(f"proposed action has {len(values)} values; TaskSpec requires {space.flat_size}")
    if space.kind != "box" or space.lower_bound is None or space.upper_bound is None:
        raise EvaluationError("publication action validation requires a bounded Box TaskSpec")
    for value in values:
        if not space.lower_bound <= value <= space.upper_bound:
            raise EvaluationError(
                f"proposed normalized action {value} lies outside " f"[{space.lower_bound}, {space.upper_bound}]"
            )


def _validate_submitted_action(
    value: Any,
    *,
    proposed_action: Sequence[float],
    task: TaskSpec,
) -> None:
    entities = task.action_profile.controlled_entities
    channels = task.action_profile.channels
    if tuple(channel.name for channel in channels) != (
        "heating_setpoint_c",
        "cooling_offset_c",
    ):
        raise EvaluationError(
            "publication submitted-action validation supports only the HOT " "heating-setpoint/cooling-offset contract"
        )
    if not isinstance(value, list) or len(value) != len(entities):
        raise EvaluationError("submitted physical action entity count does not match the TaskSpec")
    for entity_index, (entity, raw_record) in enumerate(zip(entities, value)):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "zone",
            "heating_setpoint_c",
            "cooling_setpoint_c",
        }:
            raise EvaluationError("submitted physical action record has an invalid schema")
        if raw_record.get("zone") != entity:
            raise EvaluationError("submitted physical action entity order does not match the TaskSpec")
        heating = _finite_real(
            raw_record.get("heating_setpoint_c"),
            field=f"submitted action {entity!r} heating_setpoint_c",
        )
        cooling = _finite_real(
            raw_record.get("cooling_setpoint_c"),
            field=f"submitted action {entity!r} cooling_setpoint_c",
        )
        heating_channel, offset_channel = channels
        action_index = entity_index * 2
        expected_heating = heating_channel.physical_min + proposed_action[action_index] * (
            heating_channel.physical_max - heating_channel.physical_min
        )
        expected_offset = offset_channel.physical_min + proposed_action[action_index + 1] * (
            offset_channel.physical_max - offset_channel.physical_min
        )
        transform_id = task.action_profile.transform_id
        if transform_id == "linear_quantized_0_1c_legacy":
            expected_heating = round(expected_heating, 1)
            expected_offset = round(expected_offset, 1)
        elif transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
            from building_gen.bounded_thermostat import validate_ranges
            validate_ranges((heating_channel.physical_min,heating_channel.physical_max),(offset_channel.physical_min,offset_channel.physical_max),task.action_profile.transform_id)
            expected_offset=2.+proposed_action[action_index+1]*((32. if task.action_profile.transform_id == "bounded_thermostat_12_32_v1" else 30.)-expected_heating-2.)
        elif transform_id != "linear_unquantized_v1":
            raise EvaluationError(f"unsupported publication action transform {transform_id!r}")
        if not math.isclose(heating, expected_heating, rel_tol=1e-7, abs_tol=1e-6):
            raise EvaluationError("submitted heating setpoint does not match the TaskSpec transform")
        if not math.isclose(
            cooling,
            expected_heating + expected_offset,
            rel_tol=1e-7,
            abs_tol=1e-6,
        ):
            raise EvaluationError("submitted cooling setpoint does not match the TaskSpec transform")


def _validate_policy_observation(values: Sequence[float], *, task: TaskSpec) -> None:
    space = task.observation_space
    if len(values) != space.flat_size:
        raise EvaluationError(f"policy observation has {len(values)} values; TaskSpec requires {space.flat_size}")
    if space.kind != "box" or space.lower_bound is None or space.upper_bound is None:
        raise EvaluationError("publication observation validation requires a bounded Box TaskSpec")
    for value in values:
        if not space.lower_bound <= value <= space.upper_bound:
            raise EvaluationError(
                f"normalized policy observation {value} lies outside " f"[{space.lower_bound}, {space.upper_bound}]"
            )


def _clock_from_json_and_columns(
    text: str,
    row: Mapping[str, str],
    *,
    prefix: str,
    task: TaskSpec,
    step: int,
) -> SimulationClock:
    value = _json_value(text, field="decision_clock_json", allow_strings=True)
    if not isinstance(value, Mapping) or set(value) != set(_CLOCK_FIELDS):
        raise EvaluationError("decision_clock_json must contain the exact simulation-clock schema")
    clock = _simulation_clock(value, task=task, expected_episode_step=step)
    _validate_flattened_clock(row, prefix=prefix, clock=clock)
    return clock


def _clock_from_columns(
    row: Mapping[str, str],
    *,
    prefix: str,
    task: TaskSpec,
    expected_episode_step: int,
    allow_next_year_boundary: bool = False,
) -> SimulationClock:
    values: dict[str, int | str] = {}
    for field in _CLOCK_FIELDS:
        raw = row[f"{prefix}{field}"]
        values[field] = (
            _string(raw, field=f"{prefix}{field}")
            if field == "calendar_year_semantics"
            else _int(raw, field=f"{prefix}{field}")
        )
    return _simulation_clock(
        values,
        task=task,
        expected_episode_step=expected_episode_step,
        allow_next_year_boundary=allow_next_year_boundary,
    )


def _simulation_clock(
    value: Mapping[str, Any],
    *,
    task: TaskSpec,
    expected_episode_step: int,
    allow_next_year_boundary: bool = False,
) -> SimulationClock:
    integers: dict[str, int] = {}
    for field in set(_CLOCK_FIELDS) - {"calendar_year_semantics"}:
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise EvaluationError(f"simulation clock {field} must be an integer")
        integers[field] = raw
    semantics = value.get("calendar_year_semantics")
    if not isinstance(semantics, str):
        raise EvaluationError("simulation clock calendar_year_semantics must be a string")
    try:
        clock = SimulationClock(calendar_year_semantics=semantics, **integers)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"invalid simulation clock: {error}") from error
    if clock.calendar_year <= 0:
        raise EvaluationError("publication trajectory clocks require a positive calendar year")
    try:
        date(clock.calendar_year, clock.month, clock.day_of_month)
    except ValueError as error:
        raise EvaluationError("simulation clock contains an invalid calendar date") from error
    if clock.timesteps_per_hour != task.episode.timesteps_per_hour:
        raise EvaluationError("simulation clock timesteps_per_hour does not match the TaskSpec")
    if clock.episode_step_index != expected_episode_step:
        raise EvaluationError("simulation clock episode_step_index is inconsistent with the trajectory row")
    if task.calendar_year is not None and clock.calendar_year != task.calendar_year:
        is_allowed_next_year_boundary = (
            allow_next_year_boundary
            and clock.calendar_year == task.calendar_year + 1
            and (
                clock.month,
                clock.day_of_month,
                clock.hour,
                clock.timestep_index,
                clock.day_of_year,
            )
            == (1, 1, 0, 1, 1)
        )
        if not is_allowed_next_year_boundary:
            raise EvaluationError("simulation clock calendar_year does not match the TaskSpec")
    if clock.calendar_year_semantics != task.calendar_year_semantics:
        raise EvaluationError("simulation clock calendar_year_semantics does not match the TaskSpec-derived semantics")
    if 60 % clock.timesteps_per_hour != 0:
        raise EvaluationError("simulation clock timesteps_per_hour must divide 60")
    expected_minutes = clock.timestep_index * (60 // clock.timesteps_per_hour)
    if clock.minutes != expected_minutes:
        raise EvaluationError("simulation clock minutes do not match its timestep_index")
    return clock


def _validate_flattened_clock(
    row: Mapping[str, str],
    *,
    prefix: str,
    clock: SimulationClock,
) -> None:
    for field in _CLOCK_FIELDS:
        expected = getattr(clock, field)
        raw = row[f"{prefix}{field}"]
        observed: int | str = (
            _string(raw, field=f"{prefix}{field}")
            if field == "calendar_year_semantics"
            else _int(raw, field=f"{prefix}{field}")
        )
        if observed != expected:
            raise EvaluationError(f"{prefix}{field} does not match decision_clock_json")


def _validate_scheduled_segments(components, *, row, task):
    """Recompute exposure from saved interval samples and the pinned schedule.

    Segment samples are recorded simulator evidence; this check authenticates
    their schedule coverage and arithmetic, not the underlying simulator physics.
    """
    expected_fields = {field.name for field in fields(OccupiedEnergyComfortV2Components)} | {"schema_version", "comfort_segments"}
    if not isinstance(components, Mapping) or set(components) != expected_fields:
        raise EvaluationError("scheduled reward component schema mismatch")
    if components["schema_version"] != "hot-scheduled-reward-components/1.0.0":
        raise EvaluationError("unsupported scheduled reward component version")
    segments = components["comfort_segments"]
    if not isinstance(segments, list) or not segments:
        raise EvaluationError("scheduled reward requires comfort segments")
    config = ScheduledOccupiedRewardConfig.from_mapping(task.reward.parameter_mapping)
    clock = SimulationClock(**_json_value(row["decision_clock_json"], field="decision_clock_json", allow_strings=True))
    start = 60 * (clock.hour + (clock.timestep_index - 1) / clock.timesteps_per_hour)
    duration = 60 / clock.timesteps_per_hour
    exposure = 0.0
    occupants = 0.0
    cursor = 0
    segment_fields = {"zone_id", "weekday", "start_minute", "end_minute", "minimum_c", "maximum_c",
                      "temperature_c", "occupant_count", "exposure_person_celsius_squared_hours"}
    for zone in task.action_profile.controlled_entities:
        expected = config.comfort_schedule.interval_segments(zone, weekday=clock.day_of_week,
                                                             start_minute=start, duration_minutes=duration)
        sample = None
        for day, left, right, band in expected:
            if cursor >= len(segments):
                raise EvaluationError("scheduled segments do not cover the controlled interval")
            segment = segments[cursor]
            cursor += 1
            if not isinstance(segment, Mapping) or set(segment) != segment_fields:
                raise EvaluationError("scheduled comfort segment schema mismatch")
            if segment["zone_id"] != zone:
                raise EvaluationError("scheduled segment zone order differs from the TaskSpec")
            for key, reference in {"weekday": day, "start_minute": left, "end_minute": right,
                                   "minimum_c": band.minimum_c, "maximum_c": band.maximum_c}.items():
                if _finite_real(segment[key], field=key) != reference:
                    raise EvaluationError(f"scheduled segment {key} differs from the pinned schedule or clock")
            temperature = _finite_real(segment["temperature_c"], field="segment temperature")
            count = _finite_real(segment["occupant_count"], field="segment occupants", nonnegative=True)
            if sample is None:
                sample = (temperature, count)
                occupants += count
            elif sample != (temperature, count):
                raise EvaluationError("scheduled segments change physical samples within a timestep")
            value = count * band.violation_c(temperature) ** 2 * (right - left) / 60
            recorded = _finite_real(segment["exposure_person_celsius_squared_hours"], field="segment exposure", nonnegative=True)
            if not math.isclose(recorded, value, rel_tol=1e-12, abs_tol=1e-12):
                raise EvaluationError("scheduled segment exposure does not match physical samples")
            exposure += value
    if cursor != len(segments):
        raise EvaluationError("scheduled reward contains extra comfort segments")
    area = _finite_real(components["controlled_floor_area_m2"], field="controlled area", positive=True)
    for key, expected in {"total_occupant_count": occupants,
                          "comfort_exposure_person_celsius_squared_hours_per_m2": exposure / area}.items():
        if not math.isclose(_finite_real(components[key], field=key), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise EvaluationError(f"scheduled reward {key} differs from segment evidence")


def _validate_reward_row(
    row: Mapping[str, str],
    *,
    task: TaskSpec,
    reported_carriers: set[str],
) -> None:
    text = row["reward_components_json"]
    scalar_reward = _float(row["reward"], field="reward")
    scheduled = task.reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3
    if task.reward.reward_id != HOT_OCCUPIED_ENERGY_COMFORT_V2 and not scheduled:
        if not text:
            return
        components = _json_value(text, field="reward_components_json")
        if not isinstance(components, Mapping):
            raise EvaluationError("reward_components_json must contain a JSON object")
        if "reward" in components and not math.isclose(
            _finite_real(components["reward"], field="reward_components.reward"),
            scalar_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvaluationError("reward_components.reward does not match scalar reward")
        return

    if not text:
        raise EvaluationError("reward-v2 trajectory row is missing reward_components_json")
    components = _json_value(text, field="reward_components_json", allow_strings=scheduled)
    expected_fields = {field.name for field in fields(OccupiedEnergyComfortV2Components)}
    if scheduled:
        _validate_scheduled_segments(components, row=row, task=task)
        components = {key: value for key, value in components.items()
                      if key not in {"schema_version", "comfort_segments"}}
    if not isinstance(components, Mapping) or set(components) != expected_fields:
        raise EvaluationError("reward_components_json does not match the reward-v2 schema")
    values = {name: _finite_real(components[name], field=f"reward_components.{name}") for name in expected_fields}
    try:
        config = (ScheduledOccupiedRewardConfig.from_mapping(task.reward.parameter_mapping).fixed_config()
                  if scheduled else OccupiedEnergyComfortV2Config.from_mapping(task.reward.parameter_mapping))
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"invalid reward-v2 TaskSpec parameters: {error}") from error
    interval = _float(row["interval_hours"], field="interval_hours", positive=True)
    energy = math.fsum(
        _float(
            row[f"controlled_hvac_energy_{carrier}_kwh"],
            field=f"controlled_hvac_energy_{carrier}_kwh",
            nonnegative=True,
        )
        for carrier in sorted(reported_carriers)
    )
    area = values["controlled_floor_area_m2"]
    exposure = values["comfort_exposure_person_celsius_squared_hours_per_m2"]
    occupants = values["total_occupant_count"]
    if area <= 0.0 or occupants < 0.0 or exposure < 0.0:
        raise EvaluationError("reward-v2 physical components must be nonnegative with positive floor area")
    if occupants == 0.0 and exposure != 0.0:
        raise EvaluationError("reward-v2 comfort exposure must be zero when occupancy is zero")
    expected_values = {
        "interval_controlled_hvac_site_energy_kwh": energy,
        "timestep_hours": interval,
        "controlled_hvac_energy_intensity_kwh_per_m2": energy / area,
        "normalized_energy_penalty": (
            values["controlled_hvac_energy_intensity_kwh_per_m2"] / config.energy_scale_kwh_per_m2
        ),
        "normalized_comfort_penalty": (
            values["comfort_exposure_person_celsius_squared_hours_per_m2"]
            / config.comfort_scale_person_celsius_squared_hours_per_m2
        ),
        "weighted_energy_penalty": config.energy_weight * values["normalized_energy_penalty"],
        "weighted_comfort_penalty": config.comfort_weight * values["normalized_comfort_penalty"],
    }
    for name, expected in expected_values.items():
        if not math.isclose(values[name], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise EvaluationError(f"reward-v2 component {name} is inconsistent with the trajectory")
    expected_reward = -(values["weighted_energy_penalty"] + values["weighted_comfort_penalty"])
    if not math.isclose(values["reward"], expected_reward, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(
        scalar_reward,
        expected_reward,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise EvaluationError("reward-v2 components do not reconstruct scalar reward")
    row_values = {
        "controlled_floor_area_m2": area,
        "occupied_comfort_exposure_person_celsius_squared_hours_per_m2": values[
            "comfort_exposure_person_celsius_squared_hours_per_m2"
        ],
        "occupied_person_hours": values["total_occupant_count"] * interval,
    }
    for field_name, expected in row_values.items():
        if not math.isclose(
            _float(row[field_name], field=field_name, nonnegative=True),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvaluationError(f"trajectory {field_name} does not match reward-v2 components")


def _validate_result_metrics(
    result: Mapping[str, Any],
    *,
    episode_accumulators: Sequence[_Accumulator],
    window_count: int,
    episode_count: int,
    control_steps: int,
) -> None:
    total = _combine(episode_accumulators)
    expected_values = total.metrics()
    expected_values.update(
        {
            "completed_windows": float(window_count),
            "completed_episodes": float(episode_count),
            "control_steps": float(control_steps),
        }
    )
    raw_metrics = result.get("metrics")
    if not isinstance(raw_metrics, list):
        raise EvaluationError("RunResult metrics must be a JSON list")
    metrics_by_name: dict[str, Mapping[str, Any]] = {}
    for index, metric in enumerate(raw_metrics):
        if not isinstance(metric, Mapping) or set(metric) != {
            "name",
            "value",
            "unit",
            "aggregation",
            "direction",
        }:
            raise EvaluationError(f"RunResult metric {index} does not match the supported schema")
        name = _string(metric.get("name"), field=f"metrics[{index}].name")
        if name in metrics_by_name:
            raise EvaluationError(f"RunResult contains duplicate metric {name!r}")
        metrics_by_name[name] = metric
    if set(metrics_by_name) != set(expected_values):
        raise EvaluationError(
            "RunResult metric names do not match the authenticated trajectory: "
            f"missing={sorted(set(expected_values) - set(metrics_by_name))}, "
            f"unknown={sorted(set(metrics_by_name) - set(expected_values))}"
        )
    sum_metrics = {
        "total_reward",
        "evaluated_hours",
        "legacy_24h_evaluated_zone_hours",
        "legacy_24h_comfort_violation_degree_hours",
        "controlled_hvac_energy_kwh",
        "controlled_hvac_energy_intensity_kwh_per_m2",
        "occupied_comfort_exposure_person_celsius_squared_hours",
        "occupied_comfort_exposure_person_celsius_squared_hours_per_m2",
        "occupied_person_hours",
    }
    count_metrics = {"completed_windows", "completed_episodes", "control_steps"}
    for name, expected_value in expected_values.items():
        metric = metrics_by_name[name]
        value = _finite_real(metric.get("value"), field=f"metric {name!r} value")
        if not math.isclose(value, expected_value, rel_tol=1e-10, abs_tol=1e-10):
            raise EvaluationError(f"RunResult metric {name!r} does not match the trajectory")
        if name in count_metrics:
            expected_semantics = ("count", "count", "neutral")
        else:
            unit, direction = _metric_metadata(name)
            expected_semantics = (
                unit,
                (
                    "sum"
                    if name in sum_metrics or name.startswith("controlled_hvac_energy_") and name.endswith("_kwh")
                    else "mean"
                ),
                direction,
            )
        observed_semantics = (
            metric.get("unit"),
            metric.get("aggregation"),
            metric.get("direction"),
        )
        if observed_semantics != expected_semantics:
            raise EvaluationError(f"RunResult metric {name!r} has invalid semantics")


def _combine(items: Sequence[_Accumulator]) -> _Accumulator:
    if not items:
        raise EvaluationError("cannot combine zero episodes")
    result = _Accumulator()
    for item in items:
        result.reward += item.reward
        result.interval_hours += item.interval_hours
        result.legacy_zone_hours += item.legacy_zone_hours
        result.legacy_degree_hours += item.legacy_degree_hours
        result.legacy_all_comfort_hours += item.legacy_all_comfort_hours
        result.occupied_exposure_intensity += item.occupied_exposure_intensity
        result.occupied_exposure += item.occupied_exposure
        result.occupied_person_hours += item.occupied_person_hours
        if result.has_occupied_metrics is None:
            result.has_occupied_metrics = item.has_occupied_metrics
            result.controlled_floor_area_m2 = item.controlled_floor_area_m2
        elif result.has_occupied_metrics != item.has_occupied_metrics:
            raise EvaluationError("cannot combine rollouts with different occupied-comfort schemas")
        elif item.has_occupied_metrics and not math.isclose(
            result.controlled_floor_area_m2 or 0.0,
            item.controlled_floor_area_m2 or 0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise EvaluationError("cannot combine rollouts with different controlled floor areas")
        assert result.energy is not None and item.energy is not None
        for carrier, value in item.energy.items():
            result.energy[carrier] = result.energy.get(carrier, 0.0) + value
    return result


def _add_average(target: _Accumulator, source: _Accumulator, *, divisor: float) -> None:
    target.reward += source.reward / divisor
    target.interval_hours += source.interval_hours / divisor
    target.legacy_zone_hours += source.legacy_zone_hours / divisor
    target.legacy_degree_hours += source.legacy_degree_hours / divisor
    target.legacy_all_comfort_hours += source.legacy_all_comfort_hours / divisor
    target.occupied_exposure_intensity += source.occupied_exposure_intensity / divisor
    target.occupied_exposure += source.occupied_exposure / divisor
    target.occupied_person_hours += source.occupied_person_hours / divisor
    if target.has_occupied_metrics is None:
        target.has_occupied_metrics = source.has_occupied_metrics
        target.controlled_floor_area_m2 = source.controlled_floor_area_m2
    elif target.has_occupied_metrics != source.has_occupied_metrics:
        raise EvaluationError("cannot average windows with different occupied-comfort schemas")
    elif source.has_occupied_metrics and not math.isclose(
        target.controlled_floor_area_m2 or 0.0,
        source.controlled_floor_area_m2 or 0.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise EvaluationError("cannot average windows with different controlled floor areas")
    assert target.energy is not None and source.energy is not None
    for carrier, value in source.energy.items():
        target.energy[carrier] = target.energy.get(carrier, 0.0) + value / divisor


def _reject_mixed_configs_or_duplicate_seeds(
    identities: Sequence[_RunIdentity],
    evidence_by_run_id: Mapping[str, Mapping[str, Any]],
    provenance_by_run_id: Mapping[str, SoftwareProvenance],
) -> None:
    selection_rules: dict[tuple[str, str, str], set[str | None]] = defaultdict(set)
    checkpoint_identities: dict[
        tuple[str, str, str, int],
        set[tuple[str | None, str | None, int | None, str | None]],
    ] = defaultdict(set)
    seeds: set[tuple[str, str, int]] = set()
    software_provenance: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for item in identities:
        policy = evidence_by_run_id[item.run_id]["policy"]
        selection_rules[
            (
                item.experiment_id,
                item.algorithm_id,
                item.algorithm_version,
            )
        ].add(policy.get("selection_rule"))
        checkpoint_identities[
            (
                item.experiment_id,
                item.algorithm_id,
                item.algorithm_version,
                item.seed,
            )
        ].add(
            (
                policy.get("parent_training_run_id"),
                policy.get("checkpoint_sha256"),
                policy.get("training_step"),
                (
                    evidence_by_run_id[item.run_id].get("training_bundle", {}).get("sha256")
                    if isinstance(evidence_by_run_id[item.run_id].get("training_bundle"), Mapping)
                    else None
                ),
            )
        )
        seed_key = (item.experiment_id, item.task_id, item.seed)
        if seed_key in seeds:
            raise EvaluationError(f"duplicate seed evidence for {seed_key!r}")
        seeds.add(seed_key)
        software_provenance[
            (
                item.experiment_id,
                item.algorithm_id,
                item.algorithm_version,
                item.task_id,
            )
        ].add(
            json.dumps(
                provenance_by_run_id[item.run_id].to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    mixed_selection = {key: values for key, values in selection_rules.items() if len(values) != 1}
    if mixed_selection:
        raise EvaluationError("different checkpoint-selection rules cannot be pooled: " f"{mixed_selection!r}")
    mixed_checkpoints = {key: values for key, values in checkpoint_identities.items() if len(values) != 1}
    if mixed_checkpoints:
        raise EvaluationError(
            "evaluation target tasks use different checkpoint/training-bundle/step/parent identities: "
            f"{mixed_checkpoints!r}"
        )
    mixed_provenance = {key: values for key, values in software_provenance.items() if len(values) != 1}
    if mixed_provenance:
        raise EvaluationError(
            "evaluation seeds cannot be pooled across different exact software provenance: "
            f"{sorted(mixed_provenance)!r}"
        )


def _validate_task_labels(task_labels: Mapping[str, str] | None) -> dict[str, str]:
    if task_labels is None:
        return {}
    labels: dict[str, str] = {}
    for task_id, label in task_labels.items():
        if not isinstance(task_id, str) or not task_id.strip():
            raise EvaluationError("task-label IDs must be non-empty strings")
        if not isinstance(label, str) or not label.strip():
            raise EvaluationError("task labels must be non-empty strings")
        labels[task_id] = label.strip()
    return labels


def _summarize_seeds(
    values: Sequence[tuple[_RunIdentity, str, dict[str, float]]],
    *,
    task_labels: Mapping[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for identity, window, metrics in values:
        for metric, value in metrics.items():
            grouped[
                (
                    identity.experiment_id,
                    identity.algorithm_id,
                    identity.algorithm_version,
                    identity.task_id,
                    window,
                    metric,
                )
            ].append((identity.seed, value))
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        seed_values = grouped[key]
        seeds = [seed for seed, _ in seed_values]
        if len(set(seeds)) != len(seeds):
            raise EvaluationError(f"duplicate seeds while summarizing {key!r}")
        numbers = [value for _, value in seed_values]
        if not all(math.isfinite(value) for value in numbers):
            raise EvaluationError(f"non-finite metric while summarizing {key!r}")
        n = len(numbers)
        sample_std = statistics.stdev(numbers) if n >= 2 else None
        standard_error = sample_std / math.sqrt(n) if sample_std is not None else None
        ci95 = 1.96 * standard_error if standard_error is not None else None
        metric = key[-1]
        unit, direction = _metric_metadata(metric)
        output.append(
            {
                "experiment_id": key[0],
                "algorithm_id": key[1],
                "algorithm_version": key[2],
                "evaluation_task_id": key[3],
                "evaluation_task": task_labels.get(key[3], key[3]),
                "window": key[4],
                "metric": metric,
                "unit": unit,
                "direction": direction,
                "n_seeds": n,
                "mean": statistics.fmean(numbers),
                "sample_std": sample_std,
                "standard_error": standard_error,
                "ci95_half_width": ci95,
                "min": min(numbers),
                "max": max(numbers),
            }
        )
    return output


def _metric_metadata(metric: str) -> tuple[str, str]:
    if metric.startswith("controlled_hvac_energy_") and metric.endswith("_kwh"):
        return "kWh", "minimize"
    try:
        return _METRIC_METADATA[metric]
    except KeyError as error:
        raise EvaluationError(f"unknown report metric {metric!r}") from error


def _summary_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise EvaluationError("cannot write an empty summary")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if value is None else value for key, value in row.items()})
    return stream.getvalue()


def _markdown_report(title: str, rows: Sequence[Mapping[str, Any]], run_count: int) -> str:
    if not title.strip():
        raise EvaluationError("report title cannot be empty")
    selected = [
        row
        for row in rows
        if row["window"] == "all_windows"
        and row["metric"]
        in {
            "total_reward",
            "controlled_hvac_energy_kwh",
            "controlled_hvac_energy_intensity_kwh_per_m2",
            "occupied_comfort_exposure_person_celsius_squared_hours_per_m2",
            "legacy_24h_mean_comfort_violation_c",
        }
    ]
    lines = [
        f"# {title.strip()}",
        "",
        f"Validated evaluation runs: **{run_count}**.",
        "",
        "All calendar windows were complete, and all reported values were finite. "
        "Uncertainty is a two-sided normal-approximation 95% confidence half-width "
        "computed across independent training seeds; it is omitted when only one seed is available.",
        "Energy values are controlled-HVAC EnergyPlus meter totals, not whole-building site energy. "
        "Occupied comfort values come from reward-v2 physical components weighted by "
        "EnergyPlus-reported/modelled occupancy, not measured occupancy; metrics prefixed "
        "`legacy_24h_` are unweighted diagnostics only.",
        "",
        "| Algorithm | Evaluation task | Metric | Seeds | Mean | 95% CI half-width |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        ci = row["ci95_half_width"]
        lines.append(
            "| {algorithm} | `{task}` | {metric} ({unit}) | {n} | {mean} | {ci} |".format(
                algorithm=row["algorithm_id"],
                task=row["evaluation_task"],
                metric=row["metric"],
                unit=row["unit"],
                n=row["n_seeds"],
                mean=_format_number(row["mean"]),
                ci="--" if ci is None else _format_number(ci),
            )
        )
    lines.extend(
        [
            "",
            "The machine-readable `summary.csv` also contains every monthly metric, "
            "sample standard deviation, standard error, range, and seed count.",
            "",
        ]
    )
    return "\n".join(lines)


def _expected_clock_sequence(
    window: CalendarWindowSpec,
    timesteps_per_hour: int,
    *,
    calendar_year: int | None,
) -> list[tuple[int, int, int, int]]:
    try:
        dates = window.dates_for_calendar_year(calendar_year)
    except ValueError as error:
        raise EvaluationError("invalid calendar configuration in trajectory") from error
    keys: list[tuple[int, int, int, int]] = []
    for current in dates:
        for hour in range(24):
            for timestep in range(1, timesteps_per_hour + 1):
                keys.append((current.month, current.day, hour, timestep))
    return keys


def _validate_json_numeric(text: str, *, field: str, allow_strings: bool = False) -> None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise EvaluationError(f"{field} is not valid JSON") from error

    _validate_finite_json(value, field=field, allow_strings=allow_strings)


def _portable_sibling_uri(value: Any, *, field: str) -> str:
    uri = _string(value, field=field)
    pure = PurePosixPath(uri)
    if pure.is_absolute() or pure.name != uri or uri in {".", ".."} or "\\" in uri or uri.strip() != uri:
        raise EvaluationError(f"{field} must be a portable sibling filename")
    return uri


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationError(f"{field} must contain exactly 64 lowercase hexadecimal characters")
    return value


def _finite_real(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise EvaluationError(f"{field} must be positive")
    if nonnegative and result < 0.0:
        raise EvaluationError(f"{field} must be nonnegative")
    return result


def _validate_finite_json(value: Any, *, field: str, allow_strings: bool, allow_none: bool = False) -> None:
    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, bool):
            return
        elif item is None and allow_none:
            return
        elif isinstance(item, str) and allow_strings:
            return
        elif isinstance(item, (int, float)):
            if not math.isfinite(float(item)):
                raise EvaluationError(f"{field} contains a non-finite number")
        else:
            raise EvaluationError(f"{field} contains a non-numeric JSON value")

    visit(value)


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{field} must be a non-empty string")
    return value


def _float(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise EvaluationError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise EvaluationError(f"{field} must be positive")
    if nonnegative and result < 0.0:
        raise EvaluationError(f"{field} must be nonnegative")
    return result


def _int(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{field} must be an integer") from error
    if str(result) != str(value):
        raise EvaluationError(f"{field} must be an integer")
    if positive and result <= 0:
        raise EvaluationError(f"{field} must be positive")
    if nonnegative and result < 0:
        raise EvaluationError(f"{field} must be nonnegative")
    return result


def _format_number(value: float) -> str:
    return f"{value:.6g}"


__all__ = [
    "ReportOutput",
    "ValidatedEvaluationBundle",
    "discover_results",
    "generate_report",
    "validate_evaluation_bundle",
]
