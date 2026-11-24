"""Strict, environment-agnostic rollout evaluation for HOT control policies.

The evaluator deliberately receives an environment factory instead of
constructing EnergyPlus itself.  The factory is the calendar adapter: it must
apply ``context.window`` as the model RunPeriod, configure
``context.expected_steps`` control transitions, and return a fresh Gym-like
environment.  This keeps benchmark accounting testable without EnergyPlus and
prevents calendar mutation from being hidden inside the evaluator.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

from building_gen.accounting import CONTROLLED_HVAC_METER_BY_CARRIER, EnergyCarrier
from building_gen.benchmark.artifacts import ArtifactRef, MetricValue, RunResult
from building_gen.benchmark.energyplus_diagnostics import (
    DIAGNOSTICS_FILENAME,
    DiagnosticsAllowlist,
    DiagnosticsGate,
    ScientificDiagnosticsError,
    canonical_energyplus_diagnostics_snapshot,
    energyplus_diagnostics_from_mapping,
    evaluate_scientific_diagnostics,
)
from building_gen.benchmark.policies import (
    ashrae_seasonal_action,
    static_midpoint_action,
)
from building_gen.benchmark.provenance import ResolvedRun, SoftwareProvenance
from building_gen.benchmark.specs import (
    LEGACY_TASK_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    CalendarWindowSpec,
    EvaluationProtocolSpec,
    TaskSpec,
)
from building_gen.benchmark.training_bundle import (
    TRAINING_BUNDLE_SCHEMA_VERSION,
    ValidatedTrainingBundle,
    final_checkpoint_selection_rule,
)
from building_gen.benchmark.warmup_preconditioning import (
    EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION,
    WARMUP_PRECONDITIONING_FILENAME,
    WarmupPreconditioningEvidenceError,
    expected_warmup_clock_anchor,
    validate_warmup_preconditioning_evidence,
)
from building_gen.reward_foundation import (
    HOT_OCCUPIED_ENERGY_COMFORT_V2,
    OccupiedEnergyComfortV2Components,
)

from building_gen.scheduled_reward import (
    HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3,
    ScheduledOccupiedRewardConfig, verify_scheduled_interval,
)

STEP_SCHEMA_VERSION = "hot-control-steps/2.0.0"
EVALUATION_DIAGNOSTICS_SCHEMA_VERSION = "hot-control-evaluation-diagnostics/2.0.0"
LEGACY_EVALUATION_EVIDENCE_SCHEMA_VERSION = "hot-control-evaluation-evidence/1.1.0"
EVALUATION_EVIDENCE_SCHEMA_VERSION = "hot-control-evaluation-evidence/1.2.0"
SUBMITTED_ACTION_SCHEMA_VERSION = "hot-zone-setpoint-submission/1.0.0"


class EvaluationError(ValueError):
    """Raised when rollout evidence is incomplete, inconsistent, or invalid."""


class GymEnvironment(Protocol):
    """Small subset of the Gymnasium API needed by :func:`evaluate_run`."""

    action_space: Any

    def reset(self, *, seed: int | None = None) -> tuple[Any, Mapping[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RolloutContext:
    """Inputs an environment factory must honor for one calendar episode."""

    run: ResolvedRun
    task: TaskSpec
    window: CalendarWindowSpec
    episode_index: int
    expected_steps: int


EnvironmentFactory = Callable[[RolloutContext], GymEnvironment]


@dataclass(frozen=True)
class DecisionContext:
    """State metadata supplied to a policy at one decision point."""

    rollout: RolloutContext
    step_index: int
    clock: Mapping[str, Any]
    deterministic: bool = True


Policy = Callable[[Any, DecisionContext], Any]


@dataclass(frozen=True)
class PolicyProvenance:
    """Identity of the policy callable used for an evaluation.

    Learned policies must identify immutable checkpoint bytes, the training
    step stored in that checkpoint, and the rule used to select it.  Built-in
    policies deliberately carry no checkpoint fields.
    """

    kind: str
    policy_id: str
    parent_training_run_id: str | None = None
    checkpoint_sha256: str | None = None
    training_step: int | None = None
    selection_rule: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"built_in", "learned_checkpoint"}:
            raise ValueError("policy provenance kind must be 'built_in' or 'learned_checkpoint'")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if self.kind == "built_in":
            if any(
                value is not None
                for value in (
                    self.parent_training_run_id,
                    self.checkpoint_sha256,
                    self.training_step,
                    self.selection_rule,
                )
            ):
                raise ValueError("built-in policy provenance cannot declare checkpoint fields")
            return
        if not isinstance(self.parent_training_run_id, str) or not self.parent_training_run_id:
            raise ValueError("learned policy provenance requires parent_training_run_id")
        digest = self.checkpoint_sha256
        if not isinstance(digest, str) or not _is_sha256(digest):
            raise ValueError("learned policy checkpoint_sha256 must contain 64 hexadecimal characters")
        object.__setattr__(self, "checkpoint_sha256", digest.lower())
        if isinstance(self.training_step, bool) or not isinstance(self.training_step, int) or self.training_step < 0:
            raise ValueError("learned policy training_step must be a non-negative integer")
        if not isinstance(self.selection_rule, str) or not self.selection_rule.strip():
            raise ValueError("learned policy selection_rule must be a non-empty string")
        object.__setattr__(self, "selection_rule", self.selection_rule.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "policy_id": self.policy_id,
            "parent_training_run_id": self.parent_training_run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_step": self.training_step,
            "selection_rule": self.selection_rule,
        }


@dataclass(frozen=True)
class EvaluationPolicy:
    """A callable policy bundled with publication-grade provenance."""

    actor: Policy
    provenance: PolicyProvenance
    checkpoint_path: Path | None = None
    transition_observer: Callable[..., None] | None = None

    def __post_init__(self) -> None:
        if self.provenance.kind == "built_in":
            if self.checkpoint_path is not None:
                raise ValueError("built-in policies cannot carry a checkpoint path")
            return
        if self.checkpoint_path is None:
            raise ValueError("learned checkpoint policies require an exact checkpoint path")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))

    def __call__(self, observation: Any, context: DecisionContext) -> Any:
        return self.actor(observation, context)


@dataclass(frozen=True)
class EvaluationOutput:
    """Artifacts produced for one resolved evaluation run."""

    result: RunResult
    result_path: Path
    steps_path: Path
    diagnostics_path: Path
    evidence_path: Path
    warmup_preconditioning_path: Path
    diagnostics_policy_path: Path | None = None
    policy_checkpoint_path: Path | None = None
    training_bundle_path: Path | None = None


def native_incumbent_policy(task: TaskSpec) -> EvaluationPolicy:
    """Identify native schedules without manufacturing a normalized action."""
    if (task.reward.reward_id not in (HOT_OCCUPIED_ENERGY_COMFORT_V2, HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3)
            or task.action_profile.profile_id != "thermostat_v1"
            or task.warmup_preconditioning.enabled):
        raise EvaluationError("native incumbent requires occupied reward, thermostats, and native warmup")
    return EvaluationPolicy(lambda observation, context: None,
                            PolicyProvenance("built_in", "native_incumbent"))


def static_midpoint_policy(task: TaskSpec) -> EvaluationPolicy:
    """Create the deterministic normalized midpoint policy for ``task``."""

    action = static_midpoint_action(task)

    def policy(_observation: Any, _context: DecisionContext) -> tuple[float, ...]:
        return action

    return EvaluationPolicy(policy, PolicyProvenance("built_in", "static_midpoint"))


def comfort_band_policy(task: TaskSpec) -> EvaluationPolicy:
    """Create fixed 20/24 C control with an explicit policy identity."""
    from building_gen.benchmark.policies import comfort_band_action
    action = comfort_band_action(task)
    return EvaluationPolicy(lambda observation, context: action,
                            PolicyProvenance("built_in", "comfort_band_20_24"))


def ashrae_seasonal_policy(task: TaskSpec) -> EvaluationPolicy:
    """Create the deterministic legacy HOT seasonal thermostat policy."""

    def policy(_observation: Any, context: DecisionContext) -> tuple[float, ...]:
        month = _integer(context.clock.get("month"), name="decision clock month")
        return ashrae_seasonal_action(task, month=month)

    return EvaluationPolicy(
        policy,
        PolicyProvenance("built_in", "ashrae_seasonal_setpoints"),
    )


def learned_callable_policy(
    actor: Callable[[Any], Any],
    *,
    algorithm_id: str,
    parent_training_run_id: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training_step: int,
    selection_rule: str,
) -> EvaluationPolicy:
    """Bind checkpoint provenance; sampling requires the explicit engineering diagnostic path."""

    if not callable(actor):
        raise TypeError("actor must be callable")

    def policy(observation: Any, _context: DecisionContext) -> Any:
        return actor(observation)

    return EvaluationPolicy(
        policy,
        PolicyProvenance(
            "learned_checkpoint",
            algorithm_id,
            parent_training_run_id=parent_training_run_id,
            checkpoint_sha256=checkpoint_sha256,
            training_step=training_step,
            selection_rule=selection_rule,
        ),
        Path(checkpoint_path),
    )


def ppo_callable_policy(
    actor: Callable[[Any], Any],
    *,
    parent_training_run_id: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training_step: int,
    selection_rule: str,
) -> EvaluationPolicy:
    """Adapt a deterministic callable (for example a PPO policy mean).

    ``actor`` is called once with the current policy observation.  The caller
    is responsible for loading a frozen checkpoint in evaluation mode and for
    returning its deterministic action rather than sampling from the policy.
    Bounds, shape, and finiteness are still checked by :func:`evaluate_run`.
    """

    return learned_callable_policy(
        actor,
        algorithm_id="ppo",
        parent_training_run_id=parent_training_run_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        training_step=training_step,
        selection_rule=selection_rule,
    )


def _validated_policy_provenance(
    run: ResolvedRun,
    policy: EvaluationPolicy,
) -> PolicyProvenance:
    if not isinstance(policy, EvaluationPolicy):
        raise EvaluationError("evaluation policy must be created by a provenance-bearing policy adapter")
    evidence = policy.provenance
    if evidence.policy_id != run.algorithm_id:
        raise EvaluationError(f"policy provenance identifies {evidence.policy_id!r}; run expects {run.algorithm_id!r}")
    learned_run = run.parent_training_run_id is not None
    if learned_run and evidence.kind != "learned_checkpoint":
        raise EvaluationError("learned evaluation runs require checkpoint provenance")
    if not learned_run and evidence.kind != "built_in":
        raise EvaluationError("a checkpoint policy cannot be attached to a run without a parent training run")
    if learned_run and evidence.parent_training_run_id != run.parent_training_run_id:
        raise EvaluationError("policy checkpoint provenance does not match the parent training run")
    if learned_run and policy.checkpoint_path is None:
        raise EvaluationError("learned evaluation runs require an exact policy checkpoint path")
    if not learned_run and policy.checkpoint_path is not None:
        raise EvaluationError("built-in evaluation runs cannot attach a policy checkpoint")
    return evidence


def _validated_training_bundle_link(
    *,
    run: ResolvedRun,
    policy_provenance: PolicyProvenance,
    provenance: SoftwareProvenance,
    gate: DiagnosticsGate,
    training_bundle: ValidatedTrainingBundle | None,
) -> Path | None:
    """Validate a scientific learned-policy bundle before any rollout begins."""

    learned_run = run.parent_training_run_id is not None
    if not gate.scientific:
        if training_bundle is not None:
            raise EvaluationError("training_bundle evidence is reserved for scientific evaluation")
        return None
    if not learned_run:
        if training_bundle is not None:
            raise EvaluationError("built-in scientific evaluation cannot attach a training_bundle")
        return None
    if not isinstance(training_bundle, ValidatedTrainingBundle):
        raise EvaluationError("scientific learned evaluation requires a validated training_bundle")
    if training_bundle.run.phase != "train" or training_bundle.run.run_id != run.parent_training_run_id:
        raise EvaluationError("training_bundle run does not match the evaluation parent training run")
    if run.train_task_ids != (training_bundle.task.task_id,):
        raise EvaluationError("training_bundle source TaskSpec does not match the evaluation run")
    if training_bundle.run.train_task_ids != run.train_task_ids:
        raise EvaluationError("training_bundle training task identity does not match the evaluation run")
    if training_bundle.provenance.to_dict() != provenance.to_dict():
        raise EvaluationError("training_bundle provenance does not match evaluation provenance")
    if training_bundle.checkpoint_sha256 != policy_provenance.checkpoint_sha256:
        raise EvaluationError("training_bundle checkpoint does not match learned policy provenance")
    if policy_provenance.training_step != training_bundle.contract.completed_transitions:
        raise EvaluationError("learned policy training_step does not match the training_bundle contract")
    if policy_provenance.selection_rule != final_checkpoint_selection_rule(
        training_bundle.contract.completed_transitions
    ):
        raise EvaluationError("learned policy selection_rule does not identify the sealed final checkpoint")
    source = training_bundle.bundle_path
    if source.is_symlink() or not source.is_file():
        raise EvaluationError("validated training_bundle source must remain a regular file")
    if _sha256_file(source, role="training bundle source") != training_bundle.bundle_sha256:
        raise EvaluationError("validated training_bundle bytes changed before evaluation")
    return source


def evaluate_run(
    *,
    run: ResolvedRun,
    task: TaskSpec,
    protocol: EvaluationProtocolSpec,
    environment_factory: EnvironmentFactory,
    policy: EvaluationPolicy,
    provenance: SoftwareProvenance,
    output_dir: str | Path,
    overwrite: bool = False,
    diagnostics_gate: DiagnosticsGate | None = None,
    training_bundle: ValidatedTrainingBundle | None = None,
    stochastic_diagnostic_seed: int | None = None,
) -> EvaluationOutput:
    """Evaluate one policy over every protocol window and write strict evidence.

    Every episode must contain each expected calendar timestep exactly once,
    terminate (rather than truncate) on its final transition, expose finite raw
    observations, and report a consistent set of energy carriers.  No partial
    result is emitted when these checks fail.
    """

    if run.phase != "evaluate":
        raise EvaluationError("evaluate_run requires a resolved evaluation run")
    if run.evaluation_task_id != task.task_id:
        raise EvaluationError("resolved run evaluation_task_id does not match the supplied task")
    gate = diagnostics_gate or DiagnosticsGate()
    if stochastic_diagnostic_seed is not None:
        if (type(stochastic_diagnostic_seed) is not int or not 0<=stochastic_diagnostic_seed<2**63
                or gate.mode != 'engineering' or protocol.deterministic_actions
                or training_bundle is not None or run.parent_training_run_id is None):
            raise EvaluationError('stochastic diagnostic requires an explicit sampling seed, engineering gate, learned policy and nondeterministic protocol; publication bundles are excluded')
    elif not protocol.deterministic_actions:
        raise EvaluationError("publication evaluation requires deterministic_actions=true")
    window_duration_hours: dict[str, int] = {}
    for window in protocol.windows:
        try:
            window_duration_hours[window.name] = window.duration_hours_for_calendar_year(task.calendar_year)
        except ValueError as error:
            resolved_year = task.calendar_year if task.calendar_year is not None else 2001
            raise EvaluationError(
                f"evaluation window {window.name!r} is invalid for TaskSpec calendar year {resolved_year}"
            ) from error
    policy_provenance = _validated_policy_provenance(run, policy)
    native_control = policy_provenance.policy_id == "native_incumbent"
    if native_control:
        native_incumbent_policy(task)
        if gate.mode != "engineering":
            raise EvaluationError("native incumbent scientific evidence qualification is not implemented")

    training_bundle_source = _validated_training_bundle_link(
        run=run,
        policy_provenance=policy_provenance,
        provenance=provenance,
        gate=gate,
        training_bundle=training_bundle,
    )
    uses_occupied_reward = task.reward.reward_id in (HOT_OCCUPIED_ENERGY_COMFORT_V2, HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3)
    comfort_min, comfort_max = _comfort_band(task)
    interval_hours = 1.0 / task.episode.timesteps_per_hour
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    steps_path = destination / f"{run.run_id}.steps.csv"
    result_path = destination / f"{run.run_id}.result.json"
    diagnostics_path = destination / f"{run.run_id}.energyplus-diagnostics.json"
    evidence_path = destination / f"{run.run_id}.evaluation-evidence.json"
    warmup_preconditioning_path = destination / (f"{run.run_id}.warmup-preconditioning-evidence.json")
    diagnostics_policy_path = destination / f"{run.run_id}.diagnostics-acceptance-policy.json"
    policy_checkpoint_path = destination / f"{run.run_id}.policy-checkpoint.pt"
    training_bundle_path = destination / f"{run.run_id}.training-bundle.json"
    checkpoint_source: Path | None = None
    if run.parent_training_run_id is not None:
        assert policy.checkpoint_path is not None
        assert policy_provenance.checkpoint_sha256 is not None
        checkpoint_source = policy.checkpoint_path
    evaluation_artifacts = (
        steps_path,
        diagnostics_path,
        warmup_preconditioning_path,
        evidence_path,
        diagnostics_policy_path,
        *((policy_checkpoint_path,) if checkpoint_source is not None else ()),
        *((training_bundle_path,) if training_bundle_source is not None else ()),
        result_path,
    )
    if checkpoint_source is not None:
        aliased_outputs = [path for path in evaluation_artifacts if _same_path(checkpoint_source, path)]
        if aliased_outputs:
            raise EvaluationError(
                "policy checkpoint source must not alias any evaluator output: " + ", ".join(map(str, aliased_outputs))
            )
        source_digest = _sha256_file(checkpoint_source, role="policy checkpoint source")
        if source_digest != policy_provenance.checkpoint_sha256:
            raise EvaluationError("policy checkpoint source checksum does not match PolicyProvenance")
    if training_bundle_source is not None:
        aliased_outputs = [path for path in evaluation_artifacts if _same_path(training_bundle_source, path)]
        if aliased_outputs:
            raise EvaluationError(
                "training bundle source must not alias any evaluator output: " + ", ".join(map(str, aliased_outputs))
            )
        assert training_bundle is not None
        if _sha256_file(training_bundle_source, role="training bundle source") != training_bundle.bundle_sha256:
            raise EvaluationError("validated training_bundle bytes changed before evaluation")
    if overwrite:
        _remove_exact_artifacts(evaluation_artifacts)
    else:
        existing = [path for path in evaluation_artifacts if path.exists() or path.is_symlink()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing evaluation artifact(s): " + ", ".join(map(str, existing))
            )
    with _CsvSpool() as rows:
        carrier_totals = {carrier.value: 0.0 for carrier in EnergyCarrier}
        reported_carriers: set[str] | None = None
        energy_meter_sources: dict[str, str] | None = None
        total_reward = 0.0
        total_legacy_degree_hours = 0.0
        total_legacy_zone_hours = 0.0
        total_legacy_all_comfort_hours = 0.0
        total_interval_hours = 0.0
        controlled_floor_area_m2: float | None = None
        total_occupied_comfort_exposure = 0.0
        total_occupied_comfort_exposure_intensity = 0.0
        total_occupied_person_hours = 0.0
        zone_keys: tuple[str, ...] | None = None
        try:
            scientific_allowlist = gate.policy_for(
                asset_sha256=task.building.sha256,
                weather_sha256=task.weather.sha256,
                energyplus_version=provenance.energyplus_version,
            )
        except ScientificDiagnosticsError as error:
            raise EvaluationError(str(error)) from error
        diagnostics_records: list[dict[str, Any]] = []
        warmup_preconditioning_records: list[dict[str, Any]] = []
        claimed_episode_work_dirs: set[Path] = set()
        claimed_diagnostics_paths: set[Path] = set()
        claimed_warmup_paths: set[Path] = set()

        for window in protocol.windows:
            expected_steps = window_duration_hours[window.name] * task.episode.timesteps_per_hour
            expected_clock_sequence = _expected_clock_sequence(
                window,
                task.episode.timesteps_per_hour,
                calendar_year=task.calendar_year,
            )
            expected_clock_keys = set(expected_clock_sequence)
            for episode_index in range(protocol.episodes_per_window):
                rollout = RolloutContext(run, task, window, episode_index, expected_steps)
                env = environment_factory(rollout)
                if env is None:
                    raise EvaluationError("environment_factory returned None")
                controlled_clock_keys: set[tuple[int, int, int, int]] = set()
                terminated_at: int | None = None
                try:
                    observation, decision_info = env.reset(seed=run.seed)
                    policy_observation_values = _validate_observation(
                        observation,
                        task=task,
                        name="reset policy observation",
                    )
                    decision_clock = _clock(
                        decision_info,
                        where="reset",
                        expected_calendar_year=task.calendar_year,
                        expected_calendar_year_semantics=task.calendar_year_semantics,
                    )
                    _validate_first_decision_clock_calendar(
                        decision_clock,
                        task=task,
                        window=window,
                    )
                    for step_index in range(expected_steps):
                        context = DecisionContext(rollout, step_index, decision_clock, deterministic=protocol.deterministic_actions)
                        proposed_action = policy(observation, context)
                        if native_control:
                            if proposed_action is not None:
                                raise EvaluationError("native incumbent must not propose a policy action")
                            action_values = ()
                            next_observation, reward, terminated, truncated, info = env.step_native_schedule()
                        else:
                            action_values = _flat_finite_action(proposed_action, task)
                            env_action = _environment_action(proposed_action, action_values, env)
                            next_observation, reward, terminated, truncated, info = env.step(env_action)
                        reward_value = _finite_float(reward, name="reward")
                        next_policy_observation_values = _validate_observation(
                            next_observation,
                            task=task,
                            name="next policy observation",
                        )
                        if bool(terminated) and bool(truncated):
                            raise EvaluationError("a transition cannot both terminate and truncate")
                        if bool(truncated):
                            raise EvaluationError(f"{window.name} episode {episode_index} truncated at step {step_index}")

                        raw = _raw_observation(info, where=f"{window.name} step {step_index}")
                        current_zone_keys = _controlled_temperature_keys(
                            raw,
                            controlled_entities=task.action_profile.controlled_entities,
                        )
                        if zone_keys is None:
                            zone_keys = current_zone_keys
                        elif current_zone_keys != zone_keys:
                            raise EvaluationError("zone-temperature fields changed during evaluation")
                        legacy_degree_hours, legacy_zone_hours, legacy_all_comfort = _comfort_metrics(
                            raw,
                            zone_keys,
                            comfort_min=comfort_min,
                            comfort_max=comfort_max,
                            interval_hours=interval_hours,
                        )
                        ledger, current_reported, current_meter_sources = _energy_ledger(
                            info,
                            where=f"{window.name} step {step_index}",
                            expected_meter_sources=dict(task.component_energy_meters) if task.component_energy_meters is not None else None,
                        )
                        if task.reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3:
                            interval_clock = _clock(
                                {"simulation_clock": info.get("energy_interval_clock")},
                                where=f"{window.name} step {step_index} completed interval",
                                expected_calendar_year=task.calendar_year,
                                expected_calendar_year_semantics=task.calendar_year_semantics,
                            )
                            if interval_clock != decision_clock:
                                raise EvaluationError(
                                    "scheduled reward completed-interval clock does not match the action decision clock"
                                )
                        reward_components_json, reward_components = _reward_components_json(
                            info,
                            task=task,
                            scalar_reward=reward_value,
                            ledger=ledger,
                            interval_hours=interval_hours,
                            zone_floor_areas_m2=getattr(env, "controlled_zone_floor_areas_m2", None),
                            where=f"{window.name} step {step_index}",
                        )
                        if reported_carriers is None:
                            reported_carriers = current_reported
                            energy_meter_sources = current_meter_sources
                        elif current_reported != reported_carriers:
                            raise EvaluationError("reported energy carriers changed during evaluation")
                        elif current_meter_sources != energy_meter_sources:
                            raise EvaluationError("controlled-HVAC energy meter sources changed during evaluation")
                        if native_control:
                            action_evidence = _native_action_evidence(info)
                        else:
                            action_evidence = _submitted_action_evidence(
                                info, task=task, proposed_action=action_values,
                                where=f"{window.name} step {step_index}",
                            )
                        if reward_components is not None and uses_occupied_reward:
                            area = reward_components["controlled_floor_area_m2"]
                            if controlled_floor_area_m2 is None:
                                controlled_floor_area_m2 = area
                            elif not math.isclose(area, controlled_floor_area_m2, rel_tol=0.0, abs_tol=1e-9):
                                raise EvaluationError("reward-v2 controlled floor area changed during evaluation")
                            if not math.isclose(
                                reward_components["timestep_hours"],
                                interval_hours,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            ):
                                raise EvaluationError("reward-v2 timestep_hours does not match the task interval")
                            interval_energy = math.fsum(value for value in ledger.values() if value is not None)
                            if not math.isclose(
                                reward_components["interval_controlled_hvac_site_energy_kwh"],
                                interval_energy,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            ):
                                raise EvaluationError("reward-v2 controlled-HVAC energy does not match the energy ledger")
                            exposure_intensity = reward_components["comfort_exposure_person_celsius_squared_hours_per_m2"]
                            total_occupied_comfort_exposure_intensity += exposure_intensity
                            total_occupied_comfort_exposure += exposure_intensity * area
                            total_occupied_person_hours += reward_components["total_occupant_count"] * interval_hours
                        # The action is applied at decision_clock and the returned
                        # observation is the state at the next EnergyPlus control
                        # instant.  The transition therefore belongs to the
                        # decision clock, while current_clock is retained as its
                        # next-state boundary.  A 24-hour episode naturally ends
                        # with that next-state boundary on the following day.
                        controlled_clock_key = _clock_key(
                            decision_clock,
                            timesteps_per_hour=task.episode.timesteps_per_hour,
                        )
                        current_clock = _clock(
                            info,
                            where=f"{window.name} step {step_index}",
                            expected_calendar_year=task.calendar_year,
                            expected_calendar_year_semantics=task.calendar_year_semantics,
                            allow_next_year_boundary=(
                                bool(terminated)
                                and step_index == expected_steps - 1
                                and expected_clock_sequence[step_index] == (12, 31, 23, task.episode.timesteps_per_hour)
                            ),
                        )
                        _validate_transition_clocks(
                            decision_clock,
                            current_clock,
                            expected_decision_clock=expected_clock_sequence[step_index],
                            step_index=step_index,
                            timesteps_per_hour=task.episode.timesteps_per_hour,
                        )
                        if controlled_clock_key not in expected_clock_keys:
                            raise EvaluationError(
                                f"decision clock {controlled_clock_key!r} lies outside " f"calendar window {window.name!r}"
                            )
                        if controlled_clock_key in controlled_clock_keys:
                            raise EvaluationError(
                                f"duplicate controlled calendar timestep {controlled_clock_key!r} " f"in {window.name!r}"
                            )
                        controlled_clock_keys.add(controlled_clock_key)

                        for carrier, value in ledger.items():
                            if value is not None:
                                carrier_totals[carrier] += value
                        total_reward += reward_value
                        total_legacy_degree_hours += legacy_degree_hours
                        total_legacy_zone_hours += legacy_zone_hours
                        total_legacy_all_comfort_hours += interval_hours if legacy_all_comfort else 0.0
                        total_interval_hours += interval_hours
                        if policy.transition_observer is not None:
                            policy.transition_observer(
                                next_observation=tuple(next_policy_observation_values),
                                context=context,
                                reward_components=dict(reward_components or {}),
                                energy_ledger=dict(ledger),
                                action_evidence=dict(action_evidence),
                                interval_hours=interval_hours,
                            )
                        rows.append(
                            _step_row(
                                run=run,
                                task=task,
                                protocol=protocol,
                                rollout=rollout,
                                step_index=step_index,
                                reward=reward_value,
                                reward_components_json=reward_components_json,
                                action_evidence=action_evidence,
                                policy_observation=policy_observation_values,
                                decision_clock=decision_clock,
                                clock=current_clock,
                                raw_observation=raw,
                                ledger=ledger,
                                interval_hours=interval_hours,
                                legacy_comfort_degree_hours=legacy_degree_hours,
                                legacy_zone_hours=legacy_zone_hours,
                                legacy_all_zones_comfort=legacy_all_comfort,
                                occupied_comfort_exposure_intensity=(
                                    None
                                    if reward_components is None or not uses_occupied_reward
                                    else reward_components.get("comfort_exposure_person_celsius_squared_hours_per_m2")
                                ),
                                occupied_person_hours=(
                                    None
                                    if reward_components is None or not uses_occupied_reward
                                    else reward_components.get("total_occupant_count", 0.0) * interval_hours
                                ),
                                controlled_floor_area_m2=(
                                    None
                                    if reward_components is None or not uses_occupied_reward
                                    else reward_components.get("controlled_floor_area_m2")
                                ),
                                terminated=bool(terminated),
                            )
                        )
                        observation = next_observation
                        policy_observation_values = next_policy_observation_values
                        decision_clock = current_clock
                        if terminated:
                            terminated_at = step_index + 1
                            if terminated_at != expected_steps:
                                raise EvaluationError(
                                    f"{window.name} episode {episode_index} terminated after "
                                    f"{terminated_at} of {expected_steps} steps"
                                )
                            break
                finally:
                    close = getattr(env, "close", None)
                    if callable(close):
                        close()

                _claim_episode_artifact_sources(
                    rollout=rollout,
                    environment=env,
                    claimed_episode_work_dirs=claimed_episode_work_dirs,
                    claimed_diagnostics_paths=claimed_diagnostics_paths,
                    claimed_warmup_paths=claimed_warmup_paths,
                )
                diagnostics_records.append(
                    _episode_diagnostics_record(
                        rollout=rollout,
                        environment=env,
                        gate=gate,
                        scientific_allowlist=scientific_allowlist,
                    )
                )
                warmup_preconditioning_records.append(
                    _episode_warmup_preconditioning_record(
                        rollout=rollout,
                        environment=env,
                    )
                )

                if terminated_at != expected_steps:
                    raise EvaluationError(
                        f"{window.name} episode {episode_index} did not terminate after " f"{expected_steps} steps"
                    )
                missing = expected_clock_keys - controlled_clock_keys
                extra = controlled_clock_keys - expected_clock_keys
                if missing or extra:
                    raise EvaluationError(
                        f"incomplete calendar window {window.name!r}: " f"missing={len(missing)}, extra={len(extra)}"
                    )

        if not rows or zone_keys is None or reported_carriers is None or energy_meter_sources is None:
            raise EvaluationError("evaluation produced no valid transitions")

        total_controlled_hvac_energy_kwh = math.fsum(carrier_totals.values())
        metrics = [
            MetricValue("total_reward", total_reward, "1", "sum", "maximize"),
            MetricValue("evaluated_hours", total_interval_hours, "h", "sum", "neutral"),
            MetricValue(
                "legacy_24h_evaluated_zone_hours",
                total_legacy_zone_hours,
                "zone h",
                "sum",
                "neutral",
            ),
            MetricValue(
                "legacy_24h_comfort_violation_degree_hours",
                total_legacy_degree_hours,
                "degC h",
                "sum",
                "minimize",
            ),
            MetricValue(
                "legacy_24h_mean_comfort_violation_c",
                total_legacy_degree_hours / total_legacy_zone_hours,
                "degC",
                "mean",
                "minimize",
            ),
            MetricValue(
                "legacy_24h_all_zones_comfort_fraction",
                total_legacy_all_comfort_hours / total_interval_hours,
                "1",
                "mean",
                "maximize",
            ),
            MetricValue(
                "controlled_hvac_energy_kwh",
                total_controlled_hvac_energy_kwh,
                "kWh",
                "sum",
                "minimize",
            ),
            MetricValue("completed_windows", float(len(protocol.windows)), "count", "count", "neutral"),
            MetricValue(
                "completed_episodes",
                float(len(protocol.windows) * protocol.episodes_per_window),
                "count",
                "count",
                "neutral",
            ),
            MetricValue("control_steps", float(len(rows)), "count", "count", "neutral"),
        ]
        if controlled_floor_area_m2 is not None:
            metrics.extend(
                [
                    MetricValue(
                        "controlled_hvac_energy_intensity_kwh_per_m2",
                        total_controlled_hvac_energy_kwh / controlled_floor_area_m2,
                        "kWh/m2",
                        "sum",
                        "minimize",
                    ),
                    MetricValue(
                        "occupied_comfort_exposure_person_celsius_squared_hours",
                        total_occupied_comfort_exposure,
                        "person degC2 h",
                        "sum",
                        "minimize",
                    ),
                    MetricValue(
                        "occupied_comfort_exposure_person_celsius_squared_hours_per_m2",
                        total_occupied_comfort_exposure_intensity,
                        "person degC2 h/m2",
                        "sum",
                        "minimize",
                    ),
                    MetricValue(
                        "occupied_person_hours",
                        total_occupied_person_hours,
                        "person h",
                        "sum",
                        "neutral",
                    ),
                ]
            )
            if total_occupied_person_hours > 0.0:
                metrics.append(
                    MetricValue(
                        "occupied_mean_squared_comfort_violation_celsius_squared",
                        total_occupied_comfort_exposure / total_occupied_person_hours,
                        "degC2",
                        "mean",
                        "minimize",
                    )
                )
        for carrier in sorted(reported_carriers):
            metrics.append(
                MetricValue(
                    f"controlled_hvac_energy_{carrier}_kwh",
                    carrier_totals[carrier],
                    "kWh",
                    "sum",
                    "minimize",
                )
            )

        step_digest = rows.publish(steps_path)
        diagnostics_policy_ref: dict[str, str] | None = None
        if scientific_allowlist is not None:
            diagnostics_policy_text = scientific_allowlist.canonical_json()
            diagnostics_policy_digest = hashlib.sha256(diagnostics_policy_text.encode("utf-8")).hexdigest()
            if diagnostics_policy_digest != scientific_allowlist.sha256:
                raise EvaluationError("canonical diagnostics policy digest is inconsistent")
            _atomic_write_text(diagnostics_policy_path, diagnostics_policy_text)
            diagnostics_policy_ref = {
                "uri": diagnostics_policy_path.name,
                "sha256": diagnostics_policy_digest,
                "schema_version": scientific_allowlist.schema_version,
            }
        diagnostics_text = json.dumps(
            {
                "schema_version": EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
                "run_id": run.run_id,
                "mode": gate.mode,
                "scope": {
                    "building_asset_sha256": task.building.sha256,
                    "weather_asset_sha256": task.weather.sha256,
                    "energyplus_version": provenance.energyplus_version,
                },
                "diagnostics_acceptance_policy": diagnostics_policy_ref,
                "episodes": diagnostics_records,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        diagnostics_text += "\n"
        _atomic_write_text(diagnostics_path, diagnostics_text)
        diagnostics_digest = hashlib.sha256(diagnostics_text.encode("utf-8")).hexdigest()
        warmup_preconditioning_text = json.dumps(
            _evaluation_warmup_preconditioning_artifact(
                run=run,
                task=task,
                protocol=protocol,
                records=warmup_preconditioning_records,
            ),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        warmup_preconditioning_text += "\n"
        _atomic_write_text(
            warmup_preconditioning_path,
            warmup_preconditioning_text,
        )
        warmup_preconditioning_digest = hashlib.sha256(warmup_preconditioning_text.encode("utf-8")).hexdigest()
        warmup_preconditioning_ref = {
            "uri": warmup_preconditioning_path.name,
            "sha256": warmup_preconditioning_digest,
            "schema_version": EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION,
        }
        training_bundle_digest: str | None = None
        training_bundle_ref: dict[str, str] | None = None
        if training_bundle_source is not None:
            assert training_bundle is not None
            training_bundle_digest = _atomic_copy_verified(
                training_bundle_source,
                training_bundle_path,
                expected_sha256=training_bundle.bundle_sha256,
                role="training bundle",
            )
            training_bundle_ref = {
                "uri": training_bundle_path.name,
                "sha256": training_bundle_digest,
                "schema_version": TRAINING_BUNDLE_SCHEMA_VERSION,
            }
        evidence_text = json.dumps(
            _evaluation_evidence_manifest(
                run=run,
                task=task,
                policy_provenance=policy_provenance,
                reported_carriers=reported_carriers,
                energy_meter_sources=energy_meter_sources,
                controlled_floor_area_m2=controlled_floor_area_m2,
                diagnostics_policy_ref=diagnostics_policy_ref,
                warmup_preconditioning_ref=warmup_preconditioning_ref,
                training_bundle_ref=training_bundle_ref,
                stochastic_diagnostic_seed=stochastic_diagnostic_seed,
            ),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        evidence_text += "\n"
        _atomic_write_text(evidence_path, evidence_text)
        evidence_digest = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
        policy_checkpoint_digest: str | None = None
        if checkpoint_source is not None:
            assert policy_provenance.checkpoint_sha256 is not None
            policy_checkpoint_digest = _atomic_copy_verified(
                checkpoint_source,
                policy_checkpoint_path,
                expected_sha256=policy_provenance.checkpoint_sha256,
            )
        result = RunResult(
            run=run,
            status="succeeded",
            provenance=provenance,
            metrics=tuple(metrics),
            artifacts=(
                ArtifactRef(
                    role="step_trajectory",
                    uri=steps_path.name,
                    media_type="text/csv",
                    sha256=step_digest,
                ),
                ArtifactRef(
                    role="energyplus_diagnostics",
                    uri=diagnostics_path.name,
                    media_type="application/json",
                    sha256=diagnostics_digest,
                ),
                ArtifactRef(
                    role="evaluation_evidence",
                    uri=evidence_path.name,
                    media_type="application/json",
                    sha256=evidence_digest,
                ),
                ArtifactRef(
                    role="warmup_preconditioning_evidence",
                    uri=warmup_preconditioning_path.name,
                    media_type="application/json",
                    sha256=warmup_preconditioning_digest,
                ),
                *(
                    (
                        ArtifactRef(
                            role="policy_checkpoint",
                            uri=policy_checkpoint_path.name,
                            media_type="application/octet-stream",
                            sha256=policy_checkpoint_digest,
                        ),
                    )
                    if policy_checkpoint_digest is not None
                    else ()
                ),
                *(
                    (
                        ArtifactRef(
                            role="training_bundle",
                            uri=training_bundle_path.name,
                            media_type="application/json",
                            sha256=training_bundle_digest,
                        ),
                    )
                    if training_bundle_digest is not None
                    else ()
                ),
                *(
                    (
                        ArtifactRef(
                            role="diagnostics_acceptance_policy",
                            uri=diagnostics_policy_path.name,
                            media_type="application/json",
                            sha256=diagnostics_policy_ref["sha256"],
                        ),
                    )
                    if diagnostics_policy_ref is not None
                    else ()
                ),
            ),
        )
        _atomic_write_text(result_path, result.to_json() + "\n")
        return EvaluationOutput(
            result,
            result_path,
            steps_path,
            diagnostics_path,
            evidence_path,
            warmup_preconditioning_path,
            diagnostics_policy_path if diagnostics_policy_ref is not None else None,
            policy_checkpoint_path if policy_checkpoint_digest is not None else None,
            training_bundle_path if training_bundle_digest is not None else None,
        )


def _claim_episode_artifact_sources(
    *,
    rollout: RolloutContext,
    environment: GymEnvironment,
    claimed_episode_work_dirs: set[Path],
    claimed_diagnostics_paths: set[Path],
    claimed_warmup_paths: set[Path],
) -> None:
    """Bind advertised runtime artifacts to one unique episode work directory."""

    identity = f"window {rollout.window.name!r}, episode {rollout.episode_index}"
    work_dir_value = getattr(environment, "benchmark_episode_work_dir", None)
    if work_dir_value is None:
        if rollout.task.warmup_preconditioning.enabled:
            raise EvaluationError(
                f"enabled warmup preconditioning for {identity} has no exact episode work-directory identity"
            )
        expected_energyplus_dir = None
    else:
        work_dir = Path(work_dir_value).resolve()
        if work_dir in claimed_episode_work_dirs:
            raise EvaluationError(f"evaluation reused episode work directory {work_dir} for {identity}")
        claimed_episode_work_dirs.add(work_dir)
        expected_energyplus_dir = work_dir / "energyplus"

    for attribute, filename, claimed_paths in (
        (
            "energyplus_diagnostics_path",
            DIAGNOSTICS_FILENAME,
            claimed_diagnostics_paths,
        ),
        (
            "warmup_preconditioning_evidence_path",
            WARMUP_PRECONDITIONING_FILENAME,
            claimed_warmup_paths,
        ),
    ):
        source = getattr(environment, attribute, None)
        if source is None:
            continue
        resolved = Path(source).resolve()
        if resolved in claimed_paths:
            raise EvaluationError(f"evaluation reused {filename!r} source {resolved} for {identity}")
        claimed_paths.add(resolved)
        if expected_energyplus_dir is not None:
            expected = (expected_energyplus_dir / filename).resolve()
            if resolved != expected:
                raise EvaluationError(
                    f"{filename!r} for {identity} does not belong to its exact episode work directory"
                )


def _episode_diagnostics_record(
    *,
    rollout: RolloutContext,
    environment: GymEnvironment,
    gate: DiagnosticsGate,
    scientific_allowlist: DiagnosticsAllowlist | None,
) -> dict[str, Any]:
    """Load and, in scientific mode, validate one completed episode."""

    identity = f"window {rollout.window.name!r}, episode {rollout.episode_index}"
    expected_model_sha256 = getattr(environment, "energyplus_executed_model_sha256", None)
    if expected_model_sha256 is not None:
        if not isinstance(expected_model_sha256, str) or not _is_sha256(expected_model_sha256):
            return _unavailable_diagnostics_record(
                rollout,
                gate,
                f"{identity} advertised an invalid executed-model SHA256",
            )
        expected_model_sha256 = expected_model_sha256.lower()
    elif gate.scientific:
        raise EvaluationError(f"scientific diagnostics for {identity} have no expected executed-model SHA256")

    source = getattr(environment, "energyplus_diagnostics_path", None)
    if source is None:
        return _unavailable_diagnostics_record(
            rollout,
            gate,
            f"{identity} did not advertise an EnergyPlus diagnostics artifact",
        )
    source_path = Path(source)
    if not source_path.is_file():
        return _unavailable_diagnostics_record(
            rollout,
            gate,
            f"{identity} did not produce {source_path.name!r}",
        )
    try:
        raw_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, Mapping):
            raise ValueError("diagnostics artifact must contain a JSON object")
        diagnostics = energyplus_diagnostics_from_mapping(raw_payload)
        executed_model_sha256 = raw_payload.get("executed_model_sha256")
        if executed_model_sha256 is not None:
            if not isinstance(executed_model_sha256, str) or not _is_sha256(executed_model_sha256):
                raise ValueError("executed_model_sha256 must contain 64 hexadecimal characters")
            executed_model_sha256 = executed_model_sha256.lower()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluationError(f"cannot load diagnostics for {identity}: {error}") from error

    if gate.scientific and executed_model_sha256 is None:
        raise EvaluationError(f"scientific diagnostics for {identity} do not identify the executed model bytes")
    executed_model_matches = (
        expected_model_sha256 is not None
        and executed_model_sha256 is not None
        and expected_model_sha256 == executed_model_sha256
    )
    if gate.scientific and not executed_model_matches:
        raise EvaluationError(
            f"scientific diagnostics for {identity} identify executed model {executed_model_sha256!r}; "
            f"expected {expected_model_sha256!r}"
        )

    acceptance: dict[str, Any]
    status: str
    canonical_diagnostics_json: bytes
    if gate.scientific:
        if scientific_allowlist is None:  # Defensive: policy_for fails before rollout.
            raise EvaluationError("scientific diagnostics have no scoped acceptance policy")
        result = evaluate_scientific_diagnostics(
            diagnostics,
            asset_sha256=rollout.task.building.sha256 or "",
            weather_sha256=rollout.task.weather.sha256 or "",
            allowlist=scientific_allowlist,
        )
        acceptance = {"accepted": result.accepted, "violations": list(result.violations)}
        if not result.accepted:
            raise EvaluationError(f"scientific diagnostics rejected {identity}: {'; '.join(result.violations)}")
        assert result.canonical_diagnostics_json is not None
        canonical_diagnostics_json = result.canonical_diagnostics_json
        status = "accepted"
    else:
        acceptance = {"accepted": None, "violations": []}
        canonical_diagnostics_json = canonical_energyplus_diagnostics_snapshot(diagnostics).canonical_json
        # Engineering permits warnings, but a reported simulator error must not
        # become a successful controller evaluation.
        if (diagnostics.severe_error_count or diagnostics.fatal_error_count
                or not diagnostics.completed_successfully):
            raise EvaluationError(
                f"engineering diagnostics rejected {identity}: "
                f"completed={diagnostics.completed_successfully}, "
                f"severe={diagnostics.severe_error_count}, fatal={diagnostics.fatal_error_count}"
            )
        status = "recorded"

    summary = json.loads(canonical_diagnostics_json.decode("utf-8"))
    if executed_model_sha256 is not None:
        summary["executed_model_sha256"] = executed_model_sha256
    return {
        "window": rollout.window.name,
        "episode_index": rollout.episode_index,
        "status": status,
        "acceptance": acceptance,
        "expected_executed_model_sha256": expected_model_sha256,
        "executed_model_sha256_matches": executed_model_matches,
        "diagnostics": summary,
    }


def _episode_warmup_preconditioning_record(
    *,
    rollout: RolloutContext,
    environment: GymEnvironment,
) -> dict[str, Any]:
    """Authenticate one episode's runtime warmup contract and evidence."""

    identity = f"window {rollout.window.name!r}, episode {rollout.episode_index}"
    contract = rollout.task.warmup_preconditioning
    expected_contract_payload = contract.identity_payload()
    expected_contract_sha256 = contract.sha256
    missing = object()
    runtime_contract_payload = getattr(
        environment,
        "warmup_preconditioning_contract_payload",
        missing,
    )
    runtime_contract_sha256 = getattr(
        environment,
        "warmup_preconditioning_contract_sha256",
        missing,
    )
    source = getattr(
        environment,
        "warmup_preconditioning_evidence_path",
        None,
    )

    # Only legacy task identity may omit the disabled runtime contract. Task
    # schema 1.2 makes even explicit non-intervention identity-bearing.
    if not contract.enabled:
        if source is not None:
            raise EvaluationError(
                f"disabled warmup preconditioning for {identity} unexpectedly produced runtime evidence"
            )
        runtime_contract_absent = runtime_contract_payload is missing and runtime_contract_sha256 is missing
        if (runtime_contract_payload is missing) != (runtime_contract_sha256 is missing):
            raise EvaluationError(
                f"disabled warmup preconditioning for {identity} advertises an incomplete runtime contract"
            )
        if rollout.task.schema_version == TASK_SCHEMA_VERSION:
            if runtime_contract_absent:
                raise EvaluationError(
                    f"disabled warmup preconditioning for {identity} has no complete runtime contract"
                )
        elif rollout.task.schema_version != LEGACY_TASK_SCHEMA_VERSION:
            raise EvaluationError(
                f"unsupported TaskSpec schema for disabled warmup preconditioning: " f"{rollout.task.schema_version!r}"
            )
        if not runtime_contract_absent and (
            runtime_contract_payload != expected_contract_payload or runtime_contract_sha256 != expected_contract_sha256
        ):
            raise EvaluationError(
                f"disabled warmup preconditioning runtime contract for {identity} does not match TaskSpec"
            )
        return {
            "window": rollout.window.name,
            "episode_index": rollout.episode_index,
            "status": "disabled",
            "contract_sha256": expected_contract_sha256,
            "evidence": None,
        }

    if runtime_contract_payload is missing or runtime_contract_sha256 is missing:
        raise EvaluationError(f"enabled warmup preconditioning for {identity} has no complete runtime contract")
    if not isinstance(runtime_contract_payload, Mapping):
        raise EvaluationError(f"enabled warmup preconditioning contract payload for {identity} is malformed")
    if dict(runtime_contract_payload) != expected_contract_payload:
        raise EvaluationError(f"enabled warmup preconditioning runtime contract for {identity} does not match TaskSpec")
    if runtime_contract_sha256 != expected_contract_sha256:
        raise EvaluationError(f"enabled warmup preconditioning contract SHA256 for {identity} does not match TaskSpec")
    if source is None:
        raise EvaluationError(f"enabled warmup preconditioning for {identity} produced no runtime evidence")

    expected_model_sha256 = getattr(
        environment,
        "energyplus_executed_model_sha256",
        None,
    )
    if not isinstance(expected_model_sha256, str) or not _is_sha256(expected_model_sha256):
        raise EvaluationError(f"enabled warmup preconditioning for {identity} has no valid executed-model SHA256")
    expected_model_sha256 = expected_model_sha256.lower()

    source_path = Path(source)
    diagnostics_source = getattr(environment, "energyplus_diagnostics_path", None)
    if diagnostics_source is None:
        raise EvaluationError(
            f"enabled warmup preconditioning for {identity} cannot be tied to its episode work directory"
        )
    expected_path = Path(diagnostics_source).parent / WARMUP_PRECONDITIONING_FILENAME
    try:
        resolved_path = source_path.resolve(strict=True)
    except OSError as error:
        raise EvaluationError(
            f"cannot resolve warmup preconditioning evidence for {identity}: {source_path}"
        ) from error
    if source_path.is_symlink() or resolved_path != expected_path.resolve():
        raise EvaluationError(
            f"warmup preconditioning evidence for {identity} does not belong to its exact episode work directory"
        )
    try:
        payload, raw_bytes = validate_warmup_preconditioning_evidence(
            resolved_path,
            task=rollout.task,
            window=rollout.window,
            executed_model_sha256=expected_model_sha256,
        )
    except WarmupPreconditioningEvidenceError as error:
        raise EvaluationError(f"invalid warmup preconditioning evidence for {identity}: {error}") from error
    return {
        "window": rollout.window.name,
        "episode_index": rollout.episode_index,
        "status": "accepted",
        "contract_sha256": expected_contract_sha256,
        "executed_model_sha256": expected_model_sha256,
        "evidence": {
            "source_path": f"energyplus/{WARMUP_PRECONDITIONING_FILENAME}",
            "source_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "canonical_payload_sha256": hashlib.sha256(
                json.dumps(
                    payload,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "payload": dict(payload),
        },
    }


def _evaluation_warmup_preconditioning_artifact(
    *,
    run: ResolvedRun,
    task: TaskSpec,
    protocol: EvaluationProtocolSpec,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact run-level intervention-evidence manifest."""

    expected_episode_keys = [
        (window.name, episode_index)
        for window in protocol.windows
        for episode_index in range(protocol.episodes_per_window)
    ]
    actual_episode_keys = [(record.get("window"), record.get("episode_index")) for record in records]
    if actual_episode_keys != expected_episode_keys:
        raise EvaluationError("warmup preconditioning records do not cover the exact evaluation episodes")
    expected_status = "accepted" if task.warmup_preconditioning.enabled else "disabled"
    if any(record.get("status") != expected_status for record in records):
        raise EvaluationError("warmup preconditioning record status does not match the TaskSpec contract")
    return {
        "schema_version": EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION,
        "run_id": run.run_id,
        "task": {
            "task_name": task.name,
            "task_id": task.task_id,
            "building_sha256": task.building.sha256,
            "weather_sha256": task.weather.sha256,
        },
        "contract": {
            "enabled": task.warmup_preconditioning.enabled,
            "payload": task.warmup_preconditioning.identity_payload(),
            "sha256": task.warmup_preconditioning.sha256,
        },
        "expected_episode_count": len(expected_episode_keys),
        "record_count": len(records),
        "records": [dict(record) for record in records],
    }


def _evaluation_evidence_manifest(
    *,
    run: ResolvedRun,
    task: TaskSpec,
    policy_provenance: PolicyProvenance,
    reported_carriers: set[str],
    energy_meter_sources: Mapping[str, str],
    controlled_floor_area_m2: float | None,
    diagnostics_policy_ref: Mapping[str, str] | None,
    warmup_preconditioning_ref: Mapping[str, str],
    training_bundle_ref: Mapping[str, str] | None,
    stochastic_diagnostic_seed: int | None = None,
) -> dict[str, Any]:
    if set(energy_meter_sources) != reported_carriers:
        raise EvaluationError("energy meter source carriers do not match the reported energy ledger")
    uses_occupied_reward = task.reward.reward_id in (HOT_OCCUPIED_ENERGY_COMFORT_V2, HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3)
    manifest = {
        "schema_version": (
            EVALUATION_EVIDENCE_SCHEMA_VERSION
            if training_bundle_ref is not None
            else LEGACY_EVALUATION_EVIDENCE_SCHEMA_VERSION
        ),
        "run_id": run.run_id,
        "trajectory_schema_version": STEP_SCHEMA_VERSION,
        "diagnostics_acceptance_policy": (None if diagnostics_policy_ref is None else dict(diagnostics_policy_ref)),
        "warmup_preconditioning_evidence": dict(warmup_preconditioning_ref),
        "policy": policy_provenance.to_dict(),
        "energy": {
            "metric_prefix": "controlled_hvac_energy",
            "scope": "controlled HVAC site energy only",
            "unit": "kWh",
            "interval_source": "EnergyPlus Zone Timestep meter values converted once from joules",
            "carrier_meter_names": {carrier: energy_meter_sources[carrier] for carrier in sorted(reported_carriers)},
            "excluded": "non-HVAC building loads, source energy, and carbon emissions",
            "controlled_floor_area_m2": controlled_floor_area_m2,
            "floor_area_source": (f"{task.reward.reward_id} reward components" if uses_occupied_reward else None),
        },
        "comfort": {
            "primary_scope": ("occupied, person-weighted squared temperature-band violation" if uses_occupied_reward else None),
            "primary_source": (f"{task.reward.reward_id} reward components" if uses_occupied_reward else None),
            **({"scheduled_comfort": task.reward.parameter_mapping["comfort_schedule"],
                "legacy_reference_band_c": list(_comfort_band(task))}
               if task.reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3 else {}),
            "legacy_diagnostics": (
                "24-hour unweighted zone-temperature violations are retained only in " "metrics prefixed legacy_24h_"
            ),
        },
        "observations": {
            "policy_observation_normalized_json": (
                "normalized state presented to the policy before the recorded action"
            ),
            "next_raw_observation_json": ("physical simulator state returned after the submitted action"),
            "policy_observation_encoding": task.observation_space.encoding,
        },
        "actions": {
            "proposal": "normalized policy output before environment transformation",
            "proposal_channel_order": [
                f"{entity}:{channel.name}"
                for entity in task.action_profile.controlled_entities
                for channel in task.action_profile.channels
            ],
            "proposal_units": [
                channel.unit
                for _entity in task.action_profile.controlled_entities
                for channel in task.action_profile.channels
            ],
            "transform_id": task.action_profile.transform_id,
            "submitted_schema_version": SUBMITTED_ACTION_SCHEMA_VERSION,
            "submitted_channel_order": [
                f"{entity}:{name}"
                for entity in task.action_profile.controlled_entities
                for name in ("heating_setpoint_c", "cooling_setpoint_c")
            ],
            "submitted_units": [
                "degC"
                for _entity in task.action_profile.controlled_entities
                for _name in ("heating_setpoint_c", "cooling_setpoint_c")
            ],
            "submitted_semantics": ("setpoints transformed by the environment and submitted to EnergyPlus"),
            "actuator_readback": "availability_and_values_recorded_per_step",
        },
    }
    if stochastic_diagnostic_seed is not None:
        manifest['stochastic_diagnostic']={'sampling_seed':stochastic_diagnostic_seed,'scope':'Engineering Gaussian-policy sampling only; excluded from deterministic checkpoint selection and source qualification'}
    if policy_provenance.policy_id == "native_incumbent":
        manifest["actions"] = {"mode": "native_schedule", "override_operation": "reset_actuator",
                               "controlled_entities": list(task.action_profile.controlled_entities),
                               "policy_actions_submitted": False, "actuator_readback": "unavailable"}
        manifest["observations"]["next_raw_observation_json"] = "physical state after native-schedule interval"
    if training_bundle_ref is not None:
        manifest["training_bundle"] = dict(training_bundle_ref)
    return manifest


def _unavailable_diagnostics_record(
    rollout: RolloutContext,
    gate: DiagnosticsGate,
    error: str,
) -> dict[str, Any]:
    if gate.scientific:
        raise EvaluationError(error)
    return {
        "window": rollout.window.name,
        "episode_index": rollout.episode_index,
        "status": "unavailable",
        "acceptance": {"accepted": None, "violations": []},
        "diagnostics": None,
        "error": error,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _same_path(first: Path, second: Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""

    try:
        return first.resolve(strict=False) == second.resolve(strict=False)
    except OSError as error:
        raise EvaluationError(f"cannot resolve policy checkpoint path: {error}") from error


def _sha256_file(path: Path, *, role: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise EvaluationError(f"cannot read {role} {path}: {error}") from error
    return digest.hexdigest()


def _atomic_copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    role: str = "policy checkpoint",
) -> str:
    """Atomically bundle immutable source bytes under their declared digest."""

    source_digest_before = _sha256_file(source, role=f"{role} source")
    if source_digest_before != expected_sha256:
        raise EvaluationError(f"{role} source checksum does not match its declared digest")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    copied_digest = hashlib.sha256()
    installed = False
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as destination_handle:
            descriptor = -1
            while chunk := source_handle.read(1024 * 1024):
                destination_handle.write(chunk)
                copied_digest.update(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())

        source_digest_after = _sha256_file(source, role=f"{role} source")
        if source_digest_after != source_digest_before:
            raise EvaluationError(f"{role} bytes changed while they were being bundled")
        if copied_digest.hexdigest() != expected_sha256:
            raise EvaluationError(f"bundled {role} checksum does not match its declared digest")
        os.replace(temporary_name, destination)
        installed = True
        bundled_digest = _sha256_file(destination, role=f"bundled {role}")
        if bundled_digest != expected_sha256:
            destination.unlink(missing_ok=True)
            raise EvaluationError(f"bundled {role} checksum changed after atomic copy")
        return bundled_digest
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if installed:
            destination.unlink(missing_ok=True)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _remove_exact_artifacts(paths: Sequence[Path]) -> None:
    """Remove only the named evaluator outputs, never their parent directory."""

    directories = [path for path in paths if path.is_dir() and not path.is_symlink()]
    if directories:
        raise IsADirectoryError(
            "refusing to replace evaluation artifact path(s) that are directories: " + ", ".join(map(str, directories))
        )
    for path in paths:
        path.unlink(missing_ok=True)


def _comfort_band(task: TaskSpec) -> tuple[float, float]:
    parameters = task.reward.parameter_mapping
    if task.reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3:
        # Legacy diagnostics retain a fixed reference, not the scheduled objective.
        config = ScheduledOccupiedRewardConfig.from_mapping(parameters)
        return config.comfort_schedule.default.minimum_c, config.comfort_schedule.default.maximum_c
    try:
        minimum = _finite_float(parameters["comfort_min_c"], name="comfort_min_c")
        maximum = _finite_float(parameters["comfort_max_c"], name="comfort_max_c")
    except KeyError as error:
        raise EvaluationError("task reward parameters must declare comfort_min_c and comfort_max_c") from error
    if minimum >= maximum:
        raise EvaluationError("comfort_min_c must be smaller than comfort_max_c")
    return minimum, maximum


def _validate_observation(
    observation: Any,
    *,
    task: TaskSpec,
    name: str,
) -> tuple[float, ...]:
    values = _flatten(observation)
    if not values:
        raise EvaluationError(f"{name} cannot be empty")
    if len(values) != task.observation_space.flat_size:
        raise EvaluationError(
            f"{name} contains {len(values)} values; task requires " f"{task.observation_space.flat_size}"
        )
    if task.observation_space.lower_bound != 0.0 or task.observation_space.upper_bound != 1.0:
        raise EvaluationError("publication evaluation requires normalized observation bounds [0, 1]")
    result = tuple(_finite_float(value, name=f"{name}[{index}]") for index, value in enumerate(values))
    for value in result:
        if not 0.0 <= value <= 1.0:
            raise EvaluationError(f"{name} value {value} lies outside normalized bounds [0, 1]")
    return result


def _raw_observation(info: Mapping[str, Any], *, where: str) -> dict[str, float]:
    if not isinstance(info, Mapping):
        raise EvaluationError(f"{where} info must be a mapping")
    raw = info.get("raw_observation")
    if not isinstance(raw, Mapping) or not raw:
        raise EvaluationError(f"{where} is missing a non-empty raw_observation mapping")
    result: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise EvaluationError(f"{where} raw_observation keys must be non-empty strings")
        result[key] = _finite_float(value, name=f"{where} raw_observation[{key!r}]")
    return result


def _controlled_temperature_keys(
    raw: Mapping[str, float],
    *,
    controlled_entities: Sequence[str],
) -> tuple[str, ...]:
    """Return only raw temperature channels for entities controlled by the task."""

    keys = tuple(f"{entity}_temperature" for entity in controlled_entities)
    if not keys:
        raise EvaluationError("task action profile contains no controlled entities")
    missing = [key for key in keys if key not in raw]
    if missing:
        raise EvaluationError(
            "raw observation is missing controlled-zone temperature field(s): "
            + ", ".join(repr(key) for key in missing)
        )
    return keys


def _comfort_metrics(
    raw: Mapping[str, float],
    zone_keys: Sequence[str],
    *,
    comfort_min: float,
    comfort_max: float,
    interval_hours: float,
) -> tuple[float, float, bool]:
    """Compute legacy comfort directly from authenticated raw temperatures."""

    violations = []
    for key in zone_keys:
        temperature = raw[key]
        violations.append(max(comfort_min - temperature, temperature - comfort_max, 0.0))
    return (
        sum(violations) * interval_hours,
        len(violations) * interval_hours,
        all(value == 0.0 for value in violations),
    )


def _energy_ledger(
    info: Mapping[str, Any],
    *,
    where: str,
    expected_meter_sources: Mapping[str, str] | None = None,
) -> tuple[dict[str, float | None], set[str], dict[str, str]]:
    if not isinstance(info, Mapping):
        raise EvaluationError(f"{where} info must be a mapping")
    raw = info.get("energy_ledger_kwh")
    if not isinstance(raw, Mapping):
        raise EvaluationError(f"{where} is missing energy_ledger_kwh")
    allowed = {carrier.value for carrier in EnergyCarrier}
    if set(raw) != allowed:
        missing = sorted(allowed - set(raw))
        unknown = sorted(set(raw) - allowed)
        raise EvaluationError(f"{where} energy ledger schema mismatch: missing={missing}, unknown={unknown}")
    ledger: dict[str, float | None] = {}
    reported: set[str] = set()
    for carrier in sorted(allowed):
        value = raw[carrier]
        if value is None:
            ledger[carrier] = None
            continue
        number = _finite_float(value, name=f"{where} {carrier} energy")
        if number < 0.0:
            raise EvaluationError(f"{where} {carrier} energy cannot be negative")
        ledger[carrier] = number
        reported.add(carrier)
    if not reported:
        raise EvaluationError(f"{where} reports no energy carrier")
    raw_sources = info.get("energy_meter_sources")
    if not isinstance(raw_sources, Mapping):
        raise EvaluationError(f"{where} is missing energy_meter_sources")
    if set(raw_sources) != reported:
        missing = sorted(reported - set(raw_sources))
        unknown = sorted(set(raw_sources) - reported)
        raise EvaluationError(f"{where} energy meter source schema mismatch: missing={missing}, unknown={unknown}")
    canonical_sources = {carrier.value: meter_name for carrier, meter_name in CONTROLLED_HVAC_METER_BY_CARRIER.items()}
    if expected_meter_sources is not None:
        canonical_sources = dict(expected_meter_sources)
        if reported != set(canonical_sources):
            raise EvaluationError(f'{where} must report every declared component energy carrier')
    unsupported = reported - set(canonical_sources)
    if unsupported:
        raise EvaluationError(
            f"{where} reports carrier(s) outside the controlled-HVAC meter contract: " + ", ".join(sorted(unsupported))
        )
    sources: dict[str, str] = {}
    for carrier in sorted(reported):
        meter_name = raw_sources[carrier]
        expected_meter_name = canonical_sources[carrier]
        if meter_name != expected_meter_name:
            raise EvaluationError(
                f"{where} meter source for {carrier!r} must be the controlled-HVAC meter "
                f"{expected_meter_name!r}; got {meter_name!r}"
            )
        sources[carrier] = expected_meter_name
    return ledger, reported, sources


def _clock(
    info: Mapping[str, Any],
    *,
    where: str,
    expected_calendar_year: int | None = None,
    expected_calendar_year_semantics: str | None = None,
    allow_next_year_boundary: bool = False,
) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        raise EvaluationError(f"{where} info must be a mapping")
    clock = info.get("simulation_clock")
    required = {
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
    }
    if not isinstance(clock, Mapping) or set(clock) != required:
        raise EvaluationError(f"{where} simulation_clock must contain the exact clock schema")
    result = dict(clock)
    for key in required - {"calendar_year_semantics"}:
        _integer(result[key], name=f"{where} simulation_clock[{key!r}]")
    semantics = result["calendar_year_semantics"]
    if semantics not in {"actual_weather_year", "typical_weather_year", "unknown"}:
        raise EvaluationError(f"{where} has invalid calendar_year_semantics {semantics!r}")
    calendar_year = _integer(
        result["calendar_year"],
        name=f"{where} simulation_clock['calendar_year']",
    )
    if calendar_year <= 0:
        raise EvaluationError(f"{where} simulation clock calendar_year must be positive")
    if expected_calendar_year is not None and calendar_year != expected_calendar_year:
        is_allowed_next_year_boundary = (
            allow_next_year_boundary
            and calendar_year == expected_calendar_year + 1
            and (
                _integer(result["month"], name=f"{where} simulation_clock['month']"),
                _integer(result["day_of_month"], name=f"{where} simulation_clock['day_of_month']"),
                _integer(result["hour"], name=f"{where} simulation_clock['hour']"),
                _integer(result["timestep_index"], name=f"{where} simulation_clock['timestep_index']"),
                _integer(result["day_of_year"], name=f"{where} simulation_clock['day_of_year']"),
            )
            == (1, 1, 0, 1, 1)
        )
        if not is_allowed_next_year_boundary:
            raise EvaluationError(
                f"{where} simulation clock calendar_year {calendar_year} does not match "
                f"TaskSpec calendar_year {expected_calendar_year}"
            )
    if expected_calendar_year_semantics is not None and semantics != expected_calendar_year_semantics:
        raise EvaluationError(
            f"{where} simulation clock calendar_year_semantics {semantics!r} does not match "
            f"TaskSpec-derived semantics {expected_calendar_year_semantics!r}"
        )
    return result


def _clock_key(clock: Mapping[str, Any], *, timesteps_per_hour: int) -> tuple[int, int, int, int]:
    calendar_year = _integer(clock["calendar_year"], name="clock calendar_year")
    month = _integer(clock["month"], name="clock month")
    day = _integer(clock["day_of_month"], name="clock day")
    hour = _integer(clock["hour"], name="clock hour")
    timestep = _integer(clock["timestep_index"], name="clock timestep_index")
    reported_tph = _integer(clock["timesteps_per_hour"], name="clock timesteps_per_hour")
    if reported_tph != timesteps_per_hour:
        raise EvaluationError(f"clock reports {reported_tph} timesteps/hour, task declares {timesteps_per_hour}")
    try:
        date(calendar_year, month, day)
    except ValueError as error:
        raise EvaluationError(f"invalid clock date {month}/{day}") from error
    if not 0 <= hour <= 23:
        raise EvaluationError(f"clock hour must lie in [0, 23], got {hour}")
    if not 1 <= timestep <= timesteps_per_hour:
        raise EvaluationError(f"clock timestep_index must lie in [1, {timesteps_per_hour}], got {timestep}")
    return month, day, hour, timestep


def _validate_first_decision_clock_calendar(
    clock: Mapping[str, Any],
    *,
    task: TaskSpec,
    window: CalendarWindowSpec,
) -> None:
    """Bind policy calendar inputs to the task's pinned EPW anchor."""

    if not task.warmup_preconditioning.enabled:
        return
    expected_day_of_week, expected_day_of_year = expected_warmup_clock_anchor(
        contract=task.warmup_preconditioning,
        window=window,
        calendar_year=task.calendar_year,
    )
    actual_day_of_week = _integer(
        clock["day_of_week"],
        name="first decision clock day_of_week",
    )
    actual_day_of_year = _integer(
        clock["day_of_year"],
        name="first decision clock day_of_year",
    )
    if actual_day_of_week != expected_day_of_week:
        raise EvaluationError(
            "first decision clock day_of_week does not match the identity-bearing EPW calendar anchor: "
            f"expected {expected_day_of_week}, found {actual_day_of_week}"
        )
    if actual_day_of_year != expected_day_of_year:
        raise EvaluationError(
            "first decision clock day_of_year does not match the calendar window: "
            f"expected {expected_day_of_year}, found {actual_day_of_year}"
        )


def _expected_clock_sequence(
    window: CalendarWindowSpec,
    timesteps_per_hour: int,
    *,
    calendar_year: int | None,
) -> tuple[tuple[int, int, int, int], ...]:
    keys: list[tuple[int, int, int, int]] = []
    for current in window.dates_for_calendar_year(calendar_year):
        for hour in range(24):
            for timestep in range(1, timesteps_per_hour + 1):
                keys.append((current.month, current.day, hour, timestep))
    return tuple(keys)


def _next_clock_key(
    key: tuple[int, int, int, int],
    *,
    timesteps_per_hour: int,
    calendar_year: int,
) -> tuple[int, int, int, int, int]:
    month, day, hour, timestep = key
    if timestep < timesteps_per_hour:
        return calendar_year, month, day, hour, timestep + 1
    if hour < 23:
        return calendar_year, month, day, hour + 1, 1
    following_day = date(calendar_year, month, day) + timedelta(days=1)
    return following_day.year, following_day.month, following_day.day, 0, 1


def _validate_transition_clocks(
    decision_clock: Mapping[str, Any],
    next_clock: Mapping[str, Any],
    *,
    expected_decision_clock: tuple[int, int, int, int],
    step_index: int,
    timesteps_per_hour: int,
) -> None:
    decision_key = _clock_key(decision_clock, timesteps_per_hour=timesteps_per_hour)
    next_key = _clock_key(next_clock, timesteps_per_hour=timesteps_per_hour)
    if decision_key != expected_decision_clock:
        raise EvaluationError(
            f"step {step_index} decision clock {decision_key!r} does not match "
            f"the expected chronological clock {expected_decision_clock!r}"
        )
    expected_next = _next_clock_key(
        decision_key,
        timesteps_per_hour=timesteps_per_hour,
        calendar_year=_integer(
            decision_clock["calendar_year"],
            name="decision clock calendar_year",
        ),
    )
    observed_next = (
        _integer(next_clock["calendar_year"], name="next-state clock calendar_year"),
        *next_key,
    )
    if observed_next != expected_next:
        raise EvaluationError(
            f"step {step_index} next-state clock {observed_next!r} does not immediately "
            f"follow decision clock {decision_key!r}"
        )
    decision_step = _integer(
        decision_clock["episode_step_index"],
        name="decision clock episode_step_index",
    )
    next_step = _integer(
        next_clock["episode_step_index"],
        name="next-state clock episode_step_index",
    )
    if decision_step != step_index or next_step != step_index + 1:
        raise EvaluationError(
            f"step {step_index} clock episode indices must be {step_index} and "
            f"{step_index + 1}, got {decision_step} and {next_step}"
        )


def _flat_finite_action(action: Any, task: TaskSpec) -> tuple[float, ...]:
    values = tuple(_finite_float(value, name=f"action[{index}]") for index, value in enumerate(_flatten(action)))
    if len(values) != task.action_space.flat_size:
        raise EvaluationError(
            f"policy returned {len(values)} action values; task requires {task.action_space.flat_size}"
        )
    if task.action_space.lower_bound is not None:
        assert task.action_space.upper_bound is not None
        for value in values:
            if not task.action_space.lower_bound <= value <= task.action_space.upper_bound:
                raise EvaluationError(
                    f"action {value} lies outside [{task.action_space.lower_bound}, "
                    f"{task.action_space.upper_bound}]"
                )
    return values


def _environment_action(proposed: Any, values: tuple[float, ...], env: GymEnvironment) -> Any:
    """Preserve learned-policy arrays and lazily create arrays for Gym Box spaces."""

    if hasattr(proposed, "shape"):
        return proposed
    action_space = getattr(env, "action_space", None)
    dtype = getattr(action_space, "dtype", None)
    if dtype is None:
        return values
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - only a malformed Gym runtime reaches this
        raise EvaluationError("a Gym Box action space requires NumPy") from error
    shape = getattr(action_space, "shape", (len(values),))
    return np.asarray(values, dtype=dtype).reshape(shape)


def _native_action_evidence(info: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"schema_version": "hot-native-schedule-control/1.0.0",
                "thermostat_overrides": "released", "submitted_policy_setpoints": False,
                "actuator_readback_available": False}
    if info.get("native_control") != expected or "submitted_action_physical" in info:
        raise EvaluationError("native incumbent requires explicit release evidence and no submitted setpoints")
    from building_gen.benchmark.native_demonstrations import native_setpoint_evidence
    return {"control_mode": "native_schedule", "native_control_json": json.dumps(expected, sort_keys=True),
            "native_thermostat_targets_json": json.dumps(native_setpoint_evidence(info), sort_keys=True, allow_nan=False),
            "proposed_action_normalized_json": "[]", "proposed_action_channel_names_json": "[]",
            "proposed_action_units_json": "[]", "action_transform_id": "not_applied",
            "submitted_action_schema_version": "hot-native-schedule-control/1.0.0",
            "submitted_action_physical_json": "[]", "submitted_action_channel_names_json": "[]",
            "submitted_action_units_json": "[]", "actuator_readback_status": "unavailable",
            "actuator_readback_reason": "Native schedules retain control; no policy setpoints were submitted.",
            "actuator_readback_phase": "", "actuator_readback_physical_json": "[]",
            "actuator_readback_max_abs_error_c": ""}


def _submitted_action_evidence(
    info: Mapping[str, Any],
    *,
    task: TaskSpec,
    proposed_action: tuple[float, ...],
    where: str,
) -> dict[str, Any]:
    """Validate and serialize proposed and submitted action representations."""

    if not isinstance(info, Mapping):
        raise EvaluationError(f"{where} info must be a mapping")
    if info.get("submitted_action_schema_version") != SUBMITTED_ACTION_SCHEMA_VERSION:
        raise EvaluationError(f"{where} is missing the supported submitted-action schema")
    records = info.get("submitted_action_physical")
    if not isinstance(records, list) or not records:
        raise EvaluationError(f"{where} is missing submitted_action_physical")
    readback = info.get("actuator_readback")
    if not isinstance(readback, Mapping):
        raise EvaluationError(f"{where} actuator_readback must be a mapping")
    available = readback.get('available')
    reason = ''
    if available is True:
        if set(readback) != {'available', 'phase', 'records'} or readback['phase'] != 'end_zone_timestep':
            raise EvaluationError(f'{where} invalid actuator read-back phase or fields')
    elif available is False:
        reason = readback.get('reason')
        if set(readback) != {'available', 'reason'} or not isinstance(reason, str) or not reason.strip():
            raise EvaluationError(f"{where} unavailable actuator read-back requires a reason")
    else:
        raise EvaluationError(f'{where} actuator read-back availability must be boolean')

    entities = task.action_profile.controlled_entities
    channels = task.action_profile.channels
    if tuple(channel.name for channel in channels) != (
        "heating_setpoint_c",
        "cooling_offset_c",
    ):
        raise EvaluationError(
            "submitted-action validation currently supports the HOT heating-setpoint/cooling-offset contract"
        )
    if len(records) != len(entities):
        raise EvaluationError(f"{where} submitted-action entity count does not match TaskSpec")

    canonical_records: list[dict[str, float | str]] = []
    flat_index = 0
    for entity, raw_record in zip(entities, records):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "zone",
            "heating_setpoint_c",
            "cooling_setpoint_c",
        }:
            raise EvaluationError(f"{where} submitted-action record has an invalid schema")
        if raw_record.get("zone") != entity:
            raise EvaluationError(f"{where} submitted-action entity order does not match TaskSpec")
        heating = _finite_float(
            raw_record["heating_setpoint_c"],
            name=f"{where} submitted heating setpoint",
        )
        cooling = _finite_float(
            raw_record["cooling_setpoint_c"],
            name=f"{where} submitted cooling setpoint",
        )
        heating_channel, offset_channel = channels
        expected_heating = heating_channel.physical_min + proposed_action[flat_index] * (
            heating_channel.physical_max - heating_channel.physical_min
        )
        expected_offset = offset_channel.physical_min + proposed_action[flat_index + 1] * (
            offset_channel.physical_max - offset_channel.physical_min
        )
        if task.action_profile.transform_id == "linear_quantized_0_1c_legacy":
            expected_heating = round(expected_heating, 1)
            expected_offset = round(expected_offset, 1)
        elif task.action_profile.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
            from building_gen.bounded_thermostat import validate_ranges
            validate_ranges((heating_channel.physical_min,heating_channel.physical_max),(offset_channel.physical_min,offset_channel.physical_max),task.action_profile.transform_id)
            expected_offset=2.+proposed_action[flat_index+1]*((32. if task.action_profile.transform_id == "bounded_thermostat_12_32_v1" else 30.)-expected_heating-2.)
        elif task.action_profile.transform_id != "linear_unquantized_v1":
            raise EvaluationError(f"unsupported action transform {task.action_profile.transform_id!r}")
        expected_cooling = expected_heating + expected_offset
        if not math.isclose(heating, expected_heating, rel_tol=1e-7, abs_tol=1e-6):
            raise EvaluationError(f"{where} submitted heating setpoint does not match the declared transform")
        if not math.isclose(cooling, expected_cooling, rel_tol=1e-7, abs_tol=1e-6):
            raise EvaluationError(f"{where} submitted cooling setpoint does not match the declared transform")
        canonical_records.append(
            {
                "zone": entity,
                "heating_setpoint_c": heating,
                "cooling_setpoint_c": cooling,
            }
        )
        flat_index += 2

    readback_records = []
    readback_error = ''
    if available:
        observed = readback['records']
        if not isinstance(observed, list) or len(observed) != len(canonical_records):
            raise EvaluationError(f'{where} actuator read-back must cover every controlled zone')
        errors = []
        for expected, actual in zip(canonical_records, observed):
            if not isinstance(actual, Mapping) or set(actual) != set(expected) or actual['zone'] != expected['zone']:
                raise EvaluationError(f'{where} actuator read-back zone order or fields mismatch')
            normalized = {'zone': actual['zone']}
            for key in ('heating_setpoint_c', 'cooling_setpoint_c'):
                value = actual[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise EvaluationError(f'{where} actuator read-back must be finite')
                normalized[key] = float(value)
                errors.append(abs(value - expected[key]))
            readback_records.append(normalized)
        readback_error = max(errors)

    proposed_names = [f"{entity}:{channel.name}" for entity in entities for channel in channels]
    proposed_units = [channel.unit for _entity in entities for channel in channels]
    submitted_names = [
        f"{entity}:{name}" for entity in entities for name in ("heating_setpoint_c", "cooling_setpoint_c")
    ]
    return {
        "proposed_action_normalized_json": json.dumps(
            proposed_action,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "proposed_action_channel_names_json": json.dumps(
            proposed_names,
            separators=(",", ":"),
        ),
        "proposed_action_units_json": json.dumps(proposed_units, separators=(",", ":")),
        "action_transform_id": task.action_profile.transform_id,
        "submitted_action_schema_version": SUBMITTED_ACTION_SCHEMA_VERSION,
        "submitted_action_physical_json": json.dumps(
            canonical_records,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "submitted_action_channel_names_json": json.dumps(
            submitted_names,
            separators=(",", ":"),
        ),
        "submitted_action_units_json": json.dumps(
            ["degC"] * len(submitted_names),
            separators=(",", ":"),
        ),
        "actuator_readback_status": "available" if available else "unavailable",
        "actuator_readback_reason": reason.strip(),
        "actuator_readback_phase": readback.get('phase', ''),
        "actuator_readback_physical_json": json.dumps(readback_records, separators=(',', ':'), allow_nan=False),
        "actuator_readback_max_abs_error_c": readback_error,
    }


def _step_row(
    *,
    run: ResolvedRun,
    task: TaskSpec,
    protocol: EvaluationProtocolSpec,
    rollout: RolloutContext,
    step_index: int,
    reward: float,
    reward_components_json: str,
    action_evidence: Mapping[str, Any],
    policy_observation: tuple[float, ...],
    decision_clock: Mapping[str, Any],
    clock: Mapping[str, Any],
    raw_observation: Mapping[str, float],
    ledger: Mapping[str, float | None],
    interval_hours: float,
    legacy_comfort_degree_hours: float,
    legacy_zone_hours: float,
    legacy_all_zones_comfort: bool,
    occupied_comfort_exposure_intensity: float | None,
    occupied_person_hours: float | None,
    controlled_floor_area_m2: float | None,
    terminated: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": STEP_SCHEMA_VERSION,
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "algorithm_id": run.algorithm_id,
        "algorithm_version": run.algorithm_version,
        "seed": run.seed,
        "task_id": task.task_id,
        "window": rollout.window.name,
        "window_start_month": rollout.window.start_month,
        "window_start_day": rollout.window.start_day,
        "window_end_month": rollout.window.end_month,
        "window_end_day": rollout.window.end_day,
        "episode_index": rollout.episode_index,
        "episodes_per_window": protocol.episodes_per_window,
        "step_index": step_index,
        "expected_steps": rollout.expected_steps,
        "timesteps_per_hour": task.episode.timesteps_per_hour,
        "interval_hours": interval_hours,
        "reward": reward,
        "reward_components_json": reward_components_json,
        "legacy_24h_comfort_violation_degree_hours": legacy_comfort_degree_hours,
        "legacy_24h_zone_hours": legacy_zone_hours,
        "legacy_24h_all_zones_comfort": int(legacy_all_zones_comfort),
        "occupied_comfort_exposure_person_celsius_squared_hours_per_m2": (
            "" if occupied_comfort_exposure_intensity is None else occupied_comfort_exposure_intensity
        ),
        "occupied_person_hours": "" if occupied_person_hours is None else occupied_person_hours,
        "controlled_floor_area_m2": "" if controlled_floor_area_m2 is None else controlled_floor_area_m2,
        "terminated": int(terminated),
        **action_evidence,
        "decision_clock_json": json.dumps(dict(decision_clock), sort_keys=True, separators=(",", ":"), allow_nan=False),
        "policy_observation_normalized_json": json.dumps(
            policy_observation,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "next_raw_observation_json": json.dumps(
            dict(raw_observation), sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    }
    for key, value in decision_clock.items():
        row[f"decision_clock_{key}"] = value
    for key, value in clock.items():
        row[f"clock_{key}"] = value
    for carrier in sorted(carrier.value for carrier in EnergyCarrier):
        row[f"controlled_hvac_energy_{carrier}_kwh"] = "" if ledger[carrier] is None else ledger[carrier]
    return row


def _reward_components_json(
    info: Mapping[str, Any],
    *,
    task: TaskSpec,
    scalar_reward: float,
    ledger: Mapping[str, float | None],
    interval_hours: float,
    where: str,
    zone_floor_areas_m2: Mapping[str, float] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return canonical components after verifying the versioned reward arithmetic."""

    if task.reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3:
        if zone_floor_areas_m2 is None:
            raise EvaluationError(f"{where} scheduled reward requires model-derived zone areas")
        try:
            canonical = verify_scheduled_interval(
                info, scalar_reward=scalar_reward,
                interval_controlled_hvac_site_energy_kwh=math.fsum(value for value in ledger.values() if value is not None),
                zone_floor_areas_m2=zone_floor_areas_m2,
                controlled_zones=tuple(task.action_profile.controlled_entities),
                timesteps_per_hour=task.episode.timesteps_per_hour,
                config=ScheduledOccupiedRewardConfig.from_mapping(task.reward.parameter_mapping),
            )
            _require_close(canonical["timestep_hours"], interval_hours, where=f"{where} scheduled interval")
        except (TypeError, ValueError, KeyError) as error:
            raise EvaluationError(f"{where} invalid scheduled reward evidence: {error}") from error
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False), canonical
    raw = info.get("reward_components")
    is_reward_v2 = task.reward.reward_id == HOT_OCCUPIED_ENERGY_COMFORT_V2
    if raw is None:
        if is_reward_v2:
            raise EvaluationError(f"{where} is missing reward_components for reward v2")
        return "", None
    if not isinstance(raw, Mapping):
        raise EvaluationError(f"{where} reward_components must be a mapping")

    expected_fields = tuple(field.name for field in fields(OccupiedEnergyComfortV2Components))
    if is_reward_v2:
        missing = sorted(set(expected_fields) - set(raw))
        extra = sorted(set(raw) - set(expected_fields))
        if missing or extra:
            raise EvaluationError(
                f"{where} reward_components do not match the reward-v2 schema: " f"missing={missing}, extra={extra}"
            )
        keys = expected_fields
    else:
        keys = tuple(sorted(raw))

    canonical = {key: _finite_float(raw[key], name=f"{where} reward_components.{key}") for key in keys}
    if "reward" in canonical and not math.isclose(canonical["reward"], scalar_reward, rel_tol=1e-12, abs_tol=1e-12):
        raise EvaluationError(f"{where} reward_components.reward does not match scalar reward")
    if is_reward_v2:
        parameters = task.reward.parameter_mapping
        required_parameters = {
            "energy_weight",
            "comfort_weight",
            "energy_scale_kwh_per_m2",
            "comfort_scale_person_celsius_squared_hours_per_m2",
        }
        missing_parameters = sorted(required_parameters - set(parameters))
        if missing_parameters:
            raise EvaluationError(f"{where} reward-v2 TaskSpec is missing parameters {missing_parameters}")
        energy_weight = _finite_float(parameters["energy_weight"], name="energy_weight")
        comfort_weight = _finite_float(parameters["comfort_weight"], name="comfort_weight")
        energy_scale = _finite_float(
            parameters["energy_scale_kwh_per_m2"],
            name="energy_scale_kwh_per_m2",
        )
        comfort_scale = _finite_float(
            parameters["comfort_scale_person_celsius_squared_hours_per_m2"],
            name="comfort_scale_person_celsius_squared_hours_per_m2",
        )
        if energy_weight < 0.0 or comfort_weight < 0.0:
            raise EvaluationError(f"{where} reward-v2 weights must be non-negative")
        if energy_scale <= 0.0 or comfort_scale <= 0.0:
            raise EvaluationError(f"{where} reward-v2 normalization scales must be positive")

        area = canonical["controlled_floor_area_m2"]
        occupants = canonical["total_occupant_count"]
        comfort_exposure = canonical["comfort_exposure_person_celsius_squared_hours_per_m2"]
        if area <= 0.0:
            raise EvaluationError(f"{where} reward-v2 controlled floor area must be positive")
        if occupants < 0.0 or comfort_exposure < 0.0:
            raise EvaluationError(f"{where} reward-v2 occupancy and comfort exposure must be non-negative")
        if occupants == 0.0 and comfort_exposure != 0.0:
            raise EvaluationError(f"{where} reward-v2 comfort exposure must be zero when occupancy is zero")
        _require_close(
            canonical["timestep_hours"],
            interval_hours,
            where=f"{where} reward-v2 timestep_hours",
        )
        interval_energy = math.fsum(value for value in ledger.values() if value is not None)
        _require_close(
            canonical["interval_controlled_hvac_site_energy_kwh"],
            interval_energy,
            where=f"{where} reward-v2 interval energy",
        )
        energy_intensity = interval_energy / area
        _require_close(
            canonical["controlled_hvac_energy_intensity_kwh_per_m2"],
            energy_intensity,
            where=f"{where} reward-v2 energy intensity",
        )
        normalized_energy = energy_intensity / energy_scale
        normalized_comfort = comfort_exposure / comfort_scale
        _require_close(
            canonical["normalized_energy_penalty"],
            normalized_energy,
            where=f"{where} reward-v2 normalized energy penalty",
        )
        _require_close(
            canonical["normalized_comfort_penalty"],
            normalized_comfort,
            where=f"{where} reward-v2 normalized comfort penalty",
        )
        weighted_energy = energy_weight * normalized_energy
        weighted_comfort = comfort_weight * normalized_comfort
        _require_close(
            canonical["weighted_energy_penalty"],
            weighted_energy,
            where=f"{where} reward-v2 weighted energy penalty",
        )
        _require_close(
            canonical["weighted_comfort_penalty"],
            weighted_comfort,
            where=f"{where} reward-v2 weighted comfort penalty",
        )
        reconstructed = -(weighted_energy + weighted_comfort)
        _require_close(
            scalar_reward,
            reconstructed,
            where=f"{where} reward-v2 scalar reward",
        )
    return (
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False),
        canonical,
    )


def _require_close(actual: float, expected: float, *, where: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise EvaluationError(f"{where} does not match the TaskSpec-derived value")


class _CsvSpool:
    """Bound trajectory memory while retaining CSV bytes and atomic publication."""
    def __enter__(self):
        self.stream = tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="")
        self.writer = None
        self.count = 0
        return self

    def __exit__(self, *exc):
        self.stream.close()

    def __len__(self):
        return self.count

    def append(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count += 1

    def publish(self, path):
        if not self.count:
            raise EvaluationError("cannot serialize an empty step table")
        self.stream.flush()
        self.stream.seek(0)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                while chunk := self.stream.read(1024 * 1024):
                    raw = chunk.encode("utf-8")
                    digest.update(raw)
                    handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return digest.hexdigest()


def _rows_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise EvaluationError("cannot serialize an empty step table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _flatten(value: Any) -> tuple[Any, ...]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise EvaluationError("expected a numeric array, not a mapping or string")
    if isinstance(value, Sequence):
        flattened: list[Any] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                flattened.extend(_flatten(item))
            else:
                flattened.append(item)
        return tuple(flattened)
    return (value,)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise EvaluationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise EvaluationError(f"{name} must be finite")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{name} must be an integer")
    return value


__all__ = [
    "DecisionContext",
    "EnvironmentFactory",
    "EvaluationError",
    "EvaluationOutput",
    "EvaluationPolicy",
    "GymEnvironment",
    "Policy",
    "PolicyProvenance",
    "RolloutContext",
    "ashrae_seasonal_policy",
    "evaluate_run",
    "learned_callable_policy",
    "ppo_callable_policy",
    "static_midpoint_policy",
]
