"""
EnergyPlus Interface using template-based approach with your exact state and action spaces
"""

import hashlib
import json
import logging
import math
import os
import re
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple, Union

from building_gen.accounting import (
    CONTROLLED_HVAC_METER_BY_CARRIER,
    CarrierEnergyLedger,
    EnergyCarrier,
)
from building_gen.benchmark.energyplus_diagnostics import persist_energyplus_diagnostics
from building_gen.benchmark.specs import (
    RUNPERIOD_WARMUP_KIND,
    WarmupPreconditioningSpec,
)
from building_gen.benchmark.warmup_preconditioning import (
    WARMUP_PRECONDITIONING_FILENAME,
    WarmupPreconditioningRuntimeContext,
    canonical_mapping_sha256,
    warmup_preconditioning_evidence_schema_version,
)
from building_gen.control.capabilities import ControlCapabilityManifest
from building_gen.wrapper.channel import Channel, ChannelClosed
from building_gen.wrapper.manifest_control import (
    ManifestControlRuntime,
    ManifestExecutionSnapshot,
    ManifestExecutionSnapshotCleanupError,
    materialize_manifest_execution_snapshot,
    remove_manifest_execution_snapshot,
    verify_manifest_execution_snapshot,
    verify_manifest_prepared_model,
)
from building_gen.wrapper.simulation_clock import (
    CalendarYearSemantics,
    EnergyPlusRuntimeContext,
    EnergyPlusRuntimeContextRecord,
    SimulationClock,
    SimulationClockRecord,
    advance_zone_timestep,
    energyplus_runtime_context_from_exchange,
    infer_calendar_year_semantics,
    simulation_clock_from_exchange,
    validate_calendar_year_semantics,
)

try:
    import pyenergyplus.api

    logger = logging.getLogger(__name__)
    logger.info("Successfully imported pyenergyplus.api")
    api = pyenergyplus.api.EnergyPlusAPI()
    logger.info("Successfully created EnergyPlusAPI instance")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import pyenergyplus.api: {e}")
    raise

# A stop request may arrive from the caller while the worker finishes. Both
# operations use the same native pointer, so its check/use and disposal must
# share a lock even though only the worker owns deletion.
_STATE_LIFECYCLE_LOCK = threading.RLock()
_ACTIVE_INTERFACE_STATES: set[int] = set()

# Import template utilities - if not available, create minimal versions
try:
    import src.simulator.template as template
except ImportError:
    # Create minimal template utilities if not available
    class template:
        @staticmethod
        def search_replace(data, target_type, replace_fn):
            """Minimal implementation of search_replace"""
            if isinstance(data, target_type):
                return replace_fn(data)
            elif isinstance(data, dict):
                return {k: template.search_replace(v, target_type, replace_fn) for k, v in data.items()}
            elif isinstance(data, list):
                return [template.search_replace(item, target_type, replace_fn) for item in data]
            else:
                return data


class SimulationCrashed(Exception):
    pass


@dataclass(frozen=True, slots=True)
class NativeScheduleAction:
    """Explicit command to release thermostat overrides for one control interval."""


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Any
    finished: bool
    energy_ledger: CarrierEnergyLedger
    # ``clock`` is the public next-state boundary consumed by the benchmark
    # evaluator.  Meter values in the same result belong to the separately
    # recorded, just-completed EnergyPlus interval.
    clock: SimulationClockRecord
    # Backward-compatible generic context.  It is read with the public clock at
    # reset and nonterminal next-begin states; the terminal boundary has no
    # following callback, so its generic context remains the completed one.
    # Scientific interval binding always uses the explicit field below.
    runtime_context: EnergyPlusRuntimeContextRecord
    energy_interval_clock: Optional[SimulationClockRecord]
    energy_interval_runtime_context: Optional[EnergyPlusRuntimeContextRecord]
    energy_interval_observation: Optional[Dict[str, float]]


@dataclass(frozen=True, slots=True)
class DoneResult:
    exit_code: int

    def __post_init__(self) -> None:
        if type(self.exit_code) is not int:
            raise TypeError("EnergyPlus exit code must be a native int")


@dataclass(frozen=True, slots=True)
class ExceptionResult:
    tb: traceback.TracebackException
    exn: Exception


Result = Union[StepResult, DoneResult, ExceptionResult]


HVAC_ELECTRICITY_METER_NAME = CONTROLLED_HVAC_METER_BY_CARRIER[EnergyCarrier.ELECTRICITY]
HVAC_NATURAL_GAS_METER_NAME = CONTROLLED_HVAC_METER_BY_CARRIER[EnergyCarrier.NATURAL_GAS]


def _is_allowed_weather_year_end_phase(
    begin_clock: SimulationClock,
    end_clock: SimulationClock,
) -> bool:
    """Recognize EnergyPlus' narrow Dec-31 end-callback year phase.

    EnergyPlus can increment ``CalendarYear`` before invoking the final
    end-of-zone-timestep callback while the remaining exchange fields still
    label the completed Dec 31 23:60 interval. This phase also occurs for
    actual-weather runs. Accept only that exact one-year phase drift;
    the begin callback remains the authoritative interval label.
    """

    return (
        begin_clock.calendar_year_semantics in {"typical_weather_year", "actual_weather_year"}
        and end_clock.calendar_year_semantics == begin_clock.calendar_year_semantics
        and begin_clock.month == 12
        and begin_clock.day_of_month == 31
        and begin_clock.hour == 23
        and begin_clock.minutes == 60
        and begin_clock.timestep_index == begin_clock.timesteps_per_hour
        and end_clock.calendar_year == begin_clock.calendar_year + 1
        and (
            end_clock.month,
            end_clock.day_of_month,
            end_clock.hour,
            end_clock.minutes,
            end_clock.day_of_week,
            end_clock.day_of_year,
            end_clock.timestep_index,
            end_clock.timesteps_per_hour,
            end_clock.episode_step_index,
        )
        == (
            begin_clock.month,
            begin_clock.day_of_month,
            begin_clock.hour,
            begin_clock.minutes,
            begin_clock.day_of_week,
            begin_clock.day_of_year,
            begin_clock.timestep_index,
            begin_clock.timesteps_per_hour,
            begin_clock.episode_step_index,
        )
    )


@dataclass(frozen=True, slots=True)
class VariableHole:
    variable_name: str
    variable_key: str


@dataclass(frozen=True, slots=True)
class VariableHandle:
    handle: int


@dataclass(frozen=True, slots=True)
class ActuatorHole:
    component_type: str
    control_type: str
    actuator_key: str


@dataclass(frozen=True, slots=True)
class ActuatorHandle:
    handle: int


@dataclass(frozen=True, slots=True)
class MeterHole:
    meter_name: str


@dataclass(frozen=True, slots=True)
class MeterHandle:
    handle: int


@dataclass(frozen=True, slots=True)
class FunctionHole:
    function: Any  # Callable[[int], Any]


class InvalidVariable(Exception):
    pass


class InvalidActuator(Exception):
    pass


class InvalidMeter(Exception):
    pass


@dataclass(slots=True)
class EnergyPlusInterface:
    """
    EnergyPlus interface using template-based approach with your exact state and action spaces
    """

    # Paths and configuration
    building_path: str
    weather_path: str
    config: Any
    controlled_zones: List[str]
    include_zone_occupancy: bool = False
    warmup_preconditioning: WarmupPreconditioningSpec = field(default_factory=WarmupPreconditioningSpec.disabled)
    warmup_preconditioning_context: Optional[WarmupPreconditioningRuntimeContext] = None
    control_manifest: Optional[ControlCapabilityManifest] = None
    component_meter_sources: Optional[Dict[str, str]] = None

    additional_temperature_zones: tuple[str, ...] = ()

    # Template-based observation structure
    observation_template: Dict = field(default_factory=dict)
    observation_template_with_handles: Optional[Dict] = None

    # Actuator specification
    actuators: Dict[str, ActuatorHole] = field(default_factory=dict)
    actuator_control_handles: Dict[str, int] = field(default_factory=dict)
    manifest_control_runtime: Optional[ManifestControlRuntime] = field(default=None, init=False)
    last_manifest_control_evidence: Optional[Dict[str, Any]] = field(default=None, init=False)
    pending_thermostat_setpoints: Optional[Dict[str, Tuple[float, float]]] = field(default=None, init=False)
    last_thermostat_readback: Optional[Dict[str, Any]] = field(default=None, init=False)
    manifest_execution_snapshot: Optional[ManifestExecutionSnapshot] = field(default=None, init=False)
    manifest_prepared_model_source_path: Optional[str] = field(default=None, init=False)
    manifest_prepared_model_source_sha256: Optional[str] = field(default=None, init=False)
    manifest_execution_model_sha256: Optional[str] = field(default=None, init=False)
    manifest_execution_snapshot_removed: bool = field(default=False, init=False)

    # Building info
    zone_names: List[str] = field(default_factory=list)

    # Warmup tracking
    number_of_warmup_phases_completed: int = 0
    warmup_preconditioning_handles: Dict[str, int] = field(
        default_factory=dict,
        init=False,
    )
    warmup_preconditioning_evidence: Dict[str, Any] = field(
        default_factory=dict,
        init=False,
    )

    # State management
    n_steps: int = field(default=0, init=False)
    max_steps: int = field(default=96, init=False)
    terminal_observation_emitted: bool = field(default=False, init=False)
    _terminal_callback_error: bool = field(default=False, init=False)
    verbose: bool = False

    # Channels
    obs_chan: Channel = field(default_factory=Channel)
    act_chan: Channel = field(default_factory=Channel)

    # Threading
    state: Optional[int] = field(default=None, init=False)
    simulation_thread: Optional[threading.Thread] = None
    last_clock: Optional[SimulationClockRecord] = field(default=None, init=False)
    last_runtime_context: Optional[EnergyPlusRuntimeContextRecord] = field(default=None, init=False)
    last_energy_interval_clock: Optional[SimulationClockRecord] = field(default=None, init=False)
    last_energy_interval_runtime_context: Optional[EnergyPlusRuntimeContextRecord] = field(
        default=None,
        init=False,
    )
    last_energy_interval_observation: Optional[Dict[str, float]] = field(
        default=None,
        init=False,
    )
    last_begin_clock: Optional[SimulationClock] = field(default=None, init=False)
    last_begin_runtime_context: Optional[EnergyPlusRuntimeContext] = field(default=None, init=False)
    last_transition_clock: Optional[SimulationClock] = field(default=None, init=False)
    last_transition_runtime_context: Optional[EnergyPlusRuntimeContext] = field(default=None, init=False)
    pending_transition_index: Optional[int] = field(default=None, init=False)
    buffered_energy_ledger: Optional[CarrierEnergyLedger] = field(default=None, init=False)
    buffered_energy_interval_clock: Optional[SimulationClock] = field(default=None, init=False)
    buffered_energy_interval_runtime_context: Optional[EnergyPlusRuntimeContext] = field(
        default=None,
        init=False,
    )
    buffered_energy_interval_observation: Optional[Dict[str, float]] = field(
        default=None,
        init=False,
    )
    last_exit_code: Optional[int] = field(default=None, init=False)
    calendar_year_semantics: CalendarYearSemantics = field(default="unknown", init=False)

    def __post_init__(self):
        """Initialize after creation"""
        if not isinstance(self.warmup_preconditioning, WarmupPreconditioningSpec):
            raise TypeError("warmup_preconditioning must be a WarmupPreconditioningSpec")
        if self.warmup_preconditioning.enabled:
            if not isinstance(
                self.warmup_preconditioning_context,
                WarmupPreconditioningRuntimeContext,
            ):
                raise ValueError("enabled warmup preconditioning requires a runtime context")
            if self.warmup_preconditioning_context.contract != self.warmup_preconditioning:
                raise ValueError("warmup preconditioning runtime context contract mismatch")
        elif self.warmup_preconditioning_context is not None:
            raise ValueError("disabled warmup preconditioning cannot carry a runtime context")
        if self.control_manifest is not None:
            if self.warmup_preconditioning.enabled:
                raise ValueError("manifest control cannot be combined with thermostat warmup preconditioning")
            self.manifest_control_runtime = ManifestControlRuntime(self.control_manifest)
            self.control_manifest = self.manifest_control_runtime.manifest
            verify_manifest_prepared_model(self.control_manifest, self.building_path)
        self._reset_warmup_preconditioning_state()
        # Extract zone names if not provided
        if not self.controlled_zones:
            self.zone_names = self._extract_zone_names(self.building_path)
            self.controlled_zones = self.zone_names
        else:
            self.zone_names = self.controlled_zones

        # Set max_steps based on config
        episode_length_hours = getattr(self.config, "episode_length_hours", 24)
        timesteps_per_hour = getattr(self.config, "timesteps_per_hour", 4)
        self.max_steps = episode_length_hours * timesteps_per_hour

        configured_year_semantics = getattr(self.config, "calendar_year_semantics", None)
        if configured_year_semantics is None:
            self.calendar_year_semantics = infer_calendar_year_semantics(self.weather_path)
        else:
            self.calendar_year_semantics = validate_calendar_year_semantics(configured_year_semantics)

        # Build observation template using your exact structure
        self._build_observation_template()

        # Build actuator specifications using your exact structure
        self._build_actuator_specifications()

        logger.info(f"EnergyPlus Interface created for {len(self.zone_names)} zones, {self.max_steps} max steps")

    def _reset_warmup_preconditioning_state(self) -> None:
        """Clear every per-EnergyPlus-run handle and counter."""

        self._terminal_callback_error = False
        # EnergyPlus API handles belong to one state only.  In particular, the
        # warmup and policy paths share cooling-actuator handles, so neither
        # cache may survive a new ``state_manager.new_state()`` call.
        self.observation_template_with_handles = None
        self.actuator_control_handles = {}
        self.last_manifest_control_evidence = None
        self.pending_thermostat_setpoints = None
        self.last_thermostat_readback = None
        self.manifest_execution_snapshot = None
        self.manifest_prepared_model_source_path = None
        self.manifest_prepared_model_source_sha256 = None
        self.manifest_execution_model_sha256 = None
        self.manifest_execution_snapshot_removed = False
        manifest_runtime = getattr(self, "manifest_control_runtime", None)
        if manifest_runtime is not None:
            manifest_runtime.reset_run()
        self.warmup_preconditioning_handles = {}
        self.warmup_preconditioning_evidence = {
            "callback_not_ready_count": 0,
            "ready_callback_counts_by_kind": {},
            "ready_warmup_callback_counts_by_kind": {},
            "precondition_set_calls_by_kind": {},
            "precondition_reset_calls_by_kind": {},
            "precondition_set_readback_count": 0,
            "precondition_set_readback_min_c": None,
            "precondition_set_readback_max_c": None,
            "handle_resolution_count": 0,
            "resolved_actuator_keys": [],
            "policy_action_flow_count": 0,
            "set_outside_gate_count": 0,
            "reset_before_policy_action_violation_count": 0,
            "callback_errors": [],
        }

    @staticmethod
    def _increment_kind_count(counts: Dict[str, int], kind: int, amount: int = 1) -> None:
        key = str(kind)
        counts[key] = counts.get(key, 0) + amount

    def _build_observation_template(self):
        """Build observation template matching your exact spaces.py structure"""
        self.observation_template = {}

        # Zone temperatures - direct mapping to your observation keys
        for zone in self.zone_names:
            # Your spaces.py expects: {zone_name}_temperature
            self.observation_template[f"{zone}_temperature"] = VariableHole("Zone Mean Air Temperature", zone)
            if self.include_zone_occupancy:
                # Reward v2 consumes this raw EnergyPlus output.  It is kept
                # out of the policy observation vector, so enabling it does
                # not alter the declared observation space.
                self.observation_template[f"{zone}_occupant_count"] = VariableHole("Zone People Occupant Count", zone)
                if getattr(self, 'control_manifest', None) is None:
                    # Diagnostic outputs only: these never enter the policy's
                    # declared observation vector. Unlike actuator readbacks,
                    # they also report targets when native schedules control HVAC.
                    for channel in ('heating', 'cooling'):
                        self.observation_template[f"{zone}_thermostat_{channel}_setpoint_c"] = VariableHole(
                            f"Zone Thermostat {channel.title()} Setpoint Temperature", zone)
                    self.observation_template[f"{zone}_thermostat_control_type"] = VariableHole(
                        "Zone Thermostat Control Type", zone)

        extra = getattr(self, "additional_temperature_zones", ())
        if extra:
            with open(self.building_path) as source:
                available = set(json.load(source).get("Zone", {}))
            if (len(set(extra)) != len(extra) or set(extra) & set(self.zone_names)
                    or not set(extra) <= available):
                raise ValueError("additional temperature sensors require unique canonical non-controlled zones")
            for zone in extra:
                self.observation_template[f"{zone}_temperature"] = VariableHole("Zone Mean Air Temperature", zone)

        # Outdoor conditions - matching your spaces.py keys
        self.observation_template["outdoor_temperature"] = VariableHole(
            "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        self.observation_template["outdoor_humidity"] = VariableHole(
            "Site Outdoor Air Relative Humidity", "Environment"
        )

        # Energy consumption - matching your spaces.py keys exactly
        sources = getattr(self, "component_meter_sources", None)
        self.observation_template["hvac_electricity"] = MeterHole(
            sources['electricity'] if sources is not None else HVAC_ELECTRICITY_METER_NAME)
        if sources is None or 'natural_gas' in sources:
            self.observation_template["hvac_gas"] = MeterHole(
                sources['natural_gas'] if sources is not None else HVAC_NATURAL_GAS_METER_NAME)

        # Calendar features are added once per callback from the EnergyPlus
        # exchange API. They must not be synthesized from the episode counter.

    def _build_actuator_specifications(self):
        """Build actuator specifications for your offset-based action space"""
        self.actuators = {}

        if getattr(self, "manifest_control_runtime", None) is not None:
            return

        for zone in self.zone_names:
            # Heating setpoint actuator
            heating_key = f"{zone}_heating"
            self.actuators[heating_key] = ActuatorHole("Zone Temperature Control", "Heating Setpoint", zone)

            # Cooling setpoint actuator
            cooling_key = f"{zone}_cooling"
            self.actuators[cooling_key] = ActuatorHole("Zone Temperature Control", "Cooling Setpoint", zone)

    def _materialize_manifest_execution_snapshot(
        self,
        manifest_runtime: Optional[ManifestControlRuntime],
    ) -> Optional[ManifestExecutionSnapshot]:
        if manifest_runtime is None:
            return None
        snapshot = materialize_manifest_execution_snapshot(
            manifest_runtime.manifest,
            self.building_path,
            self.config.output_dir,
        )
        self.manifest_execution_snapshot = snapshot
        self.manifest_prepared_model_source_path = snapshot.source_path
        self.manifest_prepared_model_source_sha256 = snapshot.source_sha256
        self.manifest_execution_model_sha256 = snapshot.model_sha256
        return snapshot

    @staticmethod
    def _verify_manifest_execution_snapshot(
        manifest_runtime: Optional[ManifestControlRuntime],
        snapshot: Optional[ManifestExecutionSnapshot],
    ) -> None:
        if manifest_runtime is None:
            if snapshot is not None:
                raise RuntimeError("thermostat execution cannot carry a manifest model snapshot")
            return
        if snapshot is None:
            raise RuntimeError("manifest execution is missing its pinned model snapshot")
        verify_manifest_execution_snapshot(manifest_runtime.manifest, snapshot)

    def _remove_manifest_execution_snapshot(
        self,
        manifest_runtime: Optional[ManifestControlRuntime],
        snapshot: Optional[ManifestExecutionSnapshot],
    ) -> None:
        if manifest_runtime is None:
            if snapshot is not None:
                raise RuntimeError("thermostat execution cannot remove a manifest model snapshot")
            return
        if snapshot is None:
            raise RuntimeError("manifest execution is missing its transient model snapshot")
        remove_manifest_execution_snapshot(snapshot)
        self.manifest_execution_snapshot = None
        self.manifest_execution_snapshot_removed = True

    def start(self) -> Tuple[Dict[str, float], bool, CarrierEnergyLedger | None]:
        """Start simulation using template-based approach"""
        manifest_runtime = None
        manifest_snapshot = None
        worker_started = False
        try:
            self._reset_warmup_preconditioning_state()
            manifest_runtime = getattr(self, "manifest_control_runtime", None)
            manifest_snapshot = self._materialize_manifest_execution_snapshot(manifest_runtime)
            execution_building_path = self.building_path if manifest_snapshot is None else manifest_snapshot.model_path
            self.n_steps = 0
            self.terminal_observation_emitted = False
            self.number_of_warmup_phases_completed = 0
            self.last_clock = None
            self.last_runtime_context = None
            self.last_energy_interval_clock = None
            self.last_energy_interval_runtime_context = None
            self.last_energy_interval_observation = None
            self.last_begin_clock = None
            self.last_begin_runtime_context = None
            self.last_transition_clock = None
            self.last_transition_runtime_context = None
            self.pending_transition_index = None
            self.buffered_energy_ledger = None
            self.buffered_energy_interval_clock = None
            self.buffered_energy_interval_runtime_context = None
            self.buffered_energy_interval_observation = None
            self.last_exit_code = None
            # Create EnergyPlus state
            with _STATE_LIFECYCLE_LOCK:
                self.state = api.state_manager.new_state()
                _ACTIVE_INTERFACE_STATES.add(id(self))
            logger.info(f"Created EnergyPlus state: {self.state}")

            def eplus_thread():
                """Thread running EnergyPlus with automatic cleanup"""
                exit_code = None
                snapshot_cleanup_attempted = False
                try:
                    # Run EnergyPlus
                    args = ["-d", self.config.output_dir]
                    if self.weather_path and os.path.exists(self.weather_path):
                        args.extend(["-w", self.weather_path])
                    args.append(execution_building_path)

                    logger.info(f"Starting EnergyPlus with args: {args}")
                    self._verify_manifest_execution_snapshot(manifest_runtime, manifest_snapshot)
                    try:
                        exit_code = api.runtime.run_energyplus(self.state, args)
                    finally:
                        self._verify_manifest_execution_snapshot(manifest_runtime, manifest_snapshot)
                    logger.info(f"EnergyPlus finished with exit code: {exit_code}")
                    try:
                        diagnostics_path = persist_energyplus_diagnostics(
                            self.config.output_dir,
                            model_path=execution_building_path,
                        )
                        logger.info("Persisted EnergyPlus diagnostics to %s", diagnostics_path)
                    except Exception:
                        # Recording is best-effort for legacy/engineering uses.
                        # The benchmark's scientific diagnostics gate is
                        # responsible for accepting or rejecting an artifact.
                        logger.exception("Could not persist EnergyPlus diagnostics")
                    self._verify_manifest_execution_snapshot(manifest_runtime, manifest_snapshot)
                    snapshot_cleanup_attempted = True
                    self._remove_manifest_execution_snapshot(manifest_runtime, manifest_snapshot)
                    if self.warmup_preconditioning.enabled and not self._terminal_callback_error:
                        evidence_path = self._persist_warmup_preconditioning_evidence()
                        logger.info(
                            "Persisted warmup preconditioning evidence to %s",
                            evidence_path,
                        )
                    if not self._terminal_callback_error:
                        try:
                            self.obs_chan.put(DoneResult(exit_code))
                        except ChannelClosed:
                            pass  # The caller cancelled the episode during shutdown.

                except Exception as e:
                    reported_error = e
                    if not snapshot_cleanup_attempted:
                        snapshot_cleanup_attempted = True
                        try:
                            self._remove_manifest_execution_snapshot(manifest_runtime, manifest_snapshot)
                        except Exception as cleanup_error:
                            reported_error = ManifestExecutionSnapshotCleanupError(e, cleanup_error)
                    logger.error(f"EnergyPlus thread exception: {reported_error}")
                    if not self._terminal_callback_error:
                        self._terminal_callback_error = True
                        tb_exc = traceback.TracebackException.from_exception(reported_error)
                        try:
                            self.obs_chan.put(ExceptionResult(tb_exc, reported_error))
                        except ChannelClosed:
                            logger.error("Cannot deliver worker error after channel shutdown")
                    else:
                        logger.error("Suppressing a second terminal result after a callback error")
                finally:
                    try:
                        self._dispose_state()
                    finally:
                        self.obs_chan.close()
                        self.act_chan.close()

            # Request variables before starting (template-based approach)
            template.search_replace(
                self.observation_template,
                VariableHole,
                lambda var: api.exchange.request_variable(self.state, var.variable_name, var.variable_key),
            )
            if manifest_runtime is not None:
                manifest_runtime.request_effect_variables(
                    api.exchange,
                    self.state,
                )

            # Set up callbacks
            api.runtime.callback_begin_system_timestep_before_predictor(self.state, self.callback_timestep)

            # The begin-system callback applies each action.  Every resulting
            # transition is read at the end of that same zone timestep, after
            # EnergyPlus has updated meter values and zone reports.
            api.runtime.callback_end_zone_timestep_after_zone_reporting(self.state, self.callback_end_zone_timestep)

            api.runtime.callback_after_new_environment_warmup_complete(self.state, self.warmup_callback)

            # Set console output
            api.runtime.set_console_output_status(self.state, self.verbose)

            # Start simulation thread
            self.simulation_thread = threading.Thread(target=eplus_thread, daemon=True)
            self.simulation_thread.start()
            worker_started = True

            # Get first observation
            return self._get_obs_and_filter()

        except Exception as e:
            logger.error(f"Start failed: {e}")
            traceback.print_exc()
            if not worker_started:
                self._dispose_state()
            if manifest_snapshot is not None and not worker_started:
                try:
                    self._remove_manifest_execution_snapshot(manifest_runtime, manifest_snapshot)
                except Exception as cleanup_error:
                    raise ManifestExecutionSnapshotCleanupError(e, cleanup_error) from e
            raise

    def warmup_callback(self, state: int):
        """Warmup callback - using same approach as reference"""
        self.number_of_warmup_phases_completed += 1
        logger.info(f"Warmup phase {self.number_of_warmup_phases_completed} completed")

        if self.verbose:
            print("warmup phase complete")

    @staticmethod
    def _actuator_identity(actuator: ActuatorHole) -> tuple[str, str, str]:
        """Return EnergyPlus' case-insensitive actuator identity."""

        return (
            actuator.component_type.casefold(),
            actuator.control_type.casefold(),
            actuator.actuator_key.casefold(),
        )

    def _warmup_policy_actuator_mapping(self) -> Dict[str, str]:
        """Map each contract key to its unique normal-policy actuator name."""

        contract = self.warmup_preconditioning
        if not contract.enabled:
            return {}
        policy_names_by_identity: Dict[tuple[str, str, str], List[str]] = {}
        for actuator_name, actuator in self.actuators.items():
            identity = self._actuator_identity(actuator)
            policy_names_by_identity.setdefault(identity, []).append(actuator_name)

        mapping: Dict[str, str] = {}
        claimed_policy_names: set[str] = set()
        for contract_key in contract.actuator_keys:
            identity = (
                contract.actuator_component_type.casefold(),
                contract.actuator_control_type.casefold(),
                contract_key.casefold(),
            )
            matches = policy_names_by_identity.get(identity, [])
            if len(matches) != 1:
                raise InvalidActuator(
                    "warmup cooling actuator must match exactly one policy actuator: "
                    f"{contract.actuator_component_type}/"
                    f"{contract.actuator_control_type}/{contract_key}; "
                    f"found {len(matches)}"
                )
            policy_name = matches[0]
            if policy_name in claimed_policy_names:
                raise InvalidActuator("multiple warmup cooling actuator keys match policy actuator " f"{policy_name!r}")
            claimed_policy_names.add(policy_name)
            mapping[contract_key] = policy_name
        return mapping

    def _preconditioning_handles(self, state: int) -> Dict[str, int]:
        """Bind warmup cooling handles exactly once, reusing policy handles."""

        contract = self.warmup_preconditioning
        if not contract.enabled:
            raise RuntimeError("cannot resolve preconditioning handles for a disabled contract")
        policy_mapping = self._warmup_policy_actuator_mapping()
        if self.warmup_preconditioning_handles:
            if tuple(self.warmup_preconditioning_handles) != contract.actuator_keys:
                raise InvalidActuator("cached warmup actuator keys do not match the enabled contract")
            for contract_key, policy_name in policy_mapping.items():
                handle = self.warmup_preconditioning_handles[contract_key]
                if handle < 0:
                    raise InvalidActuator(f"cached warmup cooling actuator {contract_key!r} is unavailable")
                policy_handle = self.actuator_control_handles.get(policy_name)
                if policy_handle is not None and policy_handle != handle:
                    raise InvalidActuator(
                        "warmup and policy cooling actuator handles disagree for " f"{contract_key!r}"
                    )
            return self.warmup_preconditioning_handles
        handles: Dict[str, int] = {}
        for actuator_key in contract.actuator_keys:
            policy_name = policy_mapping[actuator_key]
            handle = self.actuator_control_handles.get(policy_name)
            if handle is None:
                handle = api.exchange.get_actuator_handle(
                    state,
                    contract.actuator_component_type,
                    contract.actuator_control_type,
                    actuator_key,
                )
            if handle < 0:
                raise InvalidActuator(
                    "warmup cooling actuator unavailable: "
                    f"{contract.actuator_component_type}/"
                    f"{contract.actuator_control_type}/{actuator_key}"
                )
            handles[actuator_key] = handle
        self.warmup_preconditioning_handles = handles
        evidence = self.warmup_preconditioning_evidence
        evidence["handle_resolution_count"] += 1
        evidence["resolved_actuator_keys"] = list(handles)
        return handles

    def _apply_runperiod_warmup_preconditioning(
        self,
        state: int,
        *,
        kind: int,
        warmup: bool,
    ) -> bool:
        """Set during kind-3 warmup or reset before kind-3 policy flow.

        Returns whether a reset was completed for this callback.  Warmup
        callbacks return ``False`` because they never continue into policy
        action flow.
        """

        contract = self.warmup_preconditioning
        if not contract.enabled or kind != RUNPERIOD_WARMUP_KIND:
            return False
        handles = self._preconditioning_handles(state)
        evidence = self.warmup_preconditioning_evidence
        if warmup:
            expected_value = contract.actuator_value_c
            assert expected_value is not None
            for handle in handles.values():
                api.exchange.set_actuator_value(state, handle, expected_value)
                self._increment_kind_count(
                    evidence["precondition_set_calls_by_kind"],
                    kind,
                )
                readback = float(api.exchange.get_actuator_value(state, handle))
                evidence["precondition_set_readback_count"] += 1
                current_minimum = evidence["precondition_set_readback_min_c"]
                current_maximum = evidence["precondition_set_readback_max_c"]
                evidence["precondition_set_readback_min_c"] = (
                    readback if current_minimum is None else min(current_minimum, readback)
                )
                evidence["precondition_set_readback_max_c"] = (
                    readback if current_maximum is None else max(current_maximum, readback)
                )
                if readback != expected_value:
                    raise InvalidActuator(
                        "warmup cooling actuator readback mismatch: " f"expected {expected_value}, found {readback}"
                    )
            return False

        for handle in handles.values():
            api.exchange.reset_actuator(state, handle)
            self._increment_kind_count(
                evidence["precondition_reset_calls_by_kind"],
                kind,
            )
        return True

    def release_thermostat_overrides(self, state: int) -> None:
        """Return all registered thermostat actuators to EnergyPlus schedule control.

        This is not an evaluation policy by itself: callers must separately record
        native-control evidence and must not report submitted policy setpoints.
        """
        if getattr(self, "manifest_control_runtime", None) is not None:
            raise ValueError("native thermostat release does not support manifest equipment profiles")
        handles = self.actuator_control_handles
        expected = {f"{zone}_{channel}" for zone in self.zone_names for channel in ("heating", "cooling")}
        if (not expected or set(handles) != expected
                or any(type(handle) is not int or handle < 0 for handle in handles.values())
                or len(set(handles.values())) != len(handles)):
            raise ValueError("native thermostat release requires unique valid heating/cooling handles for every zone")
        for name in sorted(handles):
            api.exchange.reset_actuator(state, handles[name])
        self.pending_thermostat_setpoints = None
        self.last_thermostat_readback = None

    def _apply_policy_action(self, state: int, action: Any) -> None:
        manifest_runtime = getattr(self, "manifest_control_runtime", None)
        if isinstance(action, NativeScheduleAction):
            self.release_thermostat_overrides(state)
            return
        if manifest_runtime is None:
            if not isinstance(action, dict):
                raise TypeError(f"Received invalid action: {action!r}")
            self._apply_setpoint_actions(state, action)
            return
        manifest_runtime.apply_action(api.exchange, state, action)

    def callback_timestep(self, state: int) -> None:
        """Emit next-begin states, then apply one action per zone timestep.

        EnergyPlus updates ``CurTSValue`` during zone reporting, so the end
        callback captures the exact interval ledger.  For nonterminal actions,
        this callback combines that buffered ledger/evidence with the physical
        observation and exact public clock at the following begin callback.
        That preserves the benchmark's established next-begin observation
        phase without ever relabelling a stale meter read.
        """
        try:
            # Readiness is the first and only exchange call before EnergyPlus
            # confirms that API data and actuator handles are available.
            ready = api.exchange.api_data_fully_ready(state)
            if type(ready) is not bool:
                raise TypeError("EnergyPlus api_data_fully_ready must return a native bool")
            if not ready:
                self.warmup_preconditioning_evidence["callback_not_ready_count"] += 1
                logger.debug("API data not fully ready")
                return

            kind = api.exchange.kind_of_sim(state)
            if type(kind) is not int:
                raise TypeError("EnergyPlus kind_of_sim must return a native int")
            evidence = self.warmup_preconditioning_evidence
            self._increment_kind_count(
                evidence["ready_callback_counts_by_kind"],
                kind,
            )
            warmup = api.exchange.warmup_flag(state)
            if type(warmup) is not bool:
                raise TypeError("EnergyPlus warmup_flag must return a native bool")
            if warmup:
                self._increment_kind_count(
                    evidence["ready_warmup_callback_counts_by_kind"],
                    kind,
                )
            reset_completed = self._apply_runperiod_warmup_preconditioning(
                state,
                kind=kind,
                warmup=warmup,
            )

            if warmup:
                logger.debug("Still in warmup")
                return

            # Never emit observations or apply policy actions during sizing or
            # design-day environments.  ``kind_of_sim == 3`` is the exact API
            # discriminator for a Weather File Run Period.
            if kind != RUNPERIOD_WARMUP_KIND:
                return

            # Once the final action has been applied, its resulting state is
            # emitted by callback_end_zone_timestep.
            if self.n_steps >= self.max_steps or self.terminal_observation_emitted:
                return
            if self.pending_transition_index is not None:
                raise RuntimeError("EnergyPlus began a new zone timestep before the prior transition was emitted")

            transition_index = self.n_steps + 1
            decision_index = self.n_steps
            clock = simulation_clock_from_exchange(
                api.exchange,
                state,
                episode_step_index=decision_index,
                calendar_year_semantics=self.calendar_year_semantics,
            )
            runtime_context = energyplus_runtime_context_from_exchange(
                api.exchange,
                state,
                kind_of_sim=kind,
            )
            if self.last_transition_clock is not None:
                expected_clock = advance_zone_timestep(self.last_transition_clock)
                if clock != expected_clock:
                    raise RuntimeError(
                        "EnergyPlus zone clock jumped between reported transitions: "
                        f"expected {expected_clock.as_dict()}, found {clock.as_dict()}"
                    )
            elif self.n_steps != 0:
                raise RuntimeError("EnergyPlus transition history is missing before a later action")
            if self.last_transition_runtime_context is not None and (
                runtime_context.environment_num != self.last_transition_runtime_context.environment_num
                or runtime_context.kind_of_sim != self.last_transition_runtime_context.kind_of_sim
            ):
                raise RuntimeError("EnergyPlus environment changed during the policy episode")

            self.last_begin_clock = clock
            self.last_begin_runtime_context = runtime_context
            if self.n_steps == 0:
                # Reset is an initial state, not a completed policy interval.
                # Preserve the exact API interval label at public index zero;
                # action 1 is bound to this interval by the exact begin/end
                # equality check below.
                self.obs_chan.put(
                    self._build_step_result(
                        state,
                        clock=clock,
                        runtime_context=runtime_context,
                        finished=False,
                    )
                )
            else:
                if self.buffered_energy_ledger is None:
                    raise RuntimeError("Completed transition is missing its buffered energy ledger")
                if self.buffered_energy_interval_clock is None:
                    raise RuntimeError("Completed transition is missing its energy-interval clock")
                if self.buffered_energy_interval_runtime_context is None:
                    raise RuntimeError("Completed transition is missing its energy-interval runtime context")
                if self.buffered_energy_interval_observation is None:
                    raise RuntimeError("Completed transition is missing its energy-interval observation")
                self.obs_chan.put(
                    self._build_step_result(
                        state,
                        clock=clock,
                        runtime_context=runtime_context,
                        energy_interval_clock=self.buffered_energy_interval_clock,
                        energy_interval_runtime_context=(self.buffered_energy_interval_runtime_context),
                        energy_interval_observation=self.buffered_energy_interval_observation,
                        energy_ledger=self.buffered_energy_ledger,
                        finished=False,
                    )
                )
                self.buffered_energy_ledger = None
                self.buffered_energy_interval_clock = None
                self.buffered_energy_interval_runtime_context = None
                self.buffered_energy_interval_observation = None

            # Wait for actions
            act = self.act_chan.get()

            if self.warmup_preconditioning.enabled and not reset_completed:
                evidence["reset_before_policy_action_violation_count"] += 1
                raise RuntimeError("warmup preconditioning actuators were not reset before policy action")

            # Apply either the unchanged thermostat setpoint mapping or the
            # complete authenticated manifest endpoint vector.
            self._apply_policy_action(state, act)
            evidence["policy_action_flow_count"] += 1

            self.n_steps = transition_index
            self.pending_transition_index = transition_index

        except ChannelClosed:
            api.runtime.stop_simulation(state)
        except Exception as e:
            logger.error(f"Callback timestep exception: {e}")
            self.warmup_preconditioning_evidence["callback_errors"].append(repr(e))
            api.runtime.stop_simulation(state)
            if not self._terminal_callback_error:
                self._terminal_callback_error = True
                tb_exc = traceback.TracebackException.from_exception(e)
                try:
                    self.obs_chan.put(ExceptionResult(tb_exc, e))
                except ChannelClosed:
                    logger.error("Cannot deliver callback error after channel shutdown")

    def callback_end_zone_timestep(self, state: int) -> None:
        """Capture one exact post-reporting ledger for the applied action."""

        if self.pending_transition_index is None or self.terminal_observation_emitted:
            return
        try:
            ready = api.exchange.api_data_fully_ready(state)
            if type(ready) is not bool:
                raise TypeError("EnergyPlus api_data_fully_ready must return a native bool")
            if not ready:
                raise RuntimeError("EnergyPlus transition callback is not API-ready")
            warmup = api.exchange.warmup_flag(state)
            if type(warmup) is not bool:
                raise TypeError("EnergyPlus warmup_flag must return a native bool")
            if warmup:
                raise RuntimeError("EnergyPlus transition callback re-entered warmup")
            if self.last_begin_clock is None:
                raise RuntimeError("Cannot emit a transition before a begin callback")
            if self.last_begin_runtime_context is None:
                raise RuntimeError("Cannot emit transition runtime context before a begin callback")
            if self.pending_transition_index != self.n_steps:
                raise RuntimeError(
                    "Pending transition does not match the number of applied actions: "
                    f"pending={self.pending_transition_index}, actions={self.n_steps}"
                )
            observed_end_clock = simulation_clock_from_exchange(
                api.exchange,
                state,
                episode_step_index=self.pending_transition_index - 1,
                calendar_year_semantics=self.calendar_year_semantics,
            )
            transition_runtime_context = energyplus_runtime_context_from_exchange(api.exchange, state)
            if observed_end_clock != self.last_begin_clock and not _is_allowed_weather_year_end_phase(
                self.last_begin_clock,
                observed_end_clock,
            ):
                raise RuntimeError(
                    "EnergyPlus zone clock changed within one action interval: "
                    f"begin={self.last_begin_clock.as_dict()}, end={observed_end_clock.as_dict()}"
                )
            if transition_runtime_context != self.last_begin_runtime_context:
                raise RuntimeError(
                    "EnergyPlus runtime context changed within one action interval: "
                    f"begin={self.last_begin_runtime_context.as_dict()}, "
                    f"end={transition_runtime_context.as_dict()}"
                )

            transition_clock = self.last_begin_clock
            finished = self.n_steps >= self.max_steps
            completed_observation = self._read_observation(state)
            completed_energy_ledger = self._build_energy_ledger(completed_observation, meter_sources=getattr(self, "component_meter_sources", None))
            completed_interval_observation = self._process_physical_observation(
                completed_observation,
                clock=transition_clock,
            )
            manifest_runtime = getattr(self, "manifest_control_runtime", None)
            self._capture_thermostat_readback(state)
            if manifest_runtime is not None:
                self.last_manifest_control_evidence = manifest_runtime.complete_interval(
                    api.exchange,
                    state,
                )
            self.pending_transition_index = None
            self.last_transition_clock = transition_clock
            self.last_transition_runtime_context = transition_runtime_context
            next_state_clock = advance_zone_timestep(transition_clock)
            if finished:
                # Set this before the rendezvous put so a duplicate callback
                # cannot publish the terminal transition twice.
                self.terminal_observation_emitted = True
                self.obs_chan.put(
                    self._build_step_result(
                        state,
                        clock=next_state_clock,
                        runtime_context=transition_runtime_context,
                        energy_interval_clock=transition_clock,
                        energy_interval_runtime_context=transition_runtime_context,
                        energy_interval_observation=completed_interval_observation,
                        energy_ledger=completed_energy_ledger,
                        physical_observation=completed_observation,
                        finished=True,
                    )
                )
                logger.info(f"Step limit reached: {self.n_steps}/{self.max_steps}")
                api.runtime.stop_simulation(state)
            else:
                if any(
                    item is not None
                    for item in (
                        self.buffered_energy_ledger,
                        self.buffered_energy_interval_clock,
                        self.buffered_energy_interval_runtime_context,
                        self.buffered_energy_interval_observation,
                    )
                ):
                    raise RuntimeError("Refusing to overwrite an unconsumed completed transition")
                self.buffered_energy_ledger = completed_energy_ledger
                self.buffered_energy_interval_clock = transition_clock
                self.buffered_energy_interval_runtime_context = transition_runtime_context
                self.buffered_energy_interval_observation = completed_interval_observation
        except ChannelClosed:
            api.runtime.stop_simulation(state)
        except Exception as e:
            logger.error(f"End-of-zone-timestep callback exception: {e}")
            api.runtime.stop_simulation(state)
            if not self._terminal_callback_error:
                self._terminal_callback_error = True
                tb_exc = traceback.TracebackException.from_exception(e)
                try:
                    self.obs_chan.put(ExceptionResult(tb_exc, e))
                except ChannelClosed:
                    logger.error("Cannot deliver callback error after channel shutdown")

    def _read_observation(self, state: int) -> Dict[str, Any]:
        """Read physical outputs without consulting callback clock fields."""

        if self.observation_template_with_handles is None:
            logger.info("Constructing handles...")
            self.construct_handles(state)

        logger.info(f"Getting observations for step {self.n_steps}")

        var_values = template.search_replace(
            self.observation_template_with_handles,
            VariableHandle,
            lambda handle: api.exchange.get_variable_value(state, handle.handle),
        )

        meter_replaced = template.search_replace(
            var_values,
            MeterHandle,
            lambda handle: self._read_meter_value(state, handle),
        )

        actuator_replaced = template.search_replace(
            meter_replaced,
            ActuatorHandle,
            lambda han: api.exchange.get_actuator_value(state, han.handle),
        )

        function_replaced = template.search_replace(
            actuator_replaced,
            FunctionHole,
            lambda fn: fn.function(state),
        )
        return function_replaced

    def _build_step_result(
        self,
        state: int,
        *,
        clock: SimulationClock,
        runtime_context: EnergyPlusRuntimeContext,
        energy_interval_clock: SimulationClock | None = None,
        energy_interval_runtime_context: EnergyPlusRuntimeContext | None = None,
        energy_interval_observation: Dict[str, float] | None = None,
        energy_ledger: CarrierEnergyLedger | None = None,
        physical_observation: Dict[str, Any] | None = None,
        finished: bool,
    ) -> StepResult:
        """Package outputs with public and completed-interval provenance.

        ``clock`` labels the public state boundary.  When a policy action has
        completed, ``energy_interval_clock`` is the exact begin/end API label
        whose post-reporting meter values were read into ``energy_ledger``.
        Reset has no completed policy interval, so all interval fields are
        deliberately ``None``.
        """

        interval_evidence = (
            energy_interval_clock,
            energy_interval_runtime_context,
            energy_interval_observation,
        )
        if any(item is None for item in interval_evidence) and any(item is not None for item in interval_evidence):
            raise ValueError("energy interval clock, runtime context, and observation must be provided together")

        observation = dict(self._read_observation(state) if physical_observation is None else physical_observation)
        observed_energy_ledger = self._build_energy_ledger(observation, meter_sources=getattr(self, "component_meter_sources", None))
        if energy_ledger is None:
            energy_ledger = observed_energy_ledger
        elif observed_energy_ledger != energy_ledger:
            raise RuntimeError(
                "EnergyPlus next-begin meter ledger differs from the exact post-reporting capture: "
                f"end={energy_ledger.as_dict()} sources={energy_ledger.meter_source_mapping}, "
                f"begin={observed_energy_ledger.as_dict()} "
                f"sources={observed_energy_ledger.meter_source_mapping}"
            )
        obs = self._process_physical_observation(observation, clock=clock)
        logger.info(f"Got observations: {list(obs.keys())}")
        return StepResult(
            obs,
            finished,
            energy_ledger,
            clock.as_dict(),
            runtime_context.as_dict(),
            None if energy_interval_clock is None else energy_interval_clock.as_dict(),
            (None if energy_interval_runtime_context is None else energy_interval_runtime_context.as_dict()),
            None if energy_interval_observation is None else dict(energy_interval_observation),
        )

    def _process_physical_observation(
        self,
        physical_observation: Dict[str, Any],
        *,
        clock: SimulationClock,
    ) -> Dict[str, float]:
        """Attach one clock and normalize an exact callback observation."""

        observation = dict(physical_observation)
        observation.update(clock.normalized_observation())
        return self._validate_and_process_observations(observation)

    @staticmethod
    def _read_meter_value(state: int, meter_handle: MeterHandle) -> float | None:
        """Read a meter without converting an unavailable optional meter to zero."""

        if meter_handle.handle < 0:
            return None
        return float(api.exchange.get_meter_value(state, meter_handle.handle))

    @staticmethod
    def _build_energy_ledger(observation: Dict[str, Any], *, meter_sources=None) -> CarrierEnergyLedger:
        """Convert EnergyPlus interval-meter readings from joules to kWh."""

        if meter_sources is not None:
            if 'electricity' not in meter_sources or set(meter_sources) - {'electricity', 'natural_gas'}:
                raise ValueError('invalid explicit component meter carriers')
            for carrier, key in [('electricity', 'hvac_electricity'), ('natural_gas', 'hvac_gas')]:
                if carrier in meter_sources and observation.get(key) is None:
                    raise ValueError(f'Required component meter {meter_sources[carrier]!r} is unavailable')
            return CarrierEnergyLedger.from_joules(
                electricity_joules=observation['hvac_electricity'],
                natural_gas_joules=observation.get('hvac_gas') if 'natural_gas' in meter_sources else None,
                meter_sources=meter_sources)
        electricity_joules = observation.get("hvac_electricity")
        if electricity_joules is None:
            raise ValueError(f"Required EnergyPlus meter {HVAC_ELECTRICITY_METER_NAME!r} is unavailable")
        natural_gas_joules = observation.get("hvac_gas")
        meter_sources = {
            EnergyCarrier.ELECTRICITY: HVAC_ELECTRICITY_METER_NAME,
        }
        if natural_gas_joules is not None:
            meter_sources[EnergyCarrier.NATURAL_GAS] = HVAC_NATURAL_GAS_METER_NAME
        return CarrierEnergyLedger.from_joules(
            electricity_joules=electricity_joules,
            natural_gas_joules=natural_gas_joules,
            meter_sources=meter_sources,
        )

    def construct_handles(self, state: int) -> None:
        """Construct handles using template-based approach (same as reference)"""
        logger.info("Constructing handles using template approach...")

        # Convert variables to handles
        def replace_variables_with_handles(var: VariableHole) -> VariableHandle:
            handle = api.exchange.get_variable_handle(state, var.variable_name, var.variable_key)
            if handle < 0:
                logger.warning(f"Failed to get variable handle for {var.variable_name} in {var.variable_key}")
                # For zone temperature, try alternative variable names
                if var.variable_name == "Zone Mean Air Temperature":
                    alt_handle = api.exchange.get_variable_handle(state, "Zone Air Temperature", var.variable_key)
                    if alt_handle >= 0:
                        logger.info(f"Using alternative Zone Air Temperature for {var.variable_key}")
                        return VariableHandle(alt_handle)
                raise ValueError(f"Failed to get variable handle for {var.variable_name} in {var.variable_key}")
            return VariableHandle(handle)

        obs_with_variable_handles = template.search_replace(
            self.observation_template,
            VariableHole,
            replace_variables_with_handles,
        )

        # Convert meters to handles
        def replace_meters_with_handles(meter: MeterHole) -> MeterHandle:
            handle = api.exchange.get_meter_handle(state, meter.meter_name)
            if handle < 0:
                logger.warning(f"Failed to get meter handle for {meter.meter_name}")
                # For energy meters, this might be OK if building doesn't have gas
                if meter.meter_name == HVAC_NATURAL_GAS_METER_NAME:
                    logger.info("Legacy natural gas meter unavailable; ledger records absence, policy input uses a zero placeholder")
                    # Return a dummy handle that we'll handle in observation processing
                    return MeterHandle(-1)
                raise ValueError(f"Failed to get meter handle for {meter.meter_name}")
            return MeterHandle(handle)

        obs_with_variable_and_meter_handles = template.search_replace(
            obs_with_variable_handles,
            MeterHole,
            replace_meters_with_handles,
        )

        self.observation_template_with_handles = obs_with_variable_and_meter_handles

        manifest_runtime = getattr(self, "manifest_control_runtime", None)
        if manifest_runtime is not None:
            manifest_runtime.resolve_handles(api.exchange, state)
            logger.info(
                "Constructed %d manifest actuator handles and %d effect handles",
                len(manifest_runtime.actuator_handles),
                len(manifest_runtime.effect_handles),
            )
            return

        # Create actuator control handles
        warmup_policy_mapping = self._warmup_policy_actuator_mapping()
        warmup_key_by_policy_name = {
            policy_name: contract_key for contract_key, policy_name in warmup_policy_mapping.items()
        }
        self.actuator_control_handles = {}
        for actuator_name, actuator_spec in self.actuators.items():
            warmup_key = warmup_key_by_policy_name.get(actuator_name)
            handle = None if warmup_key is None else self.warmup_preconditioning_handles.get(warmup_key)
            if handle is None:
                handle = api.exchange.get_actuator_handle(
                    state, actuator_spec.component_type, actuator_spec.control_type, actuator_spec.actuator_key
                )
            if handle < 0:
                raise ValueError(f"Failed to get actuator handle for {actuator_spec.actuator_key}")
            self.actuator_control_handles[actuator_name] = handle

        logger.info(f"Constructed handles for {len(self.actuator_control_handles)} actuators")

    def _validate_and_process_observations(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate and process observations to match your exact spaces.py expectations
        """
        processed_obs = {}

        # Process all observations
        for key, value in obs.items():
            if isinstance(value, MeterHandle):
                # This shouldn't happen if template processing worked correctly
                processed_obs[key] = 0.0
            else:
                if value is None:
                    # Handle None values
                    if key == "hvac_gas":
                        processed_obs[key] = 0.0  # Gas meter often missing
                    else:
                        raise ValueError(f"Observation value is None for key: {key}")
                else:
                    processed_obs[key] = float(value)

        # STRICT VALIDATION: Check all required observations are present
        required_obs = []

        # Zone temperatures
        for zone in self.zone_names:
            temp_key = f"{zone}_temperature"
            required_obs.append(temp_key)
            if self.include_zone_occupancy:
                required_obs.append(f"{zone}_occupant_count")

        required_obs.extend(f"{zone}_temperature" for zone in getattr(self, "additional_temperature_zones", ()))

        # Outdoor conditions
        required_obs.extend(["outdoor_temperature", "outdoor_humidity"])

        # Energy (electricity is required, gas is optional)
        required_obs.append("hvac_electricity")

        # Time features
        required_obs.extend(["hour_of_day", "day_of_week", "day_of_year"])

        # Check for missing required observations
        missing_obs = [obs for obs in required_obs if obs not in processed_obs]
        if missing_obs:
            raise ValueError(f"Missing required observations: {missing_obs}. Available: {list(processed_obs.keys())}")

        for zone in getattr(self, "additional_temperature_zones", ()):
            if not math.isfinite(processed_obs[f"{zone}_temperature"]):
                raise ValueError(f"Nonfinite additional zone temperature: {zone}")

        # Add gas if missing (only fallback allowed)
        if "hvac_gas" not in processed_obs:
            processed_obs["hvac_gas"] = 0.0

        # EnergyPlus' Site Outdoor Air Relative Humidity output is always a
        # percentage. Preserve a single unambiguous public unit by validating
        # that raw API range and converting exactly once to a fraction.
        outdoor_humidity_percent = processed_obs["outdoor_humidity"]
        # Allow only floating-point roundoff at the physical endpoints.
        humidity_roundoff_percent = 1e-10
        if (not math.isfinite(outdoor_humidity_percent)
                or not -humidity_roundoff_percent <= outdoor_humidity_percent <= 100.0 + humidity_roundoff_percent):
            raise ValueError(
                "EnergyPlus outdoor_humidity percentage must be finite and lie in [0, 100], "
                f"got {outdoor_humidity_percent!r}"
            )
        processed_obs["outdoor_humidity"] = min(100.0, max(0.0, outdoor_humidity_percent)) / 100.0

        logger.debug(f"Processed observations: {list(processed_obs.keys())}")
        return processed_obs

    def _apply_setpoint_actions(self, state: int, setpoints: Dict[str, Tuple[float, float]]):
        """
        Apply setpoint actions using your exact format: Dict[zone_name, (heating_sp, cooling_sp)]
        """
        if not isinstance(setpoints, dict) or not setpoints:
            raise ValueError("Thermostat action requires a nonempty zone mapping")
        if set(setpoints) != set(self.zone_names):
            raise ValueError("Thermostat action must cover exactly the controlled zones")
        commands = []
        for zone_name, pair in setpoints.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each zone requires heating and cooling setpoints")
            heating_sp, cooling_sp = pair
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(value) for value in pair):
                raise ValueError("Thermostat setpoints must be finite real values")
            if heating_sp >= cooling_sp:
                raise ValueError("Heating setpoint must be below cooling setpoint")
            for channel, value in (("heating", heating_sp), ("cooling", cooling_sp)):
                key = f"{zone_name}_{channel}"
                handle = self.actuator_control_handles.get(key)
                if type(handle) is not int or handle < 0:
                    raise RuntimeError(f"Missing or invalid actuator handle for {key}")
                commands.append((handle, float(value)))
        if len({handle for handle, _ in commands}) != len(commands):
            raise RuntimeError("Thermostat endpoints require unique actuator handles")
        # Validate the whole action before performing any simulator writes.
        for handle, value in commands:
            api.exchange.set_actuator_value(state, handle, value)
        self.pending_thermostat_setpoints = {
            zone: (float(pair[0]), float(pair[1])) for zone, pair in setpoints.items()
        }
        self.last_thermostat_readback = None

    def _capture_thermostat_readback(self, state: int) -> None:
        """Capture actuator values for the just-completed zone interval.

        These are actuator values, not zone temperatures or proof of equipment
        response. Keep differences from submitted commands visible in artifacts.
        """
        submitted = getattr(self, 'pending_thermostat_setpoints', None)
        self.last_thermostat_readback = None
        if submitted is None:
            return
        records = []
        for zone in self.zone_names:
            record = {'zone': zone}
            for channel in ('heating', 'cooling'):
                handle = self.actuator_control_handles[f'{zone}_{channel}']
                value = api.exchange.get_actuator_value(state, handle)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise RuntimeError('Thermostat actuator readback must be finite')
                record[f'{channel}_setpoint_c'] = float(value)
            records.append(record)
        self.last_thermostat_readback = {
            'available': True, 'phase': 'end_zone_timestep', 'records': records,
        }
        self.pending_thermostat_setpoints = None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _persist_warmup_preconditioning_evidence(self) -> Path:
        """Persist the exact completed callback contract without overwriting."""

        contract = self.warmup_preconditioning
        context = self.warmup_preconditioning_context
        if not contract.enabled or context is None:
            raise RuntimeError("cannot persist evidence for disabled warmup preconditioning")

        model_path = Path(self.building_path)
        if model_path.is_symlink() or not model_path.is_file():
            raise RuntimeError("warmup-preconditioned materialized model must be a regular file")
        model_sha256 = self._sha256_file(model_path)
        if model_sha256 != context.materialized_model_sha256:
            raise RuntimeError("materialized model changed after warmup runtime context creation")
        weather_path = Path(self.weather_path)
        if weather_path.is_symlink() or not weather_path.is_file():
            raise RuntimeError("warmup preconditioning weather asset must be a regular file")
        if self._sha256_file(weather_path) != context.weather_sha256:
            raise RuntimeError("weather asset changed after warmup runtime context creation")
        try:
            model = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("cannot read materialized warmup-preconditioned model") from error
        run_period = model.get("RunPeriod") if isinstance(model, dict) else None
        if run_period != context.run_period_mapping:
            raise RuntimeError("materialized RunPeriod changed after warmup runtime context creation")
        run_period_sha256 = canonical_mapping_sha256(run_period)
        if run_period_sha256 != context.run_period_sha256:
            raise RuntimeError("materialized RunPeriod identity changed after context creation")
        buildings = model.get("Building") if isinstance(model, dict) else None
        if not isinstance(buildings, dict) or len(buildings) != 1:
            raise RuntimeError("materialized warmup evidence requires exactly one Building object")
        building = next(iter(buildings.values()))
        if not isinstance(building, dict):
            raise RuntimeError("materialized Building object is malformed")
        materialized_cap = building.get("maximum_number_of_warmup_days")
        if (
            isinstance(materialized_cap, bool)
            or not isinstance(materialized_cap, int)
            or materialized_cap != contract.maximum_warmup_days
        ):
            raise RuntimeError("materialized maximum warmup days do not match the runtime contract")
        schedules = model.get("Schedule:Constant")
        if not isinstance(schedules, dict):
            raise RuntimeError("materialized warmup evidence requires Schedule:Constant objects")
        schedule_evidence = []
        for expected_name in contract.underlying_schedule_names:
            matches = [
                value
                for name, value in schedules.items()
                if isinstance(name, str) and name.casefold() == expected_name.casefold()
            ]
            if len(matches) != 1 or not isinstance(matches[0], dict):
                raise RuntimeError(f"materialized cooling schedule {expected_name!r} is unavailable")
            actual_value = matches[0].get("hourly_value")
            if (
                isinstance(actual_value, bool)
                or not isinstance(actual_value, (int, float))
                or float(actual_value) != contract.underlying_schedule_constant_value_c
            ):
                raise RuntimeError(f"materialized cooling schedule {expected_name!r} changed")
            schedule_evidence.append(
                {
                    "name": expected_name,
                    "hourly_value_c": float(actual_value),
                }
            )

        output_dir = Path(self.config.output_dir)
        eio_path = output_dir / "eplusout.eio"
        try:
            eio_text = eio_path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise RuntimeError("cannot read EnergyPlus EIO warmup evidence") from error
        warmup_days = [int(value) for value in re.findall(r"Environment:WarmupDays,\s*(\d+)", eio_text)]
        if len(warmup_days) != 1:
            raise RuntimeError(
                "EnergyPlus EIO must contain exactly one Environment:WarmupDays record; " f"found {warmup_days!r}"
            )
        cooling_coil_lines = [line for line in eio_text.splitlines() if "Coil:Cooling:DX:SingleSpeed" in line]
        cooling_coil_payload = ("\n".join(cooling_coil_lines) + "\n").encode("utf-8") if cooling_coil_lines else b""
        sizing_fingerprint = {"line_count": len(cooling_coil_lines), "sha256": hashlib.sha256(cooling_coil_payload).hexdigest()}
        if contract.cooling_coil_types:
            from building_gen.benchmark.coil_sizing import cooling_coil_sizing_fingerprint
            typed = cooling_coil_sizing_fingerprint(eio_text, coil_types=list(contract.cooling_coil_types))
            sizing_fingerprint = {key: typed[key] for key in ("line_count", "sha256")}


        payload = {
            "schema_version": warmup_preconditioning_evidence_schema_version(contract),
            "task_id": context.task_id,
            "contract": {
                "payload": contract.identity_payload(),
                "sha256": contract.sha256,
            },
            "assets": {
                "source_building_sha256": context.source_building_sha256,
                "weather_sha256": context.weather_sha256,
            },
            "materialized_model_sha256": model_sha256,
            "materialized_model_contract": {
                "maximum_number_of_warmup_days": materialized_cap,
                "underlying_schedule_constant_values_c": schedule_evidence,
            },
            "run_period": run_period,
            "run_period_sha256": run_period_sha256,
            "callback_evidence": json.loads(
                json.dumps(
                    self.warmup_preconditioning_evidence,
                    allow_nan=False,
                    sort_keys=True,
                )
            ),
            "observed_warmup_days": warmup_days[0],
            "cooling_coil_sizing_eio": sizing_fingerprint,
        }
        contents = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        destination = output_dir / WARMUP_PRECONDITIONING_FILENAME
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"refusing to replace stale warmup preconditioning evidence {destination}")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(contents)
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise RuntimeError(
                    f"refusing to replace stale warmup preconditioning evidence {destination}"
                ) from error
        finally:
            temporary.unlink(missing_ok=True)
        return destination.resolve()

    def _filter(self, result: Result) -> Tuple[Dict[str, float], bool, CarrierEnergyLedger | None]:
        """Filter result (same logic as reference)"""
        if isinstance(result, StepResult):
            self.last_clock = result.clock
            self.last_runtime_context = result.runtime_context
            self.last_energy_interval_clock = result.energy_interval_clock
            self.last_energy_interval_runtime_context = result.energy_interval_runtime_context
            self.last_energy_interval_observation = result.energy_interval_observation
            return (result.observation, result.finished, result.energy_ledger)
        elif isinstance(result, DoneResult):
            self.last_exit_code = result.exit_code
            if result.exit_code != 0:
                raise SimulationCrashed(
                    f"energyplus exited with error code {result.exit_code}. See the eplusout.err file"
                )
            else:
                return ({}, True, None)
        elif isinstance(result, ExceptionResult):
            raise result.exn
        else:
            raise Exception("this channel should receive only a `StepResult` or a `DoneResult`.")

    def _get_obs_and_filter(
        self,
    ) -> Tuple[Dict[str, float], bool, CarrierEnergyLedger | None]:
        """Get observation and filter"""
        logger.info("Waiting for first observation from EnergyPlus...")
        result = self.obs_chan.get()
        logger.info(f"Received result type: {type(result)}")
        return self._filter(result)

    def step(self, setpoints: Union[Dict[str, Tuple[float, float]], NativeScheduleAction]) -> Tuple[Dict[str, float], bool, CarrierEnergyLedger]:
        """
        Take a step using your exact action format

        Args:
            setpoints: Dict[zone_name, (heating_setpoint, cooling_setpoint)]

        Returns:
            Tuple of (observations, done) where observations match your spaces.py structure
        """
        self.act_chan.put(setpoints)
        result = self.obs_chan.get()
        if not isinstance(result, StepResult):
            self._filter(result)
            raise SimulationCrashed("EnergyPlus ended before emitting the terminal observation")

        observation, finished, ledger = self._filter(result)
        if finished:
            # Consume and validate EnergyPlus' final exit status. This also lets
            # the simulation thread complete before the terminal transition is
            # returned to Gymnasium.
            final_result = self.obs_chan.get()
            _, done, final_ledger = self._filter(final_result)
            if not done or final_ledger is not None:
                raise RuntimeError("Expected EnergyPlus completion after terminal observation")
        if ledger is None:
            raise RuntimeError("A non-empty EnergyPlus observation must include an energy ledger")
        return observation, finished, ledger

    def _dispose_state(self):
        """Release worker-owned native state without racing an external stop."""
        with _STATE_LIFECYCLE_LOCK:
            state = self.state
            self.state = None
            if state is not None:
                api.state_manager.delete_state(state)
                owned = id(self) in _ACTIVE_INTERFACE_STATES
                _ACTIVE_INTERFACE_STATES.discard(id(self))
                if owned and not _ACTIVE_INTERFACE_STATES:
                    # Runtime retains ctypes callbacks globally. All wrapper-owned
                    # native states are now deleted, so none can invoke them.
                    api.runtime.clear_callbacks()

    def close(self):
        """Close the interface and clean up resources"""
        logger.info("EnergyPlusInterface close called")
        try:
            # Request an engine stop before closing rendezvous channels. The
            # worker owns state deletion in its ``finally`` block; this method
            # must never delete a state concurrently with EnergyPlus.
            with _STATE_LIFECYCLE_LOCK:
                state = self.state
                if state is not None:
                    api.runtime.stop_simulation(state)
        except Exception as e:
            logger.warning(f"Error requesting engine stop: {e}")
        finally:
            # Channel cancellation must happen even if the engine stop fails.
            self.obs_chan.close()
            self.act_chan.close()

    def _extract_zone_names(self, building_file: str) -> List[str]:
        """Extract controllable zone names from building file using direct JSON parsing"""
        try:
            if not os.path.exists(building_file):
                logger.error(f"Building file not found: {building_file}")
                return []

            with open(building_file, "r") as f:
                data = json.load(f)

            if "Zone" not in data:
                logger.warning("No 'Zone' section found")
                return []

            all_zones = list(data["Zone"].keys())
            logger.info(f"Found {len(all_zones)} total zones")

            # Filter for controllable zones using your approach (not RDF)
            controllable_zones = self._filter_controllable_zones(data, all_zones)

            logger.info(f"Filtered to {len(controllable_zones)} controllable zones")
            return sorted(controllable_zones) if controllable_zones else []

        except Exception as e:
            logger.error(f"Error extracting zone names: {e}")
            return []

    def _filter_controllable_zones(self, building_data: dict, all_zones: List[str]) -> List[str]:
        """Filter zones to only include controllable ones using direct JSON parsing (no RDF)"""
        conditioned_zones = set()

        # Method 1: Check for ZoneHVAC:EquipmentConnections (most reliable)
        if "ZoneHVAC:EquipmentConnections" in building_data:
            for conn_name, conn_data in building_data["ZoneHVAC:EquipmentConnections"].items():
                zone_name = conn_data.get("zone_name")
                if zone_name and zone_name in all_zones:
                    conditioned_zones.add(zone_name)

        # Method 2: Check for HVAC equipment that references zones
        hvac_equipment_types = [
            "ZoneHVAC:IdealLoadsAirSystem",
            "ZoneHVAC:PackagedTerminalAirConditioner",
            "ZoneHVAC:FourPipeFanCoil",
            "ZoneHVAC:UnitHeater",
            "ZoneHVAC:UnitVentilator",
            "ZoneHVAC:PackagedTerminalHeatPump",
            "ZoneHVAC:WindowAirConditioner",
            "ZoneHVAC:Baseboard:Convective:Electric",
            "ZoneHVAC:Baseboard:Convective:Water",
        ]

        for hvac_type in hvac_equipment_types:
            if hvac_type in building_data:
                for equip_name, equip_data in building_data[hvac_type].items():
                    zone_name = equip_data.get("zone_name") or equip_data.get("zone_or_zonelist_name")
                    if zone_name and zone_name in all_zones:
                        conditioned_zones.add(zone_name)

        # Method 3: Check for thermostats (backup method)
        if "ZoneControl:Thermostat" in building_data:
            for thermostat_name, thermostat_config in building_data["ZoneControl:Thermostat"].items():
                zone_name = thermostat_config.get("zone_or_zonelist_name", "")
                if zone_name and zone_name in all_zones:
                    conditioned_zones.add(zone_name)

        return sorted(list(conditioned_zones))
