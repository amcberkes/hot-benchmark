"""Deterministic exact-byte instrumentation for authenticated B2B VAV loops."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from building_gen.control.capabilities import (
    BUILDING2BUILDING_COMMIT,
    ActionBindingKind,
    ActionCommand,
    ActionCommandUnit,
    ActuatorDescriptor,
    ActuatorRole,
    BoundSource,
    CandidateDisposition,
    ControlCapabilityManifest,
    ControlChannel,
    DiscoveredEquipmentCandidate,
    DiscoveryEnumeratorContract,
    EquipmentDescriptor,
    EquipmentFamily,
    FixedEndpointValue,
    InstrumentationMode,
    InstrumentationReceipt,
    MutationRecord,
    MutationTargetState,
    OutputVariableDescriptor,
    OutputVariableSemantic,
    PhysicalUnit,
    SupplyOutletNodeEvidence,
    TaskActionBinding,
    TopologyEdge,
    TopologyNode,
    TopologyNodeKind,
    TopologyRelation,
    UnsupportedEquipmentRecord,
    candidate_inventory_sha256,
    energyplus_topology_node_id,
    mutation_sequence_sha256,
    supply_outlet_inventory_sha256,
)
from building_gen.control.discovery import energyplus_source_object_sha256
from building_gen.control.instrumentation import (
    ControlInstrumentationError,
    _build_mutations,
    _canonical_epjson_bytes,
    _field_text,
    _mark_change,
    _ModelIndex,
    _owned_bytes,
    _parse_epjson_bytes,
    _PlannedChange,
    _records,
    _text,
)
from building_gen.control.vav_discovery import (
    BUILDING2BUILDING_ACTUATORS_SOURCE_PATH,
    BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256,
    VAV_LOOP_OBJECT_TYPE,
    VAV_OA_CONTROLLER_OBJECT_TYPE,
    VAV_OA_SCHEDULE_FIELDS,
    VAV_PROFILE_ID,
    VAV_TERMINAL_OBJECT_TYPE,
    VAVDiscoveryError,
    VAVDiscoveryInventory,
    VAVLoopDiscoveryRecord,
    VAVObjectReference,
    VAVTerminalDiscoveryRecord,
    verify_vav_discovery_inventory_from_json_bytes,
)

VAV_INSTRUMENTATION_SCHEMA_VERSION = "hot-b2b-vav-instrumentation/2.0.0"
VAV_INSTRUMENTATION_ALGORITHM_ID = "b2b-v1-vav-exact-byte-instrumentation-v2"
VAV_INITIAL_SUPPLY_AIR_TEMPERATURE_C = 13.0
VAV_INITIAL_MINIMUM_FLOW_FRACTION = 0.3
VAV_INITIAL_HEATING_SETPOINT_C = 21.0
VAV_FIXED_COOLING_SETPOINT_C = 40.0
VAV_SUPPLY_AIR_TEMPERATURE_BOUNDS_C = (10.0, 55.0)
VAV_OUTDOOR_AIR_MASS_FLOW_BOUNDS_KG_S = (0.0, 5.0)
VAV_MINIMUM_FLOW_FRACTION_BOUNDS = (0.0, 1.0)
VAV_HEATING_SETPOINT_BOUNDS_C = (10.0, 35.0)
VAV_COOLING_SETPOINT_BOUNDS_C = (18.0, 40.0)
VAV_EFFECT_MASS_FLOW_UPPER_BOUND_KG_S = 15.0

_ERL_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ERL_SET_RE = re.compile(r"^\s*SET\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.IGNORECASE)

# EnergyPlus 24.1 exposes these exact setpoint control-variable dimensions.
# Most SetpointManager objects carry ``control_variable`` explicitly. The
# remaining public input objects encode the dimension in their exact object
# type, so they are exhaustively classified here instead of being inferred
# from an object or instance name.
_TEMPERATURE_SETPOINT_CONTROL_VARIABLES = frozenset(
    {
        "temperature",
        "maximumtemperature",
        "minimumtemperature",
    }
)
_NON_TEMPERATURE_SETPOINT_CONTROL_VARIABLES = frozenset(
    {
        "humidityratio",
        "maximumhumidityratio",
        "minimumhumidityratio",
        "massflowrate",
        "maximummassflowrate",
        "minimummassflowrate",
    }
)
_DEFAULT_TEMPERATURE_SETPOINT_MANAGER_TYPES = frozenset(
    item.casefold()
    for item in (
        "SetpointManager:Scheduled:DualSetpoint",
        "SetpointManager:OutdoorAirReset",
        "SetpointManager:SingleZone:Reheat",
        "SetpointManager:SingleZone:Heating",
        "SetpointManager:SingleZone:Cooling",
        "SetpointManager:MixedAir",
        "SetpointManager:Warmest",
        "SetpointManager:Coldest",
        "SetpointManager:FollowOutdoorAirTemperature",
        "SetpointManager:FollowSystemNodeTemperature",
        "SetpointManager:FollowGroundTemperature",
        "SetpointManager:CondenserEnteringReset",
        "SetpointManager:CondenserEnteringReset:Ideal",
    )
)
_DEFAULT_NON_TEMPERATURE_SETPOINT_MANAGER_TYPES = frozenset({"setpointmanager:returnairbypassflow"})
_IMPLICIT_TEMPERATURE_SETPOINT_MANAGER_TYPES = frozenset(
    item.casefold()
    for item in (
        "SetpointManager:MultiZone:Cooling:Average",
        "SetpointManager:MultiZone:Heating:Average",
        "SetpointManager:SingleZone:OneStageCooling",
        "SetpointManager:SingleZone:OneStageHeating",
        "SetpointManager:ReturnTemperature:ChilledWater",
        "SetpointManager:ReturnTemperature:HotWater",
    )
)
_IMPLICIT_NON_TEMPERATURE_SETPOINT_MANAGER_TYPES = frozenset(
    item.casefold()
    for item in (
        "SetpointManager:SingleZone:Humidity:Minimum",
        "SetpointManager:SingleZone:Humidity:Maximum",
        "SetpointManager:MultiZone:MinimumHumidity:Average",
        "SetpointManager:MultiZone:MaximumHumidity:Average",
        "SetpointManager:MultiZone:Humidity:Minimum",
        "SetpointManager:MultiZone:Humidity:Maximum",
    )
)


@dataclass(frozen=True, slots=True)
class _VAVLoopNames:
    token: str
    availability_type_limits: str
    always_on_schedule: str
    sat_type_limits: str
    heating_type_limits: str
    cooling_type_limits: str
    fraction_type_limits: str
    sat_schedule: str
    sat_manager: str
    availability_manager: str


@dataclass(frozen=True, slots=True)
class _VAVTerminalNames:
    token: str
    flow_schedule: str
    heating_schedule: str
    cooling_schedule: str


@dataclass(frozen=True, slots=True)
class _VAVTerminalContext:
    record: VAVTerminalDiscoveryRecord
    zone_id: str
    terminal_id: str
    names: _VAVTerminalNames
    flow_actuator: ActuatorDescriptor
    heating_actuator: ActuatorDescriptor
    cooling_actuator: ActuatorDescriptor


@dataclass(frozen=True, slots=True)
class _VAVLoopContext:
    record: VAVLoopDiscoveryRecord
    equipment: EquipmentDescriptor
    loop_id: str
    oa_id: str
    names: _VAVLoopNames
    sat_actuator: ActuatorDescriptor
    oa_actuator: ActuatorDescriptor
    terminals: tuple[_VAVTerminalContext, ...]

    @property
    def actuators(self) -> tuple[ActuatorDescriptor, ...]:
        return (
            self.sat_actuator,
            *(
                item
                for terminal in self.terminals
                for item in (
                    terminal.flow_actuator,
                    terminal.heating_actuator,
                    terminal.cooling_actuator,
                )
            ),
            self.oa_actuator,
        )

    @property
    def endpoint_ids(self) -> tuple[str, ...]:
        return tuple(item.endpoint_id for item in self.actuators)


def _loop_names(record: VAVLoopDiscoveryRecord) -> _VAVLoopNames:
    token = hashlib.sha256(
        b"HOT B2B VAV loop generated names v1\0" + record.object_name.casefold().encode("utf-8")
    ).hexdigest()[:20]
    prefix = f"HOT B2B VAV {token}"
    return _VAVLoopNames(
        token=token,
        availability_type_limits=f"{prefix} Availability Type Limits",
        always_on_schedule=f"{prefix} Always On Availability",
        sat_type_limits=f"{prefix} SAT Type Limits",
        heating_type_limits=f"{prefix} Heating Type Limits",
        cooling_type_limits=f"{prefix} Cooling Type Limits",
        fraction_type_limits=f"{prefix} Minimum Flow Type Limits",
        sat_schedule=f"{prefix} SAT Setpoint",
        sat_manager=f"{prefix} SAT Manager",
        availability_manager=f"{prefix} Loop Availability",
    )


def _terminal_names(loop: VAVLoopDiscoveryRecord, terminal: VAVTerminalDiscoveryRecord) -> _VAVTerminalNames:
    identity = (
        loop.object_name.casefold()
        + "\0"
        + terminal.terminal.object_name.casefold()
        + "\0"
        + terminal.zone.object_name.casefold()
    ).encode("utf-8")
    token = hashlib.sha256(b"HOT B2B VAV terminal generated names v1\0" + identity).hexdigest()[:20]
    prefix = f"HOT B2B VAV Terminal {token}"
    return _VAVTerminalNames(
        token=token,
        flow_schedule=f"{prefix} Minimum Flow Fraction",
        heating_schedule=f"{prefix} Heating Setpoint",
        cooling_schedule=f"{prefix} Cooling Setpoint",
    )


def _assert_object_reference(index: _ModelIndex, reference: VAVObjectReference) -> tuple[str, dict[str, Any]]:
    object_name, fields = index.lookup(reference.object_type, reference.object_name)
    if object_name != reference.object_name:
        raise ControlInstrumentationError("VAV discovery reference does not use the canonical object name")
    actual_sha = energyplus_source_object_sha256(reference.object_type, object_name, fields)
    if not hmac.compare_digest(actual_sha, reference.source_object_sha256):
        raise ControlInstrumentationError(f"VAV discovery hash does not match {reference.object_type}/{object_name}")
    return object_name, fields


def _build_contexts(
    index: _ModelIndex,
    inventory: VAVDiscoveryInventory,
    *,
    allow_unsupported_records: bool = False,
) -> tuple[_VAVLoopContext, ...]:
    if not inventory.records:
        raise ControlInstrumentationError("VAV discovery inventory is empty")
    unsupported = [item.object_name for item in inventory.unsupported_records]
    if unsupported and not allow_unsupported_records:
        raise ControlInstrumentationError(f"VAV instrumentation refuses incomplete candidate support: {unsupported!r}")
    if not inventory.supported_records:
        raise ControlInstrumentationError("VAV instrumentation requires at least one supported VAV loop")
    contexts: list[_VAVLoopContext] = []
    for record in inventory.supported_records:
        loop_name, loop = index.lookup(VAV_LOOP_OBJECT_TYPE, record.object_name)
        if loop_name != record.object_name:
            raise ControlInstrumentationError("VAV record must use the canonical AirLoopHVAC name")
        actual_loop_sha = energyplus_source_object_sha256(VAV_LOOP_OBJECT_TYPE, loop_name, loop)
        if not hmac.compare_digest(actual_loop_sha, record.source_object_sha256):
            raise ControlInstrumentationError("VAV loop object hash disagrees with exact source epJSON")
        if record.outdoor_air_controller is None:
            raise ControlInstrumentationError("supported VAV record lacks its outdoor-air controller")
        _assert_object_reference(index, record.outdoor_air_controller)
        for fan in record.fans:
            _assert_object_reference(index, fan)
        if record.availability_assignment_list is not None:
            _assert_object_reference(index, record.availability_assignment_list)
        for manager in record.availability_managers:
            _assert_object_reference(index, manager)

        loop_id = energyplus_topology_node_id(
            TopologyNodeKind.AIR_LOOP,
            VAV_LOOP_OBJECT_TYPE,
            loop_name,
        )
        zone_ids = tuple(
            energyplus_topology_node_id(TopologyNodeKind.ZONE, "Zone", item.zone.object_name)
            for item in record.terminals
        )
        equipment = EquipmentDescriptor(
            profile_id=VAV_PROFILE_ID,
            equipment_id=loop_id,
            family=EquipmentFamily.VAV,
            object_type=VAV_LOOP_OBJECT_TYPE,
            object_name=loop_name,
            served_entity_ids=zone_ids,
        )
        names = _loop_names(record)
        sat = ActuatorDescriptor(
            profile_id=VAV_PROFILE_ID,
            equipment_id=loop_id,
            control_node_id=loop_id,
            affected_entity_ids=zone_ids,
            family=EquipmentFamily.VAV,
            channel=ControlChannel.SUPPLY_AIR_TEMPERATURE_C,
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=names.sat_schedule,
            unit=PhysicalUnit.CELSIUS,
            lower_bound=VAV_SUPPLY_AIR_TEMPERATURE_BOUNDS_C[0],
            upper_bound=VAV_SUPPLY_AIR_TEMPERATURE_BOUNDS_C[1],
            lower_bound_source=BoundSource.PINNED_STATIC,
            upper_bound_source=BoundSource.PINNED_STATIC,
        )
        oa_id = energyplus_topology_node_id(
            TopologyNodeKind.OUTDOOR_AIR_CONTROLLER,
            VAV_OA_CONTROLLER_OBJECT_TYPE,
            record.outdoor_air_controller.object_name,
        )
        oa = ActuatorDescriptor(
            profile_id=VAV_PROFILE_ID,
            equipment_id=loop_id,
            control_node_id=oa_id,
            affected_entity_ids=zone_ids,
            family=EquipmentFamily.VAV,
            channel=ControlChannel.OUTDOOR_AIR_MASS_FLOW_KG_S,
            component_type="Outdoor Air Controller",
            control_type="Air Mass Flow Rate",
            component_name=record.outdoor_air_controller.object_name,
            unit=PhysicalUnit.KILOGRAMS_PER_SECOND,
            lower_bound=VAV_OUTDOOR_AIR_MASS_FLOW_BOUNDS_KG_S[0],
            upper_bound=VAV_OUTDOOR_AIR_MASS_FLOW_BOUNDS_KG_S[1],
            lower_bound_source=BoundSource.PINNED_STATIC,
            upper_bound_source=BoundSource.PINNED_STATIC,
        )

        terminals: list[_VAVTerminalContext] = []
        for terminal_record, zone_id in zip(record.terminals, zone_ids):
            for reference in (
                terminal_record.zone,
                terminal_record.terminal,
                terminal_record.thermostat,
                terminal_record.dual_setpoint,
                terminal_record.heating_schedule,
                terminal_record.cooling_schedule,
            ):
                _assert_object_reference(index, reference)
            terminal_id = energyplus_topology_node_id(
                TopologyNodeKind.TERMINAL,
                VAV_TERMINAL_OBJECT_TYPE,
                terminal_record.terminal.object_name,
            )
            terminal_names = _terminal_names(record, terminal_record)
            common = {
                "profile_id": VAV_PROFILE_ID,
                "equipment_id": loop_id,
                "affected_entity_ids": (zone_id,),
                "family": EquipmentFamily.VAV,
                "component_type": "Schedule:Constant",
                "control_type": "Schedule Value",
                "lower_bound_source": BoundSource.PINNED_STATIC,
                "upper_bound_source": BoundSource.PINNED_STATIC,
            }
            flow = ActuatorDescriptor(
                **common,
                control_node_id=terminal_id,
                channel=ControlChannel.ZONE_MINIMUM_AIR_FLOW_FRACTION,
                component_name=terminal_names.flow_schedule,
                unit=PhysicalUnit.FRACTION,
                lower_bound=VAV_MINIMUM_FLOW_FRACTION_BOUNDS[0],
                upper_bound=VAV_MINIMUM_FLOW_FRACTION_BOUNDS[1],
            )
            heating = ActuatorDescriptor(
                **common,
                control_node_id=zone_id,
                channel=ControlChannel.HEATING_SETPOINT_C,
                component_name=terminal_names.heating_schedule,
                unit=PhysicalUnit.CELSIUS,
                lower_bound=VAV_HEATING_SETPOINT_BOUNDS_C[0],
                upper_bound=VAV_HEATING_SETPOINT_BOUNDS_C[1],
            )
            cooling = ActuatorDescriptor(
                **common,
                control_node_id=zone_id,
                channel=ControlChannel.COOLING_SETPOINT_C,
                component_name=terminal_names.cooling_schedule,
                unit=PhysicalUnit.CELSIUS,
                lower_bound=VAV_COOLING_SETPOINT_BOUNDS_C[0],
                upper_bound=VAV_COOLING_SETPOINT_BOUNDS_C[1],
                role=ActuatorRole.FIXED,
                fixed_value=VAV_FIXED_COOLING_SETPOINT_C,
            )
            terminals.append(
                _VAVTerminalContext(
                    record=terminal_record,
                    zone_id=zone_id,
                    terminal_id=terminal_id,
                    names=terminal_names,
                    flow_actuator=flow,
                    heating_actuator=heating,
                    cooling_actuator=cooling,
                )
            )
        contexts.append(
            _VAVLoopContext(
                record=record,
                equipment=equipment,
                loop_id=loop_id,
                oa_id=oa_id,
                names=names,
                sat_actuator=sat,
                oa_actuator=oa,
                terminals=tuple(terminals),
            )
        )
    return tuple(contexts)


def _preflight_names(index: _ModelIndex, contexts: Sequence[_VAVLoopContext]) -> None:
    identities: list[tuple[str, str]] = []
    for context in contexts:
        names = context.names
        identities.extend(
            (
                ("ScheduleTypeLimits", names.availability_type_limits),
                ("ScheduleTypeLimits", names.sat_type_limits),
                ("ScheduleTypeLimits", names.heating_type_limits),
                ("ScheduleTypeLimits", names.cooling_type_limits),
                ("ScheduleTypeLimits", names.fraction_type_limits),
                ("Schedule:Constant", names.always_on_schedule),
                ("Schedule:Constant", names.sat_schedule),
                ("SetpointManager:Scheduled", names.sat_manager),
            )
        )
        for terminal in context.terminals:
            identities.extend(
                (
                    ("Schedule:Constant", terminal.names.flow_schedule),
                    ("Schedule:Constant", terminal.names.heating_schedule),
                    ("Schedule:Constant", terminal.names.cooling_schedule),
                )
            )
        identities.append(("AvailabilityManager:Scheduled", names.availability_manager))
    folded = [(object_type.casefold(), object_name.casefold()) for object_type, object_name in identities]
    if len(folded) != len(set(folded)):
        raise ControlInstrumentationError("deterministic VAV generated names collide with each other")
    for object_type, object_name in identities:
        index.assert_name_available(object_type, object_name)


def _add_object(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    object_type: str,
    object_name: str,
    fields: dict[str, Any],
    reason_code: str,
    endpoint_ids: Sequence[str],
) -> None:
    index.add(object_type, object_name, fields)
    _mark_change(planned, object_type, object_name, reason_code, endpoint_ids)


def _setpoint_manager_controls_temperature(
    object_type: str,
    object_name: str,
    fields: Mapping[str, Any],
) -> bool:
    """Classify one outlet manager by its exact EnergyPlus control dimension."""

    raw_control_variable = fields.get("control_variable")
    if raw_control_variable is not None:
        control_variable = _text(
            raw_control_variable,
            field_name=f"{object_type}/{object_name}.control_variable",
        ).casefold()
        if control_variable in _TEMPERATURE_SETPOINT_CONTROL_VARIABLES:
            return True
        if control_variable in _NON_TEMPERATURE_SETPOINT_CONTROL_VARIABLES:
            return False
        if object_type.casefold() == "setpointmanager:returnairbypassflow" and control_variable == "flow":
            return False
        raise ControlInstrumentationError(
            f"cannot classify {object_type}/{object_name} control_variable "
            f"{raw_control_variable!r} as a supported EnergyPlus setpoint dimension"
        )

    folded_type = object_type.casefold()
    if folded_type in _DEFAULT_TEMPERATURE_SETPOINT_MANAGER_TYPES:
        return True
    if folded_type in _DEFAULT_NON_TEMPERATURE_SETPOINT_MANAGER_TYPES:
        return False
    if folded_type in _IMPLICIT_TEMPERATURE_SETPOINT_MANAGER_TYPES:
        return True
    if folded_type in _IMPLICIT_NON_TEMPERATURE_SETPOINT_MANAGER_TYPES:
        return False
    raise ControlInstrumentationError(f"cannot classify {object_type}/{object_name} without an exact control_variable")


def _remove_conflicting_sat_managers(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    context: _VAVLoopContext,
) -> None:
    outlet = context.record.supply_outlet_node_name
    assert outlet is not None
    for object_type, object_name, fields in index.iter_objects():
        if not object_type.startswith("SetpointManager:"):
            continue
        raw_node = fields.get("setpoint_node_or_nodelist_name")
        if raw_node is None:
            continue
        node_name = _text(
            raw_node,
            field_name=f"{object_type}/{object_name}.setpoint_node_or_nodelist_name",
        )
        if node_name.casefold() != outlet.casefold():
            continue
        if not _setpoint_manager_controls_temperature(object_type, object_name, fields):
            continue
        index.remove(object_type, object_name)
        _mark_change(
            planned,
            object_type,
            object_name,
            "remove_conflicting_vav_supply_setpoint_manager",
            (context.sat_actuator.endpoint_id,),
        )


def _replace_loop_availability(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    context: _VAVLoopContext,
) -> None:
    assignment_reference = context.record.availability_assignment_list
    if assignment_reference is None:
        raise ControlInstrumentationError("supported VAV loop lacks an availability assignment list")
    list_name, assignment = _assert_object_reference(index, assignment_reference)
    manager_entries = _records(assignment, "managers", owner=f"AvailabilityManagerAssignmentList/{list_name}")
    discovered = {
        (item.object_type.casefold(), item.object_name.casefold()): item
        for item in context.record.availability_managers
    }
    resolved: list[tuple[str, str]] = []
    for manager_entry in manager_entries:
        manager_type = _field_text(
            manager_entry,
            "availability_manager_object_type",
            owner=f"AvailabilityManagerAssignmentList/{list_name}",
        )
        manager_name = _field_text(
            manager_entry,
            "availability_manager_name",
            owner=f"AvailabilityManagerAssignmentList/{list_name}",
        )
        reference = discovered.get((manager_type.casefold(), manager_name.casefold()))
        if reference is None or reference.object_type != manager_type or reference.object_name != manager_name:
            raise ControlInstrumentationError("loop availability mutation is not covered by VAV discovery")
        _assert_object_reference(index, reference)
        resolved.append((manager_type.casefold(), manager_name.casefold()))
    if len(resolved) != len(set(resolved)) or set(resolved) != set(discovered):
        raise ControlInstrumentationError("VAV loop availability inventory was not exhaustively replayed")

    _add_object(
        index,
        planned,
        "AvailabilityManager:Scheduled",
        context.names.availability_manager,
        {"schedule_name": context.names.always_on_schedule},
        "add_vav_always_on_loop_availability",
        context.endpoint_ids,
    )
    assignment["managers"] = [
        {
            "availability_manager_object_type": "AvailabilityManager:Scheduled",
            "availability_manager_name": context.names.availability_manager,
        }
    ]
    _mark_change(
        planned,
        "AvailabilityManagerAssignmentList",
        list_name,
        "replace_vav_loop_availability_with_always_on",
        context.endpoint_ids,
    )
    for reference in context.record.availability_managers:
        index.remove(reference.object_type, reference.object_name)
        _mark_change(
            planned,
            reference.object_type,
            reference.object_name,
            "remove_vav_loop_availability_manager",
            context.endpoint_ids,
        )


def _instrument_loop(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    context: _VAVLoopContext,
) -> None:
    names = context.names
    flow_endpoints = tuple(item.flow_actuator.endpoint_id for item in context.terminals)
    heating_endpoints = tuple(item.heating_actuator.endpoint_id for item in context.terminals)
    cooling_endpoints = tuple(item.cooling_actuator.endpoint_id for item in context.terminals)
    _add_object(
        index,
        planned,
        "ScheduleTypeLimits",
        names.availability_type_limits,
        {
            "lower_limit_value": 0,
            "upper_limit_value": 1,
            "numeric_type": "Discrete",
            "unit_type": "Availability",
        },
        "add_vav_availability_type_limits",
        context.endpoint_ids,
    )
    for object_name, bounds, reason, endpoint_ids, unit_type in (
        (
            names.sat_type_limits,
            VAV_SUPPLY_AIR_TEMPERATURE_BOUNDS_C,
            "add_vav_sat_type_limits",
            (context.sat_actuator.endpoint_id,),
            "Temperature",
        ),
        (
            names.heating_type_limits,
            VAV_HEATING_SETPOINT_BOUNDS_C,
            "add_vav_heating_type_limits",
            heating_endpoints,
            "Temperature",
        ),
        (
            names.cooling_type_limits,
            VAV_COOLING_SETPOINT_BOUNDS_C,
            "add_vav_cooling_type_limits",
            cooling_endpoints,
            "Temperature",
        ),
        (
            names.fraction_type_limits,
            VAV_MINIMUM_FLOW_FRACTION_BOUNDS,
            "add_vav_minimum_flow_type_limits",
            flow_endpoints,
            None,
        ),
    ):
        fields: dict[str, Any] = {
            "lower_limit_value": bounds[0],
            "upper_limit_value": bounds[1],
            "numeric_type": "Continuous",
        }
        if unit_type is not None:
            fields["unit_type"] = unit_type
        _add_object(index, planned, "ScheduleTypeLimits", object_name, fields, reason, endpoint_ids)

    _add_object(
        index,
        planned,
        "Schedule:Constant",
        names.always_on_schedule,
        {"hourly_value": 1, "schedule_type_limits_name": names.availability_type_limits},
        "add_vav_always_on_schedule",
        context.endpoint_ids,
    )
    _add_object(
        index,
        planned,
        "Schedule:Constant",
        names.sat_schedule,
        {
            "hourly_value": VAV_INITIAL_SUPPLY_AIR_TEMPERATURE_C,
            "schedule_type_limits_name": names.sat_type_limits,
        },
        "add_vav_sat_actuator_schedule",
        (context.sat_actuator.endpoint_id,),
    )
    _remove_conflicting_sat_managers(index, planned, context)
    outlet = context.record.supply_outlet_node_name
    assert outlet is not None
    _add_object(
        index,
        planned,
        "SetpointManager:Scheduled",
        names.sat_manager,
        {
            "control_variable": "Temperature",
            "schedule_name": names.sat_schedule,
            "setpoint_node_or_nodelist_name": outlet,
        },
        "add_vav_scheduled_outlet_sat_manager",
        (context.sat_actuator.endpoint_id,),
    )

    for fan_reference in context.record.fans:
        fan_name, fan = _assert_object_reference(index, fan_reference)
        fan["availability_schedule_name"] = names.always_on_schedule
        _mark_change(
            planned,
            fan_reference.object_type,
            fan_name,
            "force_vav_supply_fan_always_available",
            context.endpoint_ids,
        )
    _replace_loop_availability(index, planned, context)

    assert context.record.outdoor_air_controller is not None
    _, oa_controller = _assert_object_reference(index, context.record.outdoor_air_controller)
    discovered_fields = {item.field_name: item for item in context.record.outdoor_air_schedule_references}
    if set(discovered_fields) != {field for field in VAV_OA_SCHEDULE_FIELDS if field in oa_controller}:
        raise ControlInstrumentationError("outdoor-air schedule inventory changed before instrumentation")
    for field_name, schedule in discovered_fields.items():
        if oa_controller.get(field_name) != schedule.schedule.object_name:
            raise ControlInstrumentationError("outdoor-air schedule reference is not canonical")
    # Preserve the exact source availability/fraction schedules. The direct
    # ``Outdoor Air Controller`` API actuator remains authenticated by the
    # capability endpoint and does not require destructive schedule pinning.

    for terminal in context.terminals:
        names_terminal = terminal.names
        _add_object(
            index,
            planned,
            "Schedule:Constant",
            names_terminal.flow_schedule,
            {
                "hourly_value": VAV_INITIAL_MINIMUM_FLOW_FRACTION,
                "schedule_type_limits_name": names.fraction_type_limits,
            },
            "add_vav_minimum_flow_actuator_schedule",
            (terminal.flow_actuator.endpoint_id,),
        )
        _add_object(
            index,
            planned,
            "Schedule:Constant",
            names_terminal.heating_schedule,
            {
                "hourly_value": VAV_INITIAL_HEATING_SETPOINT_C,
                "schedule_type_limits_name": names.heating_type_limits,
            },
            "add_vav_heating_actuator_schedule",
            (terminal.heating_actuator.endpoint_id,),
        )
        _add_object(
            index,
            planned,
            "Schedule:Constant",
            names_terminal.cooling_schedule,
            {
                "hourly_value": VAV_FIXED_COOLING_SETPOINT_C,
                "schedule_type_limits_name": names.cooling_type_limits,
            },
            "add_vav_fixed_cooling_schedule",
            (terminal.cooling_actuator.endpoint_id,),
        )
        terminal_name, terminal_object = _assert_object_reference(index, terminal.record.terminal)
        terminal_object["zone_minimum_air_flow_input_method"] = "Scheduled"
        terminal_object["minimum_air_flow_fraction_schedule_name"] = names_terminal.flow_schedule
        _mark_change(
            planned,
            VAV_TERMINAL_OBJECT_TYPE,
            terminal_name,
            "instrument_vav_terminal_minimum_flow",
            (terminal.flow_actuator.endpoint_id,),
        )
        dual_name, dual = _assert_object_reference(index, terminal.record.dual_setpoint)
        dual["heating_setpoint_temperature_schedule_name"] = names_terminal.heating_schedule
        dual["cooling_setpoint_temperature_schedule_name"] = names_terminal.cooling_schedule
        _mark_change(
            planned,
            "ThermostatSetpoint:DualSetpoint",
            dual_name,
            "instrument_vav_zone_setpoints",
            (terminal.heating_actuator.endpoint_id, terminal.cooling_actuator.endpoint_id),
        )


def _program_lines(program: Mapping[str, Any], *, owner: str) -> tuple[str, ...]:
    result: list[str] = []
    for item in _records(program, "lines", owner=owner):
        result.append(_field_text(item, "program_line", owner=owner))
    return tuple(result)


def _erl_tokens(lines: Sequence[str]) -> set[str]:
    return {match.group(0).casefold() for line in lines for match in _ERL_IDENTIFIER_RE.finditer(line)}


def _thermostat_schedule_endpoint_ids(
    contexts: Sequence[_VAVLoopContext],
) -> dict[tuple[str, str], set[str]]:
    endpoint_ids: dict[tuple[str, str], set[str]] = {}
    for context in contexts:
        for terminal in context.terminals:
            for schedule, actuator in (
                (terminal.record.heating_schedule, terminal.heating_actuator),
                (terminal.record.cooling_schedule, terminal.cooling_actuator),
            ):
                endpoint_ids.setdefault(
                    (schedule.object_type.casefold(), schedule.object_name.casefold()),
                    set(),
                ).add(actuator.endpoint_id)
    return endpoint_ids


def _conflicting_ems_actuators(
    index: _ModelIndex,
    endpoint_ids_by_schedule: Mapping[tuple[str, str], set[str]],
) -> dict[str, tuple[str, dict[str, Any], set[str]]]:
    candidates: dict[str, tuple[str, dict[str, Any], set[str]]] = {}
    table = index.table("EnergyManagementSystem:Actuator") or {}
    for actuator_name, actuator in index._object_index(
        "EnergyManagementSystem:Actuator",
        table,
    ).values():
        raw_values = (
            actuator.get("actuated_component_unique_name"),
            actuator.get("actuated_component_type"),
            actuator.get("actuated_component_control_type"),
        )
        if not all(isinstance(value, str) for value in raw_values):
            continue
        component_name, component_type, control_type = (str.__str__(value) for value in raw_values)
        schedule_endpoints = endpoint_ids_by_schedule.get((component_type.casefold(), component_name.casefold()))
        if schedule_endpoints is None or control_type.casefold() != "schedule value":
            continue
        candidates[actuator_name.casefold()] = (
            actuator_name,
            actuator,
            set(schedule_endpoints),
        )
    return candidates


def _ems_program_inventory(
    index: _ModelIndex,
) -> dict[str, tuple[str, dict[str, Any], tuple[str, ...]]]:
    table = index.table("EnergyManagementSystem:Program") or {}
    return {
        name.casefold(): (
            name,
            program,
            _program_lines(
                program,
                owner=f"EnergyManagementSystem:Program/{name}",
            ),
        )
        for name, program in index._object_index(
            "EnergyManagementSystem:Program",
            table,
        ).values()
    }


def _active_conflicting_ems_programs(
    programs: Mapping[str, tuple[str, dict[str, Any], tuple[str, ...]]],
    candidate_actuators: Mapping[str, tuple[str, dict[str, Any], set[str]]],
) -> tuple[set[str], dict[str, set[str]], set[str]]:
    setters: dict[str, set[str]] = {}
    for program_key, (_, _, lines) in programs.items():
        for line in lines:
            match = _ERL_SET_RE.match(line)
            if match is None:
                continue
            actuator_key = match.group(1).casefold()
            if actuator_key in candidate_actuators:
                setters.setdefault(program_key, set()).add(actuator_key)
    active_actuator_keys = {key for values in setters.values() for key in values}
    programs_to_remove = {key for key, (_, _, lines) in programs.items() if _erl_tokens(lines) & active_actuator_keys}
    program_endpoints: dict[str, set[str]] = {}
    for program_key in programs_to_remove:
        referenced = _erl_tokens(programs[program_key][2]) & active_actuator_keys
        endpoints: set[str] = set()
        for actuator_key in referenced:
            endpoints.update(candidate_actuators[actuator_key][2])
        program_endpoints[program_key] = endpoints
    return programs_to_remove, program_endpoints, active_actuator_keys


def _remove_conflicting_ems_calling_references(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    programs_to_remove: set[str],
    program_endpoints: Mapping[str, set[str]],
) -> None:
    table = index.table("EnergyManagementSystem:ProgramCallingManager") or {}
    managers = tuple(
        index._object_index(
            "EnergyManagementSystem:ProgramCallingManager",
            table,
        ).values()
    )
    for manager_name, manager in managers:
        programs = _records(
            manager,
            "programs",
            owner=f"EnergyManagementSystem:ProgramCallingManager/{manager_name}",
        )
        retained: list[dict[str, Any]] = []
        endpoints: set[str] = set()
        for item in programs:
            program_name = _field_text(
                item,
                "program_name",
                owner=f"EnergyManagementSystem:ProgramCallingManager/{manager_name}",
            )
            if program_name.casefold() in programs_to_remove:
                endpoints.update(program_endpoints[program_name.casefold()])
            else:
                retained.append(item)
        if len(retained) == len(programs):
            continue
        if retained:
            manager["programs"] = retained
            reason = "remove_conflicting_vav_thermostat_ems_program_references"
        else:
            index.remove("EnergyManagementSystem:ProgramCallingManager", manager_name)
            reason = "remove_conflicting_vav_thermostat_ems_calling_manager"
        _mark_change(
            planned,
            "EnergyManagementSystem:ProgramCallingManager",
            manager_name,
            reason,
            tuple(endpoints),
        )


def _remove_orphaned_conflicting_ems_dependencies(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    programs: Mapping[str, tuple[str, dict[str, Any], tuple[str, ...]]],
    programs_to_remove: set[str],
    program_endpoints: Mapping[str, set[str]],
) -> None:
    removed_program_tokens = {key: _erl_tokens(programs[key][2]) for key in programs_to_remove}
    remaining_tokens = {
        token
        for key, (_, _, lines) in programs.items()
        if key not in programs_to_remove
        for token in _erl_tokens(lines)
    }
    removed_tokens = {token for tokens in removed_program_tokens.values() for token in tokens}
    for object_type in (
        "EnergyManagementSystem:Sensor",
        "EnergyManagementSystem:InternalVariable",
    ):
        table = index.table(object_type) or {}
        objects = tuple(index._object_index(object_type, table).values())
        for object_name, _ in objects:
            key = object_name.casefold()
            if key not in removed_tokens or key in remaining_tokens:
                continue
            endpoints = {
                endpoint
                for program_key, tokens in removed_program_tokens.items()
                if key in tokens
                for endpoint in program_endpoints[program_key]
            }
            index.remove(object_type, object_name)
            _mark_change(
                planned,
                object_type,
                object_name,
                "remove_orphaned_vav_thermostat_ems_dependency",
                tuple(endpoints),
            )


def _remove_conflicting_thermostat_ems(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    contexts: Sequence[_VAVLoopContext],
) -> None:
    candidate_actuators = _conflicting_ems_actuators(
        index,
        _thermostat_schedule_endpoint_ids(contexts),
    )
    if not candidate_actuators:
        return
    programs = _ems_program_inventory(index)
    programs_to_remove, program_endpoints, active_actuator_keys = _active_conflicting_ems_programs(
        programs, candidate_actuators
    )
    if not active_actuator_keys:
        return
    _remove_conflicting_ems_calling_references(
        index,
        planned,
        programs_to_remove,
        program_endpoints,
    )
    for program_key in sorted(programs_to_remove):
        program_name = programs[program_key][0]
        index.remove("EnergyManagementSystem:Program", program_name)
        _mark_change(
            planned,
            "EnergyManagementSystem:Program",
            program_name,
            "remove_conflicting_vav_thermostat_ems_program",
            tuple(program_endpoints[program_key]),
        )
    _remove_orphaned_conflicting_ems_dependencies(
        index,
        planned,
        programs,
        programs_to_remove,
        program_endpoints,
    )
    for actuator_key in sorted(active_actuator_keys):
        actuator_name, _, endpoints = candidate_actuators[actuator_key]
        index.remove("EnergyManagementSystem:Actuator", actuator_name)
        _mark_change(
            planned,
            "EnergyManagementSystem:Actuator",
            actuator_name,
            "remove_conflicting_vav_thermostat_schedule_actuator",
            tuple(endpoints),
        )


def _action_name(actuator: ActuatorDescriptor) -> str:
    if actuator.channel in {
        ControlChannel.ZONE_MINIMUM_AIR_FLOW_FRACTION,
        ControlChannel.HEATING_SETPOINT_C,
        ControlChannel.COOLING_SETPOINT_C,
    }:
        return f"{actuator.affected_entity_ids[0]}:{actuator.channel.value}"
    return f"{actuator.equipment_id}:{actuator.channel.value}"


def _build_manifest(
    source_sha256: str,
    prepared_sha256: str,
    contexts: tuple[_VAVLoopContext, ...],
    mutations: tuple[MutationRecord, ...],
    inventory: VAVDiscoveryInventory,
) -> ControlCapabilityManifest:
    equipment = tuple(item.equipment for item in contexts)
    actuators = tuple(actuator for item in contexts for actuator in item.actuators)
    nodes: list[TopologyNode] = []
    edges: list[TopologyEdge] = []
    outputs: list[OutputVariableDescriptor] = []
    candidates: list[DiscoveredEquipmentCandidate] = []
    outlets: list[SupplyOutletNodeEvidence] = []
    seen_zone_nodes: set[str] = set()
    for context in contexts:
        record = context.record
        assert record.outdoor_air_controller is not None and record.supply_outlet_node_name is not None
        nodes.extend(
            (
                TopologyNode(
                    node_id=context.loop_id,
                    kind=TopologyNodeKind.AIR_LOOP,
                    object_type=VAV_LOOP_OBJECT_TYPE,
                    object_name=record.object_name,
                ),
                TopologyNode(
                    node_id=context.oa_id,
                    kind=TopologyNodeKind.OUTDOOR_AIR_CONTROLLER,
                    object_type=VAV_OA_CONTROLLER_OBJECT_TYPE,
                    object_name=record.outdoor_air_controller.object_name,
                ),
            )
        )
        edges.append(TopologyEdge(context.loop_id, context.oa_id, TopologyRelation.CONTAINS))
        edges.append(TopologyEdge(context.oa_id, context.loop_id, TopologyRelation.CONTROLS))
        candidates.append(
            DiscoveredEquipmentCandidate(
                object_type=VAV_LOOP_OBJECT_TYPE,
                object_name=record.object_name,
                source_object_sha256=record.source_object_sha256,
                disposition=CandidateDisposition.SUPPORTED,
                served_entity_ids=context.equipment.served_entity_ids,
                prepared_object_type=VAV_LOOP_OBJECT_TYPE,
                prepared_object_name=record.object_name,
                equipment_id=context.loop_id,
            )
        )
        outlets.append(
            SupplyOutletNodeEvidence(
                observed_node_id=context.loop_id,
                source_object_type=VAV_LOOP_OBJECT_TYPE,
                source_object_name=record.object_name,
                source_object_sha256=record.source_object_sha256,
                outlet_node_name=record.supply_outlet_node_name,
            )
        )
        outputs.extend(
            (
                OutputVariableDescriptor(
                    semantic=OutputVariableSemantic.SUPPLY_AIR_TEMPERATURE,
                    observed_node_id=context.loop_id,
                    variable_name="System Node Temperature",
                    key_value=record.supply_outlet_node_name,
                    unit=PhysicalUnit.CELSIUS,
                    lower_bound=-100.0,
                    upper_bound=100.0,
                    required_observation=False,
                    actuator_endpoint_ids=(context.sat_actuator.endpoint_id,),
                ),
                OutputVariableDescriptor(
                    semantic=OutputVariableSemantic.OUTDOOR_AIR_MASS_FLOW,
                    observed_node_id=context.loop_id,
                    variable_name="Air System Outdoor Air Mass Flow Rate",
                    key_value=record.object_name,
                    unit=PhysicalUnit.KILOGRAMS_PER_SECOND,
                    lower_bound=0.0,
                    upper_bound=VAV_EFFECT_MASS_FLOW_UPPER_BOUND_KG_S,
                    required_observation=False,
                    actuator_endpoint_ids=(context.oa_actuator.endpoint_id,),
                ),
            )
        )
        for terminal in context.terminals:
            zone = terminal.record.zone
            terminal_reference = terminal.record.terminal
            if terminal.zone_id in seen_zone_nodes:
                raise ControlInstrumentationError("VAV manifest has a zone served by multiple loops")
            seen_zone_nodes.add(terminal.zone_id)
            nodes.extend(
                (
                    TopologyNode(
                        node_id=terminal.zone_id,
                        kind=TopologyNodeKind.ZONE,
                        object_type="Zone",
                        object_name=zone.object_name,
                    ),
                    TopologyNode(
                        node_id=terminal.terminal_id,
                        kind=TopologyNodeKind.TERMINAL,
                        object_type=VAV_TERMINAL_OBJECT_TYPE,
                        object_name=terminal_reference.object_name,
                    ),
                )
            )
            edges.extend(
                (
                    TopologyEdge(context.loop_id, terminal.zone_id, TopologyRelation.SERVES),
                    TopologyEdge(context.loop_id, terminal.terminal_id, TopologyRelation.CONTAINS),
                    TopologyEdge(terminal.terminal_id, terminal.zone_id, TopologyRelation.CONTROLS),
                )
            )
            outlets.append(
                SupplyOutletNodeEvidence(
                    observed_node_id=terminal.terminal_id,
                    source_object_type=VAV_TERMINAL_OBJECT_TYPE,
                    source_object_name=terminal_reference.object_name,
                    source_object_sha256=terminal_reference.source_object_sha256,
                    outlet_node_name=terminal.record.terminal_outlet_node_name,
                )
            )
            outputs.extend(
                (
                    OutputVariableDescriptor(
                        semantic=OutputVariableSemantic.SUPPLY_AIR_MASS_FLOW,
                        observed_node_id=terminal.terminal_id,
                        variable_name="System Node Mass Flow Rate",
                        key_value=terminal.record.terminal_outlet_node_name,
                        unit=PhysicalUnit.KILOGRAMS_PER_SECOND,
                        lower_bound=0.0,
                        upper_bound=VAV_EFFECT_MASS_FLOW_UPPER_BOUND_KG_S,
                        required_observation=False,
                        actuator_endpoint_ids=(terminal.flow_actuator.endpoint_id,),
                    ),
                    OutputVariableDescriptor(
                        semantic=OutputVariableSemantic.ZONE_AIR_TEMPERATURE,
                        observed_node_id=terminal.zone_id,
                        variable_name="ZONE AIR TEMPERATURE",
                        key_value=zone.object_name,
                        unit=PhysicalUnit.CELSIUS,
                        lower_bound=-100.0,
                        upper_bound=100.0,
                        required_observation=True,
                        actuator_endpoint_ids=(
                            terminal.heating_actuator.endpoint_id,
                            terminal.cooling_actuator.endpoint_id,
                        ),
                    ),
                )
            )

    unsupported = tuple(
        UnsupportedEquipmentRecord(
            object_type=record.object_type,
            object_name=record.object_name,
            reason_code=record.reason_code,
            details=(
                "The exhaustive VAV enumerator classified this AirLoopHVAC as outside "
                "the pinned VAV:Reheat topology while retaining it for composite-family accounting."
            ),
            blocked_profile_ids=(VAV_PROFILE_ID,),
        )
        for record in inventory.unsupported_records
    )
    unsupported_by_object = {item.energyplus_object: item for item in unsupported}
    candidates.extend(
        DiscoveredEquipmentCandidate(
            object_type=record.object_type,
            object_name=record.object_name,
            source_object_sha256=record.source_object_sha256,
            disposition=CandidateDisposition.UNSUPPORTED,
            served_entity_ids=(),
            unsupported_record_id=unsupported_by_object[(record.object_type, record.object_name)].record_id,
        )
        for record in inventory.unsupported_records
    )
    candidate_tuple = tuple(candidates)
    outlet_tuple = tuple(outlets)
    enumerator = DiscoveryEnumeratorContract(
        enumerator_id="hot-b2b-vav-exact-epjson-v1",
        source_model_sha256=source_sha256,
        candidate_count=len(candidate_tuple),
        candidate_inventory_sha256=candidate_inventory_sha256(candidate_tuple),
        supply_outlet_count=len(outlet_tuple),
        supply_outlet_inventory_sha256=supply_outlet_inventory_sha256(outlet_tuple),
    )
    receipt = InstrumentationReceipt(
        mode=InstrumentationMode.GENERATED,
        source_model_sha256=source_sha256,
        prepared_model_sha256=prepared_sha256,
        mutation_sequence_sha256=mutation_sequence_sha256(mutations),
        targets=tuple(
            MutationTargetState(
                object_type=item.object_type,
                object_name=item.object_name,
                source_sha256=item.before_sha256,
                prepared_sha256=item.after_sha256,
            )
            for item in mutations
        ),
    )
    commands = tuple(
        ActionCommand(
            order=index,
            name=_action_name(actuator),
            unit=ActionCommandUnit(actuator.unit.value),
            lower_bound=actuator.lower_bound,
            upper_bound=actuator.upper_bound,
            transform_id="direct_v1",
            target_endpoint_ids=(actuator.endpoint_id,),
        )
        for index, actuator in enumerate(item for item in actuators if item.role is ActuatorRole.AGENT)
    )
    action_binding = TaskActionBinding(
        binding_id="hot-b2b-vav-direct-fixed-cooling-v1",
        kind=ActionBindingKind.DIRECT,
        commands=commands,
        fixed_endpoints=tuple(
            FixedEndpointValue(item.endpoint_id, VAV_FIXED_COOLING_SETPOINT_C)
            for item in actuators
            if item.role is ActuatorRole.FIXED
        ),
    )
    return ControlCapabilityManifest(
        source_model_sha256=source_sha256,
        prepared_model_sha256=prepared_sha256,
        equipment=equipment,
        actuators=actuators,
        topology_nodes=tuple(nodes),
        topology_edges=tuple(edges),
        output_variables=tuple(outputs),
        supply_outlet_node_evidence=outlet_tuple,
        discovered_candidates=candidate_tuple,
        candidate_inventory_complete=True,
        discovery_enumerator=enumerator,
        instrumentation=receipt,
        task_action_binding=action_binding,
        unsupported_equipment=unsupported,
        mutations=mutations,
    )


@dataclass(frozen=True, slots=True)
class VAVInstrumentationResult:
    prepared_epjson_bytes: bytes
    discovery_inventory_sha256: str
    capability_manifest: ControlCapabilityManifest
    schema_version: str = VAV_INSTRUMENTATION_SCHEMA_VERSION
    algorithm_id: str = VAV_INSTRUMENTATION_ALGORITHM_ID
    upstream_commit: str = BUILDING2BUILDING_COMMIT
    upstream_source_path: str = BUILDING2BUILDING_ACTUATORS_SOURCE_PATH
    upstream_source_sha256: str = BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256

    def __post_init__(self) -> None:
        if type(self.prepared_epjson_bytes) is not bytes:
            raise TypeError("prepared_epjson_bytes must be exact bytes")
        if (
            type(self.discovery_inventory_sha256) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                self.discovery_inventory_sha256,
            )
            is None
        ):
            raise TypeError("discovery_inventory_sha256 must be a lowercase SHA-256 digest")
        if type(self.capability_manifest) is not ControlCapabilityManifest:
            raise TypeError("capability_manifest must be an exact ControlCapabilityManifest")
        canonical_manifest = ControlCapabilityManifest.from_json(self.capability_manifest.to_json())
        object.__setattr__(self, "capability_manifest", canonical_manifest)
        if self.schema_version != VAV_INSTRUMENTATION_SCHEMA_VERSION:
            raise ControlInstrumentationError("unexpected VAV instrumentation schema version")
        if self.algorithm_id != VAV_INSTRUMENTATION_ALGORITHM_ID:
            raise ControlInstrumentationError("unexpected VAV instrumentation algorithm")
        if self.upstream_commit != BUILDING2BUILDING_COMMIT:
            raise ControlInstrumentationError("VAV instrumentation must bind Building2Building v1.0.0")
        if (
            self.upstream_source_path != BUILDING2BUILDING_ACTUATORS_SOURCE_PATH
            or self.upstream_source_sha256 != BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256
        ):
            raise ControlInstrumentationError("VAV instrumentation must bind the authenticated actuators.py")
        model = _parse_epjson_bytes(self.prepared_epjson_bytes)
        _ModelIndex(model)
        if _canonical_epjson_bytes(model) != self.prepared_epjson_bytes:
            raise ControlInstrumentationError("prepared VAV epJSON must use canonical encoding")
        prepared_sha = hashlib.sha256(self.prepared_epjson_bytes).hexdigest()
        if not hmac.compare_digest(prepared_sha, canonical_manifest.prepared_model_sha256):
            raise ControlInstrumentationError("prepared VAV bytes do not match the capability")
        if any(item.profile_id != VAV_PROFILE_ID for item in canonical_manifest.equipment):
            raise ControlInstrumentationError("VAV instrumentation cannot emit a mixed-family capability")

    @property
    def source_model_sha256(self) -> str:
        return self.capability_manifest.source_model_sha256

    @property
    def prepared_model_sha256(self) -> str:
        return self.capability_manifest.prepared_model_sha256

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "upstream_commit": self.upstream_commit,
            "upstream_source_path": self.upstream_source_path,
            "upstream_source_sha256": self.upstream_source_sha256,
            "source_model_sha256": self.source_model_sha256,
            "prepared_model_sha256": self.prepared_model_sha256,
            "discovery_inventory_sha256": self.discovery_inventory_sha256,
            "capability_sha256": self.capability_manifest.capability_sha256,
        }
        return hashlib.sha256(
            b"HOT VAV instrumentation artifact v1\0"
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _instrument_vav_systems_from_json_bytes(
    source_bytes: bytes,
    expected_inventory: VAVDiscoveryInventory,
    *,
    allow_unsupported_records: bool = False,
) -> VAVInstrumentationResult:
    """Instrument every candidate only after exact discovery replay succeeds."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    try:
        inventory = verify_vav_discovery_inventory_from_json_bytes(source_bytes, expected_inventory)
    except (VAVDiscoveryError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(f"VAV discovery verification failed: {error}") from error
    source_model = _parse_epjson_bytes(source_bytes)
    source_index = _ModelIndex(source_model)
    if type(allow_unsupported_records) is not bool:
        raise TypeError("allow_unsupported_records must be an exact boolean")
    contexts = _build_contexts(
        source_index,
        inventory,
        allow_unsupported_records=allow_unsupported_records,
    )
    prepared_model = deepcopy(source_model)
    prepared_index = _ModelIndex(prepared_model)
    _preflight_names(prepared_index, contexts)
    planned: dict[tuple[str, str], _PlannedChange] = {}
    _remove_conflicting_thermostat_ems(prepared_index, planned, contexts)
    for context in contexts:
        _instrument_loop(prepared_index, planned, context)
    prepared_bytes = _canonical_epjson_bytes(prepared_model)
    prepared_sha = hashlib.sha256(prepared_bytes).hexdigest()
    mutations = _build_mutations(source_index, prepared_index, planned)
    try:
        manifest = _build_manifest(
            hashlib.sha256(source_bytes).hexdigest(),
            prepared_sha,
            contexts,
            mutations,
            inventory,
        )
    except (TypeError, ValueError) as error:
        raise ControlInstrumentationError(
            f"frozen capability contract rejected VAV instrumentation: {error}"
        ) from error
    return VAVInstrumentationResult(
        prepared_epjson_bytes=prepared_bytes,
        discovery_inventory_sha256=inventory.inventory_sha256,
        capability_manifest=manifest,
    )


def instrument_vav_systems_from_json_bytes(
    source_bytes: bytes,
    expected_inventory: VAVDiscoveryInventory,
) -> VAVInstrumentationResult:
    """Instrument a native VAV profile only when every air loop is supported."""

    return _instrument_vav_systems_from_json_bytes(source_bytes, expected_inventory)


def instrument_vav_systems_for_composite_from_json_bytes(
    source_bytes: bytes,
    expected_inventory: VAVDiscoveryInventory,
    *,
    unitary_owned_air_loop_names: tuple[str, ...],
) -> VAVInstrumentationResult:
    """Instrument supported VAV loops while accounting exact unitary-owned loops."""

    if type(unitary_owned_air_loop_names) is not tuple or any(
        type(item) is not str or not item or item != item.strip() for item in unitary_owned_air_loop_names
    ):
        raise TypeError("unitary_owned_air_loop_names must be an exact tuple of nonempty strings")
    folded = tuple(item.casefold() for item in unitary_owned_air_loop_names)
    if len(folded) != len(set(folded)):
        raise ValueError("unitary_owned_air_loop_names cannot repeat ignoring case")
    unsupported = expected_inventory.unsupported_records
    actual = {item.object_name.casefold(): item for item in unsupported}
    if set(actual) != set(folded) or any(
        item.reason_code != "vav_loop_has_no_reheat_terminals" for item in unsupported
    ):
        raise ControlInstrumentationError(
            "composite VAV instrumentation requires unsupported records to equal the "
            "authenticated unitary-owned non-VAV loops"
        )
    return _instrument_vav_systems_from_json_bytes(
        source_bytes,
        expected_inventory,
        allow_unsupported_records=True,
    )


def verify_vav_instrumentation_replay(
    source_bytes: bytes,
    expected_inventory: VAVDiscoveryInventory,
    expected_result: VAVInstrumentationResult,
) -> VAVInstrumentationResult:
    """Recompute exact VAV instrumentation and reject artifact substitution."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    if type(expected_result) is not VAVInstrumentationResult:
        raise TypeError("expected_result must be an exact VAVInstrumentationResult")
    try:
        canonical_expected = VAVInstrumentationResult(
            prepared_epjson_bytes=expected_result.prepared_epjson_bytes,
            discovery_inventory_sha256=expected_result.discovery_inventory_sha256,
            capability_manifest=expected_result.capability_manifest,
            schema_version=expected_result.schema_version,
            algorithm_id=expected_result.algorithm_id,
            upstream_commit=expected_result.upstream_commit,
            upstream_source_path=expected_result.upstream_source_path,
            upstream_source_sha256=expected_result.upstream_source_sha256,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(f"expected VAV instrumentation is malformed: {error}") from error
    recomputed = instrument_vav_systems_from_json_bytes(source_bytes, expected_inventory)
    if recomputed != canonical_expected or not hmac.compare_digest(
        recomputed.artifact_sha256,
        canonical_expected.artifact_sha256,
    ):
        raise ControlInstrumentationError("VAV instrumentation does not match deterministic replay")
    return recomputed


__all__ = [
    "VAV_COOLING_SETPOINT_BOUNDS_C",
    "VAV_EFFECT_MASS_FLOW_UPPER_BOUND_KG_S",
    "VAV_FIXED_COOLING_SETPOINT_C",
    "VAV_HEATING_SETPOINT_BOUNDS_C",
    "VAV_INITIAL_HEATING_SETPOINT_C",
    "VAV_INITIAL_MINIMUM_FLOW_FRACTION",
    "VAV_INITIAL_SUPPLY_AIR_TEMPERATURE_C",
    "VAV_INSTRUMENTATION_ALGORITHM_ID",
    "VAV_INSTRUMENTATION_SCHEMA_VERSION",
    "VAV_MINIMUM_FLOW_FRACTION_BOUNDS",
    "VAV_OUTDOOR_AIR_MASS_FLOW_BOUNDS_KG_S",
    "VAV_SUPPLY_AIR_TEMPERATURE_BOUNDS_C",
    "VAVInstrumentationResult",
    "instrument_vav_systems_from_json_bytes",
    "instrument_vav_systems_for_composite_from_json_bytes",
    "verify_vav_instrumentation_replay",
]
