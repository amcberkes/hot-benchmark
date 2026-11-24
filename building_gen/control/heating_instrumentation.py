"""Deterministic exact-byte instrumentation for B2B heating-only zones."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from building_gen.control.capabilities import (
    BUILDING2BUILDING_COMMIT,
    ActionBindingKind,
    ActionCommand,
    ActionCommandUnit,
    ActuatorDescriptor,
    BoundSource,
    CandidateDisposition,
    ControlCapabilityManifest,
    ControlChannel,
    DiscoveredEquipmentCandidate,
    DiscoveryEnumeratorContract,
    EquipmentDescriptor,
    EquipmentFamily,
    InstrumentationMode,
    InstrumentationReceipt,
    MutationRecord,
    MutationTargetState,
    OutputVariableDescriptor,
    OutputVariableSemantic,
    PhysicalUnit,
    TaskActionBinding,
    TopologyEdge,
    TopologyNode,
    TopologyNodeKind,
    TopologyRelation,
    candidate_inventory_sha256,
    energyplus_topology_node_id,
    mutation_sequence_sha256,
    supply_outlet_inventory_sha256,
)
from building_gen.control.discovery import energyplus_source_object_sha256
from building_gen.control.heating_discovery import (
    BUILDING2BUILDING_ACTUATORS_SOURCE_PATH,
    BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256,
    HEATING_PROFILE_ID,
    HeatingDiscoveryError,
    HeatingDiscoveryInventory,
    HeatingObjectReference,
    HeatingZoneDiscoveryRecord,
    verify_heating_discovery_inventory_from_json_bytes,
)
from building_gen.control.instrumentation import (
    ControlInstrumentationError,
    _build_mutations,
    _canonical_epjson_bytes,
    _mark_change,
    _ModelIndex,
    _owned_bytes,
    _parse_epjson_bytes,
    _PlannedChange,
    _sha256,
)

HEATING_INSTRUMENTATION_SCHEMA_VERSION = "hot-b2b-heating-only-instrumentation/1.1.0"
HEATING_INSTRUMENTATION_ALGORITHM_ID = "b2b-v1-heating-only-exact-byte-instrumentation-v2"
HEATING_SETPOINT_BOUNDS_C = (10.0, 35.0)
HEATING_INITIAL_SETPOINT_C = 18.0


@dataclass(frozen=True, slots=True)
class _HeatingGlobalNames:
    availability_type_limits: str
    always_on_schedule: str


@dataclass(frozen=True, slots=True)
class _HeatingZoneNames:
    heating_schedule: str
    heating_type_limits: str


@dataclass(frozen=True, slots=True)
class _HeatingZoneContext:
    record: HeatingZoneDiscoveryRecord
    equipment: EquipmentDescriptor
    zone_id: str
    names: _HeatingZoneNames
    actuator: ActuatorDescriptor


def _global_names(inventory: HeatingDiscoveryInventory) -> _HeatingGlobalNames:
    token = inventory.inventory_sha256[:20]
    prefix = f"HOT B2B Heating {token}"
    return _HeatingGlobalNames(
        availability_type_limits=f"{prefix} Availability Type Limits",
        always_on_schedule=f"{prefix} Always On Availability",
    )


def _zone_names(record: HeatingZoneDiscoveryRecord) -> _HeatingZoneNames:
    identity = json.dumps(
        {
            "zone": record.zone.energyplus_object,
            "equipment": [item.energyplus_object for item in record.equipment],
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    token = hashlib.sha256(b"HOT B2B heating-only generated names v1\0" + identity).hexdigest()[:20]
    prefix = f"HOT B2B Heating Zone {token}"
    return _HeatingZoneNames(
        heating_schedule=f"{prefix} Setpoint",
        heating_type_limits=f"{prefix} Type Limits",
    )


def _assert_reference(index: _ModelIndex, reference: HeatingObjectReference) -> tuple[str, dict[str, Any]]:
    try:
        object_name, fields = index.lookup(reference.object_type, reference.object_name)
    except ControlInstrumentationError:
        raise
    if object_name != reference.object_name:
        raise ControlInstrumentationError("heating discovery reference does not use the canonical object name")
    actual = energyplus_source_object_sha256(reference.object_type, object_name, fields)
    if not hmac.compare_digest(actual, reference.source_object_sha256):
        raise ControlInstrumentationError(
            f"heating discovery hash does not match {reference.object_type}/{object_name}"
        )
    return object_name, fields


def _build_contexts(index: _ModelIndex, inventory: HeatingDiscoveryInventory) -> tuple[_HeatingZoneContext, ...]:
    if not inventory.records:
        raise ControlInstrumentationError("heating discovery inventory is empty")
    contexts: list[_HeatingZoneContext] = []
    for record in inventory.records:
        references = (
            record.zone,
            record.equipment_connection,
            record.equipment_list,
            *record.equipment,
            record.thermostat,
            record.thermostat_setpoint,
            record.heating_schedule,
        ) + (() if record.cooling_schedule is None else (record.cooling_schedule,))
        for reference in references:
            _assert_reference(index, reference)
        anchor = record.equipment[0]
        zone_id = energyplus_topology_node_id(TopologyNodeKind.ZONE, "Zone", record.zone.object_name)
        equipment_id = energyplus_topology_node_id(
            TopologyNodeKind.EQUIPMENT,
            anchor.object_type,
            anchor.object_name,
        )
        equipment = EquipmentDescriptor(
            profile_id=HEATING_PROFILE_ID,
            equipment_id=equipment_id,
            family=EquipmentFamily.HEATING_ONLY,
            object_type=anchor.object_type,
            object_name=anchor.object_name,
            served_entity_ids=(zone_id,),
        )
        names = _zone_names(record)
        if record.thermostat_setpoint.object_type == "ThermostatSetpoint:SingleHeating":
            upper_bound = HEATING_SETPOINT_BOUNDS_C[1]
            upper_bound_source = BoundSource.PINNED_STATIC
        else:
            if record.cooling_schedule is None:  # pragma: no cover - discovery/dataclass invariant
                raise ControlInstrumentationError("dual-setpoint heating zone has no cooling schedule evidence")
            if record.cooling_schedule.object_type != "Schedule:Constant":
                raise ControlInstrumentationError(
                    "dual-setpoint heating upper-bound derivation currently requires an exact "
                    f"Schedule:Constant cooling schedule; found {record.cooling_schedule.object_type}/"
                    f"{record.cooling_schedule.object_name}"
                )
            _cooling_name, cooling_schedule = _assert_reference(index, record.cooling_schedule)
            raw_cooling_setpoint = cooling_schedule.get("hourly_value")
            if (
                isinstance(raw_cooling_setpoint, bool)
                or not isinstance(raw_cooling_setpoint, (int, float))
                or not math.isfinite(float(raw_cooling_setpoint))
            ):
                raise ControlInstrumentationError(
                    "dual-setpoint Schedule:Constant cooling hourly_value must be a finite number"
                )
            cooling_setpoint = float(raw_cooling_setpoint)
            # EnergyPlus' DualSetPointWithDeadBand fatal condition is the
            # strict predicate heating > cooling.  Equality is valid, so the
            # preserved cooling value is the exact safe envelope; subtracting
            # an invented margin would silently shrink the policy surface.
            upper_bound = min(HEATING_SETPOINT_BOUNDS_C[1], cooling_setpoint)
            upper_bound_source = (
                BoundSource.MODEL_DESIGN
                if cooling_setpoint < HEATING_SETPOINT_BOUNDS_C[1]
                else BoundSource.PINNED_STATIC
            )
            if upper_bound < HEATING_INITIAL_SETPOINT_C:
                raise ControlInstrumentationError(
                    "dual-setpoint cooling schedule cannot contain the pinned initial heating setpoint: "
                    f"cooling={cooling_setpoint:g} C, initial={HEATING_INITIAL_SETPOINT_C:g} C"
                )
        actuator = ActuatorDescriptor(
            profile_id=HEATING_PROFILE_ID,
            equipment_id=equipment_id,
            control_node_id=zone_id,
            affected_entity_ids=(zone_id,),
            family=EquipmentFamily.HEATING_ONLY,
            channel=ControlChannel.HEATING_SETPOINT_C,
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=names.heating_schedule,
            unit=PhysicalUnit.CELSIUS,
            lower_bound=HEATING_SETPOINT_BOUNDS_C[0],
            upper_bound=upper_bound,
            lower_bound_source=BoundSource.PINNED_STATIC,
            upper_bound_source=upper_bound_source,
        )
        contexts.append(_HeatingZoneContext(record, equipment, zone_id, names, actuator))
    return tuple(contexts)


def _preflight_names(
    index: _ModelIndex,
    global_names: _HeatingGlobalNames,
    contexts: tuple[_HeatingZoneContext, ...],
) -> None:
    generated = [
        ("ScheduleTypeLimits", global_names.availability_type_limits),
        ("Schedule:Constant", global_names.always_on_schedule),
        *(("ScheduleTypeLimits", item.names.heating_type_limits) for item in contexts),
        *(("Schedule:Constant", item.names.heating_schedule) for item in contexts),
    ]
    folded = [(object_type.casefold(), object_name.casefold()) for object_type, object_name in generated]
    if len(folded) != len(set(folded)):
        raise ControlInstrumentationError("generated heating-only object names collide")

    # EnergyPlus schedule names occupy one global namespace even though
    # epJSON stores the concrete schedule classes in separate tables.  The
    # generic ModelIndex availability check is intentionally table-local, so
    # reject a generated Schedule:Constant name that aliases *any* existing
    # Schedule:* object before making the first mutation.
    generated_schedule_names = {
        object_name.casefold(): object_name
        for object_type, object_name in generated
        if object_type.casefold().startswith("schedule:")
    }
    for object_type, object_name, _fields in index.iter_objects():
        if object_type.casefold().startswith("schedule:") and object_name.casefold() in generated_schedule_names:
            raise ControlInstrumentationError(
                "generated heating-only schedule name collides with source Schedule:* object "
                f"{object_type}/{object_name}"
            )
    for object_type, object_name in generated:
        index.assert_name_available(object_type, object_name)


def _add_object(
    index: _ModelIndex,
    planned: dict[tuple[str, str], _PlannedChange],
    object_type: str,
    object_name: str,
    fields: dict[str, Any],
    reason_code: str,
    endpoint_ids: tuple[str, ...],
) -> None:
    index.add(object_type, object_name, fields)
    _mark_change(planned, object_type, object_name, reason_code, endpoint_ids)


def _instrument(
    index: _ModelIndex,
    inventory: HeatingDiscoveryInventory,
    contexts: tuple[_HeatingZoneContext, ...],
    planned: dict[tuple[str, str], _PlannedChange],
) -> None:
    global_names = _global_names(inventory)
    _preflight_names(index, global_names, contexts)
    all_endpoints = tuple(item.actuator.endpoint_id for item in contexts)
    _add_object(
        index,
        planned,
        "ScheduleTypeLimits",
        global_names.availability_type_limits,
        {
            "lower_limit_value": 0,
            "upper_limit_value": 1,
            "numeric_type": "Discrete",
            "unit_type": "Availability",
        },
        "add_heating_only_availability_type_limits",
        all_endpoints,
    )
    _add_object(
        index,
        planned,
        "Schedule:Constant",
        global_names.always_on_schedule,
        {
            "hourly_value": 1,
            "schedule_type_limits_name": global_names.availability_type_limits,
        },
        "add_heating_only_always_on_schedule",
        all_endpoints,
    )
    for context in contexts:
        endpoint = (context.actuator.endpoint_id,)
        _add_object(
            index,
            planned,
            "ScheduleTypeLimits",
            context.names.heating_type_limits,
            {
                "lower_limit_value": context.actuator.lower_bound,
                "upper_limit_value": context.actuator.upper_bound,
                "numeric_type": "Continuous",
                "unit_type": "Temperature",
            },
            "add_heating_only_zone_setpoint_type_limits",
            endpoint,
        )
        _add_object(
            index,
            planned,
            "Schedule:Constant",
            context.names.heating_schedule,
            {
                "hourly_value": HEATING_INITIAL_SETPOINT_C,
                "schedule_type_limits_name": context.names.heating_type_limits,
            },
            "add_heating_only_setpoint_schedule",
            endpoint,
        )
        setpoint_name, setpoint = _assert_reference(index, context.record.thermostat_setpoint)
        if context.record.thermostat_setpoint.object_type == "ThermostatSetpoint:DualSetpoint":
            field_name = "heating_setpoint_temperature_schedule_name"
        elif context.record.thermostat_setpoint.object_type == "ThermostatSetpoint:SingleHeating":
            field_name = "setpoint_temperature_schedule_name"
        else:  # pragma: no cover - discovery/dataclass invariant
            raise ControlInstrumentationError("unsupported heating thermostat type")
        setpoint[field_name] = context.names.heating_schedule
        _mark_change(
            planned,
            context.record.thermostat_setpoint.object_type,
            setpoint_name,
            "instrument_heating_only_zone_setpoint",
            endpoint,
        )
        for equipment_reference in context.record.equipment:
            equipment_name, equipment = _assert_reference(index, equipment_reference)
            equipment["availability_schedule_name"] = global_names.always_on_schedule
            _mark_change(
                planned,
                equipment_reference.object_type,
                equipment_name,
                "pin_heating_only_equipment_available",
                endpoint,
            )


def _build_manifest(
    source_sha256: str,
    prepared_sha256: str,
    contexts: tuple[_HeatingZoneContext, ...],
    mutations: tuple[MutationRecord, ...],
) -> ControlCapabilityManifest:
    # Capability canonicalization orders equipment by profile/object identity;
    # construct the direct command vector in that same order so H commands
    # remain exactly aligned with the H physical endpoints after decoding.
    contexts = tuple(
        sorted(
            contexts,
            key=lambda item: (
                item.equipment.object_type.casefold(),
                item.equipment.object_name.casefold(),
                item.equipment.equipment_id.casefold(),
            ),
        )
    )
    equipment = tuple(item.equipment for item in contexts)
    actuators = tuple(item.actuator for item in contexts)
    nodes: list[TopologyNode] = []
    edges: list[TopologyEdge] = []
    outputs: list[OutputVariableDescriptor] = []
    candidates: list[DiscoveredEquipmentCandidate] = []
    for context in contexts:
        record = context.record
        nodes.extend(
            (
                TopologyNode(
                    node_id=context.equipment.equipment_id,
                    kind=TopologyNodeKind.EQUIPMENT,
                    object_type=context.equipment.object_type,
                    object_name=context.equipment.object_name,
                ),
                TopologyNode(
                    node_id=context.zone_id,
                    kind=TopologyNodeKind.ZONE,
                    object_type="Zone",
                    object_name=record.zone.object_name,
                ),
            )
        )
        edges.append(
            TopologyEdge(
                context.equipment.equipment_id,
                context.zone_id,
                TopologyRelation.CONTROLS,
            )
        )
        outputs.append(
            OutputVariableDescriptor(
                semantic=OutputVariableSemantic.ZONE_AIR_TEMPERATURE,
                observed_node_id=context.zone_id,
                variable_name="ZONE AIR TEMPERATURE",
                key_value=record.zone.object_name,
                unit=PhysicalUnit.CELSIUS,
                lower_bound=-100.0,
                upper_bound=100.0,
                required_observation=True,
                actuator_endpoint_ids=(context.actuator.endpoint_id,),
            )
        )
        for reference in record.equipment:
            candidates.append(
                DiscoveredEquipmentCandidate(
                    object_type=reference.object_type,
                    object_name=reference.object_name,
                    source_object_sha256=reference.source_object_sha256,
                    disposition=CandidateDisposition.SUPPORTED,
                    served_entity_ids=(context.zone_id,),
                    prepared_object_type=reference.object_type,
                    prepared_object_name=reference.object_name,
                    equipment_id=context.equipment.equipment_id,
                )
            )
    candidate_tuple = tuple(candidates)
    enumerator = DiscoveryEnumeratorContract(
        enumerator_id="hot-b2b-heating-only-exact-epjson-v1",
        source_model_sha256=source_sha256,
        candidate_count=len(candidate_tuple),
        candidate_inventory_sha256=candidate_inventory_sha256(candidate_tuple),
        supply_outlet_count=0,
        supply_outlet_inventory_sha256=supply_outlet_inventory_sha256(()),
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
            name=f"{context.zone_id}:{ControlChannel.HEATING_SETPOINT_C.value}",
            unit=ActionCommandUnit.CELSIUS,
            lower_bound=context.actuator.lower_bound,
            upper_bound=context.actuator.upper_bound,
            transform_id="direct_v1",
            target_endpoint_ids=(context.actuator.endpoint_id,),
        )
        for index, context in enumerate(contexts)
    )
    return ControlCapabilityManifest(
        source_model_sha256=source_sha256,
        prepared_model_sha256=prepared_sha256,
        equipment=equipment,
        actuators=actuators,
        topology_nodes=tuple(nodes),
        topology_edges=tuple(edges),
        output_variables=tuple(outputs),
        supply_outlet_node_evidence=(),
        discovered_candidates=candidate_tuple,
        candidate_inventory_complete=True,
        discovery_enumerator=enumerator,
        instrumentation=receipt,
        task_action_binding=TaskActionBinding(
            binding_id="hot-b2b-heating-only-direct-v1",
            kind=ActionBindingKind.DIRECT,
            commands=commands,
            fixed_endpoints=(),
        ),
        mutations=mutations,
    )


@dataclass(frozen=True, slots=True)
class HeatingInstrumentationResult:
    prepared_epjson_bytes: bytes
    discovery_inventory_sha256: str
    capability_manifest: ControlCapabilityManifest
    schema_version: str = HEATING_INSTRUMENTATION_SCHEMA_VERSION
    algorithm_id: str = HEATING_INSTRUMENTATION_ALGORITHM_ID
    upstream_commit: str = BUILDING2BUILDING_COMMIT
    upstream_source_path: str = BUILDING2BUILDING_ACTUATORS_SOURCE_PATH
    upstream_source_sha256: str = BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256

    def __post_init__(self) -> None:
        if type(self.prepared_epjson_bytes) is not bytes:
            raise TypeError("prepared_epjson_bytes must be exact bytes")
        if type(self.discovery_inventory_sha256) is not str:
            raise TypeError("discovery_inventory_sha256 must be an exact string")
        _sha256(
            self.discovery_inventory_sha256,
            field_name="discovery_inventory_sha256",
        )
        if type(self.capability_manifest) is not ControlCapabilityManifest:
            raise TypeError("capability_manifest must be an exact ControlCapabilityManifest")
        manifest = ControlCapabilityManifest.from_json(self.capability_manifest.to_json())
        object.__setattr__(self, "capability_manifest", manifest)
        expected_strings = {
            "schema_version": HEATING_INSTRUMENTATION_SCHEMA_VERSION,
            "algorithm_id": HEATING_INSTRUMENTATION_ALGORITHM_ID,
            "upstream_commit": BUILDING2BUILDING_COMMIT,
            "upstream_source_path": BUILDING2BUILDING_ACTUATORS_SOURCE_PATH,
            "upstream_source_sha256": BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256,
        }
        for field_name, expected in expected_strings.items():
            value = getattr(self, field_name)
            if type(value) is not str or value != expected:
                raise ValueError(f"unexpected heating instrumentation {field_name}")
        prepared_model = _parse_epjson_bytes(self.prepared_epjson_bytes)
        _ModelIndex(prepared_model)
        if _canonical_epjson_bytes(prepared_model) != self.prepared_epjson_bytes:
            raise ControlInstrumentationError("prepared heating epJSON must use canonical encoding")
        prepared_sha = hashlib.sha256(self.prepared_epjson_bytes).hexdigest()
        if not hmac.compare_digest(prepared_sha, manifest.prepared_model_sha256):
            raise ValueError("prepared heating bytes do not match the capability manifest")
        if manifest.profile_ids != (HEATING_PROFILE_ID,):
            raise ValueError("heating instrumentation must emit only the heating-only profile")

    @property
    def source_model_sha256(self) -> str:
        return self.capability_manifest.source_model_sha256

    @property
    def prepared_model_sha256(self) -> str:
        return self.capability_manifest.prepared_model_sha256

    @property
    def artifact_sha256(self) -> str:
        identity = {
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
            b"HOT heating-only instrumentation artifact v1\0"
            + json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def instrument_heating_only_from_json_bytes(
    source_bytes: bytes,
    inventory: HeatingDiscoveryInventory,
) -> HeatingInstrumentationResult:
    """Instrument every authenticated heating-only candidate as zone groups."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    try:
        verified_inventory = verify_heating_discovery_inventory_from_json_bytes(source_bytes, inventory)
    except (HeatingDiscoveryError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(f"heating discovery verification failed: {error}") from error
    try:
        source_model = _parse_epjson_bytes(source_bytes)
        prepared_model = deepcopy(source_model)
        source_index = _ModelIndex(source_model)
        prepared_index = _ModelIndex(prepared_model)
        contexts = _build_contexts(prepared_index, verified_inventory)
        planned: dict[tuple[str, str], _PlannedChange] = {}
        _instrument(prepared_index, verified_inventory, contexts, planned)
        prepared_bytes = _canonical_epjson_bytes(prepared_model)
        mutations = _build_mutations(source_index, prepared_index, planned)
        manifest = _build_manifest(
            hashlib.sha256(source_bytes).hexdigest(),
            hashlib.sha256(prepared_bytes).hexdigest(),
            contexts,
            mutations,
        )
    except (ControlInstrumentationError, TypeError, ValueError) as error:
        if isinstance(error, ControlInstrumentationError):
            raise
        raise ControlInstrumentationError(f"heating instrumentation failed: {error}") from error
    return HeatingInstrumentationResult(
        prepared_epjson_bytes=prepared_bytes,
        discovery_inventory_sha256=verified_inventory.inventory_sha256,
        capability_manifest=manifest,
    )


def _canonical_result(value: Any) -> HeatingInstrumentationResult:
    if type(value) is not HeatingInstrumentationResult:
        raise TypeError("expected_result must be an exact HeatingInstrumentationResult")
    try:
        return HeatingInstrumentationResult(
            prepared_epjson_bytes=value.prepared_epjson_bytes,
            discovery_inventory_sha256=value.discovery_inventory_sha256,
            capability_manifest=value.capability_manifest,
            schema_version=value.schema_version,
            algorithm_id=value.algorithm_id,
            upstream_commit=value.upstream_commit,
            upstream_source_path=value.upstream_source_path,
            upstream_source_sha256=value.upstream_source_sha256,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(f"expected heating artifact is malformed or tampered: {error}") from error


def verify_heating_instrumentation_replay(
    source_bytes: bytes,
    inventory: HeatingDiscoveryInventory,
    expected_result: HeatingInstrumentationResult,
) -> HeatingInstrumentationResult:
    """Recompute instrumentation and reject any artifact substitution."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    expected = _canonical_result(expected_result)
    recomputed = instrument_heating_only_from_json_bytes(source_bytes, inventory)
    comparisons = (
        (expected.prepared_epjson_bytes, recomputed.prepared_epjson_bytes),
        (
            expected.capability_manifest.to_json().encode("utf-8"),
            recomputed.capability_manifest.to_json().encode("utf-8"),
        ),
        (expected.artifact_sha256.encode("ascii"), recomputed.artifact_sha256.encode("ascii")),
    )
    if any(not hmac.compare_digest(expected_value, actual_value) for expected_value, actual_value in comparisons):
        raise ControlInstrumentationError("heating instrumentation artifact does not match deterministic replay")
    return recomputed


__all__ = [
    "HEATING_INITIAL_SETPOINT_C",
    "HEATING_INSTRUMENTATION_ALGORITHM_ID",
    "HEATING_INSTRUMENTATION_SCHEMA_VERSION",
    "HEATING_SETPOINT_BOUNDS_C",
    "HeatingInstrumentationResult",
    "instrument_heating_only_from_json_bytes",
    "verify_heating_instrumentation_replay",
]
