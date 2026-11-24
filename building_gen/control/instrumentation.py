"""Pure, deterministic instrumentation for supported B2B unitary systems.

The mutation logic in this module is adapted from the pinned MIT-licensed
Building2Building v1.0.0 ``pipeline/actuators.py`` implementation, without
importing its RDF, Minergym, or runtime dependencies.  Discovery remains the
authority for topology and bounds.  This module verifies that discovery result
against exact source bytes before making a deep-copy mutation and produces a
complete draft :class:`ControlCapabilityManifest` as its receipt.

No function in this module reads or writes the filesystem.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
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
    MutationOperation,
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
    candidate_inventory_sha256,
    energyplus_topology_node_id,
    mutation_sequence_sha256,
    supply_outlet_inventory_sha256,
)
from building_gen.control.discovery import (
    UNITARY_OBJECT_TYPE,
    UNITARY_PROFILE_ID,
    UNITARY_SAT_LOWER_BOUND_C,
    ControlDiscoveryError,
    UnitaryDiscoveryInventory,
    UnitaryDiscoveryRecord,
    energyplus_source_object_sha256,
    verify_unitary_discovery_inventory_from_json_bytes,
)

UNITARY_INSTRUMENTATION_SCHEMA_VERSION = "hot-b2b-unitary-instrumentation/0.1.0-draft"
UNITARY_INSTRUMENTATION_ALGORITHM_ID = "b2b-v1-unitary-exact-byte-instrumentation-v1"
AIR_TO_AIR_UNITARY_INSTRUMENTATION_SCHEMA_VERSION = "hot-b2b-air-to-air-unitary-instrumentation/1.0.0"
AIR_TO_AIR_UNITARY_INSTRUMENTATION_ALGORITHM_ID = "b2b-v1-air-to-air-conversion-unitary-exact-byte-instrumentation-v1"
DX_COOLING_COMPRESSOR_MIN_OAT_C = 10.0
INITIAL_SAT_SETPOINT_C = 22.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DX_COIL_TYPES = (
    "Coil:Cooling:DX:SingleSpeed",
    "Coil:Cooling:DX:TwoSpeed",
    "Coil:Cooling:DX:MultiSpeed",
    "Coil:Cooling:DX:TwoStageWithHumidityControlMode",
)
_FAN_OBJECT_TYPES = (
    "Fan:SystemModel",
    "Fan:OnOff",
    "Fan:ConstantVolume",
)
_CANONICAL_TABLE_TYPES = (
    UNITARY_OBJECT_TYPE,
    "AirLoopHVAC",
    "BranchList",
    "Branch",
    "Zone",
    "ZoneHVAC:EquipmentConnections",
    "ZoneHVAC:EquipmentList",
    "ZoneHVAC:AirDistributionUnit",
    "AirTerminal:SingleDuct:ConstantVolume:NoReheat",
    "AirLoopHVAC:ZoneSplitter",
    "NodeList",
    *_FAN_OBJECT_TYPES,
    *_DX_COIL_TYPES,
    "ScheduleTypeLimits",
    "Schedule:Constant",
    "SetpointManager:Scheduled",
    "AvailabilityManagerAssignmentList",
    "AvailabilityManager:NightCycle",
    "AvailabilityManager:Scheduled",
)
_CANONICAL_TABLE_BY_CASEFOLD = {item.casefold(): item for item in _CANONICAL_TABLE_TYPES}
if len(_CANONICAL_TABLE_BY_CASEFOLD) != len(_CANONICAL_TABLE_TYPES):  # pragma: no cover - module invariant
    raise RuntimeError("instrumentation table types must be unique ignoring case")


class ControlInstrumentationError(ValueError):
    """Raised when exact source bytes cannot be instrumented unambiguously."""


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ControlInstrumentationError(f"{field_name} must be a nonempty exact string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ControlInstrumentationError(f"{field_name} contains a control character")
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ControlInstrumentationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlInstrumentationError(f"epJSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ControlInstrumentationError(f"epJSON contains non-finite constant {value!r}")


def _finite_json_float(value: str) -> float:
    exact = Decimal(value)
    result = float(exact)
    if not math.isfinite(result) or (result == 0.0 and exact != 0):
        raise ControlInstrumentationError(f"epJSON floating-point value is not finite or representable: {value!r}")
    return result


def _owned_bytes(value: Any, *, field_name: str) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{field_name} must be bytes")
    # Copy through the buffer protocol so provenance and parsing use the same
    # exact base bytes without invoking virtual methods on a bytes subclass.
    # This also works on the Python version in the pinned EnergyPlus image,
    # where ``bytes.__bytes__`` is unavailable.
    return memoryview(value).tobytes()


def _parse_epjson_bytes(source_bytes: bytes) -> dict[str, Any]:
    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    if source_bytes.startswith(b"\xef\xbb\xbf"):
        raise ControlInstrumentationError("epJSON may not contain a UTF-8 byte-order mark")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ControlInstrumentationError(f"epJSON is not UTF-8: {error}") from error
    try:
        model = json.loads(
            source_text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_finite_json_float,
        )
    except ControlInstrumentationError:
        raise
    except json.JSONDecodeError as error:
        raise ControlInstrumentationError(f"epJSON is not valid JSON: {error}") from error
    if not isinstance(model, dict):
        raise ControlInstrumentationError("epJSON root must be an object")
    return model


def _canonical_epjson_bytes(model: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            model,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ControlInstrumentationError(f"prepared epJSON is not canonical JSON data: {error}") from error


class _ModelIndex:
    """Casefold index over a mutable epJSON model with canonical table keys."""

    def __init__(self, model: dict[str, Any]) -> None:
        self.model = model
        self._tables: dict[str, tuple[str, dict[str, Any]]] = {}
        for raw_type, raw_table in model.items():
            object_type = _text(raw_type, field_name="epJSON root object-table name")
            folded = object_type.casefold()
            if folded in self._tables:
                raise ControlInstrumentationError("epJSON root contains object-table names that collide ignoring case")
            canonical = _CANONICAL_TABLE_BY_CASEFOLD.get(folded)
            if canonical is not None and object_type != canonical:
                raise ControlInstrumentationError(
                    f"epJSON instrumentation table {object_type!r} must use canonical spelling {canonical!r}"
                )
            if folded.startswith("setpointmanager:") and not object_type.startswith("SetpointManager:"):
                raise ControlInstrumentationError(
                    f"epJSON setpoint-manager table {object_type!r} must use canonical prefix spelling"
                )
            if not isinstance(raw_table, dict):
                raise ControlInstrumentationError(f"epJSON {object_type!r} table must be an object")
            self._validate_object_table(object_type, raw_table)
            self._tables[folded] = (object_type, raw_table)

    @staticmethod
    def _validate_object_table(object_type: str, table: Mapping[str, Any]) -> None:
        names: dict[str, str] = {}
        for raw_name, fields in table.items():
            name = _text(raw_name, field_name=f"{object_type} object name")
            folded = name.casefold()
            previous = names.get(folded)
            if previous is not None:
                raise ControlInstrumentationError(
                    f"epJSON {object_type!r} contains names that collide ignoring case: " f"{previous!r}, {name!r}"
                )
            if not isinstance(fields, dict):
                raise ControlInstrumentationError(f"epJSON object {object_type}/{name} must be an object")
            names[folded] = name

    def table(self, object_type: str, *, create: bool = False) -> dict[str, Any] | None:
        canonical_type = _text(object_type, field_name="object_type")
        entry = self._tables.get(canonical_type.casefold())
        if entry is not None:
            if entry[0] != canonical_type:
                raise ControlInstrumentationError(
                    f"epJSON table lookup {canonical_type!r} resolved noncanonically to {entry[0]!r}"
                )
            return entry[1]
        if not create:
            return None
        table: dict[str, Any] = {}
        self.model[canonical_type] = table
        self._tables[canonical_type.casefold()] = (canonical_type, table)
        return table

    @staticmethod
    def _object_index(object_type: str, table: Mapping[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
        result: dict[str, tuple[str, dict[str, Any]]] = {}
        for raw_name, fields in table.items():
            name = _text(raw_name, field_name=f"{object_type} object name")
            if not isinstance(fields, dict):
                raise ControlInstrumentationError(f"epJSON object {object_type}/{name} must be an object")
            folded = name.casefold()
            if folded in result:
                raise ControlInstrumentationError(f"epJSON {object_type!r} contains names that collide ignoring case")
            result[folded] = (name, fields)
        return result

    def lookup(self, object_type: str, object_name: str) -> tuple[str, dict[str, Any]]:
        table = self.table(object_type)
        if table is None:
            raise ControlInstrumentationError(f"required epJSON table {object_type!r} is absent")
        entry = self._object_index(object_type, table).get(object_name.casefold())
        if entry is None:
            raise ControlInstrumentationError(f"required epJSON object {object_type}/{object_name} is absent")
        return entry

    def maybe_lookup(self, object_type: str, object_name: str) -> tuple[str, dict[str, Any]] | None:
        table = self.table(object_type)
        if table is None:
            return None
        return self._object_index(object_type, table).get(object_name.casefold())

    def assert_name_available(self, object_type: str, object_name: str) -> None:
        if self.maybe_lookup(object_type, object_name) is not None:
            raise ControlInstrumentationError(
                f"generated EnergyPlus object name collides with source object {object_type}/{object_name}"
            )

    def add(self, object_type: str, object_name: str, fields: dict[str, Any]) -> None:
        self.assert_name_available(object_type, object_name)
        table = self.table(object_type, create=True)
        assert table is not None  # construction invariant
        table[object_name] = fields

    def remove(self, object_type: str, object_name: str) -> tuple[str, dict[str, Any]]:
        canonical_name, fields = self.lookup(object_type, object_name)
        table = self.table(object_type)
        assert table is not None  # lookup already proved this
        del table[canonical_name]
        return canonical_name, fields

    def iter_objects(self) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        values: list[tuple[str, str, dict[str, Any]]] = []
        for object_type, table in self._tables.values():
            for object_name, fields in self._object_index(object_type, table).values():
                values.append((object_type, object_name, fields))
        return tuple(
            sorted(
                values,
                key=lambda item: (item[0].casefold(), item[1].casefold()),
            )
        )

    def object_map(self) -> dict[tuple[str, str], tuple[str, str, dict[str, Any]]]:
        return {
            (object_type.casefold(), object_name.casefold()): (object_type, object_name, fields)
            for object_type, object_name, fields in self.iter_objects()
        }


@dataclass(frozen=True, slots=True)
class _GeneratedNames:
    availability_type_limits: str
    always_on_schedule: str
    fan_mode_schedule: str
    sat_type_limits: str
    sat_schedule: str
    sat_manager: str
    namespace_token: str


@dataclass(frozen=True, slots=True)
class _SystemContext:
    record: UnitaryDiscoveryRecord
    equipment: EquipmentDescriptor
    fan_actuator: ActuatorDescriptor
    sat_actuator: ActuatorDescriptor
    zone_name: str
    zone_id: str
    air_loop_name: str
    names: _GeneratedNames


@dataclass(slots=True)
class _PlannedChange:
    reason_code: str
    endpoint_ids: set[str]


def _generated_names(record: UnitaryDiscoveryRecord) -> _GeneratedNames:
    identity = f"{record.object_type.casefold()}\0{record.object_name.casefold()}".encode("utf-8")
    token = hashlib.sha256(b"HOT B2B unitary generated names v1\0" + identity).hexdigest()[:20]
    prefix = f"HOT B2B Unitary {token}"
    return _GeneratedNames(
        availability_type_limits=f"{prefix} Availability Type Limits",
        always_on_schedule=f"{prefix} Always On Availability",
        fan_mode_schedule=f"{prefix} Fan Operating Mode",
        sat_type_limits=f"{prefix} SAT Type Limits",
        sat_schedule=f"{prefix} SAT Setpoint",
        sat_manager=f"{prefix} SAT Manager",
        namespace_token=token,
    )


def _mark_change(
    planned: dict[tuple[str, str], _PlannedChange],
    object_type: str,
    object_name: str,
    reason_code: str,
    endpoint_ids: Sequence[str],
) -> None:
    key = (object_type.casefold(), object_name.casefold())
    existing = planned.get(key)
    if existing is None:
        planned[key] = _PlannedChange(reason_code, set(endpoint_ids))
        return
    if existing.reason_code != reason_code:
        existing.reason_code = "instrument_unitary_control_interface"
    existing.endpoint_ids.update(endpoint_ids)


def _records(fields: Mapping[str, Any], key: str, *, owner: str) -> list[dict[str, Any]]:
    raw = fields.get(key, [])
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ControlInstrumentationError(f"{owner}.{key} must be an array")
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ControlInstrumentationError(f"{owner}.{key}[{index}] must be an object")
    return raw


def _field_text(fields: Mapping[str, Any], key: str, *, owner: str) -> str:
    return _text(fields.get(key), field_name=f"{owner}.{key}")


def _build_contexts(
    source_index: _ModelIndex,
    inventory: UnitaryDiscoveryInventory,
) -> tuple[_SystemContext, ...]:
    if not inventory.records:
        raise ControlInstrumentationError("unitary discovery inventory is empty")
    unsupported = [item.object_name for item in inventory.unsupported_records]
    if unsupported:
        raise ControlInstrumentationError(
            f"unitary instrumentation refuses unsupported discovery records: {unsupported!r}"
        )

    seen_loops: dict[str, str] = {}
    seen_fans: dict[tuple[str, str], str] = {}
    seen_outlets: dict[str, str] = {}
    contexts: list[_SystemContext] = []
    for record in inventory.supported_records:
        if any(
            value is None
            for value in (
                record.zone_name,
                record.air_loop_name,
                record.supply_outlet_node_name,
                record.fan_object_type,
                record.fan_object_name,
                record.fan_upper_bound_kg_s,
                record.fan_upper_bound_source,
                record.sat_upper_bound_c,
                record.sat_upper_bound_source,
            )
        ):
            raise ControlInstrumentationError("supported discovery record is incomplete")

        system_name, system = source_index.lookup(UNITARY_OBJECT_TYPE, record.object_name)
        if system_name != record.object_name:
            raise ControlInstrumentationError("discovery record must use the canonical unitary object name")
        source_object_sha = energyplus_source_object_sha256(UNITARY_OBJECT_TYPE, system_name, system)
        if not hmac.compare_digest(source_object_sha, record.source_object_sha256):
            raise ControlInstrumentationError("discovery record unitary object hash does not match source epJSON")

        loop_name, _ = source_index.lookup("AirLoopHVAC", record.air_loop_name)  # type: ignore[arg-type]
        fan_name, _ = source_index.lookup(record.fan_object_type, record.fan_object_name)  # type: ignore[arg-type]
        zone_name, _ = source_index.lookup("Zone", record.zone_name)  # type: ignore[arg-type]
        outlet_name = _text(record.supply_outlet_node_name, field_name="supply_outlet_node_name")

        previous_loop = seen_loops.get(loop_name.casefold())
        if previous_loop is not None:
            raise ControlInstrumentationError(
                f"air loop {loop_name!r} ambiguously contains multiple supported unitary systems"
            )
        seen_loops[loop_name.casefold()] = system_name
        fan_key = (record.fan_object_type.casefold(), fan_name.casefold())  # type: ignore[union-attr]
        previous_fan = seen_fans.get(fan_key)
        if previous_fan is not None:
            raise ControlInstrumentationError(
                f"fan {record.fan_object_type}/{fan_name} is shared by multiple supported unitary systems"
            )
        seen_fans[fan_key] = system_name
        previous_outlet = seen_outlets.get(outlet_name.casefold())
        if previous_outlet is not None:
            raise ControlInstrumentationError(
                f"supply outlet {outlet_name!r} is shared by multiple supported unitary systems"
            )
        seen_outlets[outlet_name.casefold()] = system_name

        equipment_id = energyplus_topology_node_id(
            TopologyNodeKind.EQUIPMENT,
            UNITARY_OBJECT_TYPE,
            system_name,
        )
        zone_id = energyplus_topology_node_id(TopologyNodeKind.ZONE, "Zone", zone_name)
        equipment = EquipmentDescriptor(
            profile_id=UNITARY_PROFILE_ID,
            equipment_id=equipment_id,
            family=EquipmentFamily.UNITARY,
            object_type=UNITARY_OBJECT_TYPE,
            object_name=system_name,
            served_entity_ids=(zone_id,),
        )
        names = _generated_names(record)
        fan_actuator = ActuatorDescriptor(
            profile_id=UNITARY_PROFILE_ID,
            equipment_id=equipment_id,
            control_node_id=equipment_id,
            affected_entity_ids=(zone_id,),
            family=EquipmentFamily.UNITARY,
            channel=ControlChannel.FAN_AIR_MASS_FLOW_KG_S,
            component_type="Fan",
            control_type="Fan Air Mass Flow Rate",
            component_name=fan_name,
            unit=PhysicalUnit.KILOGRAMS_PER_SECOND,
            lower_bound=0.0,
            upper_bound=record.fan_upper_bound_kg_s,  # type: ignore[arg-type]
            lower_bound_source=BoundSource.PINNED_STATIC,
            upper_bound_source=record.fan_upper_bound_source,  # type: ignore[arg-type]
        )
        sat_actuator = ActuatorDescriptor(
            profile_id=UNITARY_PROFILE_ID,
            equipment_id=equipment_id,
            control_node_id=equipment_id,
            affected_entity_ids=(zone_id,),
            family=EquipmentFamily.UNITARY,
            channel=ControlChannel.SUPPLY_AIR_TEMPERATURE_C,
            component_type="Schedule:Constant",
            control_type="Schedule Value",
            component_name=names.sat_schedule,
            unit=PhysicalUnit.CELSIUS,
            lower_bound=UNITARY_SAT_LOWER_BOUND_C,
            upper_bound=record.sat_upper_bound_c,  # type: ignore[arg-type]
            lower_bound_source=BoundSource.PINNED_STATIC,
            upper_bound_source=record.sat_upper_bound_source,  # type: ignore[arg-type]
        )
        contexts.append(
            _SystemContext(
                record=record,
                equipment=equipment,
                fan_actuator=fan_actuator,
                sat_actuator=sat_actuator,
                zone_name=zone_name,
                zone_id=zone_id,
                air_loop_name=loop_name,
                names=names,
            )
        )
    return tuple(contexts)


def _preflight_generated_names(index: _ModelIndex, context: _SystemContext) -> None:
    names = context.names
    for object_type, object_name in (
        ("ScheduleTypeLimits", names.availability_type_limits),
        ("ScheduleTypeLimits", names.sat_type_limits),
        ("Schedule:Constant", names.always_on_schedule),
        ("Schedule:Constant", names.fan_mode_schedule),
        ("Schedule:Constant", names.sat_schedule),
        ("SetpointManager:Scheduled", names.sat_manager),
    ):
        index.assert_name_available(object_type, object_name)


def _set_dx_lockout(
    index: _ModelIndex,
    system: dict[str, Any],
    context: _SystemContext,
    planned: dict[tuple[str, str], _PlannedChange],
) -> None:
    raw_type = system.get("cooling_coil_object_type")
    raw_name = system.get("cooling_coil_name")
    type_present = raw_type not in (None, "")
    name_present = raw_name not in (None, "")
    if type_present != name_present:
        raise ControlInstrumentationError(
            f"{UNITARY_OBJECT_TYPE}/{context.record.object_name} has incomplete cooling-coil identity"
        )
    if not type_present:
        return
    coil_type = _text(raw_type, field_name="cooling_coil_object_type")
    coil_name_ref = _text(raw_name, field_name="cooling_coil_name")
    canonical_dx = {item.casefold(): item for item in _DX_COIL_TYPES}.get(coil_type.casefold())
    if canonical_dx is None:
        return
    if coil_type != canonical_dx:
        raise ControlInstrumentationError(
            f"DX cooling coil type {coil_type!r} must use canonical spelling {canonical_dx!r}"
        )
    coil_name, coil = index.lookup(canonical_dx, coil_name_ref)
    if coil.get("minimum_outdoor_dry_bulb_temperature_for_compressor_operation") == (DX_COOLING_COMPRESSOR_MIN_OAT_C):
        return
    coil["minimum_outdoor_dry_bulb_temperature_for_compressor_operation"] = DX_COOLING_COMPRESSOR_MIN_OAT_C
    _mark_change(
        planned,
        canonical_dx,
        coil_name,
        "set_dx_compressor_lockout",
        (context.sat_actuator.endpoint_id,),
    )


def _remove_conflicting_outlet_managers(
    index: _ModelIndex,
    context: _SystemContext,
    planned: dict[tuple[str, str], _PlannedChange],
) -> None:
    outlet = context.record.supply_outlet_node_name
    assert outlet is not None
    for object_type, object_name, fields in index.iter_objects():
        if not object_type.startswith("SetpointManager:"):
            continue
        raw_node = fields.get("setpoint_node_or_nodelist_name")
        if raw_node is None:
            continue
        node = _text(
            raw_node,
            field_name=f"{object_type}/{object_name}.setpoint_node_or_nodelist_name",
        )
        if node.casefold() != outlet.casefold():
            continue
        index.remove(object_type, object_name)
        _mark_change(
            planned,
            object_type,
            object_name,
            "remove_conflicting_outlet_setpoint_manager",
            (context.sat_actuator.endpoint_id,),
        )


def _availability_list_reference_count(index: _ModelIndex, list_name: str) -> int:
    loops = index.table("AirLoopHVAC") or {}
    count = 0
    for loop_name, fields in loops.items():
        raw = fields.get("availability_manager_list_name")
        if raw in (None, ""):
            continue
        reference = _text(raw, field_name=f"AirLoopHVAC/{loop_name}.availability_manager_list_name")
        if reference.casefold() == list_name.casefold():
            count += 1
    return count


def _availability_manager_reference_count(
    index: _ModelIndex,
    manager_type: str,
    manager_name: str,
) -> int:
    assignment_lists = index.table("AvailabilityManagerAssignmentList") or {}
    count = 0
    for list_name, fields in assignment_lists.items():
        for manager in _records(
            fields,
            "managers",
            owner=f"AvailabilityManagerAssignmentList/{list_name}",
        ):
            object_type = _field_text(
                manager,
                "availability_manager_object_type",
                owner=f"AvailabilityManagerAssignmentList/{list_name}",
            )
            object_name = _field_text(
                manager,
                "availability_manager_name",
                owner=f"AvailabilityManagerAssignmentList/{list_name}",
            )
            if object_type.casefold() == manager_type.casefold() and object_name.casefold() == manager_name.casefold():
                count += 1
    return count


def _scheduled_availability_name(context: _SystemContext, old_name: str) -> str:
    token = hashlib.sha256(
        b"HOT B2B scheduled availability manager v1\0"
        + context.names.namespace_token.encode("ascii")
        + b"\0"
        + old_name.casefold().encode("utf-8")
    ).hexdigest()[:12]
    return f"HOT B2B Unitary {context.names.namespace_token} Loop Availability {token}"


def _replace_night_cycle_availability(
    index: _ModelIndex,
    context: _SystemContext,
    planned: dict[tuple[str, str], _PlannedChange],
) -> None:
    loop_name, loop = index.lookup("AirLoopHVAC", context.air_loop_name)
    raw_list_name = loop.get("availability_manager_list_name")
    if raw_list_name in (None, ""):
        return
    list_reference = _text(
        raw_list_name,
        field_name=f"AirLoopHVAC/{loop_name}.availability_manager_list_name",
    )
    list_name, assignment_list = index.lookup("AvailabilityManagerAssignmentList", list_reference)
    if _availability_list_reference_count(index, list_name) != 1:
        raise ControlInstrumentationError(
            f"availability-manager list {list_name!r} is shared or ambiguously referenced"
        )
    managers = _records(
        assignment_list,
        "managers",
        owner=f"AvailabilityManagerAssignmentList/{list_name}",
    )
    changed = False
    for manager in managers:
        object_type = _field_text(
            manager,
            "availability_manager_object_type",
            owner=f"AvailabilityManagerAssignmentList/{list_name}",
        )
        object_name_ref = _field_text(
            manager,
            "availability_manager_name",
            owner=f"AvailabilityManagerAssignmentList/{list_name}",
        )
        if object_type.casefold() != "AvailabilityManager:NightCycle".casefold():
            continue
        old_name, _ = index.lookup("AvailabilityManager:NightCycle", object_name_ref)
        if (
            _availability_manager_reference_count(
                index,
                "AvailabilityManager:NightCycle",
                old_name,
            )
            != 1
        ):
            raise ControlInstrumentationError(
                f"night-cycle availability manager {old_name!r} is shared or ambiguously referenced"
            )
        new_name = _scheduled_availability_name(context, old_name)
        index.assert_name_available("AvailabilityManager:Scheduled", new_name)
        index.add(
            "AvailabilityManager:Scheduled",
            new_name,
            {"schedule_name": context.names.always_on_schedule},
        )
        _mark_change(
            planned,
            "AvailabilityManager:Scheduled",
            new_name,
            "add_always_on_loop_availability",
            (context.fan_actuator.endpoint_id,),
        )
        index.remove("AvailabilityManager:NightCycle", old_name)
        _mark_change(
            planned,
            "AvailabilityManager:NightCycle",
            old_name,
            "remove_night_cycle_availability",
            (context.fan_actuator.endpoint_id,),
        )
        manager["availability_manager_object_type"] = "AvailabilityManager:Scheduled"
        manager["availability_manager_name"] = new_name
        changed = True
    if changed:
        _mark_change(
            planned,
            "AvailabilityManagerAssignmentList",
            list_name,
            "replace_night_cycle_with_always_on",
            (context.fan_actuator.endpoint_id,),
        )


def _instrument_context(
    index: _ModelIndex,
    context: _SystemContext,
    planned: dict[tuple[str, str], _PlannedChange],
) -> None:
    _preflight_generated_names(index, context)
    record = context.record
    fan_endpoint = context.fan_actuator.endpoint_id
    sat_endpoint = context.sat_actuator.endpoint_id
    names = context.names

    index.add(
        "ScheduleTypeLimits",
        names.availability_type_limits,
        {
            "lower_limit_value": 0,
            "upper_limit_value": 1,
            "numeric_type": "Discrete",
            "unit_type": "Availability",
        },
    )
    _mark_change(
        planned,
        "ScheduleTypeLimits",
        names.availability_type_limits,
        "add_fan_availability_type_limits",
        (fan_endpoint,),
    )
    index.add(
        "Schedule:Constant",
        names.always_on_schedule,
        {
            "hourly_value": 1,
            "schedule_type_limits_name": names.availability_type_limits,
        },
    )
    _mark_change(
        planned,
        "Schedule:Constant",
        names.always_on_schedule,
        "add_fan_always_on_schedule",
        (fan_endpoint,),
    )
    index.add(
        "Schedule:Constant",
        names.fan_mode_schedule,
        {
            "hourly_value": 1,
            "schedule_type_limits_name": names.availability_type_limits,
        },
    )
    _mark_change(
        planned,
        "Schedule:Constant",
        names.fan_mode_schedule,
        "add_fan_operating_mode_schedule",
        (fan_endpoint,),
    )

    sat_upper = record.sat_upper_bound_c
    assert sat_upper is not None
    index.add(
        "ScheduleTypeLimits",
        names.sat_type_limits,
        {
            "lower_limit_value": UNITARY_SAT_LOWER_BOUND_C,
            "upper_limit_value": sat_upper,
            "numeric_type": "Continuous",
            "unit_type": "Temperature",
        },
    )
    _mark_change(
        planned,
        "ScheduleTypeLimits",
        names.sat_type_limits,
        "add_sat_type_limits",
        (sat_endpoint,),
    )
    initial_sat = min(INITIAL_SAT_SETPOINT_C, sat_upper)
    index.add(
        "Schedule:Constant",
        names.sat_schedule,
        {
            "hourly_value": initial_sat,
            "schedule_type_limits_name": names.sat_type_limits,
        },
    )
    _mark_change(
        planned,
        "Schedule:Constant",
        names.sat_schedule,
        "add_sat_actuator_schedule",
        (sat_endpoint,),
    )

    system_name, system = index.lookup(UNITARY_OBJECT_TYPE, record.object_name)
    system["control_type"] = "SetPoint"
    system["supply_air_fan_operating_mode_schedule_name"] = names.fan_mode_schedule
    _mark_change(
        planned,
        UNITARY_OBJECT_TYPE,
        system_name,
        "instrument_unitary_control_interface",
        (fan_endpoint, sat_endpoint),
    )
    _set_dx_lockout(index, system, context, planned)

    fan_type = record.fan_object_type
    fan_name_ref = record.fan_object_name
    assert fan_type is not None and fan_name_ref is not None
    fan_name, fan = index.lookup(fan_type, fan_name_ref)
    fan["availability_schedule_name"] = names.always_on_schedule
    _mark_change(
        planned,
        fan_type,
        fan_name,
        "force_fan_always_available",
        (fan_endpoint,),
    )

    _replace_night_cycle_availability(index, context, planned)
    _remove_conflicting_outlet_managers(index, context, planned)
    outlet = record.supply_outlet_node_name
    assert outlet is not None
    index.add(
        "SetpointManager:Scheduled",
        names.sat_manager,
        {
            "control_variable": "Temperature",
            "schedule_name": names.sat_schedule,
            "setpoint_node_or_nodelist_name": outlet,
        },
    )
    _mark_change(
        planned,
        "SetpointManager:Scheduled",
        names.sat_manager,
        "add_scheduled_outlet_sat_manager",
        (sat_endpoint,),
    )


def _build_mutations(
    source_index: _ModelIndex,
    prepared_index: _ModelIndex,
    planned: Mapping[tuple[str, str], _PlannedChange],
) -> tuple[MutationRecord, ...]:
    source_objects = source_index.object_map()
    prepared_objects = prepared_index.object_map()
    changed_keys = [
        key
        for key in set(source_objects) | set(prepared_objects)
        if source_objects.get(key) != prepared_objects.get(key)
    ]
    changed_keys.sort()
    unplanned = [key for key in changed_keys if key not in planned]
    if unplanned:
        raise ControlInstrumentationError(f"instrumentation produced unreceipted object mutations: {unplanned!r}")

    mutations: list[MutationRecord] = []
    for sequence, key in enumerate(changed_keys):
        source = source_objects.get(key)
        prepared = prepared_objects.get(key)
        if source is None:
            assert prepared is not None
            object_type, object_name, after_fields = prepared
            operation = MutationOperation.ADD
            before_sha = None
            after_sha = energyplus_source_object_sha256(object_type, object_name, after_fields)
        elif prepared is None:
            object_type, object_name, before_fields = source
            operation = MutationOperation.REMOVE
            before_sha = energyplus_source_object_sha256(object_type, object_name, before_fields)
            after_sha = None
        else:
            source_type, source_name, before_fields = source
            object_type, object_name, after_fields = prepared
            if (source_type, source_name) != (object_type, object_name):
                raise ControlInstrumentationError("instrumentation cannot rename EnergyPlus objects implicitly")
            operation = MutationOperation.REPLACE
            before_sha = energyplus_source_object_sha256(object_type, object_name, before_fields)
            after_sha = energyplus_source_object_sha256(object_type, object_name, after_fields)
        metadata = planned[key]
        mutations.append(
            MutationRecord(
                sequence=sequence,
                operation=operation,
                object_type=object_type,
                object_name=object_name,
                reason_code=metadata.reason_code,
                before_sha256=before_sha,
                after_sha256=after_sha,
                affected_endpoint_ids=tuple(sorted(metadata.endpoint_ids, key=str.casefold)),
            )
        )
    if not mutations:
        raise ControlInstrumentationError("unitary instrumentation unexpectedly produced no mutations")
    return tuple(mutations)


def _build_manifest(
    source_sha256: str,
    prepared_sha256: str,
    contexts: tuple[_SystemContext, ...],
    mutations: tuple[MutationRecord, ...],
) -> ControlCapabilityManifest:
    equipment = tuple(item.equipment for item in contexts)
    actuators = tuple(actuator for item in contexts for actuator in (item.fan_actuator, item.sat_actuator))

    zone_nodes: dict[str, TopologyNode] = {}
    equipment_nodes: list[TopologyNode] = []
    topology_edges: list[TopologyEdge] = []
    outputs: list[OutputVariableDescriptor] = []
    candidates: list[DiscoveredEquipmentCandidate] = []
    supply_outlets: list[SupplyOutletNodeEvidence] = []
    for item in contexts:
        zone_nodes.setdefault(
            item.zone_id,
            TopologyNode(
                node_id=item.zone_id,
                kind=TopologyNodeKind.ZONE,
                object_type="Zone",
                object_name=item.zone_name,
            ),
        )
        equipment_nodes.append(
            TopologyNode(
                node_id=item.equipment.equipment_id,
                kind=TopologyNodeKind.EQUIPMENT,
                object_type=UNITARY_OBJECT_TYPE,
                object_name=item.record.object_name,
            )
        )
        topology_edges.append(
            TopologyEdge(
                source_id=item.equipment.equipment_id,
                target_id=item.zone_id,
                relation=TopologyRelation.CONTROLS,
            )
        )
        candidates.append(
            DiscoveredEquipmentCandidate(
                object_type=UNITARY_OBJECT_TYPE,
                object_name=item.record.object_name,
                source_object_sha256=item.record.source_object_sha256,
                disposition=CandidateDisposition.SUPPORTED,
                served_entity_ids=(item.zone_id,),
                prepared_object_type=UNITARY_OBJECT_TYPE,
                prepared_object_name=item.record.object_name,
                equipment_id=item.equipment.equipment_id,
            )
        )
        outlet = item.record.supply_outlet_node_name
        assert outlet is not None
        supply_outlets.append(
            SupplyOutletNodeEvidence(
                observed_node_id=item.equipment.equipment_id,
                source_object_type=UNITARY_OBJECT_TYPE,
                source_object_name=item.record.object_name,
                source_object_sha256=item.record.source_object_sha256,
                outlet_node_name=outlet,
            )
        )
        outputs.extend(
            (
                OutputVariableDescriptor(
                    semantic=OutputVariableSemantic.SUPPLY_AIR_MASS_FLOW,
                    observed_node_id=item.equipment.equipment_id,
                    variable_name="System Node Mass Flow Rate",
                    key_value=outlet,
                    unit=PhysicalUnit.KILOGRAMS_PER_SECOND,
                    lower_bound=0.0,
                    upper_bound=item.fan_actuator.upper_bound,
                    required_observation=False,
                    actuator_endpoint_ids=(item.fan_actuator.endpoint_id,),
                ),
                OutputVariableDescriptor(
                    semantic=OutputVariableSemantic.SUPPLY_AIR_TEMPERATURE,
                    observed_node_id=item.equipment.equipment_id,
                    variable_name="System Node Temperature",
                    key_value=outlet,
                    unit=PhysicalUnit.CELSIUS,
                    lower_bound=-100.0,
                    upper_bound=max(100.0, item.sat_actuator.upper_bound),
                    required_observation=False,
                    actuator_endpoint_ids=(item.sat_actuator.endpoint_id,),
                ),
            )
        )

    for zone_id, node in sorted(zone_nodes.items(), key=lambda value: value[0].casefold()):
        outputs.append(
            OutputVariableDescriptor(
                semantic=OutputVariableSemantic.ZONE_AIR_TEMPERATURE,
                observed_node_id=zone_id,
                variable_name="ZONE AIR TEMPERATURE",
                key_value=node.object_name,  # type: ignore[arg-type]
                unit=PhysicalUnit.CELSIUS,
                lower_bound=-100.0,
                upper_bound=100.0,
                required_observation=True,
            )
        )

    candidate_tuple = tuple(candidates)
    supply_outlet_tuple = tuple(supply_outlets)
    enumerator = DiscoveryEnumeratorContract(
        enumerator_id="hot-b2b-unitary-exact-epjson-v1",
        source_model_sha256=source_sha256,
        candidate_count=len(candidate_tuple),
        candidate_inventory_sha256=candidate_inventory_sha256(candidate_tuple),
        supply_outlet_count=len(supply_outlet_tuple),
        supply_outlet_inventory_sha256=supply_outlet_inventory_sha256(supply_outlet_tuple),
    )
    targets = tuple(
        MutationTargetState(
            object_type=item.object_type,
            object_name=item.object_name,
            source_sha256=item.before_sha256,
            prepared_sha256=item.after_sha256,
        )
        for item in mutations
    )
    receipt = InstrumentationReceipt(
        mode=InstrumentationMode.GENERATED,
        source_model_sha256=source_sha256,
        prepared_model_sha256=prepared_sha256,
        mutation_sequence_sha256=mutation_sequence_sha256(mutations),
        targets=targets,
    )
    commands = tuple(
        ActionCommand(
            order=index,
            name=f"{actuator.equipment_id}:{actuator.channel.value}",
            unit=ActionCommandUnit(actuator.unit.value),
            lower_bound=actuator.lower_bound,
            upper_bound=actuator.upper_bound,
            transform_id="direct_v1",
            target_endpoint_ids=(actuator.endpoint_id,),
        )
        for index, actuator in enumerate(actuators)
    )
    action_binding = TaskActionBinding(
        binding_id="hot-b2b-unitary-direct-v1",
        kind=ActionBindingKind.DIRECT,
        commands=commands,
        fixed_endpoints=(),
    )
    return ControlCapabilityManifest(
        source_model_sha256=source_sha256,
        prepared_model_sha256=prepared_sha256,
        equipment=equipment,
        actuators=actuators,
        topology_nodes=tuple(zone_nodes.values()) + tuple(equipment_nodes),
        topology_edges=tuple(topology_edges),
        output_variables=tuple(outputs),
        supply_outlet_node_evidence=supply_outlet_tuple,
        discovered_candidates=candidate_tuple,
        candidate_inventory_complete=True,
        discovery_enumerator=enumerator,
        instrumentation=receipt,
        task_action_binding=action_binding,
        mutations=mutations,
    )


def _build_air_to_air_composite_manifest(
    conversion_source_bytes: bytes,
    conversion_result: Any,
    unitary_result: UnitaryInstrumentationResult,
) -> ControlCapabilityManifest:
    """Rebase converted-unitary evidence onto the exact AirToAir source.

    ``unitary_result`` proves the converted-model-to-instrumented-model slice.
    The conversion artifact proves the preceding source-to-converted slice.
    This function constructs one capability mutation receipt spanning both,
    while keeping the field-level conversion receipt independently replayable.
    """

    from building_gen.control.air_to_air_conversion import (
        AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
        BRANCH_OBJECT_TYPE,
        UNITARY_SYSTEM_OBJECT_TYPE,
        AirToAirConversionResult,
    )

    if type(conversion_result) is not AirToAirConversionResult:
        raise TypeError("conversion_result must be an exact AirToAirConversionResult")
    if type(unitary_result) is not UnitaryInstrumentationResult:
        raise TypeError("unitary_result must be an exact UnitaryInstrumentationResult")
    base_manifest = unitary_result.capability_manifest
    conversion_candidates = {item.object_name.casefold(): item for item in conversion_result.report.candidates}
    base_candidates = {item.object_name.casefold(): item for item in base_manifest.discovered_candidates}
    if not set(conversion_candidates).issubset(base_candidates):
        missing = sorted(item.object_name for key, item in conversion_candidates.items() if key not in base_candidates)
        raise ControlInstrumentationError(
            f"converted AirToAir candidates are absent from unitary discovery: {missing!r}"
        )

    source_model = _parse_epjson_bytes(conversion_source_bytes)
    final_model = _parse_epjson_bytes(unitary_result.prepared_epjson_bytes)
    source_index = _ModelIndex(source_model)
    final_index = _ModelIndex(final_model)
    planned: dict[tuple[str, str], _PlannedChange] = {
        (mutation.object_type.casefold(), mutation.object_name.casefold()): _PlannedChange(
            mutation.reason_code,
            set(mutation.affected_endpoint_ids),
        )
        for mutation in base_manifest.mutations
    }
    endpoint_ids_by_equipment = {
        equipment.equipment_id.casefold(): {
            actuator.endpoint_id
            for actuator in base_manifest.actuators
            if actuator.equipment_id == equipment.equipment_id
        }
        for equipment in base_manifest.equipment
    }
    for folded_name, conversion_candidate in conversion_candidates.items():
        candidate = base_candidates[folded_name]
        if (
            candidate.object_type != UNITARY_SYSTEM_OBJECT_TYPE
            or candidate.object_name != conversion_candidate.object_name
            or candidate.prepared_energyplus_object != (UNITARY_SYSTEM_OBJECT_TYPE, conversion_candidate.object_name)
            or candidate.disposition is not CandidateDisposition.SUPPORTED
            or candidate.equipment_id is None
        ):
            raise ControlInstrumentationError(
                f"converted candidate {conversion_candidate.object_name!r} has a noncanonical unitary binding"
            )
        if not hmac.compare_digest(
            candidate.source_object_sha256,
            conversion_candidate.converted_object_sha256,
        ):
            raise ControlInstrumentationError(
                f"converted candidate {conversion_candidate.object_name!r} object hash disagrees with discovery"
            )
        endpoints = endpoint_ids_by_equipment.get(candidate.equipment_id.casefold(), set())
        if not endpoints:
            raise ControlInstrumentationError(
                f"converted candidate {conversion_candidate.object_name!r} has no actuator endpoints"
            )
        planned[(AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE.casefold(), folded_name)] = _PlannedChange(
            "convert_air_to_air_heat_pump",
            set(endpoints),
        )
        unitary_key = (UNITARY_SYSTEM_OBJECT_TYPE.casefold(), folded_name)
        existing_unitary_plan = planned.get(unitary_key)
        unitary_endpoints = set(endpoints)
        if existing_unitary_plan is not None:
            unitary_endpoints.update(existing_unitary_plan.endpoint_ids)
        planned[unitary_key] = _PlannedChange(
            "convert_and_instrument_air_to_air_heat_pump",
            unitary_endpoints,
        )
        branch_key = (BRANCH_OBJECT_TYPE.casefold(), conversion_candidate.branch_name.casefold())
        existing_branch_plan = planned.get(branch_key)
        branch_endpoints = set(endpoints)
        if existing_branch_plan is not None:
            branch_endpoints.update(existing_branch_plan.endpoint_ids)
        planned[branch_key] = _PlannedChange(
            "rewrite_air_to_air_branch_reference",
            branch_endpoints,
        )

    mutations = _build_mutations(source_index, final_index, planned)
    mutation_by_key = {
        (mutation.object_type.casefold(), mutation.object_name.casefold()): mutation for mutation in mutations
    }
    candidates: list[DiscoveredEquipmentCandidate] = []
    for base_candidate in base_manifest.discovered_candidates:
        candidate_conversion = conversion_candidates.get(base_candidate.object_name.casefold())
        if candidate_conversion is None:
            candidates.append(base_candidate)
            continue
        conversion_keys = (
            (
                AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE.casefold(),
                candidate_conversion.object_name.casefold(),
            ),
            (
                UNITARY_SYSTEM_OBJECT_TYPE.casefold(),
                candidate_conversion.object_name.casefold(),
            ),
            (BRANCH_OBJECT_TYPE.casefold(), candidate_conversion.branch_name.casefold()),
        )
        try:
            conversion_mutation_ids = tuple(mutation_by_key[key].mutation_id for key in conversion_keys)
        except KeyError as error:  # pragma: no cover - guarded by the composite diff
            raise ControlInstrumentationError(
                f"converted candidate {candidate_conversion.object_name!r} lacks a composite mutation"
            ) from error
        candidates.append(
            DiscoveredEquipmentCandidate(
                object_type=AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
                object_name=candidate_conversion.object_name,
                source_object_sha256=candidate_conversion.source_object_sha256,
                disposition=base_candidate.disposition,
                served_entity_ids=base_candidate.served_entity_ids,
                prepared_object_type=UNITARY_SYSTEM_OBJECT_TYPE,
                prepared_object_name=candidate_conversion.object_name,
                conversion_mutation_ids=conversion_mutation_ids,
                equipment_id=base_candidate.equipment_id,
            )
        )
    candidate_tuple = tuple(candidates)

    supply_outlets: list[SupplyOutletNodeEvidence] = []
    for evidence in base_manifest.supply_outlet_node_evidence:
        outlet_conversion = conversion_candidates.get(evidence.source_object_name.casefold())
        if outlet_conversion is None:
            supply_outlets.append(evidence)
            continue
        supply_outlets.append(
            SupplyOutletNodeEvidence(
                observed_node_id=evidence.observed_node_id,
                source_object_type=AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
                source_object_name=outlet_conversion.object_name,
                source_object_sha256=outlet_conversion.source_object_sha256,
                outlet_node_name=evidence.outlet_node_name,
            )
        )
    supply_outlet_tuple = tuple(supply_outlets)
    source_sha256 = hashlib.sha256(conversion_source_bytes).hexdigest()
    if not hmac.compare_digest(source_sha256, conversion_result.report.source_model_sha256):
        raise ControlInstrumentationError("conversion receipt does not bind the composite source bytes")
    enumerator = DiscoveryEnumeratorContract(
        enumerator_id="hot-b2b-air-to-air-unitary-exact-epjson-v1",
        source_model_sha256=source_sha256,
        candidate_count=len(candidate_tuple),
        candidate_inventory_sha256=candidate_inventory_sha256(candidate_tuple),
        supply_outlet_count=len(supply_outlet_tuple),
        supply_outlet_inventory_sha256=supply_outlet_inventory_sha256(supply_outlet_tuple),
    )
    targets = tuple(
        MutationTargetState(
            object_type=mutation.object_type,
            object_name=mutation.object_name,
            source_sha256=mutation.before_sha256,
            prepared_sha256=mutation.after_sha256,
        )
        for mutation in mutations
    )
    receipt = InstrumentationReceipt(
        mode=InstrumentationMode.GENERATED,
        source_model_sha256=source_sha256,
        prepared_model_sha256=base_manifest.prepared_model_sha256,
        mutation_sequence_sha256=mutation_sequence_sha256(mutations),
        targets=targets,
    )
    try:
        return ControlCapabilityManifest(
            source_model_sha256=source_sha256,
            prepared_model_sha256=base_manifest.prepared_model_sha256,
            equipment=base_manifest.equipment,
            actuators=base_manifest.actuators,
            topology_nodes=base_manifest.topology_nodes,
            topology_edges=base_manifest.topology_edges,
            output_variables=base_manifest.output_variables,
            supply_outlet_node_evidence=supply_outlet_tuple,
            discovered_candidates=candidate_tuple,
            candidate_inventory_complete=True,
            discovery_enumerator=enumerator,
            instrumentation=receipt,
            task_action_binding=base_manifest.task_action_binding,
            unsupported_equipment=base_manifest.unsupported_equipment,
            mutations=mutations,
            upstream=base_manifest.upstream,
            publication_ready=base_manifest.publication_ready,
            profile_registry_sha256=base_manifest.profile_registry_sha256,
            profile_registry_version=base_manifest.profile_registry_version,
            schema_version=base_manifest.schema_version,
        )
    except (TypeError, ValueError) as error:
        raise ControlInstrumentationError(
            f"frozen capability contract rejected composite AirToAir instrumentation: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class UnitaryInstrumentationResult:
    """Canonical prepared bytes and their complete structural capability receipt."""

    prepared_epjson_bytes: bytes
    discovery_inventory_sha256: str
    capability_manifest: ControlCapabilityManifest
    schema_version: str = UNITARY_INSTRUMENTATION_SCHEMA_VERSION
    algorithm_id: str = UNITARY_INSTRUMENTATION_ALGORITHM_ID
    upstream_commit: str = BUILDING2BUILDING_COMMIT

    def __post_init__(self) -> None:
        if type(self.prepared_epjson_bytes) is not bytes:
            raise TypeError("prepared_epjson_bytes must be exact bytes")
        if type(self.discovery_inventory_sha256) is not str:
            raise TypeError("discovery_inventory_sha256 must be an exact string")
        _sha256(self.discovery_inventory_sha256, field_name="discovery_inventory_sha256")
        if type(self.capability_manifest) is not ControlCapabilityManifest:
            raise TypeError("capability_manifest must be an exact ControlCapabilityManifest")
        try:
            canonical_manifest = ControlCapabilityManifest.from_json(self.capability_manifest.to_json())
        except (TypeError, ValueError) as error:
            raise ControlInstrumentationError(
                f"capability_manifest cannot be canonically revalidated: {error}"
            ) from error
        object.__setattr__(self, "capability_manifest", canonical_manifest)
        for field_name in ("schema_version", "algorithm_id", "upstream_commit"):
            if type(getattr(self, field_name)) is not str:
                raise TypeError(f"{field_name} must be an exact string")
        version_identity = (self.schema_version, self.algorithm_id)
        supported_identities = {
            (
                UNITARY_INSTRUMENTATION_SCHEMA_VERSION,
                UNITARY_INSTRUMENTATION_ALGORITHM_ID,
            ),
            (
                AIR_TO_AIR_UNITARY_INSTRUMENTATION_SCHEMA_VERSION,
                AIR_TO_AIR_UNITARY_INSTRUMENTATION_ALGORITHM_ID,
            ),
        }
        if version_identity not in supported_identities:
            raise ControlInstrumentationError("unexpected unitary instrumentation schema/algorithm identity")
        converted_candidates = tuple(
            candidate
            for candidate in canonical_manifest.discovered_candidates
            if candidate.energyplus_object != candidate.prepared_energyplus_object
        )
        is_air_to_air_identity = version_identity == (
            AIR_TO_AIR_UNITARY_INSTRUMENTATION_SCHEMA_VERSION,
            AIR_TO_AIR_UNITARY_INSTRUMENTATION_ALGORITHM_ID,
        )
        if bool(converted_candidates) != is_air_to_air_identity:
            raise ControlInstrumentationError(
                "AirToAir conversion candidates and instrumentation identity must be paired"
            )
        if is_air_to_air_identity:
            from building_gen.control.air_to_air_conversion import (
                AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
                UNITARY_SYSTEM_OBJECT_TYPE,
            )

            for candidate in converted_candidates:
                if (
                    candidate.object_type != AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE
                    or candidate.prepared_energyplus_object != (UNITARY_SYSTEM_OBJECT_TYPE, candidate.object_name)
                    or not candidate.conversion_mutation_ids
                ):
                    raise ControlInstrumentationError(
                        "AirToAir instrumentation identity requires canonical HeatPump-to-Unitary provenance"
                    )
        if self.upstream_commit != BUILDING2BUILDING_COMMIT:
            raise ControlInstrumentationError("unitary instrumentation must bind Building2Building v1.0.0")
        parsed = _parse_epjson_bytes(self.prepared_epjson_bytes)
        _ModelIndex(parsed)
        if _canonical_epjson_bytes(parsed) != self.prepared_epjson_bytes:
            raise ControlInstrumentationError("prepared_epjson_bytes must use canonical JSON encoding")
        prepared_sha = hashlib.sha256(self.prepared_epjson_bytes).hexdigest()
        if not hmac.compare_digest(prepared_sha, self.capability_manifest.prepared_model_sha256):
            raise ControlInstrumentationError("prepared bytes do not match the capability prepared-model hash")

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
            "source_model_sha256": self.source_model_sha256,
            "prepared_model_sha256": self.prepared_model_sha256,
            "discovery_inventory_sha256": self.discovery_inventory_sha256,
            "capability_sha256": self.capability_manifest.capability_sha256,
        }
        return hashlib.sha256(
            b"HOT unitary instrumentation artifact v1\0"
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def instrument_unitary_systems_from_json_bytes(
    source_bytes: bytes,
    expected_inventory: UnitaryDiscoveryInventory,
) -> UnitaryInstrumentationResult:
    """Instrument every supported discovered unitary system from exact bytes."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    if not isinstance(expected_inventory, UnitaryDiscoveryInventory):
        raise TypeError("expected_inventory must be a UnitaryDiscoveryInventory")
    try:
        inventory = verify_unitary_discovery_inventory_from_json_bytes(source_bytes, expected_inventory)
    except (ControlDiscoveryError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(f"unitary discovery inventory verification failed: {error}") from error

    source_model = _parse_epjson_bytes(source_bytes)
    source_index = _ModelIndex(source_model)
    contexts = _build_contexts(source_index, inventory)
    prepared_model = deepcopy(source_model)
    prepared_index = _ModelIndex(prepared_model)
    planned: dict[tuple[str, str], _PlannedChange] = {}
    for context in contexts:
        _instrument_context(prepared_index, context, planned)

    prepared_bytes = _canonical_epjson_bytes(prepared_model)
    prepared_sha = hashlib.sha256(prepared_bytes).hexdigest()
    mutations = _build_mutations(source_index, prepared_index, planned)
    try:
        manifest = _build_manifest(
            hashlib.sha256(source_bytes).hexdigest(),
            prepared_sha,
            contexts,
            mutations,
        )
    except (TypeError, ValueError) as error:
        raise ControlInstrumentationError(f"frozen capability contract rejected instrumentation: {error}") from error
    return UnitaryInstrumentationResult(
        prepared_epjson_bytes=prepared_bytes,
        discovery_inventory_sha256=inventory.inventory_sha256,
        capability_manifest=manifest,
    )


def instrument_air_to_air_heat_pumps_from_json_bytes(
    source_bytes: bytes,
    conversion_result: Any,
    expected_inventory: UnitaryDiscoveryInventory,
) -> UnitaryInstrumentationResult:
    """Instrument an authenticated AirToAir conversion as one composite slice."""

    from building_gen.control.air_to_air_conversion import (
        AirToAirConversionError,
        AirToAirConversionResult,
        verify_air_to_air_conversion_replay,
    )

    if type(source_bytes) is not bytes:
        raise TypeError("source_bytes must be exact bytes")
    if type(conversion_result) is not AirToAirConversionResult:
        raise TypeError("conversion_result must be an exact AirToAirConversionResult")
    try:
        verified_conversion = verify_air_to_air_conversion_replay(source_bytes, conversion_result)
    except (AirToAirConversionError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(f"AirToAir conversion verification failed: {error}") from error
    unitary_result = instrument_unitary_systems_from_json_bytes(
        verified_conversion.converted_model_bytes,
        expected_inventory,
    )
    manifest = _build_air_to_air_composite_manifest(
        source_bytes,
        verified_conversion,
        unitary_result,
    )
    return UnitaryInstrumentationResult(
        prepared_epjson_bytes=unitary_result.prepared_epjson_bytes,
        discovery_inventory_sha256=unitary_result.discovery_inventory_sha256,
        capability_manifest=manifest,
        schema_version=AIR_TO_AIR_UNITARY_INSTRUMENTATION_SCHEMA_VERSION,
        algorithm_id=AIR_TO_AIR_UNITARY_INSTRUMENTATION_ALGORITHM_ID,
    )


def verify_unitary_instrumentation_replay(
    source_bytes: bytes,
    expected_inventory: UnitaryDiscoveryInventory,
    expected_result: UnitaryInstrumentationResult,
) -> UnitaryInstrumentationResult:
    """Recompute exact instrumentation and reject any artifact tampering."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    if type(expected_result) is not UnitaryInstrumentationResult:
        raise TypeError("expected_result must be an exact UnitaryInstrumentationResult")
    try:
        canonical_expected = UnitaryInstrumentationResult(
            prepared_epjson_bytes=expected_result.prepared_epjson_bytes,
            discovery_inventory_sha256=expected_result.discovery_inventory_sha256,
            capability_manifest=expected_result.capability_manifest,
            schema_version=expected_result.schema_version,
            algorithm_id=expected_result.algorithm_id,
            upstream_commit=expected_result.upstream_commit,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(
            f"expected unitary instrumentation artifact is malformed or tampered: {error}"
        ) from error
    recomputed = instrument_unitary_systems_from_json_bytes(source_bytes, expected_inventory)
    if recomputed != canonical_expected or not hmac.compare_digest(
        recomputed.artifact_sha256,
        canonical_expected.artifact_sha256,
    ):
        raise ControlInstrumentationError("unitary instrumentation artifact does not match deterministic replay")
    return recomputed


def verify_air_to_air_unitary_instrumentation_replay(
    source_bytes: bytes,
    conversion_result: Any,
    expected_inventory: UnitaryDiscoveryInventory,
    expected_result: UnitaryInstrumentationResult,
) -> UnitaryInstrumentationResult:
    """Recompute the conversion-aware instrumentation and reject substitution."""

    from building_gen.control.air_to_air_conversion import AirToAirConversionResult

    if type(source_bytes) is not bytes:
        raise TypeError("source_bytes must be exact bytes")
    if type(conversion_result) is not AirToAirConversionResult:
        raise TypeError("conversion_result must be an exact AirToAirConversionResult")
    if type(expected_result) is not UnitaryInstrumentationResult:
        raise TypeError("expected_result must be an exact UnitaryInstrumentationResult")
    try:
        canonical_expected = UnitaryInstrumentationResult(
            prepared_epjson_bytes=expected_result.prepared_epjson_bytes,
            discovery_inventory_sha256=expected_result.discovery_inventory_sha256,
            capability_manifest=expected_result.capability_manifest,
            schema_version=expected_result.schema_version,
            algorithm_id=expected_result.algorithm_id,
            upstream_commit=expected_result.upstream_commit,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ControlInstrumentationError(
            f"expected AirToAir instrumentation artifact is malformed or tampered: {error}"
        ) from error
    recomputed = instrument_air_to_air_heat_pumps_from_json_bytes(
        source_bytes,
        conversion_result,
        expected_inventory,
    )
    if recomputed != canonical_expected or not hmac.compare_digest(
        recomputed.artifact_sha256,
        canonical_expected.artifact_sha256,
    ):
        raise ControlInstrumentationError(
            "AirToAir unitary instrumentation artifact does not match deterministic replay"
        )
    return recomputed


__all__ = [
    "AIR_TO_AIR_UNITARY_INSTRUMENTATION_ALGORITHM_ID",
    "AIR_TO_AIR_UNITARY_INSTRUMENTATION_SCHEMA_VERSION",
    "ControlInstrumentationError",
    "DX_COOLING_COMPRESSOR_MIN_OAT_C",
    "INITIAL_SAT_SETPOINT_C",
    "UNITARY_INSTRUMENTATION_ALGORITHM_ID",
    "UNITARY_INSTRUMENTATION_SCHEMA_VERSION",
    "UnitaryInstrumentationResult",
    "instrument_air_to_air_heat_pumps_from_json_bytes",
    "instrument_unitary_systems_from_json_bytes",
    "verify_air_to_air_unitary_instrumentation_replay",
    "verify_unitary_instrumentation_replay",
]
