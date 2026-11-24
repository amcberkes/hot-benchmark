"""Exact-byte discovery for the pinned Building2Building VAV control slice.

The topology walk is adapted from ``make_vav_system_controllable`` in the
authenticated Building2Building v1.0.0 source and strengthened with complete
``AirLoopHVAC`` candidate coverage, case-insensitive collision rejection,
canonical reference ownership, exact source-object hashes, and deterministic
inventory replay.  Discovery never mutates the model or reads the filesystem.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from building_gen.control.capabilities import (
    BUILDING2BUILDING_COMMIT,
    CandidateDisposition,
)
from building_gen.control.discovery import energyplus_source_object_sha256
from building_gen.control.instrumentation import (
    ControlInstrumentationError,
    _ModelIndex,
    _owned_bytes,
    _parse_epjson_bytes,
)

VAV_PROFILE_ID = "b2b_vavsystem_v1"
VAV_LOOP_OBJECT_TYPE = "AirLoopHVAC"
VAV_TERMINAL_OBJECT_TYPE = "AirTerminal:SingleDuct:VAV:Reheat"
VAV_OA_CONTROLLER_OBJECT_TYPE = "Controller:OutdoorAir"
VAV_DISCOVERY_SCHEMA_VERSION = "hot-b2b-vav-discovery/1.0.0"
VAV_DISCOVERY_ALGORITHM_ID = "b2b-v1-vav-exact-topology-v1"
BUILDING2BUILDING_ACTUATORS_SOURCE_PATH = "building2building/pipeline/actuators.py"
BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256 = "4e96d35f40bc9589a241e6f3a776c7dbbeac6dda220f635bbda90bc16189847d"

VAV_FAN_OBJECT_TYPES = (
    "Fan:SystemModel",
    "Fan:OnOff",
    "Fan:ConstantVolume",
    "Fan:VariableVolume",
)
VAV_OA_SCHEDULE_FIELDS = (
    "minimum_outdoor_air_schedule_name",
    "minimum_fraction_of_outdoor_air_schedule_name",
    "maximum_fraction_of_outdoor_air_schedule_name",
    "time_of_day_economizer_control_schedule_name",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class VAVDiscoveryError(ValueError):
    """Raised when a VAV candidate inventory is ambiguous or unauthenticated."""


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise VAVDiscoveryError(f"{field_name} must be a nonempty exact string")
    value = str.__str__(value)
    if not value or value != str.strip(value):
        raise VAVDiscoveryError(f"{field_name} must be a nonempty exact string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise VAVDiscoveryError(f"{field_name} contains a control character")
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    value = _text(value, field_name=field_name)
    if _SHA256_RE.fullmatch(value) is None:
        raise VAVDiscoveryError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise VAVDiscoveryError(f"value is not canonical JSON data: {error}") from error


def _records(fields: Mapping[str, Any], key: str, *, owner: str) -> tuple[Mapping[str, Any], ...]:
    raw = fields.get(key, ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise VAVDiscoveryError(f"{owner}.{key} must be an array")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise VAVDiscoveryError(f"{owner}.{key}[{index}] must be an object")
        result.append(item)
    return tuple(result)


def _field(fields: Mapping[str, Any], key: str, *, owner: str) -> str:
    return _text(fields.get(key), field_name=f"{owner}.{key}")


def _table(index: _ModelIndex, object_type: str) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    table = index.table(object_type) or {}
    values: list[tuple[str, Mapping[str, Any]]] = []
    for object_name, fields in table.items():
        if not isinstance(fields, Mapping):  # pragma: no cover - _ModelIndex already proves this
            raise VAVDiscoveryError(f"{object_type}/{object_name} must be an object")
        values.append((object_name, fields))
    return tuple(sorted(values, key=lambda item: item[0].casefold()))


def _lookup(index: _ModelIndex, object_type: str, object_name: Any, *, label: str) -> tuple[str, Mapping[str, Any]]:
    reference = _text(object_name, field_name=label)
    try:
        name, fields = index.lookup(object_type, reference)
    except ControlInstrumentationError as error:
        raise VAVDiscoveryError(f"{label} does not resolve exactly: {reference!r}") from error
    if name != reference:
        raise VAVDiscoveryError(f"{label} must use canonical object name {name!r}")
    return name, fields


def _canonical_type_reference(value: Any, expected: str, *, field_name: str) -> str:
    actual = _text(value, field_name=field_name)
    if actual.casefold() != expected.casefold() or actual != expected:
        raise VAVDiscoveryError(f"{field_name} must use canonical object type {expected!r}")
    return actual


def _object_sha(object_type: str, object_name: str, fields: Mapping[str, Any]) -> str:
    return energyplus_source_object_sha256(object_type, object_name, fields)


@dataclass(frozen=True, slots=True)
class VAVObjectReference:
    object_type: str
    object_name: str
    source_object_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_type", _text(self.object_type, field_name="object_type"))
        object.__setattr__(self, "object_name", _text(self.object_name, field_name="object_name"))
        object.__setattr__(
            self,
            "source_object_sha256",
            _sha256(self.source_object_sha256, field_name="source_object_sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "object_type": self.object_type,
            "object_name": self.object_name,
            "source_object_sha256": self.source_object_sha256,
        }


@dataclass(frozen=True, slots=True)
class VAVScheduleReference:
    field_name: str
    schedule: VAVObjectReference

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_name", _text(self.field_name, field_name="field_name"))
        if type(self.schedule) is not VAVObjectReference:
            raise TypeError("schedule must be an exact VAVObjectReference")

    def to_dict(self) -> dict[str, Any]:
        return {"field_name": self.field_name, "schedule": self.schedule.to_dict()}


@dataclass(frozen=True, slots=True)
class VAVTerminalDiscoveryRecord:
    zone: VAVObjectReference
    terminal: VAVObjectReference
    terminal_inlet_node_name: str
    terminal_outlet_node_name: str
    thermostat: VAVObjectReference
    dual_setpoint: VAVObjectReference
    heating_schedule: VAVObjectReference
    cooling_schedule: VAVObjectReference

    def __post_init__(self) -> None:
        for field_name in (
            "zone",
            "terminal",
            "thermostat",
            "dual_setpoint",
            "heating_schedule",
            "cooling_schedule",
        ):
            if type(getattr(self, field_name)) is not VAVObjectReference:
                raise TypeError(f"{field_name} must be an exact VAVObjectReference")
        if self.zone.object_type != "Zone":
            raise VAVDiscoveryError("VAV terminal zone reference must target Zone")
        if self.terminal.object_type != VAV_TERMINAL_OBJECT_TYPE:
            raise VAVDiscoveryError("VAV terminal reference has the wrong object type")
        if self.thermostat.object_type != "ZoneControl:Thermostat":
            raise VAVDiscoveryError("VAV thermostat reference has the wrong object type")
        if self.dual_setpoint.object_type != "ThermostatSetpoint:DualSetpoint":
            raise VAVDiscoveryError("VAV dual-setpoint reference has the wrong object type")
        object.__setattr__(
            self,
            "terminal_inlet_node_name",
            _text(self.terminal_inlet_node_name, field_name="terminal_inlet_node_name"),
        )
        object.__setattr__(
            self,
            "terminal_outlet_node_name",
            _text(self.terminal_outlet_node_name, field_name="terminal_outlet_node_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone.to_dict(),
            "terminal": self.terminal.to_dict(),
            "terminal_inlet_node_name": self.terminal_inlet_node_name,
            "terminal_outlet_node_name": self.terminal_outlet_node_name,
            "thermostat": self.thermostat.to_dict(),
            "dual_setpoint": self.dual_setpoint.to_dict(),
            "heating_schedule": self.heating_schedule.to_dict(),
            "cooling_schedule": self.cooling_schedule.to_dict(),
        }


def _validate_canonical_loop_collections(
    schedule_references: tuple[VAVScheduleReference, ...],
    fans: tuple[VAVObjectReference, ...],
    availability_managers: tuple[VAVObjectReference, ...],
    terminals: tuple[VAVTerminalDiscoveryRecord, ...],
) -> None:
    schedule_fields = tuple(item.field_name for item in schedule_references)
    expected_schedule_fields = tuple(field for field in VAV_OA_SCHEDULE_FIELDS if field in schedule_fields)
    if schedule_fields != expected_schedule_fields or len(schedule_fields) != len(set(schedule_fields)):
        raise VAVDiscoveryError("outdoor-air schedule references must use canonical unique field order")
    fan_keys = tuple((item.object_type.casefold(), item.object_name.casefold()) for item in fans)
    if any(item.object_type not in VAV_FAN_OBJECT_TYPES for item in fans):
        raise VAVDiscoveryError("VAV fan references must use a supported canonical fan type")
    if fan_keys != tuple(sorted(fan_keys)) or len(fan_keys) != len(set(fan_keys)):
        raise VAVDiscoveryError("VAV fan references must use canonical unique order")
    manager_keys = tuple((item.object_type.casefold(), item.object_name.casefold()) for item in availability_managers)
    if any(not item.object_type.startswith("AvailabilityManager:") for item in availability_managers):
        raise VAVDiscoveryError("VAV availability references must use canonical manager types")
    if manager_keys != tuple(sorted(manager_keys)) or len(manager_keys) != len(set(manager_keys)):
        raise VAVDiscoveryError("VAV availability references must use canonical unique order")
    terminal_keys = tuple(item.zone.object_name.casefold() for item in terminals)
    if terminal_keys != tuple(sorted(terminal_keys)) or len(terminal_keys) != len(set(terminal_keys)):
        raise VAVDiscoveryError("VAV terminals must use canonical unique zone order")
    if len({item.terminal.object_name.casefold() for item in terminals}) != len(terminals):
        raise VAVDiscoveryError("VAV terminal object references must be unique per loop")


@dataclass(frozen=True, slots=True)
class VAVLoopDiscoveryRecord:
    object_name: str
    source_object_sha256: str
    disposition: CandidateDisposition
    reason_code: str | None
    demand_inlet_node_name: str | None
    supply_outlet_node_name: str | None
    outdoor_air_controller: VAVObjectReference | None
    outdoor_air_schedule_references: tuple[VAVScheduleReference, ...]
    fans: tuple[VAVObjectReference, ...]
    availability_assignment_list: VAVObjectReference | None
    availability_managers: tuple[VAVObjectReference, ...]
    terminals: tuple[VAVTerminalDiscoveryRecord, ...]
    object_type: str = VAV_LOOP_OBJECT_TYPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_name", _text(self.object_name, field_name="object_name"))
        object.__setattr__(
            self,
            "source_object_sha256",
            _sha256(self.source_object_sha256, field_name="source_object_sha256"),
        )
        if self.object_type != VAV_LOOP_OBJECT_TYPE:
            raise VAVDiscoveryError(f"object_type must equal {VAV_LOOP_OBJECT_TYPE!r}")
        try:
            disposition = CandidateDisposition(self.disposition)
        except (TypeError, ValueError) as error:
            raise VAVDiscoveryError("disposition is invalid") from error
        object.__setattr__(self, "disposition", disposition)
        for field_name, expected_type in (
            ("outdoor_air_schedule_references", VAVScheduleReference),
            ("fans", VAVObjectReference),
            ("availability_managers", VAVObjectReference),
            ("terminals", VAVTerminalDiscoveryRecord),
        ):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(type(item) is not expected_type for item in values):
                raise TypeError(f"{field_name} must be an exact tuple of {expected_type.__name__}")
        _validate_canonical_loop_collections(
            self.outdoor_air_schedule_references,
            self.fans,
            self.availability_managers,
            self.terminals,
        )
        if self.disposition is CandidateDisposition.SUPPORTED:
            if self.reason_code is not None:
                raise VAVDiscoveryError("supported VAV records cannot have a reason_code")
            if (
                self.demand_inlet_node_name is None
                or self.supply_outlet_node_name is None
                or self.outdoor_air_controller is None
                or self.availability_assignment_list is None
                or not self.fans
                or not self.availability_managers
                or not self.terminals
            ):
                raise VAVDiscoveryError("supported VAV records require complete topology")
            object.__setattr__(
                self,
                "demand_inlet_node_name",
                _text(self.demand_inlet_node_name, field_name="demand_inlet_node_name"),
            )
            object.__setattr__(
                self,
                "supply_outlet_node_name",
                _text(self.supply_outlet_node_name, field_name="supply_outlet_node_name"),
            )
            if type(self.outdoor_air_controller) is not VAVObjectReference:
                raise TypeError("outdoor_air_controller must be an exact VAVObjectReference")
            if self.outdoor_air_controller.object_type != VAV_OA_CONTROLLER_OBJECT_TYPE:
                raise VAVDiscoveryError("outdoor-air controller has the wrong object type")
            if self.availability_assignment_list is not None and (
                type(self.availability_assignment_list) is not VAVObjectReference
                or self.availability_assignment_list.object_type != "AvailabilityManagerAssignmentList"
            ):
                raise VAVDiscoveryError("availability assignment list has the wrong object type")
        else:
            if self.reason_code is None:
                raise VAVDiscoveryError("unsupported VAV records require a reason_code")
            object.__setattr__(self, "reason_code", _text(self.reason_code, field_name="reason_code"))
            if any(
                value is not None
                for value in (
                    self.demand_inlet_node_name,
                    self.supply_outlet_node_name,
                    self.outdoor_air_controller,
                    self.availability_assignment_list,
                )
            ) or any((self.outdoor_air_schedule_references, self.fans, self.availability_managers, self.terminals)):
                raise VAVDiscoveryError("unsupported VAV records cannot claim partial topology")

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(b"HOT VAV discovery candidate v1\0" + _canonical_json(self.to_dict())).hexdigest()
        return f"vav-candidate-{digest[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "object_name": self.object_name,
            "source_object_sha256": self.source_object_sha256,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "demand_inlet_node_name": self.demand_inlet_node_name,
            "supply_outlet_node_name": self.supply_outlet_node_name,
            "outdoor_air_controller": (
                None if self.outdoor_air_controller is None else self.outdoor_air_controller.to_dict()
            ),
            "outdoor_air_schedule_references": [item.to_dict() for item in self.outdoor_air_schedule_references],
            "fans": [item.to_dict() for item in self.fans],
            "availability_assignment_list": (
                None if self.availability_assignment_list is None else self.availability_assignment_list.to_dict()
            ),
            "availability_managers": [item.to_dict() for item in self.availability_managers],
            "terminals": [item.to_dict() for item in self.terminals],
        }


def _canonical_object_reference(value: Any, *, field_name: str) -> VAVObjectReference:
    if type(value) is not VAVObjectReference:
        raise TypeError(f"{field_name} must be an exact VAVObjectReference")
    try:
        return VAVObjectReference(
            object_type=value.object_type,
            object_name=value.object_name,
            source_object_sha256=value.source_object_sha256,
        )
    except AttributeError as error:
        raise VAVDiscoveryError(f"{field_name} is missing canonical reference state") from error


def _canonical_schedule_reference(value: Any, *, field_name: str) -> VAVScheduleReference:
    if type(value) is not VAVScheduleReference:
        raise TypeError(f"{field_name} must be an exact VAVScheduleReference")
    try:
        return VAVScheduleReference(
            field_name=value.field_name,
            schedule=_canonical_object_reference(
                value.schedule,
                field_name=f"{field_name}.schedule",
            ),
        )
    except AttributeError as error:
        raise VAVDiscoveryError(f"{field_name} is missing canonical schedule state") from error


def _canonical_terminal_record(value: Any, *, field_name: str) -> VAVTerminalDiscoveryRecord:
    if type(value) is not VAVTerminalDiscoveryRecord:
        raise TypeError(f"{field_name} must be an exact VAVTerminalDiscoveryRecord")
    try:
        return VAVTerminalDiscoveryRecord(
            zone=_canonical_object_reference(value.zone, field_name=f"{field_name}.zone"),
            terminal=_canonical_object_reference(
                value.terminal,
                field_name=f"{field_name}.terminal",
            ),
            terminal_inlet_node_name=value.terminal_inlet_node_name,
            terminal_outlet_node_name=value.terminal_outlet_node_name,
            thermostat=_canonical_object_reference(
                value.thermostat,
                field_name=f"{field_name}.thermostat",
            ),
            dual_setpoint=_canonical_object_reference(
                value.dual_setpoint,
                field_name=f"{field_name}.dual_setpoint",
            ),
            heating_schedule=_canonical_object_reference(
                value.heating_schedule,
                field_name=f"{field_name}.heating_schedule",
            ),
            cooling_schedule=_canonical_object_reference(
                value.cooling_schedule,
                field_name=f"{field_name}.cooling_schedule",
            ),
        )
    except AttributeError as error:
        raise VAVDiscoveryError(f"{field_name} is missing canonical terminal state") from error


def _canonical_loop_record(value: Any, *, field_name: str) -> VAVLoopDiscoveryRecord:
    if type(value) is not VAVLoopDiscoveryRecord:
        raise TypeError(f"{field_name} must be an exact VAVLoopDiscoveryRecord")
    try:
        return VAVLoopDiscoveryRecord(
            object_name=value.object_name,
            source_object_sha256=value.source_object_sha256,
            disposition=value.disposition,
            reason_code=value.reason_code,
            demand_inlet_node_name=value.demand_inlet_node_name,
            supply_outlet_node_name=value.supply_outlet_node_name,
            outdoor_air_controller=(
                None
                if value.outdoor_air_controller is None
                else _canonical_object_reference(
                    value.outdoor_air_controller,
                    field_name=f"{field_name}.outdoor_air_controller",
                )
            ),
            outdoor_air_schedule_references=tuple(
                _canonical_schedule_reference(
                    item,
                    field_name=f"{field_name}.outdoor_air_schedule_references[{index}]",
                )
                for index, item in enumerate(value.outdoor_air_schedule_references)
            ),
            fans=tuple(
                _canonical_object_reference(
                    item,
                    field_name=f"{field_name}.fans[{index}]",
                )
                for index, item in enumerate(value.fans)
            ),
            availability_assignment_list=(
                None
                if value.availability_assignment_list is None
                else _canonical_object_reference(
                    value.availability_assignment_list,
                    field_name=f"{field_name}.availability_assignment_list",
                )
            ),
            availability_managers=tuple(
                _canonical_object_reference(
                    item,
                    field_name=f"{field_name}.availability_managers[{index}]",
                )
                for index, item in enumerate(value.availability_managers)
            ),
            terminals=tuple(
                _canonical_terminal_record(
                    item,
                    field_name=f"{field_name}.terminals[{index}]",
                )
                for index, item in enumerate(value.terminals)
            ),
            object_type=value.object_type,
        )
    except AttributeError as error:
        raise VAVDiscoveryError(f"{field_name} is missing canonical loop state") from error


def _inventory_payload(
    *,
    source_model_sha256: str,
    records: tuple[VAVLoopDiscoveryRecord, ...],
    schema_version: str,
    algorithm_id: str,
    upstream_commit: str,
    upstream_source_path: str,
    upstream_source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "algorithm_id": algorithm_id,
        "upstream_commit": upstream_commit,
        "upstream_source_path": upstream_source_path,
        "upstream_source_sha256": upstream_source_sha256,
        "source_model_sha256": source_model_sha256,
        "records": [item.to_dict() for item in records],
    }


@dataclass(frozen=True, slots=True)
class VAVDiscoveryInventory:
    source_model_sha256: str
    records: tuple[VAVLoopDiscoveryRecord, ...]
    schema_version: str = VAV_DISCOVERY_SCHEMA_VERSION
    algorithm_id: str = VAV_DISCOVERY_ALGORITHM_ID
    upstream_commit: str = BUILDING2BUILDING_COMMIT
    upstream_source_path: str = BUILDING2BUILDING_ACTUATORS_SOURCE_PATH
    upstream_source_sha256: str = BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_model_sha256",
            _sha256(self.source_model_sha256, field_name="source_model_sha256"),
        )
        if type(self.records) is not tuple:
            raise TypeError("records must be an exact tuple")
        records = tuple(
            _canonical_loop_record(item, field_name=f"records[{index}]") for index, item in enumerate(self.records)
        )
        ordered = tuple(sorted(records, key=lambda item: item.object_name.casefold()))
        if ordered != records:
            raise VAVDiscoveryError("VAV discovery records must use canonical object-name order")
        if len({item.object_name.casefold() for item in ordered}) != len(ordered):
            raise VAVDiscoveryError("VAV discovery record names collide ignoring case")
        object.__setattr__(self, "records", records)
        schema_version = _text(self.schema_version, field_name="schema_version")
        algorithm_id = _text(self.algorithm_id, field_name="algorithm_id")
        upstream_commit = _text(self.upstream_commit, field_name="upstream_commit")
        upstream_source_path = _text(self.upstream_source_path, field_name="upstream_source_path")
        upstream_source_sha256 = _sha256(
            self.upstream_source_sha256,
            field_name="upstream_source_sha256",
        )
        if schema_version != VAV_DISCOVERY_SCHEMA_VERSION:
            raise VAVDiscoveryError("unexpected VAV discovery schema version")
        if algorithm_id != VAV_DISCOVERY_ALGORITHM_ID:
            raise VAVDiscoveryError("unexpected VAV discovery algorithm")
        if upstream_commit != BUILDING2BUILDING_COMMIT:
            raise VAVDiscoveryError("VAV discovery must bind Building2Building v1.0.0")
        if upstream_source_path != BUILDING2BUILDING_ACTUATORS_SOURCE_PATH:
            raise VAVDiscoveryError("VAV discovery has the wrong upstream source path")
        if upstream_source_sha256 != BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256:
            raise VAVDiscoveryError("VAV discovery has the wrong upstream source hash")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "algorithm_id", algorithm_id)
        object.__setattr__(self, "upstream_commit", upstream_commit)
        object.__setattr__(self, "upstream_source_path", upstream_source_path)
        object.__setattr__(self, "upstream_source_sha256", upstream_source_sha256)

    @property
    def supported_records(self) -> tuple[VAVLoopDiscoveryRecord, ...]:
        return tuple(item for item in self.records if item.disposition is CandidateDisposition.SUPPORTED)

    @property
    def unsupported_records(self) -> tuple[VAVLoopDiscoveryRecord, ...]:
        return tuple(item for item in self.records if item.disposition is CandidateDisposition.UNSUPPORTED)

    @property
    def inventory_sha256(self) -> str:
        payload = _inventory_payload(
            source_model_sha256=self.source_model_sha256,
            records=self.records,
            schema_version=self.schema_version,
            algorithm_id=self.algorithm_id,
            upstream_commit=self.upstream_commit,
            upstream_source_path=self.upstream_source_path,
            upstream_source_sha256=self.upstream_source_sha256,
        )
        return hashlib.sha256(b"HOT VAV discovery inventory v1\0" + _canonical_json(payload)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = _inventory_payload(
            source_model_sha256=self.source_model_sha256,
            records=self.records,
            schema_version=self.schema_version,
            algorithm_id=self.algorithm_id,
            upstream_commit=self.upstream_commit,
            upstream_source_path=self.upstream_source_path,
            upstream_source_sha256=self.upstream_source_sha256,
        )
        return {**payload, "inventory_sha256": self.inventory_sha256}

    def verify_source_bytes(self, source_bytes: bytes) -> VAVDiscoveryInventory:
        """Recompute this inventory against exact source bytes."""

        return verify_vav_discovery_inventory_from_json_bytes(source_bytes, self)


def _reference(object_type: str, object_name: str, fields: Mapping[str, Any]) -> VAVObjectReference:
    return VAVObjectReference(object_type, object_name, _object_sha(object_type, object_name, fields))


def _schedule_reference(index: _ModelIndex, schedule_name: Any, *, label: str) -> VAVObjectReference:
    reference = _text(schedule_name, field_name=label)
    matches = [
        (object_type, object_name, fields)
        for object_type, object_name, fields in index.iter_objects()
        if object_type.startswith("Schedule:") and object_name.casefold() == reference.casefold()
    ]
    if len(matches) != 1:
        raise VAVDiscoveryError(f"{label} must resolve to exactly one Schedule:* object: {reference!r}")
    object_type, object_name, fields = matches[0]
    if object_name != reference:
        raise VAVDiscoveryError(f"{label} must use canonical schedule name {object_name!r}")
    return _reference(object_type, object_name, fields)


def _zone_air_inlet_nodes(
    index: _ModelIndex,
    connection_name: str,
    connection: Mapping[str, Any],
) -> tuple[str, ...]:
    reference = _field(
        connection,
        "zone_air_inlet_node_or_nodelist_name",
        owner=connection_name,
    )
    node_list = index.maybe_lookup("NodeList", reference)
    if node_list is None:
        return (reference,)
    node_list_name, node_list_fields = node_list
    if node_list_name != reference:
        raise VAVDiscoveryError(
            f"{connection_name}.zone_air_inlet_node_or_nodelist_name must use canonical NodeList name "
            f"{node_list_name!r}"
        )
    nodes = tuple(
        _field(item, "node_name", owner=f"NodeList/{node_list_name}")
        for item in _records(node_list_fields, "nodes", owner=f"NodeList/{node_list_name}")
    )
    if not nodes:
        raise VAVDiscoveryError(f"zone inlet NodeList {node_list_name!r} is empty")
    if len({item.casefold() for item in nodes}) != len(nodes):
        raise VAVDiscoveryError(f"zone inlet NodeList {node_list_name!r} repeats a node")
    return nodes


def _zone_terminal_inventory(
    index: _ModelIndex,
) -> tuple[dict[str, tuple[VAVObjectReference, VAVObjectReference, str, str]], int]:
    terminal_table = {name.casefold(): (name, fields) for name, fields in _table(index, VAV_TERMINAL_OBJECT_TYPE)}
    occurrences: Counter[str] = Counter()
    by_terminal: dict[str, tuple[VAVObjectReference, VAVObjectReference, str, str]] = {}
    connections: list[tuple[str, Mapping[str, Any], str, Mapping[str, Any]]] = []
    zone_inlet_owners: dict[str, tuple[str, str]] = {}
    seen_connection_zones: set[str] = set()
    for connection_name, connection in _table(index, "ZoneHVAC:EquipmentConnections"):
        zone_name, zone_fields = _lookup(
            index,
            "Zone",
            _field(connection, "zone_name", owner=connection_name),
            label=f"{connection_name} zone",
        )
        if zone_name.casefold() in seen_connection_zones:
            raise VAVDiscoveryError(f"zone {zone_name!r} has multiple equipment connections")
        seen_connection_zones.add(zone_name.casefold())
        for inlet_node in _zone_air_inlet_nodes(index, connection_name, connection):
            inlet_key = inlet_node.casefold()
            previous = zone_inlet_owners.get(inlet_key)
            if previous is not None:
                raise VAVDiscoveryError(
                    f"zone air inlet {inlet_node!r} is owned by multiple equipment connections: "
                    f"{previous[0]!r}, {connection_name!r}"
                )
            zone_inlet_owners[inlet_key] = (connection_name, inlet_node)
        connections.append((connection_name, connection, zone_name, zone_fields))

    seen_zones: set[str] = set()
    seen_terminal_inlets: dict[str, str] = {}
    seen_terminal_outlets: dict[str, str] = {}
    for connection_name, connection, zone_name, zone_fields in connections:
        equipment_list_name, equipment_list = _lookup(
            index,
            "ZoneHVAC:EquipmentList",
            _field(connection, "zone_conditioning_equipment_list_name", owner=connection_name),
            label=f"{connection_name} equipment list",
        )
        for item in _records(equipment_list, "equipment", owner=equipment_list_name):
            raw_type = item.get("zone_equipment_object_type")
            if not isinstance(raw_type, str) or str.__str__(raw_type).casefold() != "zonehvac:airdistributionunit":
                continue
            _canonical_type_reference(
                raw_type,
                "ZoneHVAC:AirDistributionUnit",
                field_name=f"{equipment_list_name}.zone_equipment_object_type",
            )
            distribution_name, distribution = _lookup(
                index,
                "ZoneHVAC:AirDistributionUnit",
                item.get("zone_equipment_name"),
                label=f"{equipment_list_name} air distribution unit",
            )
            terminal_type = _field(distribution, "air_terminal_object_type", owner=distribution_name)
            if terminal_type.casefold() != VAV_TERMINAL_OBJECT_TYPE.casefold():
                continue
            _canonical_type_reference(
                terminal_type,
                VAV_TERMINAL_OBJECT_TYPE,
                field_name=f"{distribution_name}.air_terminal_object_type",
            )
            terminal_name, terminal = _lookup(
                index,
                VAV_TERMINAL_OBJECT_TYPE,
                _field(distribution, "air_terminal_name", owner=distribution_name),
                label=f"{distribution_name} VAV terminal",
            )
            terminal_inlet = _field(terminal, "air_inlet_node_name", owner=terminal_name)
            terminal_outlet = _field(terminal, "air_outlet_node_name", owner=terminal_name)
            distribution_outlet = _field(
                distribution,
                "air_distribution_unit_outlet_node_name",
                owner=distribution_name,
            )
            if distribution_outlet != terminal_outlet:
                raise VAVDiscoveryError(
                    f"air distribution unit {distribution_name!r} outlet does not exactly match "
                    f"VAV terminal {terminal_name!r} outlet"
                )
            inlet_owner = zone_inlet_owners.get(terminal_outlet.casefold())
            if inlet_owner is None or inlet_owner != (connection_name, terminal_outlet):
                raise VAVDiscoveryError(
                    f"VAV terminal {terminal_name!r} outlet does not belong to owning zone "
                    f"equipment connection {connection_name!r}"
                )
            previous_inlet_terminal = seen_terminal_inlets.get(terminal_inlet.casefold())
            if previous_inlet_terminal is not None:
                raise VAVDiscoveryError(
                    f"VAV terminal inlet {terminal_inlet!r} is shared by terminals "
                    f"{previous_inlet_terminal!r} and {terminal_name!r}"
                )
            previous_outlet_terminal = seen_terminal_outlets.get(terminal_outlet.casefold())
            if previous_outlet_terminal is not None:
                raise VAVDiscoveryError(
                    f"VAV terminal outlet {terminal_outlet!r} is shared by terminals "
                    f"{previous_outlet_terminal!r} and {terminal_name!r}"
                )
            seen_terminal_inlets[terminal_inlet.casefold()] = terminal_name
            seen_terminal_outlets[terminal_outlet.casefold()] = terminal_name
            terminal_key = terminal_name.casefold()
            occurrences[terminal_key] += 1
            if terminal_key in by_terminal:
                raise VAVDiscoveryError(f"VAV terminal {terminal_name!r} resolves from multiple zone walks")
            if zone_name.casefold() in seen_zones:
                raise VAVDiscoveryError(f"VAV zone {zone_name!r} resolves to multiple VAV terminals")
            seen_zones.add(zone_name.casefold())
            by_terminal[terminal_key] = (
                _reference("Zone", zone_name, zone_fields),
                _reference(VAV_TERMINAL_OBJECT_TYPE, terminal_name, terminal),
                terminal_inlet,
                terminal_outlet,
            )
    expected = set(terminal_table)
    if set(occurrences) != expected or any(occurrences[key] != 1 for key in expected):
        missing = sorted(expected - set(occurrences))
        repeated = sorted(key for key, count in occurrences.items() if count != 1)
        raise VAVDiscoveryError(
            "VAV terminal inventory is not an exact one-to-one zone equipment walk: "
            f"missing={missing!r}, repeated={repeated!r}"
        )
    return by_terminal, len(terminal_table)


def _terminal_loop_groups(
    index: _ModelIndex,
    terminals: Mapping[str, tuple[VAVObjectReference, VAVObjectReference, str, str]],
) -> dict[str, list[tuple[VAVObjectReference, VAVObjectReference, str, str]]]:
    outlet_to_demand: dict[str, tuple[str, str]] = {}
    for splitter_name, splitter in _table(index, "AirLoopHVAC:ZoneSplitter"):
        demand = _field(splitter, "inlet_node_name", owner=splitter_name)
        for item in _records(splitter, "nodes", owner=splitter_name):
            outlet = _field(item, "outlet_node_name", owner=splitter_name)
            if outlet.casefold() in outlet_to_demand:
                raise VAVDiscoveryError(f"zone-splitter outlet {outlet!r} is repeated")
            outlet_to_demand[outlet.casefold()] = (outlet, demand)

    demand_to_loop: dict[str, tuple[str, str]] = {}
    for loop_name, loop in _table(index, VAV_LOOP_OBJECT_TYPE):
        demand = _field(loop, "demand_side_inlet_node_names", owner=loop_name)
        if demand.casefold() in demand_to_loop:
            raise VAVDiscoveryError(f"air-loop demand inlet {demand!r} is repeated")
        demand_to_loop[demand.casefold()] = (demand, loop_name)

    groups: dict[str, list[tuple[VAVObjectReference, VAVObjectReference, str, str]]] = {}
    consumed_splitter_outlets: Counter[str] = Counter()
    for terminal_data in terminals.values():
        terminal = terminal_data[1]
        inlet = terminal_data[2]
        outlet_match = outlet_to_demand.get(inlet.casefold())
        if outlet_match is None:
            raise VAVDiscoveryError(f"VAV terminal {terminal.object_name!r} has no exact splitter outlet")
        canonical_outlet, resolved_demand = outlet_match
        if canonical_outlet != inlet:
            raise VAVDiscoveryError(f"VAV terminal {terminal.object_name!r} uses noncanonical splitter outlet spelling")
        consumed_splitter_outlets[canonical_outlet.casefold()] += 1
        if consumed_splitter_outlets[canonical_outlet.casefold()] != 1:
            raise VAVDiscoveryError(f"zone-splitter outlet {canonical_outlet!r} is consumed by multiple VAV terminals")
        loop_match = demand_to_loop.get(resolved_demand.casefold())
        if loop_match is None:
            raise VAVDiscoveryError(f"VAV terminal {terminal.object_name!r} has no exact air-loop owner")
        canonical_demand, resolved_loop = loop_match
        if canonical_demand != resolved_demand:
            raise VAVDiscoveryError(f"VAV terminal {terminal.object_name!r} uses noncanonical demand inlet spelling")
        groups.setdefault(resolved_loop.casefold(), []).append(terminal_data)
    return groups


def _thermostat_by_zone(index: _ModelIndex) -> dict[str, tuple[str, Mapping[str, Any]]]:
    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for thermostat_name, thermostat in _table(index, "ZoneControl:Thermostat"):
        zone_name = _field(thermostat, "zone_or_zonelist_name", owner=thermostat_name)
        zone, _ = _lookup(index, "Zone", zone_name, label=f"{thermostat_name} zone")
        if zone.casefold() in result:
            raise VAVDiscoveryError(f"zone {zone!r} has multiple ZoneControl:Thermostat objects")
        result[zone.casefold()] = (thermostat_name, thermostat)
    return result


def _terminal_record(
    index: _ModelIndex,
    raw: tuple[VAVObjectReference, VAVObjectReference, str, str],
    thermostats: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> VAVTerminalDiscoveryRecord:
    zone, terminal, inlet, outlet = raw
    thermostat_entry = thermostats.get(zone.object_name.casefold())
    if thermostat_entry is None:
        raise VAVDiscoveryError(f"VAV zone {zone.object_name!r} has no exact thermostat")
    thermostat_name, thermostat_fields = thermostat_entry
    control_type = _field(thermostat_fields, "control_1_object_type", owner=thermostat_name)
    _canonical_type_reference(
        control_type,
        "ThermostatSetpoint:DualSetpoint",
        field_name=f"{thermostat_name}.control_1_object_type",
    )
    dual_name, dual = _lookup(
        index,
        "ThermostatSetpoint:DualSetpoint",
        _field(thermostat_fields, "control_1_name", owner=thermostat_name),
        label=f"{thermostat_name} dual setpoint",
    )
    heating = _schedule_reference(
        index,
        _field(dual, "heating_setpoint_temperature_schedule_name", owner=dual_name),
        label=f"{dual_name} heating schedule",
    )
    cooling = _schedule_reference(
        index,
        _field(dual, "cooling_setpoint_temperature_schedule_name", owner=dual_name),
        label=f"{dual_name} cooling schedule",
    )
    return VAVTerminalDiscoveryRecord(
        zone=zone,
        terminal=terminal,
        terminal_inlet_node_name=inlet,
        terminal_outlet_node_name=outlet,
        thermostat=_reference("ZoneControl:Thermostat", thermostat_name, thermostat_fields),
        dual_setpoint=_reference("ThermostatSetpoint:DualSetpoint", dual_name, dual),
        heating_schedule=heating,
        cooling_schedule=cooling,
    )


def _branch_components(
    index: _ModelIndex,
    loop_name: str,
    loop: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    branch_list_name, branch_list = _lookup(
        index,
        "BranchList",
        _field(loop, "branch_list_name", owner=loop_name),
        label=f"{loop_name} branch list",
    )
    branch_occurrences: Counter[str] = Counter()
    components: list[Mapping[str, Any]] = []
    for item in _records(branch_list, "branches", owner=branch_list_name):
        branch_name, branch = _lookup(
            index,
            "Branch",
            item.get("branch_name"),
            label=f"{branch_list_name} branch",
        )
        branch_occurrences[branch_name.casefold()] += 1
        components.extend(_records(branch, "components", owner=branch_name))
    repeated = sorted(name for name, count in branch_occurrences.items() if count != 1)
    if repeated:
        raise VAVDiscoveryError(f"VAV loop {loop_name!r} repeats branch references: {repeated!r}")
    return tuple(components)


def _availability_records(
    index: _ModelIndex,
    loop_name: str,
    loop: Mapping[str, Any],
) -> tuple[VAVObjectReference | None, tuple[VAVObjectReference, ...]]:
    raw_list_name = loop.get("availability_manager_list_name")
    if raw_list_name in (None, ""):
        raise VAVDiscoveryError(f"VAV loop {loop_name!r} has no availability manager list")
    list_name, assignment = _lookup(
        index,
        "AvailabilityManagerAssignmentList",
        raw_list_name,
        label=f"{loop_name} availability manager list",
    )
    references = 0
    for other_loop_name, other_loop in _table(index, VAV_LOOP_OBJECT_TYPE):
        raw = other_loop.get("availability_manager_list_name")
        if raw in (None, ""):
            continue
        other_reference = _text(raw, field_name=f"{other_loop_name}.availability_manager_list_name")
        references += int(other_reference.casefold() == list_name.casefold())
    if references != 1:
        raise VAVDiscoveryError(f"availability list {list_name!r} is shared or ambiguously referenced")

    manager_entries = _records(assignment, "managers", owner=list_name)
    if not manager_entries:
        raise VAVDiscoveryError(f"availability list {list_name!r} is empty")
    managers: list[VAVObjectReference] = []
    for manager in manager_entries:
        manager_type = _field(manager, "availability_manager_object_type", owner=list_name)
        if not manager_type.startswith("AvailabilityManager:"):
            raise VAVDiscoveryError(
                f"{list_name}.availability_manager_object_type must use the canonical AvailabilityManager: prefix"
            )
        manager_name, fields = _lookup(
            index,
            manager_type,
            _field(manager, "availability_manager_name", owner=list_name),
            label=f"{list_name} availability manager",
        )
        reference_count = 0
        for other_list_name, other_list in _table(index, "AvailabilityManagerAssignmentList"):
            for other in _records(other_list, "managers", owner=other_list_name):
                raw_type = other.get("availability_manager_object_type")
                raw_name = other.get("availability_manager_name")
                if isinstance(raw_type, str) and isinstance(raw_name, str):
                    reference_count += int(
                        str.__str__(raw_type).casefold() == manager_type.casefold()
                        and str.__str__(raw_name).casefold() == manager_name.casefold()
                    )
        if reference_count != 1:
            raise VAVDiscoveryError(
                f"availability manager {manager_type}/{manager_name} is shared or ambiguously referenced"
            )
        managers.append(_reference(manager_type, manager_name, fields))
    manager_keys = [(item.object_type.casefold(), item.object_name.casefold()) for item in managers]
    if len(manager_keys) != len(set(manager_keys)):
        raise VAVDiscoveryError(f"availability list {list_name!r} repeats a manager reference")
    return (
        _reference("AvailabilityManagerAssignmentList", list_name, assignment),
        tuple(sorted(managers, key=lambda item: (item.object_type.casefold(), item.object_name.casefold()))),
    )


def _supported_loop_record(
    index: _ModelIndex,
    loop_name: str,
    loop: Mapping[str, Any],
    raw_terminals: Sequence[tuple[VAVObjectReference, VAVObjectReference, str, str]],
    thermostats: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> VAVLoopDiscoveryRecord:
    demand = _field(loop, "demand_side_inlet_node_names", owner=loop_name)
    supply_outlet = _field(loop, "supply_side_outlet_node_names", owner=loop_name)
    components = _branch_components(index, loop_name, loop)

    fans: list[VAVObjectReference] = []
    outdoor_system_references: list[str] = []
    canonical_fans = {item.casefold(): item for item in VAV_FAN_OBJECT_TYPES}
    for component in components:
        raw_type = component.get("component_object_type")
        if not isinstance(raw_type, str):
            continue
        component_type = str.__str__(raw_type)
        canonical_fan = canonical_fans.get(component_type.casefold())
        if canonical_fan is not None:
            _canonical_type_reference(
                component_type,
                canonical_fan,
                field_name=f"{loop_name} branch component type",
            )
            fan_name, fan = _lookup(
                index,
                canonical_fan,
                component.get("component_name"),
                label=f"{loop_name} fan",
            )
            fans.append(_reference(canonical_fan, fan_name, fan))
        elif component_type.casefold() == "airloophvac:outdoorairsystem":
            _canonical_type_reference(
                component_type,
                "AirLoopHVAC:OutdoorAirSystem",
                field_name=f"{loop_name} branch component type",
            )
            outdoor_system_references.append(_field(component, "component_name", owner=f"{loop_name} branch component"))
    fan_keys = [(item.object_type.casefold(), item.object_name.casefold()) for item in fans]
    if not fans or len(fan_keys) != len(set(fan_keys)):
        raise VAVDiscoveryError(f"VAV loop {loop_name!r} requires unique referenced supply fans")
    if len(outdoor_system_references) != 1:
        raise VAVDiscoveryError(f"VAV loop {loop_name!r} requires exactly one outdoor-air system reference")
    outdoor_system_name, outdoor_system = _lookup(
        index,
        "AirLoopHVAC:OutdoorAirSystem",
        outdoor_system_references[0],
        label=f"{loop_name} outdoor-air system",
    )
    controller_list_name, controller_list = _lookup(
        index,
        "AirLoopHVAC:ControllerList",
        _field(outdoor_system, "controller_list_name", owner=outdoor_system_name),
        label=f"{outdoor_system_name} controller list",
    )
    controller_names: list[str] = []
    for field_name, raw_type in controller_list.items():
        if not field_name.endswith("_object_type") or not isinstance(raw_type, str):
            continue
        if str.__str__(raw_type).casefold() != VAV_OA_CONTROLLER_OBJECT_TYPE.casefold():
            continue
        _canonical_type_reference(
            raw_type,
            VAV_OA_CONTROLLER_OBJECT_TYPE,
            field_name=f"{controller_list_name}.{field_name}",
        )
        controller_names.append(
            _field(
                controller_list,
                field_name[: -len("_object_type")] + "_name",
                owner=controller_list_name,
            )
        )
    if len(controller_names) != 1:
        raise VAVDiscoveryError(f"VAV loop {loop_name!r} requires exactly one Controller:OutdoorAir reference")
    controller_name, controller = _lookup(
        index,
        VAV_OA_CONTROLLER_OBJECT_TYPE,
        controller_names[0],
        label=f"{controller_list_name} outdoor-air controller",
    )
    oa_schedules = tuple(
        VAVScheduleReference(
            field_name,
            _schedule_reference(
                index,
                controller[field_name],
                label=f"{controller_name}.{field_name}",
            ),
        )
        for field_name in VAV_OA_SCHEDULE_FIELDS
        if field_name in controller
    )
    terminal_records = tuple(
        sorted(
            (_terminal_record(index, item, thermostats) for item in raw_terminals),
            key=lambda item: item.zone.object_name.casefold(),
        )
    )
    availability_list, availability_managers = _availability_records(index, loop_name, loop)
    return VAVLoopDiscoveryRecord(
        object_name=loop_name,
        source_object_sha256=_object_sha(VAV_LOOP_OBJECT_TYPE, loop_name, loop),
        disposition=CandidateDisposition.SUPPORTED,
        reason_code=None,
        demand_inlet_node_name=demand,
        supply_outlet_node_name=supply_outlet,
        outdoor_air_controller=_reference(VAV_OA_CONTROLLER_OBJECT_TYPE, controller_name, controller),
        outdoor_air_schedule_references=oa_schedules,
        fans=tuple(sorted(fans, key=lambda item: (item.object_type.casefold(), item.object_name.casefold()))),
        availability_assignment_list=availability_list,
        availability_managers=availability_managers,
        terminals=terminal_records,
    )


def _unsupported_loop(loop_name: str, loop: Mapping[str, Any], reason_code: str) -> VAVLoopDiscoveryRecord:
    return VAVLoopDiscoveryRecord(
        object_name=loop_name,
        source_object_sha256=_object_sha(VAV_LOOP_OBJECT_TYPE, loop_name, loop),
        disposition=CandidateDisposition.UNSUPPORTED,
        reason_code=reason_code,
        demand_inlet_node_name=None,
        supply_outlet_node_name=None,
        outdoor_air_controller=None,
        outdoor_air_schedule_references=(),
        fans=(),
        availability_assignment_list=None,
        availability_managers=(),
        terminals=(),
    )


def _discover_from_model(model: dict[str, Any], *, source_model_sha256: str) -> VAVDiscoveryInventory:
    try:
        index = _ModelIndex(model)
    except ControlInstrumentationError as error:
        raise VAVDiscoveryError(f"epJSON indexing failed: {error}") from error
    terminals, _ = _zone_terminal_inventory(index)
    groups = _terminal_loop_groups(index, terminals)
    thermostats = _thermostat_by_zone(index)
    records: list[VAVLoopDiscoveryRecord] = []
    for loop_name, loop in _table(index, VAV_LOOP_OBJECT_TYPE):
        raw_terminals = groups.get(loop_name.casefold(), ())
        if not raw_terminals:
            records.append(_unsupported_loop(loop_name, loop, "vav_loop_has_no_reheat_terminals"))
            continue
        try:
            records.append(
                _supported_loop_record(
                    index,
                    loop_name,
                    loop,
                    raw_terminals,
                    thermostats,
                )
            )
        except VAVDiscoveryError as error:
            raise VAVDiscoveryError(f"VAV loop {loop_name!r} is structurally unsupported: {error}") from error

    supported = [item for item in records if item.disposition is CandidateDisposition.SUPPORTED]
    seen_fans: set[tuple[str, str]] = set()
    seen_oa: set[str] = set()
    seen_outlets: set[str] = set()
    seen_zones: set[str] = set()
    for record in supported:
        assert record.outdoor_air_controller is not None and record.supply_outlet_node_name is not None
        oa_key = record.outdoor_air_controller.object_name.casefold()
        outlet_key = record.supply_outlet_node_name.casefold()
        if oa_key in seen_oa:
            raise VAVDiscoveryError("a Controller:OutdoorAir is owned by multiple VAV loops")
        if outlet_key in seen_outlets:
            raise VAVDiscoveryError("a supply outlet is owned by multiple VAV loops")
        seen_oa.add(oa_key)
        seen_outlets.add(outlet_key)
        for fan in record.fans:
            fan_key = (fan.object_type.casefold(), fan.object_name.casefold())
            if fan_key in seen_fans:
                raise VAVDiscoveryError("a supply fan is owned by multiple VAV loops")
            seen_fans.add(fan_key)
        for terminal in record.terminals:
            zone_key = terminal.zone.object_name.casefold()
            if zone_key in seen_zones:
                raise VAVDiscoveryError("a served zone is owned by multiple VAV loops")
            seen_zones.add(zone_key)
    return VAVDiscoveryInventory(source_model_sha256, tuple(records))


def discover_vav_systems_from_json_bytes(source_bytes: bytes) -> VAVDiscoveryInventory:
    """Return an exhaustive authenticated inventory for exact epJSON bytes."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    try:
        model = _parse_epjson_bytes(source_bytes)
    except (ControlInstrumentationError, TypeError, ValueError) as error:
        raise VAVDiscoveryError(f"cannot parse VAV source epJSON: {error}") from error
    return _discover_from_model(model, source_model_sha256=hashlib.sha256(source_bytes).hexdigest())


def _canonical_inventory(value: Any) -> VAVDiscoveryInventory:
    if type(value) is not VAVDiscoveryInventory:
        raise TypeError("expected_inventory must be an exact VAVDiscoveryInventory")
    try:
        return VAVDiscoveryInventory(
            source_model_sha256=value.source_model_sha256,
            records=value.records,
            schema_version=value.schema_version,
            algorithm_id=value.algorithm_id,
            upstream_commit=value.upstream_commit,
            upstream_source_path=value.upstream_source_path,
            upstream_source_sha256=value.upstream_source_sha256,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise VAVDiscoveryError(f"expected VAV inventory is malformed: {error}") from error


def verify_vav_discovery_inventory_from_json_bytes(
    source_bytes: bytes,
    expected_inventory: VAVDiscoveryInventory,
) -> VAVDiscoveryInventory:
    """Recompute discovery and reject exact-byte or inventory substitution."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    expected = _canonical_inventory(expected_inventory)
    recomputed = discover_vav_systems_from_json_bytes(source_bytes)
    expected_payload = _canonical_json(expected.to_dict())
    recomputed_payload = _canonical_json(recomputed.to_dict())
    if not hmac.compare_digest(expected_payload, recomputed_payload) or not hmac.compare_digest(
        expected.inventory_sha256,
        recomputed.inventory_sha256,
    ):
        raise VAVDiscoveryError("VAV discovery inventory does not match exact-byte recomputation")
    return recomputed


__all__ = [
    "BUILDING2BUILDING_ACTUATORS_SOURCE_PATH",
    "BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256",
    "VAV_DISCOVERY_ALGORITHM_ID",
    "VAV_DISCOVERY_SCHEMA_VERSION",
    "VAV_FAN_OBJECT_TYPES",
    "VAV_LOOP_OBJECT_TYPE",
    "VAV_OA_CONTROLLER_OBJECT_TYPE",
    "VAV_OA_SCHEDULE_FIELDS",
    "VAV_PROFILE_ID",
    "VAV_TERMINAL_OBJECT_TYPE",
    "VAVDiscoveryError",
    "VAVDiscoveryInventory",
    "VAVLoopDiscoveryRecord",
    "VAVObjectReference",
    "VAVScheduleReference",
    "VAVTerminalDiscoveryRecord",
    "discover_vav_systems_from_json_bytes",
    "verify_vav_discovery_inventory_from_json_bytes",
]
