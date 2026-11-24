"""Exact-byte discovery for the pinned Building2Building heating-only slice.

The topology walk follows ``make_heating_only_controllable`` in the
authenticated Building2Building v1.0.0 source.  HOT strengthens that walk by
requiring complete candidate coverage, exact case-insensitive ownership, a
single canonical thermostat per served zone, and deterministic replay against
the original epJSON bytes.  Discovery is pure and never reads the filesystem.
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
    control_profile_by_id,
)
from building_gen.control.discovery import energyplus_source_object_sha256
from building_gen.control.instrumentation import (
    ControlInstrumentationError,
    _ModelIndex,
    _owned_bytes,
    _parse_epjson_bytes,
)
from building_gen.control.vav_discovery import (
    BUILDING2BUILDING_ACTUATORS_SOURCE_PATH,
    BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256,
)

HEATING_PROFILE_ID = "b2b_heating_only_v1"
HEATING_DISCOVERY_SCHEMA_VERSION = "hot-b2b-heating-only-discovery/1.1.0"
HEATING_DISCOVERY_ALGORITHM_ID = "b2b-v1-heating-only-exact-topology-v2"
HEATING_EQUIPMENT_OBJECT_TYPES = (
    "ZoneHVAC:Baseboard:Convective:Electric",
    "ZoneHVAC:Baseboard:Convective:Water",
    "ZoneHVAC:Baseboard:RadiantConvective:Electric",
    "ZoneHVAC:Baseboard:RadiantConvective:Water",
    "ZoneHVAC:HighTemperatureRadiant",
    "ZoneHVAC:UnitHeater",
)
HEATING_THERMOSTAT_OBJECT_TYPES = (
    "ThermostatSetpoint:DualSetpoint",
    "ThermostatSetpoint:SingleHeating",
)
if (  # pragma: no cover - module invariant
    control_profile_by_id(HEATING_PROFILE_ID).equipment_object_types != HEATING_EQUIPMENT_OBJECT_TYPES
):
    raise RuntimeError("heating discovery object types drifted from the authenticated profile registry")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HeatingDiscoveryError(ValueError):
    """Raised when a heating-only candidate cannot be owned exactly."""


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise HeatingDiscoveryError(f"{field_name} must be a nonempty exact string")
    value = str.__str__(value)
    if not value or value != value.strip():
        raise HeatingDiscoveryError(f"{field_name} must be a nonempty exact string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise HeatingDiscoveryError(f"{field_name} contains a control character")
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    value = _text(value, field_name=field_name)
    if _SHA256_RE.fullmatch(value) is None:
        raise HeatingDiscoveryError(f"{field_name} must be a lowercase SHA-256 digest")
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
        raise HeatingDiscoveryError(f"value is not canonical JSON data: {error}") from error


def _records(fields: Mapping[str, Any], key: str, *, owner: str) -> tuple[Mapping[str, Any], ...]:
    raw = fields.get(key, ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise HeatingDiscoveryError(f"{owner}.{key} must be an array")
    result: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise HeatingDiscoveryError(f"{owner}.{key}[{index}] must be an object")
        result.append(value)
    return tuple(result)


def _field(fields: Mapping[str, Any], key: str, *, owner: str) -> str:
    return _text(fields.get(key), field_name=f"{owner}.{key}")


def _table(index: _ModelIndex, object_type: str) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    try:
        table = index.table(object_type) or {}
        indexed = index._object_index(object_type, table)
    except ControlInstrumentationError as error:
        raise HeatingDiscoveryError(str(error)) from error
    return tuple(sorted(indexed.values(), key=lambda item: item[0].casefold()))


def _lookup(
    index: _ModelIndex,
    object_type: str,
    object_name: Any,
    *,
    label: str,
) -> tuple[str, Mapping[str, Any]]:
    reference = _text(object_name, field_name=label)
    try:
        canonical_name, fields = index.lookup(object_type, reference)
    except ControlInstrumentationError as error:
        raise HeatingDiscoveryError(str(error)) from error
    if canonical_name != reference:
        raise HeatingDiscoveryError(f"{label} must use canonical object name {canonical_name!r}")
    return canonical_name, fields


def _canonical_type_reference(value: Any, expected: str, *, field_name: str) -> str:
    actual = _text(value, field_name=field_name)
    if actual.casefold() != expected.casefold():
        raise HeatingDiscoveryError(f"{field_name} must reference {expected!r}")
    if actual != expected:
        raise HeatingDiscoveryError(f"{field_name} must use canonical spelling {expected!r}")
    return actual


def _object_sha(object_type: str, object_name: str, fields: Mapping[str, Any]) -> str:
    try:
        return energyplus_source_object_sha256(object_type, object_name, fields)
    except (TypeError, ValueError) as error:
        raise HeatingDiscoveryError(f"cannot hash {object_type}/{object_name}: {error}") from error


@dataclass(frozen=True, slots=True)
class HeatingObjectReference:
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

    @property
    def energyplus_object(self) -> tuple[str, str]:
        return self.object_type, self.object_name

    def to_dict(self) -> dict[str, str]:
        return {
            "object_type": self.object_type,
            "object_name": self.object_name,
            "source_object_sha256": self.source_object_sha256,
        }


def _canonical_reference(value: Any, *, field_name: str) -> HeatingObjectReference:
    if type(value) is not HeatingObjectReference:
        raise TypeError(f"{field_name} must be an exact HeatingObjectReference")
    try:
        return HeatingObjectReference(value.object_type, value.object_name, value.source_object_sha256)
    except (AttributeError, TypeError, ValueError) as error:
        raise HeatingDiscoveryError(f"{field_name} is malformed: {error}") from error


@dataclass(frozen=True, slots=True)
class HeatingZoneDiscoveryRecord:
    """One deduplicated served zone and all heating-only equipment it owns."""

    zone: HeatingObjectReference
    equipment_connection: HeatingObjectReference
    equipment_list: HeatingObjectReference
    equipment: tuple[HeatingObjectReference, ...]
    thermostat: HeatingObjectReference
    thermostat_setpoint: HeatingObjectReference
    heating_schedule: HeatingObjectReference
    cooling_schedule: HeatingObjectReference | None

    def __post_init__(self) -> None:
        zone = _canonical_reference(self.zone, field_name="zone")
        connection = _canonical_reference(self.equipment_connection, field_name="equipment_connection")
        equipment_list = _canonical_reference(self.equipment_list, field_name="equipment_list")
        thermostat = _canonical_reference(self.thermostat, field_name="thermostat")
        setpoint = _canonical_reference(self.thermostat_setpoint, field_name="thermostat_setpoint")
        schedule = _canonical_reference(self.heating_schedule, field_name="heating_schedule")
        cooling_schedule = (
            None
            if self.cooling_schedule is None
            else _canonical_reference(self.cooling_schedule, field_name="cooling_schedule")
        )
        if zone.object_type != "Zone":
            raise HeatingDiscoveryError("heating zone reference must be a Zone")
        if connection.object_type != "ZoneHVAC:EquipmentConnections":
            raise HeatingDiscoveryError("heating connection reference has an unexpected object type")
        if equipment_list.object_type != "ZoneHVAC:EquipmentList":
            raise HeatingDiscoveryError("heating equipment-list reference has an unexpected object type")
        if thermostat.object_type != "ZoneControl:Thermostat":
            raise HeatingDiscoveryError("heating thermostat reference has an unexpected object type")
        if setpoint.object_type not in HEATING_THERMOSTAT_OBJECT_TYPES:
            raise HeatingDiscoveryError("heating thermostat setpoint has an unsupported object type")
        if not schedule.object_type.startswith("Schedule:"):
            raise HeatingDiscoveryError("heating schedule must resolve to exactly one Schedule:* object")
        if setpoint.object_type == "ThermostatSetpoint:DualSetpoint":
            if cooling_schedule is None:
                raise HeatingDiscoveryError("a dual-setpoint heating zone requires its exact cooling schedule")
            if not cooling_schedule.object_type.startswith("Schedule:"):
                raise HeatingDiscoveryError("cooling schedule must resolve to exactly one Schedule:* object")
        elif cooling_schedule is not None:
            raise HeatingDiscoveryError("a single-heating setpoint cannot declare a cooling schedule")
        if type(self.equipment) is not tuple or not self.equipment:
            raise HeatingDiscoveryError("a heating zone requires a nonempty exact equipment tuple")
        equipment = tuple(
            _canonical_reference(item, field_name=f"equipment[{index}]") for index, item in enumerate(self.equipment)
        )
        if any(item.object_type not in HEATING_EQUIPMENT_OBJECT_TYPES for item in equipment):
            raise HeatingDiscoveryError("heating equipment lies outside the pinned object-type family")
        ordered = tuple(sorted(equipment, key=lambda item: (item.object_type.casefold(), item.object_name.casefold())))
        if equipment != ordered:
            raise HeatingDiscoveryError("heating equipment must use canonical object order")
        identities = [(item.object_type.casefold(), item.object_name.casefold()) for item in equipment]
        if len(identities) != len(set(identities)):
            raise HeatingDiscoveryError("heating equipment identities collide")
        for field_name, value in (
            ("zone", zone),
            ("equipment_connection", connection),
            ("equipment_list", equipment_list),
            ("thermostat", thermostat),
            ("thermostat_setpoint", setpoint),
            ("heating_schedule", schedule),
        ):
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "cooling_schedule", cooling_schedule)
        object.__setattr__(self, "equipment", equipment)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone.to_dict(),
            "equipment_connection": self.equipment_connection.to_dict(),
            "equipment_list": self.equipment_list.to_dict(),
            "equipment": [item.to_dict() for item in self.equipment],
            "thermostat": self.thermostat.to_dict(),
            "thermostat_setpoint": self.thermostat_setpoint.to_dict(),
            "heating_schedule": self.heating_schedule.to_dict(),
            "cooling_schedule": None if self.cooling_schedule is None else self.cooling_schedule.to_dict(),
        }


def _canonical_zone_record(value: Any, *, field_name: str) -> HeatingZoneDiscoveryRecord:
    if type(value) is not HeatingZoneDiscoveryRecord:
        raise TypeError(f"{field_name} must be an exact HeatingZoneDiscoveryRecord")
    try:
        return HeatingZoneDiscoveryRecord(
            zone=value.zone,
            equipment_connection=value.equipment_connection,
            equipment_list=value.equipment_list,
            equipment=value.equipment,
            thermostat=value.thermostat,
            thermostat_setpoint=value.thermostat_setpoint,
            heating_schedule=value.heating_schedule,
            cooling_schedule=value.cooling_schedule,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise HeatingDiscoveryError(f"{field_name} is malformed: {error}") from error


def _inventory_payload(
    source_model_sha256: str,
    records: Sequence[HeatingZoneDiscoveryRecord],
    *,
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
class HeatingDiscoveryInventory:
    source_model_sha256: str
    records: tuple[HeatingZoneDiscoveryRecord, ...]
    schema_version: str = HEATING_DISCOVERY_SCHEMA_VERSION
    algorithm_id: str = HEATING_DISCOVERY_ALGORITHM_ID
    upstream_commit: str = BUILDING2BUILDING_COMMIT
    upstream_source_path: str = BUILDING2BUILDING_ACTUATORS_SOURCE_PATH
    upstream_source_sha256: str = BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_model_sha256",
            _sha256(self.source_model_sha256, field_name="source_model_sha256"),
        )
        for field_name in (
            "schema_version",
            "algorithm_id",
            "upstream_commit",
            "upstream_source_path",
            "upstream_source_sha256",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name=field_name))
        if self.schema_version != HEATING_DISCOVERY_SCHEMA_VERSION:
            raise HeatingDiscoveryError("unexpected heating discovery schema version")
        if self.algorithm_id != HEATING_DISCOVERY_ALGORITHM_ID:
            raise HeatingDiscoveryError("unexpected heating discovery algorithm")
        if self.upstream_commit != BUILDING2BUILDING_COMMIT:
            raise HeatingDiscoveryError("heating discovery must bind the pinned upstream commit")
        if self.upstream_source_path != BUILDING2BUILDING_ACTUATORS_SOURCE_PATH:
            raise HeatingDiscoveryError("heating discovery must bind the pinned upstream source path")
        if self.upstream_source_sha256 != BUILDING2BUILDING_ACTUATORS_SOURCE_SHA256:
            raise HeatingDiscoveryError("heating discovery must bind the pinned upstream source bytes")
        if type(self.records) is not tuple:
            raise TypeError("records must be an exact tuple")
        records = tuple(
            _canonical_zone_record(item, field_name=f"records[{index}]") for index, item in enumerate(self.records)
        )
        ordered = tuple(sorted(records, key=lambda item: item.zone.object_name.casefold()))
        if records != ordered:
            raise HeatingDiscoveryError("heating zone records must use canonical zone order")
        zones = [item.zone.object_name.casefold() for item in records]
        if len(zones) != len(set(zones)):
            raise HeatingDiscoveryError("heating zone records repeat a served zone")
        equipment = [
            (reference.object_type.casefold(), reference.object_name.casefold())
            for record in records
            for reference in record.equipment
        ]
        if len(equipment) != len(set(equipment)):
            raise HeatingDiscoveryError("heating equipment is owned by multiple served zones")
        setpoints = [item.thermostat_setpoint.energyplus_object for item in records]
        if len({(item[0].casefold(), item[1].casefold()) for item in setpoints}) != len(setpoints):
            raise HeatingDiscoveryError("a thermostat setpoint object is shared by multiple heating zones")
        object.__setattr__(self, "records", records)

    @property
    def equipment_count(self) -> int:
        return sum(len(item.equipment) for item in self.records)

    @property
    def inventory_sha256(self) -> str:
        return hashlib.sha256(
            b"HOT B2B heating-only discovery inventory v1\0"
            + _canonical_json(
                _inventory_payload(
                    self.source_model_sha256,
                    self.records,
                    schema_version=self.schema_version,
                    algorithm_id=self.algorithm_id,
                    upstream_commit=self.upstream_commit,
                    upstream_source_path=self.upstream_source_path,
                    upstream_source_sha256=self.upstream_source_sha256,
                )
            )
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = _inventory_payload(
            self.source_model_sha256,
            self.records,
            schema_version=self.schema_version,
            algorithm_id=self.algorithm_id,
            upstream_commit=self.upstream_commit,
            upstream_source_path=self.upstream_source_path,
            upstream_source_sha256=self.upstream_source_sha256,
        )
        return {**payload, "equipment_count": self.equipment_count, "inventory_sha256": self.inventory_sha256}

    def verify_source_bytes(self, source_bytes: bytes) -> HeatingDiscoveryInventory:
        return verify_heating_discovery_inventory_from_json_bytes(source_bytes, self)


def _reference(object_type: str, object_name: str, fields: Mapping[str, Any]) -> HeatingObjectReference:
    return HeatingObjectReference(object_type, object_name, _object_sha(object_type, object_name, fields))


def _schedule_reference(index: _ModelIndex, schedule_name: Any, *, label: str) -> HeatingObjectReference:
    reference = _text(schedule_name, field_name=label)
    matches = [
        (object_type, object_name, fields)
        for object_type, object_name, fields in index.iter_objects()
        if object_type.startswith("Schedule:") and object_name.casefold() == reference.casefold()
    ]
    if len(matches) != 1:
        raise HeatingDiscoveryError(f"{label} must resolve to exactly one Schedule:* object: {reference!r}")
    object_type, object_name, fields = matches[0]
    if object_name != reference:
        raise HeatingDiscoveryError(f"{label} must use canonical schedule name {object_name!r}")
    return _reference(object_type, object_name, fields)


def _thermostat_by_zone(
    index: _ModelIndex,
    served_zone_names: Sequence[str],
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    served_zones: dict[str, str] = {}
    for index_value, raw_zone_name in enumerate(served_zone_names):
        zone_name, _ = _lookup(
            index,
            "Zone",
            raw_zone_name,
            label=f"served heating zone[{index_value}]",
        )
        folded = zone_name.casefold()
        if folded in served_zones:
            raise HeatingDiscoveryError(f"served heating zone {zone_name!r} is repeated")
        served_zones[folded] = zone_name

    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for thermostat_name, thermostat in _table(index, "ZoneControl:Thermostat"):
        zone_reference = _field(thermostat, "zone_or_zonelist_name", owner=thermostat_name)
        served_zone_name = served_zones.get(zone_reference.casefold())
        if served_zone_name is None:
            # ZoneControl:Thermostat may validly target an unrelated Zone or
            # ZoneList.  The pinned B2B heating walk searches only for the
            # exact served zone name, so unrelated thermostat topology lies
            # outside this profile's ownership and validation surface.
            continue
        zone_name, _ = _lookup(index, "Zone", zone_reference, label=f"{thermostat_name} zone")
        if zone_name != served_zone_name:
            raise HeatingDiscoveryError(f"{thermostat_name} must target canonical served zone {served_zone_name!r}")
        if zone_name.casefold() in result:
            raise HeatingDiscoveryError(f"zone {zone_name!r} has multiple ZoneControl:Thermostat objects")
        result[zone_name.casefold()] = (thermostat_name, thermostat)
    return result


def _thermostat_record(
    index: _ModelIndex,
    zone_name: str,
    thermostats: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> tuple[
    HeatingObjectReference,
    HeatingObjectReference,
    HeatingObjectReference,
    HeatingObjectReference | None,
]:
    thermostat_entry = thermostats.get(zone_name.casefold())
    if thermostat_entry is None:
        raise HeatingDiscoveryError(f"heating-only zone {zone_name!r} has no exact thermostat")
    thermostat_name, thermostat = thermostat_entry
    control_type = _field(thermostat, "control_1_object_type", owner=thermostat_name)
    canonical_types = {item.casefold(): item for item in HEATING_THERMOSTAT_OBJECT_TYPES}
    canonical_type = canonical_types.get(control_type.casefold())
    if canonical_type is None:
        raise HeatingDiscoveryError(
            f"heating-only zone {zone_name!r} uses unsupported thermostat type {control_type!r}"
        )
    _canonical_type_reference(
        control_type,
        canonical_type,
        field_name=f"{thermostat_name}.control_1_object_type",
    )
    setpoint_name, setpoint = _lookup(
        index,
        canonical_type,
        _field(thermostat, "control_1_name", owner=thermostat_name),
        label=f"{thermostat_name} setpoint",
    )
    schedule_field = (
        "heating_setpoint_temperature_schedule_name"
        if canonical_type == "ThermostatSetpoint:DualSetpoint"
        else "setpoint_temperature_schedule_name"
    )
    schedule = _schedule_reference(
        index,
        _field(setpoint, schedule_field, owner=setpoint_name),
        label=f"{setpoint_name} heating schedule",
    )
    cooling_schedule = None
    if canonical_type == "ThermostatSetpoint:DualSetpoint":
        cooling_schedule = _schedule_reference(
            index,
            _field(
                setpoint,
                "cooling_setpoint_temperature_schedule_name",
                owner=setpoint_name,
            ),
            label=f"{setpoint_name} cooling schedule",
        )
    return (
        _reference("ZoneControl:Thermostat", thermostat_name, thermostat),
        _reference(canonical_type, setpoint_name, setpoint),
        schedule,
        cooling_schedule,
    )


def _discover_from_model(model: dict[str, Any], *, source_model_sha256: str) -> HeatingDiscoveryInventory:
    try:
        index = _ModelIndex(model)
    except ControlInstrumentationError as error:
        raise HeatingDiscoveryError(f"epJSON indexing failed: {error}") from error
    canonical_types = {item.casefold(): item for item in HEATING_EQUIPMENT_OBJECT_TYPES}
    equipment_tables = {
        object_type: {name.casefold(): (name, fields) for name, fields in _table(index, object_type)}
        for object_type in HEATING_EQUIPMENT_OBJECT_TYPES
    }
    expected = {
        (object_type.casefold(), object_name.casefold())
        for object_type, table in equipment_tables.items()
        for object_name, _ in table.values()
    }
    occurrences: Counter[tuple[str, str]] = Counter()
    equipment_list_owners: dict[str, str] = {}
    zone_groups: dict[str, tuple[str, Mapping[str, Any], str, Mapping[str, Any], list[HeatingObjectReference]]] = {}
    for connection_name, connection in _table(index, "ZoneHVAC:EquipmentConnections"):
        zone_name, zone = _lookup(
            index,
            "Zone",
            _field(connection, "zone_name", owner=connection_name),
            label=f"{connection_name}.zone_name",
        )
        equipment_list_name, equipment_list = _lookup(
            index,
            "ZoneHVAC:EquipmentList",
            _field(connection, "zone_conditioning_equipment_list_name", owner=connection_name),
            label=f"{connection_name} equipment list",
        )
        prior_owner = equipment_list_owners.get(equipment_list_name.casefold())
        if prior_owner is not None:
            raise HeatingDiscoveryError(
                f"equipment list {equipment_list_name!r} is shared by zones {prior_owner!r} and {zone_name!r}"
            )
        equipment_list_owners[equipment_list_name.casefold()] = zone_name
        heating_references: list[HeatingObjectReference] = []
        for item in _records(equipment_list, "equipment", owner=equipment_list_name):
            raw_type = item.get("zone_equipment_object_type")
            if not isinstance(raw_type, str):
                raise HeatingDiscoveryError(
                    f"{equipment_list_name}.zone_equipment_object_type must be a nonempty exact string"
                )
            object_type = canonical_types.get(str.__str__(raw_type).casefold())
            if object_type is None:
                # The pinned B2B walk selects supported heating devices from
                # the complete zone-equipment list; it does not require those
                # devices to be the zone's only equipment.  Keep validating
                # the list entry as exact text, but leave unrelated equipment
                # outside this profile's ownership and mutation surface.
                _text(raw_type, field_name=f"{equipment_list_name} equipment type")
                continue
            _canonical_type_reference(
                raw_type,
                object_type,
                field_name=f"{equipment_list_name}.zone_equipment_object_type",
            )
            object_name, fields = _lookup(
                index,
                object_type,
                item.get("zone_equipment_name"),
                label=f"{equipment_list_name} heating-only equipment",
            )
            identity = (object_type.casefold(), object_name.casefold())
            occurrences[identity] += 1
            heating_references.append(_reference(object_type, object_name, fields))
        if not heating_references:
            continue
        if zone_name.casefold() in zone_groups:
            raise HeatingDiscoveryError(f"heating-only zone {zone_name!r} has multiple equipment connections")
        zone_groups[zone_name.casefold()] = (
            connection_name,
            connection,
            equipment_list_name,
            equipment_list,
            heating_references,
        )
    if set(occurrences) != expected or any(occurrences[item] != 1 for item in expected):
        missing = sorted(expected - set(occurrences))
        repeated = sorted(item for item, count in occurrences.items() if count != 1)
        raise HeatingDiscoveryError(
            "heating-only candidate inventory is not an exact one-to-one zone equipment walk: "
            f"missing={missing!r}, repeated={repeated!r}"
        )
    thermostats = _thermostat_by_zone(
        index,
        tuple(
            _field(connection, "zone_name", owner=connection_name)
            for connection_name, connection, _equipment_list_name, _equipment_list, _equipment in zone_groups.values()
        ),
    )
    records: list[HeatingZoneDiscoveryRecord] = []
    for _zone_key, group in sorted(zone_groups.items()):
        connection_name, connection, equipment_list_name, equipment_list, equipment = group
        zone_name, zone = _lookup(
            index,
            "Zone",
            _field(connection, "zone_name", owner=connection_name),
            label=f"{connection_name}.zone_name",
        )
        thermostat, setpoint, schedule, cooling_schedule = _thermostat_record(index, zone_name, thermostats)
        records.append(
            HeatingZoneDiscoveryRecord(
                zone=_reference("Zone", zone_name, zone),
                equipment_connection=_reference(
                    "ZoneHVAC:EquipmentConnections",
                    connection_name,
                    connection,
                ),
                equipment_list=_reference("ZoneHVAC:EquipmentList", equipment_list_name, equipment_list),
                equipment=tuple(
                    sorted(equipment, key=lambda item: (item.object_type.casefold(), item.object_name.casefold()))
                ),
                thermostat=thermostat,
                thermostat_setpoint=setpoint,
                heating_schedule=schedule,
                cooling_schedule=cooling_schedule,
            )
        )
    return HeatingDiscoveryInventory(source_model_sha256, tuple(records))


def discover_heating_only_from_json_bytes(source_bytes: bytes) -> HeatingDiscoveryInventory:
    """Return the exhaustive authenticated heating-only inventory."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    try:
        model = _parse_epjson_bytes(source_bytes)
    except (ControlInstrumentationError, TypeError, ValueError) as error:
        raise HeatingDiscoveryError(f"cannot parse heating-only source epJSON: {error}") from error
    return _discover_from_model(model, source_model_sha256=hashlib.sha256(source_bytes).hexdigest())


def _canonical_inventory(value: Any) -> HeatingDiscoveryInventory:
    if type(value) is not HeatingDiscoveryInventory:
        raise TypeError("expected_inventory must be an exact HeatingDiscoveryInventory")
    try:
        return HeatingDiscoveryInventory(
            source_model_sha256=value.source_model_sha256,
            records=value.records,
            schema_version=value.schema_version,
            algorithm_id=value.algorithm_id,
            upstream_commit=value.upstream_commit,
            upstream_source_path=value.upstream_source_path,
            upstream_source_sha256=value.upstream_source_sha256,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise HeatingDiscoveryError(f"expected heating inventory is malformed: {error}") from error


def verify_heating_discovery_inventory_from_json_bytes(
    source_bytes: bytes,
    expected_inventory: HeatingDiscoveryInventory,
) -> HeatingDiscoveryInventory:
    """Recompute discovery and reject source-byte or inventory substitution."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    expected_inventory = _canonical_inventory(expected_inventory)
    recomputed = discover_heating_only_from_json_bytes(source_bytes)
    if not hmac.compare_digest(
        _canonical_json(expected_inventory.to_dict()),
        _canonical_json(recomputed.to_dict()),
    ) or not hmac.compare_digest(expected_inventory.inventory_sha256, recomputed.inventory_sha256):
        raise HeatingDiscoveryError("heating discovery inventory does not match exact-byte recomputation")
    return recomputed


__all__ = [
    "HEATING_DISCOVERY_ALGORITHM_ID",
    "HEATING_DISCOVERY_SCHEMA_VERSION",
    "HEATING_EQUIPMENT_OBJECT_TYPES",
    "HEATING_PROFILE_ID",
    "HEATING_THERMOSTAT_OBJECT_TYPES",
    "HeatingDiscoveryError",
    "HeatingDiscoveryInventory",
    "HeatingObjectReference",
    "HeatingZoneDiscoveryRecord",
    "discover_heating_only_from_json_bytes",
    "verify_heating_discovery_inventory_from_json_bytes",
]
