"""Pure epJSON discovery for the first Building2Building unitary slice.

This adapts the one-zone ``AirLoopHVAC:UnitarySystem`` topology walk from the
pinned Building2Building v1.0.0 source.  It performs no model mutation and no
EnergyPlus run.  Every target unitary object is inventoried as supported or
unsupported so callers cannot silently drop systems that fail topology
resolution.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from building_gen.control.capabilities import (
    BUILDING2BUILDING_COMMIT,
    BoundSource,
    CandidateDisposition,
    ControlChannel,
    control_profile_by_id,
)

UNITARY_DISCOVERY_SCHEMA_VERSION = "hot-b2b-unitary-discovery/0.1.0-draft"
UNITARY_DISCOVERY_ALGORITHM_ID = "b2b-v1-one-zone-unitary-topology-adaptation-v1"
UNITARY_PROFILE_ID = "b2b_unitarysystem_v1"
UNITARY_OBJECT_TYPE = "AirLoopHVAC:UnitarySystem"
CONSTANT_VOLUME_TERMINAL_TYPE = "AirTerminal:SingleDuct:ConstantVolume:NoReheat"
STANDARD_AIR_DENSITY_KG_M3 = 1.2
FALLBACK_FAN_MAX_KG_S = 15.0
FALLBACK_SAT_MAX_C = 60.0
_UNITARY_SAT_PROFILE_LOWER_BOUND = (
    control_profile_by_id(UNITARY_PROFILE_ID)
    .channel_contract(ControlChannel.SUPPLY_AIR_TEMPERATURE_C)
    .lower_bound.exact_value
)
if _UNITARY_SAT_PROFILE_LOWER_BOUND is None:  # pragma: no cover - bundled-profile invariant
    raise RuntimeError("unitary SAT profile must declare an exact lower bound")
UNITARY_SAT_LOWER_BOUND_C = _UNITARY_SAT_PROFILE_LOWER_BOUND
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FAN_OBJECT_TYPES = (
    "Fan:SystemModel",
    "Fan:OnOff",
    "Fan:ConstantVolume",
)
_FAN_DESIGN_FLOW_FIELD_BY_TYPE = {
    "Fan:SystemModel": "design_maximum_air_flow_rate",
    "Fan:OnOff": "maximum_flow_rate",
    "Fan:ConstantVolume": "maximum_flow_rate",
}
_DISCOVERY_RELEVANT_OBJECT_TYPES = (
    UNITARY_OBJECT_TYPE,
    "AirLoopHVAC",
    "BranchList",
    "Branch",
    "ZoneHVAC:EquipmentConnections",
    "ZoneHVAC:EquipmentList",
    "ZoneHVAC:AirDistributionUnit",
    CONSTANT_VOLUME_TERMINAL_TYPE,
    "AirLoopHVAC:ZoneSplitter",
    "NodeList",
    *_FAN_OBJECT_TYPES,
)
_DISCOVERY_RELEVANT_OBJECT_TYPE_BY_CASEFOLD = {item.casefold(): item for item in _DISCOVERY_RELEVANT_OBJECT_TYPES}
if len(_DISCOVERY_RELEVANT_OBJECT_TYPE_BY_CASEFOLD) != len(  # pragma: no cover - module invariant
    _DISCOVERY_RELEVANT_OBJECT_TYPES
):
    raise RuntimeError("discovery-relevant EnergyPlus object types must be unique ignoring case")
_MISSING = object()


class ControlDiscoveryError(ValueError):
    """Raised when epJSON discovery inputs are structurally ambiguous."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlDiscoveryError(f"epJSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ControlDiscoveryError(f"epJSON contains non-finite constant {value!r}")


def _finite_json_float(value: str) -> float:
    exact = Decimal(value)
    result = float(exact)
    if not math.isfinite(result) or (result == 0.0 and exact != 0):
        raise ControlDiscoveryError(f"epJSON floating-point value is not finite or representable: {value!r}")
    return result


def _sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ControlDiscoveryError(f"{field_name} must be a lowercase SHA-256 digest")
    value = str.__str__(value)
    if _SHA256_RE.fullmatch(value) is None:
        raise ControlDiscoveryError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _enum(value: Any, enum_type: type[Any], *, field_name: str) -> Any:
    try:
        if isinstance(value, enum_type):
            return value
        if isinstance(value, str):
            value = str.__str__(value)
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ControlDiscoveryError(f"{field_name} is not a valid {enum_type.__name__}") from error


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ControlDiscoveryError(f"{field_name} must be a nonempty exact string")
    value = str.__str__(value)
    if not value or value != str.strip(value):
        raise ControlDiscoveryError(f"{field_name} must be a nonempty exact string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ControlDiscoveryError(f"{field_name} contains a control character")
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
        raise ControlDiscoveryError(f"epJSON object is not canonical JSON data: {error}") from error


def _owned_bytes(value: Any, *, field_name: str) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{field_name} must be bytes")
    # ``bytes.__bytes__`` is not available on every supported Python version
    # (notably the Python shipped in the pinned EnergyPlus container).  A
    # memoryview copies the immutable buffer without dispatching a hostile
    # bytes-subclass ``__bytes__``, ``decode``, or ``startswith`` override.
    return memoryview(value).tobytes()


def energyplus_source_object_sha256(
    object_type: str,
    object_name: str,
    fields: Mapping[str, Any],
) -> str:
    """Hash one exact source object with an explicit EnergyPlus identity."""

    object_type = _text(object_type, field_name="object_type")
    object_name = _text(object_name, field_name="object_name")
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping")
    payload = {
        "object_type": object_type,
        "object_name": object_name,
        "fields": dict(fields),
    }
    return hashlib.sha256(b"HOT EnergyPlus source object v1\0" + _canonical_json(payload)).hexdigest()


def _root_table_index(model: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    """Index root object tables without permitting case-spelling ambiguity.

    EnergyPlus object references are case-insensitive, but epJSON object-table
    keys are part of the discovery grammar.  A misspelled relevant table must
    therefore fail closed instead of looking like an absent table.
    """

    result: dict[str, tuple[str, Any]] = {}
    for raw_object_type, raw_table in model.items():
        object_type = _text(raw_object_type, field_name="epJSON root object-table name")
        folded = object_type.casefold()
        if folded in result:
            raise ControlDiscoveryError("epJSON root contains object-table names that collide ignoring case")
        canonical = _DISCOVERY_RELEVANT_OBJECT_TYPE_BY_CASEFOLD.get(folded)
        if canonical is not None and object_type != canonical:
            raise ControlDiscoveryError(
                f"epJSON discovery-relevant object table {object_type!r} must use canonical spelling " f"{canonical!r}"
            )
        result[folded] = (object_type, raw_table)
    return result


def _table(
    root_tables: Mapping[str, tuple[str, Any]],
    object_type: str,
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    entry = root_tables.get(object_type.casefold())
    raw = {} if entry is None else entry[1]
    if not isinstance(raw, Mapping):
        raise ControlDiscoveryError(f"epJSON {object_type!r} table must be a mapping")
    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for raw_name, fields in raw.items():
        name = _text(raw_name, field_name=f"{object_type} object name")
        if not isinstance(fields, Mapping):
            raise ControlDiscoveryError(f"epJSON object {object_type}/{name} must be a mapping")
        folded = name.casefold()
        if folded in result:
            raise ControlDiscoveryError(f"epJSON {object_type!r} contains names that collide ignoring case")
        result[folded] = (name, fields)
    return result


def _records(fields: Mapping[str, Any], key: str, *, owner: str) -> tuple[Mapping[str, Any], ...]:
    raw = fields.get(key, ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ControlDiscoveryError(f"{owner}.{key} must be an array")
    records: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ControlDiscoveryError(f"{owner}.{key}[{index}] must be an object")
        records.append(value)
    return tuple(records)


def _field_text(fields: Mapping[str, Any], key: str, *, owner: str) -> str:
    return _text(fields.get(key), field_name=f"{owner}.{key}")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_exact_autosize(value: Any) -> bool:
    return isinstance(value, str) and str.__str__(value) == "Autosize"


@dataclass(frozen=True, slots=True)
class UnitaryDiscoveryRecord:
    object_type: str
    object_name: str
    source_object_sha256: str
    disposition: CandidateDisposition
    reason_code: str | None
    zone_name: str | None
    air_loop_name: str | None
    supply_outlet_node_name: str | None
    fan_object_type: str | None
    fan_object_name: str | None
    fan_upper_bound_kg_s: float | None
    fan_upper_bound_source: BoundSource | None
    sat_upper_bound_c: float | None
    sat_upper_bound_source: BoundSource | None

    def __post_init__(self) -> None:
        object_type = _text(self.object_type, field_name="object_type")
        if object_type != UNITARY_OBJECT_TYPE:
            raise ControlDiscoveryError(f"object_type must be {UNITARY_OBJECT_TYPE!r}")
        object.__setattr__(self, "object_type", object_type)
        object.__setattr__(self, "object_name", _text(self.object_name, field_name="object_name"))
        object.__setattr__(
            self,
            "source_object_sha256",
            _sha256(self.source_object_sha256, field_name="source_object_sha256"),
        )
        disposition = _enum(
            self.disposition,
            CandidateDisposition,
            field_name="disposition",
        )
        object.__setattr__(self, "disposition", disposition)
        if self.disposition is CandidateDisposition.SUPPORTED:
            if self.reason_code is not None:
                raise ControlDiscoveryError("supported unitary records cannot carry a reason_code")
            required = (
                self.zone_name,
                self.air_loop_name,
                self.supply_outlet_node_name,
                self.fan_object_type,
                self.fan_object_name,
                self.fan_upper_bound_kg_s,
                self.fan_upper_bound_source,
                self.sat_upper_bound_c,
                self.sat_upper_bound_source,
            )
            if any(value is None for value in required):
                raise ControlDiscoveryError("supported unitary records require complete topology and bounds")
            for field_name in (
                "zone_name",
                "air_loop_name",
                "supply_outlet_node_name",
                "fan_object_type",
                "fan_object_name",
            ):
                object.__setattr__(
                    self,
                    field_name,
                    _text(getattr(self, field_name), field_name=field_name),
                )
            if self.fan_object_type not in _FAN_OBJECT_TYPES:
                raise ControlDiscoveryError("supported unitary records require a supported fan type")
            for field_name in ("fan_upper_bound_source", "sat_upper_bound_source"):
                source = _enum(
                    getattr(self, field_name),
                    BoundSource,
                    field_name=field_name,
                )
                if source not in {
                    BoundSource.MODEL_DESIGN,
                    BoundSource.ENGINEERING_FALLBACK,
                }:
                    raise ControlDiscoveryError(f"{field_name} must be model_design or engineering_fallback")
                object.__setattr__(self, field_name, source)

            fan_upper = _finite_float(self.fan_upper_bound_kg_s)
            if fan_upper is None or fan_upper <= 0.0:
                raise ControlDiscoveryError("fan_upper_bound_kg_s must be finite and positive")
            object.__setattr__(self, "fan_upper_bound_kg_s", fan_upper)
            if self.fan_upper_bound_source is BoundSource.ENGINEERING_FALLBACK and fan_upper != FALLBACK_FAN_MAX_KG_S:
                raise ControlDiscoveryError(f"fallback fan_upper_bound_kg_s must equal {FALLBACK_FAN_MAX_KG_S!r}")

            sat_upper = _finite_float(self.sat_upper_bound_c)
            if sat_upper is None or sat_upper <= UNITARY_SAT_LOWER_BOUND_C:
                raise ControlDiscoveryError(
                    "sat_upper_bound_c must be finite and exceed the unitary profile lower bound"
                )
            object.__setattr__(self, "sat_upper_bound_c", sat_upper)
            if self.sat_upper_bound_source is BoundSource.ENGINEERING_FALLBACK and sat_upper != FALLBACK_SAT_MAX_C:
                raise ControlDiscoveryError(f"fallback sat_upper_bound_c must equal {FALLBACK_SAT_MAX_C!r}")
        elif self.reason_code is None:
            raise ControlDiscoveryError("unsupported unitary records require a reason_code")
        else:
            object.__setattr__(
                self,
                "reason_code",
                _text(self.reason_code, field_name="reason_code"),
            )
            details = (
                self.zone_name,
                self.air_loop_name,
                self.supply_outlet_node_name,
                self.fan_object_type,
                self.fan_object_name,
                self.fan_upper_bound_kg_s,
                self.fan_upper_bound_source,
                self.sat_upper_bound_c,
                self.sat_upper_bound_source,
            )
            if any(value is not None for value in details):
                raise ControlDiscoveryError("unsupported unitary records cannot claim resolved topology or bounds")

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(
            b"HOT unitary discovery candidate v1\0" + _canonical_json(_unitary_record_payload(self))
        ).hexdigest()
        return f"unitary-candidate-{digest[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return _unitary_record_payload(self)


def _unitary_record_payload(record: UnitaryDiscoveryRecord) -> dict[str, Any]:
    """Serialize one canonical base record without virtual dispatch."""

    return {
        "object_type": record.object_type,
        "object_name": record.object_name,
        "source_object_sha256": record.source_object_sha256,
        "disposition": record.disposition.value,
        "reason_code": record.reason_code,
        "zone_name": record.zone_name,
        "air_loop_name": record.air_loop_name,
        "supply_outlet_node_name": record.supply_outlet_node_name,
        "fan_object_type": record.fan_object_type,
        "fan_object_name": record.fan_object_name,
        "fan_upper_bound_kg_s": record.fan_upper_bound_kg_s,
        "fan_upper_bound_source": (
            None if record.fan_upper_bound_source is None else record.fan_upper_bound_source.value
        ),
        "sat_upper_bound_c": record.sat_upper_bound_c,
        "sat_upper_bound_source": (
            None if record.sat_upper_bound_source is None else record.sat_upper_bound_source.value
        ),
    }


def _canonical_unitary_record(
    record: Any,
    *,
    field_name: str,
) -> UnitaryDiscoveryRecord:
    """Own a revalidated exact-base copy of one discovery record."""

    if type(record) is not UnitaryDiscoveryRecord:
        raise TypeError(f"{field_name} must be an exact UnitaryDiscoveryRecord")
    try:
        return UnitaryDiscoveryRecord(
            object_type=record.object_type,
            object_name=record.object_name,
            source_object_sha256=record.source_object_sha256,
            disposition=record.disposition,
            reason_code=record.reason_code,
            zone_name=record.zone_name,
            air_loop_name=record.air_loop_name,
            supply_outlet_node_name=record.supply_outlet_node_name,
            fan_object_type=record.fan_object_type,
            fan_object_name=record.fan_object_name,
            fan_upper_bound_kg_s=record.fan_upper_bound_kg_s,
            fan_upper_bound_source=record.fan_upper_bound_source,
            sat_upper_bound_c=record.sat_upper_bound_c,
            sat_upper_bound_source=record.sat_upper_bound_source,
        )
    except AttributeError as error:
        raise ControlDiscoveryError(f"{field_name} is missing canonical record state") from error


def _canonical_unitary_record_tuple(
    records: Any,
    *,
    field_name: str,
) -> tuple[UnitaryDiscoveryRecord, ...]:
    if type(records) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    return tuple(
        _canonical_unitary_record(
            record,
            field_name=f"{field_name}[{index}]",
        )
        for index, record in enumerate(records)
    )


def _unitary_inventory_payload(
    *,
    source_model_sha256: str,
    records: tuple[UnitaryDiscoveryRecord, ...],
    schema_version: str,
    algorithm_id: str,
    upstream_commit: str,
) -> dict[str, Any]:
    """Build the explicit base payload authenticated by an inventory hash."""

    return {
        "schema_version": schema_version,
        "algorithm_id": algorithm_id,
        "upstream_commit": upstream_commit,
        "source_model_sha256": source_model_sha256,
        "records": [_unitary_record_payload(record) for record in records],
    }


def _unitary_inventory_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(b"HOT unitary discovery inventory v1\0" + _canonical_json(payload)).hexdigest()


def _canonical_unitary_inventory(
    inventory: Any,
    *,
    field_name: str,
) -> UnitaryDiscoveryInventory:
    """Reconstruct one exact inventory after class definition is available."""

    if type(inventory) is not UnitaryDiscoveryInventory:
        raise TypeError(f"{field_name} must be an exact UnitaryDiscoveryInventory")
    records = _canonical_unitary_record_tuple(
        inventory.records,
        field_name=f"{field_name}.records",
    )
    try:
        return UnitaryDiscoveryInventory(
            source_model_sha256=inventory.source_model_sha256,
            records=records,
            schema_version=inventory.schema_version,
            algorithm_id=inventory.algorithm_id,
            upstream_commit=inventory.upstream_commit,
        )
    except AttributeError as error:
        raise ControlDiscoveryError(f"{field_name} is missing canonical inventory state") from error


def _unsupported_record(
    object_name: str,
    source_object_sha256: str,
    reason_code: str,
) -> UnitaryDiscoveryRecord:
    """Return one explicit unsupported outcome without partial topology claims."""

    return UnitaryDiscoveryRecord(
        UNITARY_OBJECT_TYPE,
        object_name,
        source_object_sha256,
        CandidateDisposition.UNSUPPORTED,
        reason_code,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@dataclass(frozen=True, slots=True)
class UnitaryDiscoveryInventory:
    """One deterministic inventory that still requires source-byte verification.

    Constructor validation establishes internal invariants but cannot prove
    that ``source_model_sha256`` or the records came from a particular file.
    Downstream consumers of a constructed or deserialized inventory must call
    :meth:`verify_source_bytes` (or the equivalent module-level verifier)
    before trusting it.
    """

    source_model_sha256: str
    records: tuple[UnitaryDiscoveryRecord, ...]
    schema_version: str = UNITARY_DISCOVERY_SCHEMA_VERSION
    algorithm_id: str = UNITARY_DISCOVERY_ALGORITHM_ID
    upstream_commit: str = BUILDING2BUILDING_COMMIT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_model_sha256",
            _sha256(self.source_model_sha256, field_name="source_model_sha256"),
        )
        schema_version = _text(self.schema_version, field_name="schema_version")
        if schema_version != UNITARY_DISCOVERY_SCHEMA_VERSION:
            raise ControlDiscoveryError("unexpected unitary discovery schema version")
        object.__setattr__(self, "schema_version", schema_version)
        algorithm_id = _text(self.algorithm_id, field_name="algorithm_id")
        if algorithm_id != UNITARY_DISCOVERY_ALGORITHM_ID:
            raise ControlDiscoveryError("unexpected unitary discovery algorithm")
        object.__setattr__(self, "algorithm_id", algorithm_id)
        upstream_commit = _text(self.upstream_commit, field_name="upstream_commit")
        if upstream_commit != BUILDING2BUILDING_COMMIT:
            raise ControlDiscoveryError("unitary discovery must bind Building2Building v1.0.0")
        object.__setattr__(self, "upstream_commit", upstream_commit)
        records = _canonical_unitary_record_tuple(
            self.records,
            field_name="records",
        )
        ordered = tuple(sorted(records, key=lambda item: item.object_name.casefold()))
        if records != ordered:
            raise ControlDiscoveryError("unitary discovery records must use canonical object-name order")
        names = [item.object_name.casefold() for item in records]
        if len(names) != len(set(names)):
            raise ControlDiscoveryError("unitary discovery records contain duplicate object names")
        object.__setattr__(self, "records", records)

    @property
    def inventory_sha256(self) -> str:
        payload = _unitary_inventory_payload(
            source_model_sha256=self.source_model_sha256,
            records=self.records,
            schema_version=self.schema_version,
            algorithm_id=self.algorithm_id,
            upstream_commit=self.upstream_commit,
        )
        return _unitary_inventory_sha256(payload)

    @property
    def supported_records(self) -> tuple[UnitaryDiscoveryRecord, ...]:
        return tuple(item for item in self.records if item.disposition is CandidateDisposition.SUPPORTED)

    @property
    def unsupported_records(self) -> tuple[UnitaryDiscoveryRecord, ...]:
        return tuple(item for item in self.records if item.disposition is CandidateDisposition.UNSUPPORTED)

    def verify_source_bytes(self, source_bytes: bytes) -> UnitaryDiscoveryInventory:
        """Recompute this inventory from exact bytes and return the verified result."""

        return verify_unitary_discovery_inventory_from_json_bytes(source_bytes, self)


def _loop_components(
    root_tables: Mapping[str, tuple[str, Any]],
) -> dict[str, list[tuple[str, str, str, str]]]:
    loops = _table(root_tables, "AirLoopHVAC")
    branch_lists = _table(root_tables, "BranchList")
    branches = _table(root_tables, "Branch")
    result: dict[str, list[tuple[str, str, str, str]]] = {}
    for loop_name, loop in loops.values():
        owner = f"AirLoopHVAC/{loop_name}"
        branch_list_ref = _field_text(loop, "branch_list_name", owner=owner)
        branch_list_entry = branch_lists.get(branch_list_ref.casefold())
        if branch_list_entry is None:
            continue
        branch_list_name, branch_list = branch_list_entry
        for branch_ref in _records(
            branch_list,
            "branches",
            owner=f"BranchList/{branch_list_name}",
        ):
            branch_name_ref = _field_text(
                branch_ref,
                "branch_name",
                owner=f"BranchList/{branch_list_name}",
            )
            branch_entry = branches.get(branch_name_ref.casefold())
            if branch_entry is None:
                continue
            branch_name, branch = branch_entry
            for component in _records(branch, "components", owner=f"Branch/{branch_name}"):
                component_type = _field_text(
                    component,
                    "component_object_type",
                    owner=f"Branch/{branch_name}",
                )
                component_name = _field_text(
                    component,
                    "component_name",
                    owner=f"Branch/{branch_name}",
                )
                inlet = _field_text(
                    component,
                    "component_inlet_node_name",
                    owner=f"Branch/{branch_name}",
                )
                outlet = _field_text(
                    component,
                    "component_outlet_node_name",
                    owner=f"Branch/{branch_name}",
                )
                result.setdefault(loop_name.casefold(), []).append((component_type, component_name, inlet, outlet))
    return result


def _loop_demand_nodes(root_tables: Mapping[str, tuple[str, Any]]) -> dict[str, set[str]]:
    loops = _table(root_tables, "AirLoopHVAC")
    node_lists = _table(root_tables, "NodeList")
    result: dict[str, set[str]] = {}
    for loop_name, loop in loops.values():
        owner = f"AirLoopHVAC/{loop_name}"
        reference = _field_text(loop, "demand_side_inlet_node_names", owner=owner)
        node_list_entry = node_lists.get(reference.casefold())
        if node_list_entry is None:
            result[loop_name.casefold()] = {reference.casefold()}
            continue
        node_list_name, node_list = node_list_entry
        result[loop_name.casefold()] = {
            _field_text(item, "node_name", owner=f"NodeList/{node_list_name}").casefold()
            for item in _records(node_list, "nodes", owner=f"NodeList/{node_list_name}")
        }
    return result


def _zones_by_loop(root_tables: Mapping[str, tuple[str, Any]]) -> dict[str, set[str]]:
    connections = _table(root_tables, "ZoneHVAC:EquipmentConnections")
    equipment_lists = _table(root_tables, "ZoneHVAC:EquipmentList")
    adus = _table(root_tables, "ZoneHVAC:AirDistributionUnit")
    terminals = _table(root_tables, CONSTANT_VOLUME_TERMINAL_TYPE)
    splitters = _table(root_tables, "AirLoopHVAC:ZoneSplitter")
    loop_nodes = _loop_demand_nodes(root_tables)

    terminal_inlet_to_zones: dict[str, set[str]] = {}
    for connection_name, connection in connections.values():
        owner = f"ZoneHVAC:EquipmentConnections/{connection_name}"
        zone_name = _field_text(connection, "zone_name", owner=owner)
        equipment_list_ref = _field_text(
            connection,
            "zone_conditioning_equipment_list_name",
            owner=owner,
        )
        equipment_list_entry = equipment_lists.get(equipment_list_ref.casefold())
        if equipment_list_entry is None:
            continue
        equipment_list_name, equipment_list = equipment_list_entry
        for equipment_ref in _records(
            equipment_list,
            "equipment",
            owner=f"ZoneHVAC:EquipmentList/{equipment_list_name}",
        ):
            if (
                _field_text(
                    equipment_ref,
                    "zone_equipment_object_type",
                    owner=f"ZoneHVAC:EquipmentList/{equipment_list_name}",
                ).casefold()
                != "zonehvac:airdistributionunit".casefold()
            ):
                continue
            adu_ref = _field_text(
                equipment_ref,
                "zone_equipment_name",
                owner=f"ZoneHVAC:EquipmentList/{equipment_list_name}",
            )
            adu_entry = adus.get(adu_ref.casefold())
            if adu_entry is None:
                continue
            adu_name, adu = adu_entry
            terminal_type = _field_text(
                adu,
                "air_terminal_object_type",
                owner=f"ZoneHVAC:AirDistributionUnit/{adu_name}",
            )
            if terminal_type.casefold() != CONSTANT_VOLUME_TERMINAL_TYPE.casefold():
                continue
            terminal_ref = _field_text(
                adu,
                "air_terminal_name",
                owner=f"ZoneHVAC:AirDistributionUnit/{adu_name}",
            )
            terminal_entry = terminals.get(terminal_ref.casefold())
            if terminal_entry is None:
                continue
            terminal_name, terminal = terminal_entry
            inlet = _field_text(
                terminal,
                "air_inlet_node_name",
                owner=f"{CONSTANT_VOLUME_TERMINAL_TYPE}/{terminal_name}",
            )
            terminal_inlet_to_zones.setdefault(inlet.casefold(), set()).add(zone_name)

    zones_by_loop: dict[str, set[str]] = {}
    for splitter_name, splitter in splitters.values():
        owner = f"AirLoopHVAC:ZoneSplitter/{splitter_name}"
        splitter_inlet = _field_text(splitter, "inlet_node_name", owner=owner)
        zones: set[str] = set()
        for outlet in _records(splitter, "nodes", owner=owner):
            outlet_name = _field_text(outlet, "outlet_node_name", owner=owner)
            zones.update(terminal_inlet_to_zones.get(outlet_name.casefold(), ()))
        if not zones:
            continue
        matching_loops = [
            loop_name for loop_name, demand_nodes in loop_nodes.items() if splitter_inlet.casefold() in demand_nodes
        ]
        for loop_name in matching_loops:
            zones_by_loop.setdefault(loop_name, set()).update(zones)
    return zones_by_loop


def _fan_identity(
    root_tables: Mapping[str, tuple[str, Any]],
    system: Mapping[str, Any],
    *,
    owner: str,
) -> tuple[str, str, Mapping[str, Any]] | None:
    fan_name = _field_text(system, "supply_fan_name", owner=owner)
    declared_type = system.get("supply_fan_object_type")
    if declared_type is not None:
        declared_type = _text(declared_type, field_name=f"{owner}.supply_fan_object_type")
        candidate_types = tuple(item for item in _FAN_OBJECT_TYPES if item.casefold() == declared_type.casefold())
        if not candidate_types:
            return None
    else:
        candidate_types = _FAN_OBJECT_TYPES
    matches: list[tuple[str, str, Mapping[str, Any]]] = []
    for object_type in candidate_types:
        entry = _table(root_tables, object_type).get(fan_name.casefold())
        if entry is not None:
            canonical_name, fields = entry
            matches.append((object_type, canonical_name, fields))
    return matches[0] if len(matches) == 1 else None


def _discover_unitary_systems_from_trusted_mapping(
    model: Mapping[str, Any],
    *,
    source_model_sha256: str,
) -> UnitaryDiscoveryInventory:
    """Inventory parsed bytes after the public entry point authenticates them.

    This helper is deliberately private: a mapping cannot prove which source
    bytes it came from. Production callers must use
    :func:`discover_unitary_systems_from_json_bytes`.
    """

    if not isinstance(model, Mapping):
        raise TypeError("model must be an epJSON mapping")
    source_model_sha256 = _sha256(source_model_sha256, field_name="source_model_sha256")
    root_tables = _root_table_index(model)
    systems = _table(root_tables, UNITARY_OBJECT_TYPE)
    air_loops = _table(root_tables, "AirLoopHVAC")
    components_by_loop = _loop_components(root_tables)
    zones_by_loop = _zones_by_loop(root_tables)
    records: list[UnitaryDiscoveryRecord] = []

    for system_name, system in sorted(systems.values(), key=lambda item: item[0].casefold()):
        source_hash = energyplus_source_object_sha256(UNITARY_OBJECT_TYPE, system_name, system)
        occurrences = [
            (loop_name, inlet, outlet)
            for loop_name, components in components_by_loop.items()
            for object_type, object_name, inlet, outlet in components
            if object_type.casefold() == UNITARY_OBJECT_TYPE.casefold()
            and object_name.casefold() == system_name.casefold()
        ]
        if len(occurrences) != 1:
            records.append(
                _unsupported_record(
                    system_name,
                    source_hash,
                    "unitary_branch_occurrence_not_unique",
                )
            )
            continue
        loop_key, branch_inlet, branch_outlet = occurrences[0]
        zones = sorted(zones_by_loop.get(loop_key, ()), key=str.casefold)
        if len(zones) != 1:
            records.append(
                _unsupported_record(
                    system_name,
                    source_hash,
                    "unitary_served_zone_not_unique",
                )
            )
            continue
        owner = f"{UNITARY_OBJECT_TYPE}/{system_name}"
        try:
            system_inlet = _field_text(system, "air_inlet_node_name", owner=owner)
            system_outlet = _field_text(system, "air_outlet_node_name", owner=owner)
        except ControlDiscoveryError:
            records.append(
                _unsupported_record(
                    system_name,
                    source_hash,
                    "unitary_air_node_fields_missing_or_invalid",
                )
            )
            continue
        if system_inlet.casefold() != branch_inlet.casefold() or system_outlet.casefold() != branch_outlet.casefold():
            records.append(
                _unsupported_record(
                    system_name,
                    source_hash,
                    "unitary_air_nodes_do_not_match_branch",
                )
            )
            continue

        fan = _fan_identity(root_tables, system, owner=owner)
        if fan is None:
            records.append(
                _unsupported_record(
                    system_name,
                    source_hash,
                    "unitary_supply_fan_not_unique_or_unsupported",
                )
            )
            continue

        fan_type, fan_name, fan_fields = fan
        design_flow_field = _FAN_DESIGN_FLOW_FIELD_BY_TYPE[fan_type]
        raw_design_flow = fan_fields.get(design_flow_field, _MISSING)
        if raw_design_flow is _MISSING or _is_exact_autosize(raw_design_flow):
            fan_upper = FALLBACK_FAN_MAX_KG_S
            fan_source = BoundSource.ENGINEERING_FALLBACK
        else:
            design_flow_m3_s = _finite_float(raw_design_flow)
            if design_flow_m3_s is None or design_flow_m3_s <= 0.0:
                records.append(
                    _unsupported_record(
                        system_name,
                        source_hash,
                        "unitary_fan_design_flow_invalid",
                    )
                )
                continue
            fan_upper = design_flow_m3_s * STANDARD_AIR_DENSITY_KG_M3
            if not math.isfinite(fan_upper) or fan_upper <= 0.0:
                records.append(
                    _unsupported_record(
                        system_name,
                        source_hash,
                        "unitary_fan_design_flow_invalid",
                    )
                )
                continue
            fan_source = BoundSource.MODEL_DESIGN

        raw_sat_upper = system.get("maximum_supply_air_temperature", _MISSING)
        if raw_sat_upper is _MISSING or _is_exact_autosize(raw_sat_upper):
            sat_upper = FALLBACK_SAT_MAX_C
            sat_source = BoundSource.ENGINEERING_FALLBACK
        else:
            parsed_sat_upper = _finite_float(raw_sat_upper)
            if parsed_sat_upper is None:
                records.append(
                    _unsupported_record(
                        system_name,
                        source_hash,
                        "unitary_sat_upper_bound_invalid",
                    )
                )
                continue
            sat_upper = parsed_sat_upper
            if sat_upper <= UNITARY_SAT_LOWER_BOUND_C:
                records.append(
                    _unsupported_record(
                        system_name,
                        source_hash,
                        "unitary_sat_upper_bound_not_above_profile_lower_bound",
                    )
                )
                continue
            sat_source = BoundSource.MODEL_DESIGN

        loop_name = next(
            canonical_name for canonical_name, _ in air_loops.values() if canonical_name.casefold() == loop_key
        )
        records.append(
            UnitaryDiscoveryRecord(
                UNITARY_OBJECT_TYPE,
                system_name,
                source_hash,
                CandidateDisposition.SUPPORTED,
                None,
                zones[0],
                loop_name,
                system_outlet,
                fan_type,
                fan_name,
                fan_upper,
                fan_source,
                sat_upper,
                sat_source,
            )
        )

    return UnitaryDiscoveryInventory(source_model_sha256, tuple(records))


def discover_unitary_systems_from_json_bytes(source_bytes: bytes) -> UnitaryDiscoveryInventory:
    """Parse and inventory exact epJSON bytes with an authenticated file hash."""

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    if source_bytes.startswith(b"\xef\xbb\xbf"):
        raise ControlDiscoveryError("epJSON may not contain a UTF-8 byte-order mark")
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ControlDiscoveryError(f"epJSON is not UTF-8: {error}") from error
    try:
        model = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_finite_json_float,
        )
    except ControlDiscoveryError:
        raise
    except json.JSONDecodeError as error:
        raise ControlDiscoveryError(f"epJSON is not valid JSON: {error}") from error
    if not isinstance(model, Mapping):
        raise ControlDiscoveryError("epJSON root must be an object")
    return _discover_unitary_systems_from_trusted_mapping(
        model,
        source_model_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def verify_unitary_discovery_inventory_from_json_bytes(
    source_bytes: bytes,
    expected_inventory: UnitaryDiscoveryInventory,
) -> UnitaryDiscoveryInventory:
    """Recompute and authenticate an inventory against exact epJSON bytes.

    ``UnitaryDiscoveryInventory`` construction alone does not establish source
    provenance.  Downstream consumers must use this verifier before trusting a
    constructed or deserialized inventory.  The returned value is the freshly
    recomputed inventory, never the caller-supplied object.
    """

    source_bytes = _owned_bytes(source_bytes, field_name="source_bytes")
    expected = _canonical_unitary_inventory(
        expected_inventory,
        field_name="expected_inventory",
    )
    recomputed = discover_unitary_systems_from_json_bytes(source_bytes)
    if not hmac.compare_digest(
        expected.source_model_sha256,
        recomputed.source_model_sha256,
    ):
        raise ControlDiscoveryError("unitary discovery inventory does not authenticate the exact source bytes")

    expected_payload = _unitary_inventory_payload(
        source_model_sha256=expected.source_model_sha256,
        records=expected.records,
        schema_version=expected.schema_version,
        algorithm_id=expected.algorithm_id,
        upstream_commit=expected.upstream_commit,
    )
    recomputed_payload = _unitary_inventory_payload(
        source_model_sha256=recomputed.source_model_sha256,
        records=recomputed.records,
        schema_version=recomputed.schema_version,
        algorithm_id=recomputed.algorithm_id,
        upstream_commit=recomputed.upstream_commit,
    )
    expected_payload_bytes = _canonical_json(expected_payload)
    recomputed_payload_bytes = _canonical_json(recomputed_payload)
    expected_sha256 = _unitary_inventory_sha256(expected_payload)
    recomputed_sha256 = _unitary_inventory_sha256(recomputed_payload)
    if not hmac.compare_digest(expected_payload_bytes, recomputed_payload_bytes) or not hmac.compare_digest(
        expected_sha256,
        recomputed_sha256,
    ):
        raise ControlDiscoveryError("unitary discovery inventory does not match recomputation from source bytes")
    return recomputed


__all__ = [
    "CONSTANT_VOLUME_TERMINAL_TYPE",
    "ControlDiscoveryError",
    "FALLBACK_FAN_MAX_KG_S",
    "FALLBACK_SAT_MAX_C",
    "STANDARD_AIR_DENSITY_KG_M3",
    "UNITARY_DISCOVERY_ALGORITHM_ID",
    "UNITARY_DISCOVERY_SCHEMA_VERSION",
    "UNITARY_OBJECT_TYPE",
    "UNITARY_PROFILE_ID",
    "UNITARY_SAT_LOWER_BOUND_C",
    "UnitaryDiscoveryInventory",
    "UnitaryDiscoveryRecord",
    "discover_unitary_systems_from_json_bytes",
    "energyplus_source_object_sha256",
    "verify_unitary_discovery_inventory_from_json_bytes",
]
