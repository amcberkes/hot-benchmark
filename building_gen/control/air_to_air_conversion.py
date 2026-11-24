"""Authenticated Building2Building AirToAir-to-UnitarySystem conversion.

The pinned Building2Building paper release converts
``AirLoopHVAC:UnitaryHeatPump:AirToAir`` objects into
``AirLoopHVAC:UnitarySystem`` objects before adding the direct fan-flow and
supply-air-temperature interface.  This module preserves that transformation
while making the conversion an exact-byte, exhaustive, replayable artifact.

Unlike the upstream convenience transform, this implementation refuses name
collisions, ambiguous field ownership, dangling references, and candidates
that do not occur exactly once in a ``Branch`` component list.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from building_gen.control.capabilities import (
    BUILDING2BUILDING_COMMIT,
    BUILDING2BUILDING_LICENSE_SPDX,
    BUILDING2BUILDING_LICENSE_URL,
    BUILDING2BUILDING_REPOSITORY_URL,
    BUILDING2BUILDING_TAG,
)
from building_gen.control.discovery import energyplus_source_object_sha256

AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE = "AirLoopHVAC:UnitaryHeatPump:AirToAir"
UNITARY_SYSTEM_OBJECT_TYPE = "AirLoopHVAC:UnitarySystem"
BRANCH_OBJECT_TYPE = "Branch"
AIR_TO_AIR_CONVERSION_SCHEMA_VERSION = "hot-b2b-air-to-air-unitary-conversion/1.0.0"
AIR_TO_AIR_CONVERSION_ALGORITHM_ID = "b2b-v1-air-to-air-to-unitarysystem-exact-conversion-v1"
BUILDING2BUILDING_CONVERSION_SOURCE_PATH = "building2building/pipeline/actuators.py"
BUILDING2BUILDING_CONVERSION_FUNCTION = "convert_heat_pumps_to_unitary_systems"
BUILDING2BUILDING_CONVERSION_SOURCE_SHA256 = "4e96d35f40bc9589a241e6f3a776c7dbbeac6dda220f635bbda90bc16189847d"

_FIELD_RENAMES = {
    "supply_air_fan_name": "supply_fan_name",
    "supply_air_fan_object_type": "supply_fan_object_type",
    "maximum_supply_air_temperature_from_supplemental_heater": "maximum_supply_air_temperature",
}
_FLOW_RATE_METHOD_FIELDS = {
    "cooling_supply_air_flow_rate": "cooling_supply_air_flow_rate_method",
    "heating_supply_air_flow_rate": "heating_supply_air_flow_rate_method",
    "no_load_supply_air_flow_rate": "no_load_supply_air_flow_rate_method",
}
_ASSIGNED_FIELDS = {
    "control_type": "SetPoint",
    "dehumidification_control_type": "None",
}
_REMOVED_FIELDS = ("maximum_outdoor_dry_bulb_temperature_for_supplemental_heater_operation",)
_CONTROLLED_FIELD_NAMES = frozenset(
    {
        *_FIELD_RENAMES,
        *_FIELD_RENAMES.values(),
        *_FLOW_RATE_METHOD_FIELDS,
        *_FLOW_RATE_METHOD_FIELDS.values(),
        *_ASSIGNED_FIELDS,
        *_REMOVED_FIELDS,
    }
)
_CONTROLLED_FIELD_BY_CASEFOLD = {item.casefold(): item for item in _CONTROLLED_FIELD_NAMES}
_RELEVANT_OBJECT_TYPES = (
    AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
    UNITARY_SYSTEM_OBJECT_TYPE,
    BRANCH_OBJECT_TYPE,
)
_RELEVANT_OBJECT_TYPE_BY_CASEFOLD = {item.casefold(): item for item in _RELEVANT_OBJECT_TYPES}
_MUTATION_OPERATIONS = frozenset(
    {
        "move_object",
        "rename_field",
        "assign_field",
        "remove_field",
        "rewrite_branch_reference",
    }
)


class AirToAirConversionError(ValueError):
    """Raised when exact conversion cannot be proved exhaustively."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AirToAirConversionError(f"epJSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise AirToAirConversionError(f"epJSON contains non-finite constant {value!r}")


def _finite_json_float(value: str) -> float:
    exact = Decimal(value)
    result = float(exact)
    if not math.isfinite(result) or (result == 0.0 and exact != 0):
        raise AirToAirConversionError(f"epJSON floating-point value is not finite or representable: {value!r}")
    return result


def _exact_text(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    if not value or value != value.strip():
        raise AirToAirConversionError(f"{field_name} must be a nonempty trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AirToAirConversionError(f"{field_name} contains a control character")
    return value


def _sha256(value: object, *, field_name: str) -> str:
    value = _exact_text(value, field_name=field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AirToAirConversionError(f"{field_name} must be a lowercase SHA-256 digest")
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
        raise AirToAirConversionError(f"value is not canonical JSON data: {error}") from error


def _canonical_epjson_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise AirToAirConversionError(f"converted model is not canonical JSON data: {error}") from error


def _parse_epjson_bytes(source_bytes: bytes) -> dict[str, Any]:
    if type(source_bytes) is not bytes:
        raise TypeError("source_bytes must be exact bytes")
    if source_bytes.startswith(b"\xef\xbb\xbf"):
        raise AirToAirConversionError("epJSON may not contain a UTF-8 byte-order mark")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AirToAirConversionError(f"epJSON is not UTF-8: {error}") from error
    try:
        parsed = json.loads(
            source_text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_finite_json_float,
        )
    except AirToAirConversionError:
        raise
    except json.JSONDecodeError as error:
        raise AirToAirConversionError(f"epJSON is not valid JSON: {error}") from error
    if type(parsed) is not dict:
        raise AirToAirConversionError("epJSON root must be an object")
    return parsed


def _canonical_value_json(value: Any) -> str:
    return _canonical_json(value).decode("utf-8")


def _canonical_optional_value_json(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    value = _exact_text(value, field_name=field_name)
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_finite_json_float,
        )
    except (AirToAirConversionError, json.JSONDecodeError) as error:
        raise AirToAirConversionError(f"{field_name} is not canonical JSON: {error}") from error
    if _canonical_value_json(parsed) != value:
        raise AirToAirConversionError(f"{field_name} must use canonical JSON encoding")
    return value


def _root_table_index(model: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    result: dict[str, tuple[str, Any]] = {}
    for raw_type, raw_table in model.items():
        object_type = _exact_text(raw_type, field_name="epJSON root object-table name")
        folded = object_type.casefold()
        if folded in result:
            raise AirToAirConversionError("epJSON root object-table names collide ignoring case")
        canonical_type = _RELEVANT_OBJECT_TYPE_BY_CASEFOLD.get(folded)
        if canonical_type is not None and object_type != canonical_type:
            raise AirToAirConversionError(
                f"conversion-relevant table {object_type!r} must use canonical spelling {canonical_type!r}"
            )
        result[folded] = (object_type, raw_table)
    return result


def _field_index(fields: Mapping[str, Any], *, owner: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_field_name in fields:
        field_name = _exact_text(raw_field_name, field_name=f"{owner} field name")
        folded = field_name.casefold()
        if folded in result:
            raise AirToAirConversionError(f"{owner} field names collide ignoring case")
        canonical = _CONTROLLED_FIELD_BY_CASEFOLD.get(folded)
        if canonical is not None and field_name != canonical:
            raise AirToAirConversionError(
                f"conversion-controlled field {field_name!r} on {owner} must use canonical spelling {canonical!r}"
            )
        result[folded] = field_name
    return result


def _object_table(
    root_tables: Mapping[str, tuple[str, Any]],
    object_type: str,
) -> dict[str, tuple[str, dict[str, Any]]]:
    entry = root_tables.get(object_type.casefold())
    raw_table = {} if entry is None else entry[1]
    if type(raw_table) is not dict:
        raise AirToAirConversionError(f"epJSON {object_type!r} table must be an object")
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for raw_name, raw_fields in raw_table.items():
        name = _exact_text(raw_name, field_name=f"{object_type} object name")
        if type(raw_fields) is not dict:
            raise AirToAirConversionError(f"epJSON object {object_type}/{name} must be an object")
        folded = name.casefold()
        if folded in result:
            raise AirToAirConversionError(f"epJSON {object_type!r} names collide ignoring case")
        _field_index(raw_fields, owner=f"{object_type}/{name}")
        result[folded] = (name, raw_fields)
    return result


def _branch_components(branch_name: str, branch: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_components = branch.get("components", ())
    if isinstance(raw_components, (str, bytes)) or not isinstance(raw_components, Sequence):
        raise AirToAirConversionError(f"Branch/{branch_name}.components must be an array")
    result: list[dict[str, Any]] = []
    for index, component in enumerate(raw_components):
        if type(component) is not dict:
            raise AirToAirConversionError(f"Branch/{branch_name}.components[{index}] must be an object")
        _field_index(component, owner=f"Branch/{branch_name}.components[{index}]")
        result.append(component)
    return result


@dataclass(frozen=True, slots=True)
class AirToAirFieldMutation:
    """One ordered field or ownership transition in a candidate conversion."""

    sequence: int
    operation: str
    owner_object_type: str
    owner_object_name: str
    field_path: str
    before_json: str | None
    after_json: str | None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise TypeError("conversion mutation sequence must be a nonnegative native int")
        operation = _exact_text(self.operation, field_name="conversion mutation operation")
        if operation not in _MUTATION_OPERATIONS:
            raise AirToAirConversionError(f"unsupported conversion mutation operation {operation!r}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(
            self,
            "owner_object_type",
            _exact_text(self.owner_object_type, field_name="conversion mutation owner object type"),
        )
        object.__setattr__(
            self,
            "owner_object_name",
            _exact_text(self.owner_object_name, field_name="conversion mutation owner object name"),
        )
        object.__setattr__(
            self,
            "field_path",
            _exact_text(self.field_path, field_name="conversion mutation field path"),
        )
        before = _canonical_optional_value_json(self.before_json, field_name="conversion mutation before_json")
        after = _canonical_optional_value_json(self.after_json, field_name="conversion mutation after_json")
        if operation == "remove_field":
            if before is None or after is not None:
                raise AirToAirConversionError("remove_field requires only before_json")
        elif operation == "assign_field":
            if after is None:
                raise AirToAirConversionError("assign_field requires after_json")
        elif before is None or after is None:
            raise AirToAirConversionError(f"{operation} requires before_json and after_json")
        object.__setattr__(self, "before_json", before)
        object.__setattr__(self, "after_json", after)

    @property
    def mutation_id(self) -> str:
        digest = hashlib.sha256(
            b"HOT B2B AirToAir field mutation v1\0" + _canonical_json(self.identity_payload())
        ).hexdigest()
        return f"air-to-air-mutation-{digest[:24]}"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "operation": self.operation,
            "owner": {
                "object_type": self.owner_object_type,
                "object_name": self.owner_object_name,
            },
            "field_path": self.field_path,
            "before_json": self.before_json,
            "after_json": self.after_json,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"mutation_id": self.mutation_id, **self.identity_payload()}


def _canonical_mutation(value: object, *, field_name: str) -> AirToAirFieldMutation:
    if type(value) is not AirToAirFieldMutation:
        raise TypeError(f"{field_name} must be an exact AirToAirFieldMutation")
    try:
        return AirToAirFieldMutation(
            sequence=value.sequence,
            operation=value.operation,
            owner_object_type=value.owner_object_type,
            owner_object_name=value.owner_object_name,
            field_path=value.field_path,
            before_json=value.before_json,
            after_json=value.after_json,
        )
    except AttributeError as error:
        raise AirToAirConversionError(f"{field_name} is missing canonical mutation state") from error


@dataclass(frozen=True, slots=True)
class AirToAirConversionCandidate:
    """Exact source, converted object, Branch occurrence, and mutation chain."""

    object_name: str
    source_object_sha256: str
    converted_object_sha256: str
    branch_name: str
    branch_component_index: int
    branch_before_sha256: str
    branch_after_sha256: str
    mutations: tuple[AirToAirFieldMutation, ...]
    source_object_type: str = AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE
    converted_object_type: str = UNITARY_SYSTEM_OBJECT_TYPE

    def __post_init__(self) -> None:
        object_name = _exact_text(self.object_name, field_name="conversion candidate object_name")
        object.__setattr__(self, "object_name", object_name)
        source_type = _exact_text(self.source_object_type, field_name="conversion candidate source_object_type")
        converted_type = _exact_text(
            self.converted_object_type,
            field_name="conversion candidate converted_object_type",
        )
        if source_type != AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE:
            raise AirToAirConversionError("conversion candidate has an unexpected source object type")
        if converted_type != UNITARY_SYSTEM_OBJECT_TYPE:
            raise AirToAirConversionError("conversion candidate has an unexpected converted object type")
        object.__setattr__(self, "source_object_type", source_type)
        object.__setattr__(self, "converted_object_type", converted_type)
        for field_name in (
            "source_object_sha256",
            "converted_object_sha256",
            "branch_before_sha256",
            "branch_after_sha256",
        ):
            object.__setattr__(self, field_name, _sha256(getattr(self, field_name), field_name=field_name))
        if hmac.compare_digest(self.source_object_sha256, self.converted_object_sha256):
            raise AirToAirConversionError("source and converted object hashes must differ")
        branch_name = _exact_text(self.branch_name, field_name="conversion candidate branch_name")
        object.__setattr__(self, "branch_name", branch_name)
        if type(self.branch_component_index) is not int or self.branch_component_index < 0:
            raise TypeError("branch_component_index must be a nonnegative native int")
        if hmac.compare_digest(self.branch_before_sha256, self.branch_after_sha256):
            raise AirToAirConversionError("Branch conversion must change the Branch object hash")
        if type(self.mutations) is not tuple:
            raise TypeError("conversion candidate mutations must be an exact tuple")
        mutations = tuple(
            _canonical_mutation(value, field_name=f"conversion candidate mutations[{index}]")
            for index, value in enumerate(self.mutations)
        )
        if tuple(item.sequence for item in mutations) != tuple(range(len(mutations))):
            raise AirToAirConversionError("conversion candidate mutation sequence must be contiguous from zero")
        if not mutations or mutations[0].operation != "move_object":
            raise AirToAirConversionError("conversion candidate mutation chain must begin with move_object")
        branch_mutations = [item for item in mutations if item.operation == "rewrite_branch_reference"]
        if len(branch_mutations) != 1:
            raise AirToAirConversionError("conversion candidate requires exactly one Branch-reference mutation")
        branch_mutation = branch_mutations[0]
        expected_branch_path = f"components[{self.branch_component_index}].component_object_type"
        if (
            branch_mutation.owner_object_type != BRANCH_OBJECT_TYPE
            or branch_mutation.owner_object_name != branch_name
            or branch_mutation.field_path != expected_branch_path
        ):
            raise AirToAirConversionError("conversion candidate Branch mutation ownership is inconsistent")
        mutation_ids = [item.mutation_id.casefold() for item in mutations]
        if len(mutation_ids) != len(set(mutation_ids)):
            raise AirToAirConversionError("conversion candidate mutation identities collide")
        object.__setattr__(self, "mutations", mutations)

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(
            b"HOT B2B AirToAir conversion candidate v1\0" + _canonical_json(self.identity_payload())
        ).hexdigest()
        return f"air-to-air-candidate-{digest[:24]}"

    @property
    def mutation_chain_sha256(self) -> str:
        return hashlib.sha256(
            b"HOT B2B AirToAir mutation chain v1\0" + _canonical_json([item.to_dict() for item in self.mutations])
        ).hexdigest()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "source_energyplus": {
                "object_type": self.source_object_type,
                "object_name": self.object_name,
            },
            "source_object_sha256": self.source_object_sha256,
            "converted_energyplus": {
                "object_type": self.converted_object_type,
                "object_name": self.object_name,
            },
            "converted_object_sha256": self.converted_object_sha256,
            "branch_reference": {
                "branch_name": self.branch_name,
                "component_index": self.branch_component_index,
                "before_sha256": self.branch_before_sha256,
                "after_sha256": self.branch_after_sha256,
            },
            "mutations": [item.to_dict() for item in self.mutations],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mutation_chain_sha256": self.mutation_chain_sha256,
            **self.identity_payload(),
        }


def _canonical_candidate(value: object, *, field_name: str) -> AirToAirConversionCandidate:
    if type(value) is not AirToAirConversionCandidate:
        raise TypeError(f"{field_name} must be an exact AirToAirConversionCandidate")
    try:
        return AirToAirConversionCandidate(
            object_name=value.object_name,
            source_object_sha256=value.source_object_sha256,
            converted_object_sha256=value.converted_object_sha256,
            branch_name=value.branch_name,
            branch_component_index=value.branch_component_index,
            branch_before_sha256=value.branch_before_sha256,
            branch_after_sha256=value.branch_after_sha256,
            mutations=value.mutations,
            source_object_type=value.source_object_type,
            converted_object_type=value.converted_object_type,
        )
    except AttributeError as error:
        raise AirToAirConversionError(f"{field_name} is missing canonical candidate state") from error


@dataclass(frozen=True, slots=True)
class AirToAirConversionReport:
    """Content-addressed receipt for an exhaustive exact-byte conversion."""

    source_model_sha256: str
    converted_model_sha256: str
    candidates: tuple[AirToAirConversionCandidate, ...]
    schema_version: str = AIR_TO_AIR_CONVERSION_SCHEMA_VERSION
    algorithm_id: str = AIR_TO_AIR_CONVERSION_ALGORITHM_ID
    building2building_repository_url: str = BUILDING2BUILDING_REPOSITORY_URL
    building2building_tag: str = BUILDING2BUILDING_TAG
    building2building_commit: str = BUILDING2BUILDING_COMMIT
    building2building_license_spdx: str = BUILDING2BUILDING_LICENSE_SPDX
    building2building_license_url: str = BUILDING2BUILDING_LICENSE_URL
    building2building_conversion_source_path: str = BUILDING2BUILDING_CONVERSION_SOURCE_PATH
    building2building_conversion_function: str = BUILDING2BUILDING_CONVERSION_FUNCTION
    building2building_conversion_source_sha256: str = BUILDING2BUILDING_CONVERSION_SOURCE_SHA256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_model_sha256",
            _sha256(self.source_model_sha256, field_name="source_model_sha256"),
        )
        object.__setattr__(
            self,
            "converted_model_sha256",
            _sha256(self.converted_model_sha256, field_name="converted_model_sha256"),
        )
        if hmac.compare_digest(self.source_model_sha256, self.converted_model_sha256):
            raise AirToAirConversionError("conversion must change the model hash")
        for field_name in (
            "schema_version",
            "algorithm_id",
            "building2building_repository_url",
            "building2building_tag",
            "building2building_commit",
            "building2building_license_spdx",
            "building2building_license_url",
            "building2building_conversion_source_path",
            "building2building_conversion_function",
        ):
            object.__setattr__(self, field_name, _exact_text(getattr(self, field_name), field_name=field_name))
        if self.schema_version != AIR_TO_AIR_CONVERSION_SCHEMA_VERSION:
            raise AirToAirConversionError("unexpected AirToAir conversion schema version")
        if self.algorithm_id != AIR_TO_AIR_CONVERSION_ALGORITHM_ID:
            raise AirToAirConversionError("unexpected AirToAir conversion algorithm")
        expected_pins = (
            BUILDING2BUILDING_REPOSITORY_URL,
            BUILDING2BUILDING_TAG,
            BUILDING2BUILDING_COMMIT,
            BUILDING2BUILDING_LICENSE_SPDX,
            BUILDING2BUILDING_LICENSE_URL,
        )
        if (
            self.building2building_repository_url,
            self.building2building_tag,
            self.building2building_commit,
            self.building2building_license_spdx,
            self.building2building_license_url,
        ) != expected_pins:
            raise AirToAirConversionError("AirToAir conversion must bind the pinned Building2Building source")
        if (
            self.building2building_conversion_source_path != BUILDING2BUILDING_CONVERSION_SOURCE_PATH
            or self.building2building_conversion_function != BUILDING2BUILDING_CONVERSION_FUNCTION
            or _sha256(
                self.building2building_conversion_source_sha256,
                field_name="building2building_conversion_source_sha256",
            )
            != BUILDING2BUILDING_CONVERSION_SOURCE_SHA256
        ):
            raise AirToAirConversionError("AirToAir conversion must bind the pinned upstream conversion implementation")
        if type(self.candidates) is not tuple:
            raise TypeError("conversion report candidates must be an exact tuple")
        candidates = tuple(
            _canonical_candidate(value, field_name=f"conversion report candidates[{index}]")
            for index, value in enumerate(self.candidates)
        )
        if not candidates:
            raise AirToAirConversionError("AirToAir conversion requires at least one source candidate")
        ordered = tuple(sorted(candidates, key=lambda item: item.object_name.casefold()))
        if candidates != ordered:
            raise AirToAirConversionError("conversion candidates must use canonical object-name order")
        names = [item.object_name.casefold() for item in candidates]
        if len(names) != len(set(names)):
            raise AirToAirConversionError("conversion candidate names collide ignoring case")
        object.__setattr__(self, "candidates", candidates)

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            b"HOT B2B AirToAir conversion artifact v1\0" + _canonical_json(self.identity_payload())
        ).hexdigest()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "source_model_sha256": self.source_model_sha256,
            "converted_model_sha256": self.converted_model_sha256,
            "candidate_count": len(self.candidates),
            "candidates": [item.to_dict() for item in self.candidates],
            "building2building": {
                "repository_url": self.building2building_repository_url,
                "tag": self.building2building_tag,
                "commit": self.building2building_commit,
                "license_spdx": self.building2building_license_spdx,
                "license_url": self.building2building_license_url,
                "conversion_source": {
                    "path": self.building2building_conversion_source_path,
                    "function": self.building2building_conversion_function,
                    "file_sha256": self.building2building_conversion_source_sha256,
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_sha256": self.artifact_sha256, **self.identity_payload()}


def canonical_air_to_air_conversion_report(value: object) -> AirToAirConversionReport:
    """Own a deeply revalidated exact-base copy of a conversion report."""

    if type(value) is not AirToAirConversionReport:
        raise TypeError("conversion_report must be an exact AirToAirConversionReport")
    try:
        return AirToAirConversionReport(
            source_model_sha256=value.source_model_sha256,
            converted_model_sha256=value.converted_model_sha256,
            candidates=value.candidates,
            schema_version=value.schema_version,
            algorithm_id=value.algorithm_id,
            building2building_repository_url=value.building2building_repository_url,
            building2building_tag=value.building2building_tag,
            building2building_commit=value.building2building_commit,
            building2building_license_spdx=value.building2building_license_spdx,
            building2building_license_url=value.building2building_license_url,
            building2building_conversion_source_path=value.building2building_conversion_source_path,
            building2building_conversion_function=value.building2building_conversion_function,
            building2building_conversion_source_sha256=value.building2building_conversion_source_sha256,
        )
    except AttributeError as error:
        raise AirToAirConversionError("conversion_report is missing canonical report state") from error


@dataclass(frozen=True, slots=True)
class AirToAirConversionResult:
    """Canonical converted bytes and their exact conversion receipt."""

    converted_model_bytes: bytes
    report: AirToAirConversionReport

    def __post_init__(self) -> None:
        if type(self.converted_model_bytes) is not bytes:
            raise TypeError("converted_model_bytes must be exact bytes")
        report = canonical_air_to_air_conversion_report(self.report)
        object.__setattr__(self, "report", report)
        if not hmac.compare_digest(
            hashlib.sha256(self.converted_model_bytes).hexdigest(),
            report.converted_model_sha256,
        ):
            raise AirToAirConversionError("converted_model_bytes do not match the conversion report")
        parsed = _parse_epjson_bytes(self.converted_model_bytes)
        if _canonical_epjson_bytes(parsed) != self.converted_model_bytes:
            raise AirToAirConversionError("converted_model_bytes must use canonical JSON encoding")


def _mutation(
    sequence: int,
    operation: str,
    owner_object_type: str,
    owner_object_name: str,
    field_path: str,
    before: Any,
    after: Any,
    *,
    before_present: bool = True,
    after_present: bool = True,
) -> AirToAirFieldMutation:
    return AirToAirFieldMutation(
        sequence=sequence,
        operation=operation,
        owner_object_type=owner_object_type,
        owner_object_name=owner_object_name,
        field_path=field_path,
        before_json=_canonical_value_json(before) if before_present else None,
        after_json=_canonical_value_json(after) if after_present else None,
    )


def _convert_fields(
    object_name: str,
    source_fields: Mapping[str, Any],
) -> tuple[dict[str, Any], list[AirToAirFieldMutation]]:
    owner = f"{AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE}/{object_name}"
    _field_index(source_fields, owner=owner)
    target_only_fields = {
        *_FIELD_RENAMES.values(),
        *_FLOW_RATE_METHOD_FIELDS.values(),
        "control_type",
    }
    invalid_target_owners = sorted(
        (field for field in target_only_fields if field in source_fields),
        key=str.casefold,
    )
    if invalid_target_owners:
        raise AirToAirConversionError(f"{owner} already owns conversion-target-only fields {invalid_target_owners!r}")

    converted_fields: dict[str, Any] = {}
    converted_field_owners: dict[str, str] = {}
    for field_name in sorted(source_fields, key=str.casefold):
        target_field = _FIELD_RENAMES.get(field_name, field_name)
        folded = target_field.casefold()
        previous = converted_field_owners.get(folded)
        if previous is not None:
            raise AirToAirConversionError(f"{owner} fields {previous!r} and {field_name!r} collide after conversion")
        converted_field_owners[folded] = field_name
        converted_fields[target_field] = copy.deepcopy(source_fields[field_name])

    mutations = [
        _mutation(
            0,
            "move_object",
            AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
            object_name,
            "@object_type",
            AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
            UNITARY_SYSTEM_OBJECT_TYPE,
        )
    ]
    sequence = 1
    for source_field, target_field in sorted(_FIELD_RENAMES.items()):
        if source_field not in source_fields:
            continue
        mutations.append(
            _mutation(
                sequence,
                "rename_field",
                AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
                object_name,
                f"{source_field}->{target_field}",
                source_fields[source_field],
                converted_fields[target_field],
            )
        )
        sequence += 1

    for flow_field, method_field in sorted(_FLOW_RATE_METHOD_FIELDS.items()):
        if flow_field not in converted_fields:
            continue
        converted_fields[method_field] = "SupplyAirFlowRate"
        mutations.append(
            _mutation(
                sequence,
                "assign_field",
                UNITARY_SYSTEM_OBJECT_TYPE,
                object_name,
                method_field,
                None,
                "SupplyAirFlowRate",
                before_present=False,
            )
        )
        sequence += 1

    for field_name, assigned_value in _ASSIGNED_FIELDS.items():
        before_present = field_name in converted_fields
        before = converted_fields.get(field_name)
        converted_fields[field_name] = assigned_value
        mutations.append(
            _mutation(
                sequence,
                "assign_field",
                UNITARY_SYSTEM_OBJECT_TYPE,
                object_name,
                field_name,
                before,
                assigned_value,
                before_present=before_present,
            )
        )
        sequence += 1

    for field_name in _REMOVED_FIELDS:
        if field_name not in converted_fields:
            continue
        before = converted_fields.pop(field_name)
        mutations.append(
            _mutation(
                sequence,
                "remove_field",
                AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
                object_name,
                field_name,
                before,
                None,
                after_present=False,
            )
        )
        sequence += 1
    return converted_fields, mutations


def convert_air_to_air_heat_pumps_from_json_bytes(source_bytes: bytes) -> AirToAirConversionResult:
    """Convert every exact AirToAir candidate and authenticate all mutations."""

    model = _parse_epjson_bytes(source_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    root_tables = _root_table_index(model)
    heat_pumps = _object_table(root_tables, AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE)
    if not heat_pumps:
        raise AirToAirConversionError("AirToAir conversion requires at least one source candidate")
    unitary_systems = _object_table(root_tables, UNITARY_SYSTEM_OBJECT_TYPE)
    branches = _object_table(root_tables, BRANCH_OBJECT_TYPE)

    existing_unitary_names = set(unitary_systems)
    collisions = sorted(
        (name for folded, (name, _) in heat_pumps.items() if folded in existing_unitary_names),
        key=str.casefold,
    )
    if collisions:
        raise AirToAirConversionError(
            f"AirToAir candidates collide with existing UnitarySystem names ignoring case: {collisions!r}"
        )

    references: dict[str, list[tuple[str, int]]] = {folded: [] for folded in heat_pumps}
    for branch_name, branch in sorted(branches.values(), key=lambda item: item[0].casefold()):
        for component_index, component in enumerate(_branch_components(branch_name, branch)):
            component_type = _exact_text(
                component.get("component_object_type"),
                field_name=f"Branch/{branch_name}.components[{component_index}].component_object_type",
            )
            component_type_folded = component_type.casefold()
            if component_type_folded not in {
                AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE.casefold(),
                UNITARY_SYSTEM_OBJECT_TYPE.casefold(),
            }:
                continue
            expected_component_type = (
                AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE
                if component_type_folded == AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE.casefold()
                else UNITARY_SYSTEM_OBJECT_TYPE
            )
            if component_type != expected_component_type:
                raise AirToAirConversionError(
                    f"Branch/{branch_name}.components[{component_index}] conversion-relevant type "
                    "must use canonical spelling"
                )
            component_name = _exact_text(
                component.get("component_name"),
                field_name=f"Branch/{branch_name}.components[{component_index}].component_name",
            )
            candidate = heat_pumps.get(component_name.casefold())
            if component_type == UNITARY_SYSTEM_OBJECT_TYPE:
                if candidate is not None:
                    raise AirToAirConversionError(
                        f"Branch/{branch_name}.components[{component_index}] already owns a converted-type "
                        f"reference to AirToAir candidate {component_name!r}"
                    )
                continue
            if candidate is None:
                raise AirToAirConversionError(
                    f"Branch/{branch_name}.components[{component_index}] references unknown AirToAir "
                    f"candidate {component_name!r}"
                )
            canonical_name = candidate[0]
            if component_name != canonical_name:
                raise AirToAirConversionError(
                    f"Branch/{branch_name}.components[{component_index}] must use canonical candidate name "
                    f"{canonical_name!r}"
                )
            references[component_name.casefold()].append((branch_name, component_index))

    invalid_reference_counts = {
        heat_pumps[folded][0]: len(occurrences) for folded, occurrences in references.items() if len(occurrences) != 1
    }
    if invalid_reference_counts:
        raise AirToAirConversionError(
            "every AirToAir candidate must have exactly one Branch reference; " f"counts={invalid_reference_counts!r}"
        )

    converted_model = copy.deepcopy(model)
    converted_heat_pumps = converted_model[AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE]
    converted_unitary_table = converted_model.setdefault(UNITARY_SYSTEM_OBJECT_TYPE, {})
    if type(converted_unitary_table) is not dict:  # defended above; keeps the mutation site explicit
        raise AirToAirConversionError("converted UnitarySystem table must be an object")
    converted_branch_table = converted_model.get(BRANCH_OBJECT_TYPE, {})
    if type(converted_branch_table) is not dict:  # defended above
        raise AirToAirConversionError("converted Branch table must be an object")

    candidates: list[AirToAirConversionCandidate] = []
    for object_name, source_fields in sorted(heat_pumps.values(), key=lambda item: item[0].casefold()):
        converted_fields, mutations = _convert_fields(object_name, source_fields)
        converted_unitary_table[object_name] = converted_fields
        branch_name, component_index = references[object_name.casefold()][0]
        branch = converted_branch_table[branch_name]
        branch_before_sha256 = energyplus_source_object_sha256(BRANCH_OBJECT_TYPE, branch_name, branch)
        component = _branch_components(branch_name, branch)[component_index]
        before_type = component["component_object_type"]
        component["component_object_type"] = UNITARY_SYSTEM_OBJECT_TYPE
        branch_after_sha256 = energyplus_source_object_sha256(BRANCH_OBJECT_TYPE, branch_name, branch)
        mutations.append(
            _mutation(
                len(mutations),
                "rewrite_branch_reference",
                BRANCH_OBJECT_TYPE,
                branch_name,
                f"components[{component_index}].component_object_type",
                before_type,
                UNITARY_SYSTEM_OBJECT_TYPE,
            )
        )
        del converted_heat_pumps[object_name]
        candidates.append(
            AirToAirConversionCandidate(
                object_name=object_name,
                source_object_sha256=energyplus_source_object_sha256(
                    AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE,
                    object_name,
                    source_fields,
                ),
                converted_object_sha256=energyplus_source_object_sha256(
                    UNITARY_SYSTEM_OBJECT_TYPE,
                    object_name,
                    converted_fields,
                ),
                branch_name=branch_name,
                branch_component_index=component_index,
                branch_before_sha256=branch_before_sha256,
                branch_after_sha256=branch_after_sha256,
                mutations=tuple(mutations),
            )
        )

    if converted_heat_pumps:
        raise AirToAirConversionError("AirToAir conversion left unconverted source candidates")
    del converted_model[AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE]
    converted_bytes = _canonical_epjson_bytes(converted_model)
    report = AirToAirConversionReport(
        source_model_sha256=source_sha256,
        converted_model_sha256=hashlib.sha256(converted_bytes).hexdigest(),
        candidates=tuple(candidates),
    )
    return AirToAirConversionResult(converted_bytes, report)


def verify_air_to_air_conversion_replay(
    source_bytes: bytes,
    expected_result: AirToAirConversionResult,
) -> AirToAirConversionResult:
    """Recompute conversion and reject byte, report, or mutation substitution."""

    if type(source_bytes) is not bytes:
        raise TypeError("source_bytes must be exact bytes")
    if type(expected_result) is not AirToAirConversionResult:
        raise TypeError("expected_result must be an exact AirToAirConversionResult")
    try:
        canonical_expected = AirToAirConversionResult(
            converted_model_bytes=expected_result.converted_model_bytes,
            report=expected_result.report,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise AirToAirConversionError(f"expected conversion artifact is malformed or tampered: {error}") from error
    recomputed = convert_air_to_air_heat_pumps_from_json_bytes(source_bytes)
    if recomputed != canonical_expected or not hmac.compare_digest(
        recomputed.report.artifact_sha256,
        canonical_expected.report.artifact_sha256,
    ):
        raise AirToAirConversionError("AirToAir conversion artifact does not match deterministic replay")
    return recomputed


__all__ = [
    "AIR_TO_AIR_CONVERSION_ALGORITHM_ID",
    "AIR_TO_AIR_CONVERSION_SCHEMA_VERSION",
    "AIR_TO_AIR_HEAT_PUMP_OBJECT_TYPE",
    "BRANCH_OBJECT_TYPE",
    "BUILDING2BUILDING_CONVERSION_FUNCTION",
    "BUILDING2BUILDING_CONVERSION_SOURCE_PATH",
    "BUILDING2BUILDING_CONVERSION_SOURCE_SHA256",
    "UNITARY_SYSTEM_OBJECT_TYPE",
    "AirToAirConversionCandidate",
    "AirToAirConversionError",
    "AirToAirConversionReport",
    "AirToAirConversionResult",
    "AirToAirFieldMutation",
    "canonical_air_to_air_conversion_report",
    "convert_air_to_air_heat_pumps_from_json_bytes",
    "verify_air_to_air_conversion_replay",
]
