"""Auditable, fail-closed repairs for EnergyPlus epJSON models.

The functions in this module operate on in-memory mappings and never mutate
their input.  They intentionally require enough EnergyPlus topology to justify
every inferred field; a missing or ambiguous connection is an error rather
than an invitation to guess.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

MODEL_REPAIR_REPORT_SCHEMA_VERSION = "hot-epjson-model-repair-report-v1"
MIXED_WATER_HEATER_USE_SIDE_REPAIR_ID = "infer-mixed-water-heater-use-side-nodes"
MIXED_WATER_HEATER_USE_SIDE_REPAIR_VERSION = 1
PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_ID = "restore-plant-connected-mixed-water-heater"
PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_VERSION = 1
OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_ID = "replace-officesmall-defrost-eir-with-neutral-reference"
OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_VERSION = 1

OFFICESMALL_AUDITED_DEFROST_CURVE_NAME = "HPACDefrostCAPFT"
OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME = "HPACDefrostEIRFT_Constant1"
OFFICESMALL_DEFROST_HEATING_COIL_NAMES = (
    "PSZ-AC:1 Heating Coil",
    "PSZ-AC:2 Heating Coil",
    "PSZ-AC:3 Heating Coil",
    "PSZ-AC:4 Heating Coil",
    "PSZ-AC:5 Heating Coil",
)

_DEFROST_CURVE_OBJECT_TYPE = "Curve:Biquadratic"
_DEFROST_COIL_OBJECT_TYPE = "Coil:Heating:DX:SingleSpeed"
_DEFROST_CURVE_REFERENCE_FIELD = "defrost_energy_input_ratio_function_of_temperature_curve_name"
_AUDITED_DEFROST_CURVE_FIELDS = (
    ("coefficient1_constant", 0.297145),
    ("coefficient2_x", 0.0430933),
    ("coefficient3_x_2", -0.000748766),
    ("coefficient4_y", 0.00597727),
    ("coefficient5_y_2", 0.000482112),
    ("coefficient6_x_y", -0.000956448),
    ("maximum_value_of_x", 23.88889),
    ("maximum_value_of_y", 46.11111),
    ("minimum_value_of_x", 12.77778),
    ("minimum_value_of_y", 21.11111),
)
_NEUTRAL_DEFROST_CURVE_FIELDS = (
    ("coefficient1_constant", 1.0),
    ("coefficient2_x", 0.0),
    ("coefficient3_x_2", 0.0),
    ("coefficient4_y", 0.0),
    ("coefficient5_y_2", 0.0),
    ("coefficient6_x_y", 0.0),
    ("maximum_value_of_x", 50.0),
    ("maximum_value_of_y", 50.0),
    ("minimum_value_of_x", 0.0),
    ("minimum_value_of_y", 0.0),
)

_DOE_OFFICESMALL_ARCHIVE_URL = (
    "https://www.energycodes.gov/sites/default/files/2023-10/" "ASHRAE901_OfficeSmall__STD2013.zip"
)
_DOE_OFFICESMALL_ARCHIVE_SHA256 = "2c543bf57696d68af53fbd37de2dfce41032d9fcab2fe0d44a5a86e02017e492"
_DOE_OFFICESMALL_MEMBER = "ASHRAE901_OfficeSmall_STD2013_Seattle.idf"
_ENERGYPLUS_DEFROST_REFERENCE_TAG = "v24.1.0"
_ENERGYPLUS_DEFROST_REFERENCE_COMMIT = "9d7789a3accef2ed968a6198a90b6218206340e5"
_ENERGYPLUS_DEFROST_REFERENCE_URL = (
    "https://raw.githubusercontent.com/NREL/EnergyPlus/v24.1.0/" "testfiles/UnitarySystem_SingleSpeedDX.idf"
)
_ENERGYPLUS_DEFROST_REFERENCE_FILE_SHA256 = "f761213b324af991151f420b72414c4a76c6b2adcc5416fcaa6d29a6dd926129"
_ENERGYPLUS_DEFROST_REFERENCE_CURVE_NAMES = (
    "Sys 3 Heat Pump Air Source HP Heating Coil DefrEIR-FT",
    "Sys 4 Heat Pump Air Source HP Heating Coil DefrEIR-FT",
)
_NEUTRAL_DEFROST_ASSUMPTION = (
    "Use the constant-one curves in an EnergyPlus v24.1.0 test/example model "
    "as a neutral reference-model assumption for defrost EIR. This is not manufacturer "
    "calibration and does not establish physical accuracy for the installed equipment."
)

_HEATER_OBJECT_TYPE = "WaterHeater:Mixed"
_USE_SIDE_INLET_FIELD = "use_side_inlet_node_name"
_USE_SIDE_OUTLET_FIELD = "use_side_outlet_node_name"


@dataclass(frozen=True)
class NodePair:
    """The inlet and outlet node names at one stage of a repair."""

    inlet: str | None
    outlet: str | None

    def to_json_data(self) -> dict[str, str | None]:
        return {"inlet": self.inlet, "outlet": self.outlet}


@dataclass(frozen=True)
class PlantSupplyComponentReference:
    """The exact supply-side component from which node names were inferred."""

    plant_loop_name: str
    branch_list_name: str
    branch_name: str
    component_index: int

    def to_json_data(self) -> dict[str, str | int]:
        return {
            "plant_loop_name": self.plant_loop_name,
            "branch_list_name": self.branch_list_name,
            "branch_name": self.branch_name,
            "component_index": self.component_index,
        }


@dataclass(frozen=True)
class MixedWaterHeaterUseSideRepairReport:
    """Deterministic audit record for one mixed-water-heater repair."""

    heater_name: str
    status: Literal["repaired", "already_consistent"]
    source: PlantSupplyComponentReference
    before: NodePair
    inferred: NodePair
    after: NodePair
    changed_fields: tuple[str, ...]
    schema_version: str = MODEL_REPAIR_REPORT_SCHEMA_VERSION
    repair_id: str = MIXED_WATER_HEATER_USE_SIDE_REPAIR_ID
    repair_version: int = MIXED_WATER_HEATER_USE_SIDE_REPAIR_VERSION

    def to_json_data(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable field names."""

        return {
            "schema_version": self.schema_version,
            "repair_id": self.repair_id,
            "repair_version": self.repair_version,
            "status": self.status,
            "target": {
                "object_type": _HEATER_OBJECT_TYPE,
                "object_name": self.heater_name,
            },
            "source_topology": self.source.to_json_data(),
            "nodes": {
                "before": self.before.to_json_data(),
                "inferred": self.inferred.to_json_data(),
                "after": self.after.to_json_data(),
            },
            "changed_fields": list(self.changed_fields),
        }


class MixedWaterHeaterUseSideRepairError(ValueError):
    """Structured error raised when the repair cannot be justified safely."""

    def __init__(self, code: str, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_REPAIR_REPORT_SCHEMA_VERSION,
            "repair_id": MIXED_WATER_HEATER_USE_SIDE_REPAIR_ID,
            "repair_version": MIXED_WATER_HEATER_USE_SIDE_REPAIR_VERSION,
            "status": "rejected",
            "error": {
                "code": self.code,
                "message": str(self),
                "context": copy.deepcopy(self.context),
            },
        }


@dataclass(frozen=True)
class PlantConnectedMixedWaterHeaterRepairReport:
    """Audit record for restoring one legacy plant-connected water heater."""

    heater_name: str
    plant_loop_name: str
    water_use_connection_name: str
    status: Literal["repaired", "already_consistent"]
    node_repair: MixedWaterHeaterUseSideRepairReport
    sizing_plant_name: str
    scheduled_setpoint_maximum_c: float
    loop_maximum_temperature_before_c: float
    loop_maximum_temperature_after_c: float
    removed_standalone_fields: tuple[str, ...]
    changed_fields: tuple[str, ...]
    schema_version: str = MODEL_REPAIR_REPORT_SCHEMA_VERSION
    repair_id: str = PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_ID
    repair_version: int = PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_VERSION

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repair_id": self.repair_id,
            "repair_version": self.repair_version,
            "status": self.status,
            "target": {
                "object_type": _HEATER_OBJECT_TYPE,
                "object_name": self.heater_name,
                "plant_loop_name": self.plant_loop_name,
                "water_use_connection_name": self.water_use_connection_name,
            },
            "node_repair": self.node_repair.to_json_data(),
            "sizing_plant_name": self.sizing_plant_name,
            "scheduled_setpoint_maximum_c": self.scheduled_setpoint_maximum_c,
            "loop_maximum_temperature_c": {
                "before": self.loop_maximum_temperature_before_c,
                "after": self.loop_maximum_temperature_after_c,
            },
            "removed_standalone_fields": list(self.removed_standalone_fields),
            "changed_fields": list(self.changed_fields),
        }


class PlantConnectedMixedWaterHeaterRepairError(ValueError):
    """Raised when a complete plant-connected repair is not justified."""

    def __init__(self, code: str, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_REPAIR_REPORT_SCHEMA_VERSION,
            "repair_id": PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_ID,
            "repair_version": PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_VERSION,
            "status": "rejected",
            "error": {
                "code": self.code,
                "message": str(self),
                "context": copy.deepcopy(self.context),
            },
        }


@dataclass(frozen=True)
class BiquadraticCurveSnapshot:
    """An exact, deterministically ordered epJSON biquadratic curve."""

    object_name: str
    fields: tuple[tuple[str, float], ...]

    def to_json_data(self) -> dict[str, Any]:
        return {
            "object_type": _DEFROST_CURVE_OBJECT_TYPE,
            "object_name": self.object_name,
            "fields": dict(self.fields),
        }


@dataclass(frozen=True)
class HeatingCoilDefrostCurveReference:
    """One exact heating-coil field that selects a defrost EIR curve."""

    coil_name: str
    curve_name: str

    def to_json_data(self) -> dict[str, str]:
        return {
            "object_type": _DEFROST_COIL_OBJECT_TYPE,
            "object_name": self.coil_name,
            "field": _DEFROST_CURVE_REFERENCE_FIELD,
            "curve_name": self.curve_name,
        }


@dataclass(frozen=True)
class OfficeSmallDefrostEIRCurveRepairReport:
    """Versioned audit record for the narrow OfficeSmall defrost repair."""

    status: Literal["repaired", "already_consistent"]
    before_curve: BiquadraticCurveSnapshot
    after_curve: BiquadraticCurveSnapshot
    before_references: tuple[HeatingCoilDefrostCurveReference, ...]
    after_references: tuple[HeatingCoilDefrostCurveReference, ...]
    changed_coils: tuple[str, ...]
    schema_version: str = MODEL_REPAIR_REPORT_SCHEMA_VERSION
    repair_id: str = OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_ID
    repair_version: int = OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_VERSION

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repair_id": self.repair_id,
            "repair_version": self.repair_version,
            "status": self.status,
            "before": {
                "curve": self.before_curve.to_json_data(),
                "heating_coil_references": [reference.to_json_data() for reference in self.before_references],
            },
            "after": {
                "curve": self.after_curve.to_json_data(),
                "heating_coil_references": [reference.to_json_data() for reference in self.after_references],
            },
            "changed_coils": list(self.changed_coils),
            "assumption": {
                "kind": "neutral_reference_model_assumption",
                "statement": _NEUTRAL_DEFROST_ASSUMPTION,
            },
            "lineage": {
                "doe_pnnl_reference_building": {
                    "archive_url": _DOE_OFFICESMALL_ARCHIVE_URL,
                    "archive_sha256": _DOE_OFFICESMALL_ARCHIVE_SHA256,
                    "member": _DOE_OFFICESMALL_MEMBER,
                    "member_defrost_curve_reference": "HPACCOOLEIRFT",
                    "audited_epjson_curve_name": OFFICESMALL_AUDITED_DEFROST_CURVE_NAME,
                    "scope": (
                        "Lineage of the inherited five defrost references only; "
                        "not evidence that the source curve is valid for defrost."
                    ),
                },
                "energyplus_test_example": {
                    "tag": _ENERGYPLUS_DEFROST_REFERENCE_TAG,
                    "commit": _ENERGYPLUS_DEFROST_REFERENCE_COMMIT,
                    "url": _ENERGYPLUS_DEFROST_REFERENCE_URL,
                    "member": "testfiles/UnitarySystem_SingleSpeedDX.idf",
                    "file_sha256": _ENERGYPLUS_DEFROST_REFERENCE_FILE_SHA256,
                    "curve_names": list(_ENERGYPLUS_DEFROST_REFERENCE_CURVE_NAMES),
                    "source_line_ranges": ["3385-3396", "3831-3842"],
                    "kind": "EnergyPlus test/example model",
                },
            },
        }


class OfficeSmallDefrostEIRCurveRepairError(ValueError):
    """Raised when the audited OfficeSmall signature is not exact."""

    def __init__(self, code: str, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})

    def to_json_data(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_REPAIR_REPORT_SCHEMA_VERSION,
            "repair_id": OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_ID,
            "repair_version": OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_VERSION,
            "status": "rejected",
            "error": {
                "code": self.code,
                "message": str(self),
                "context": copy.deepcopy(self.context),
            },
        }


@dataclass(frozen=True)
class _SupplyCandidate:
    source: PlantSupplyComponentReference
    nodes: NodePair


def _fail(code: str, message: str, **context: Any) -> None:
    raise MixedWaterHeaterUseSideRepairError(code, message, context=context)


def _object_collection(model: Mapping[str, Any], object_type: str) -> Mapping[str, Any]:
    raw = model.get(object_type)
    if raw is None:
        _fail(
            "missing_object_collection",
            f"epJSON model does not declare {object_type!r} objects",
            object_type=object_type,
        )
    if not isinstance(raw, Mapping):
        _fail(
            "malformed_object_collection",
            f"epJSON {object_type!r} collection must be a mapping",
            object_type=object_type,
        )
    return raw


def _casefold_name(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_name", f"{description} must be a non-empty string", value=value)
    return value.casefold()


def _resolve_named_object(
    collection: Mapping[str, Any],
    requested_name: Any,
    *,
    object_type: str,
) -> tuple[str, Mapping[str, Any]]:
    requested_key = _casefold_name(requested_name, description=f"{object_type} object name")
    matches: list[tuple[str, Any]] = []
    for declared_name, parameters in collection.items():
        declared_key = _casefold_name(declared_name, description=f"declared {object_type} object name")
        if declared_key == requested_key:
            matches.append((declared_name, parameters))

    if not matches:
        _fail(
            "object_not_found",
            f"{object_type} object {requested_name!r} was not found",
            object_type=object_type,
            object_name=requested_name,
        )
    if len(matches) != 1:
        _fail(
            "ambiguous_object_name",
            f"multiple {object_type} objects match {requested_name!r} ignoring case",
            object_type=object_type,
            object_name=requested_name,
            matches=[name for name, _ in matches],
        )

    canonical_name, parameters = matches[0]
    if not isinstance(parameters, Mapping):
        _fail(
            "malformed_object",
            f"{object_type} object {canonical_name!r} must be a mapping",
            object_type=object_type,
            object_name=canonical_name,
        )
    return canonical_name, parameters


def _validate_unique_declared_names(collection: Mapping[str, Any], *, object_type: str) -> None:
    """Reject collections that are ambiguous under EnergyPlus name matching."""

    seen: dict[str, str] = {}
    for declared_name in collection:
        key = _casefold_name(declared_name, description=f"declared {object_type} object name")
        previous = seen.get(key)
        if previous is not None:
            _fail(
                "ambiguous_object_name",
                f"multiple {object_type} objects share a name ignoring case",
                object_type=object_type,
                matches=[previous, declared_name],
            )
        seen[key] = declared_name


def _required_string(parameters: Mapping[str, Any], field_name: str, *, owner: str) -> str:
    value = parameters.get(field_name)
    if not isinstance(value, str) or not value.strip():
        _fail(
            "missing_topology_field",
            f"{owner} must declare a non-empty {field_name!r}",
            owner=owner,
            field_name=field_name,
            value=value,
        )
    return value


def _optional_node(parameters: Mapping[str, Any], field_name: str, *, owner: str) -> str | None:
    if field_name not in parameters or parameters[field_name] is None:
        return None
    value = parameters[field_name]
    if not isinstance(value, str):
        _fail(
            "invalid_node_name",
            f"{owner} field {field_name!r} must be a string when present",
            owner=owner,
            field_name=field_name,
            value=value,
        )
    return value if value.strip() else None


def _sequence(value: Any, *, description: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("malformed_topology", f"{description} must be a sequence", value=value)
    return value


def _supply_candidates(model: Mapping[str, Any], *, heater_name: str) -> tuple[_SupplyCandidate, ...]:
    """Find exact references to ``heater_name`` on PlantLoop supply branches."""

    plant_loops = _object_collection(model, "PlantLoop")
    branch_lists = _object_collection(model, "BranchList")
    branches = _object_collection(model, "Branch")
    if not plant_loops:
        _fail("missing_supply_topology", "epJSON model does not declare any PlantLoop objects")
    _validate_unique_declared_names(plant_loops, object_type="PlantLoop")
    _validate_unique_declared_names(branch_lists, object_type="BranchList")
    _validate_unique_declared_names(branches, object_type="Branch")

    target_key = heater_name.casefold()
    candidates: list[_SupplyCandidate] = []
    for raw_loop_name, raw_loop in plant_loops.items():
        if not isinstance(raw_loop, Mapping):
            _fail(
                "malformed_object",
                f"PlantLoop object {raw_loop_name!r} must be a mapping",
                object_type="PlantLoop",
                object_name=raw_loop_name,
            )
        branch_list_reference = _required_string(
            raw_loop,
            "plant_side_branch_list_name",
            owner=f"PlantLoop {raw_loop_name!r}",
        )
        branch_list_name, branch_list = _resolve_named_object(
            branch_lists,
            branch_list_reference,
            object_type="BranchList",
        )
        branch_entries = _sequence(
            branch_list.get("branches"),
            description=f"BranchList {branch_list_name!r} field 'branches'",
        )
        if not branch_entries:
            _fail(
                "missing_supply_topology",
                f"PlantLoop {raw_loop_name!r} supply BranchList {branch_list_name!r} is empty",
                plant_loop_name=raw_loop_name,
                branch_list_name=branch_list_name,
            )

        for branch_entry_index, branch_entry in enumerate(branch_entries):
            if not isinstance(branch_entry, Mapping):
                _fail(
                    "malformed_topology",
                    f"BranchList {branch_list_name!r} entry {branch_entry_index} must be a mapping",
                    branch_list_name=branch_list_name,
                    branch_entry_index=branch_entry_index,
                )
            branch_reference = _required_string(
                branch_entry,
                "branch_name",
                owner=f"BranchList {branch_list_name!r} entry {branch_entry_index}",
            )
            branch_name, branch = _resolve_named_object(
                branches,
                branch_reference,
                object_type="Branch",
            )
            components = _sequence(
                branch.get("components"),
                description=f"Branch {branch_name!r} field 'components'",
            )
            if not components:
                _fail(
                    "missing_supply_topology",
                    f"PlantLoop supply Branch {branch_name!r} has no components",
                    plant_loop_name=raw_loop_name,
                    branch_name=branch_name,
                )

            for component_index, component in enumerate(components):
                if not isinstance(component, Mapping):
                    _fail(
                        "malformed_topology",
                        f"Branch {branch_name!r} component {component_index} must be a mapping",
                        branch_name=branch_name,
                        component_index=component_index,
                    )
                component_type = _required_string(
                    component,
                    "component_object_type",
                    owner=f"Branch {branch_name!r} component {component_index}",
                )
                component_name = _required_string(
                    component,
                    "component_name",
                    owner=f"Branch {branch_name!r} component {component_index}",
                )
                if (
                    component_type.casefold() != _HEATER_OBJECT_TYPE.casefold()
                    or component_name.casefold() != target_key
                ):
                    continue

                inlet = _required_string(
                    component,
                    "component_inlet_node_name",
                    owner=f"Branch {branch_name!r} component {component_index}",
                )
                outlet = _required_string(
                    component,
                    "component_outlet_node_name",
                    owner=f"Branch {branch_name!r} component {component_index}",
                )
                candidates.append(
                    _SupplyCandidate(
                        source=PlantSupplyComponentReference(
                            plant_loop_name=raw_loop_name,
                            branch_list_name=branch_list_name,
                            branch_name=branch_name,
                            component_index=component_index,
                        ),
                        nodes=NodePair(inlet=inlet, outlet=outlet),
                    )
                )

    return tuple(candidates)


def _nodes_match(first: str, second: str) -> bool:
    return first.casefold() == second.casefold()


def repair_mixed_water_heater_use_side_nodes(
    model: Mapping[str, Any],
    *,
    heater_name: str,
) -> tuple[dict[str, Any], MixedWaterHeaterUseSideRepairReport]:
    """Infer a mixed water heater's use-side nodes from exact plant topology.

    Only an exact ``WaterHeater:Mixed`` component reference reached through a
    ``PlantLoop``'s *plant-side* (supply-side) branch list is accepted.  The
    target heater must have exactly one such reference.  Existing node names
    must either be absent or match that reference using EnergyPlus's
    case-insensitive name semantics.

    The returned model is a deep copy; ``model`` is unchanged on success and
    on every error.  A standalone tank, incomplete topology, conflicting node
    name, or multiple supply references raises
    :class:`MixedWaterHeaterUseSideRepairError` rather than being guessed.
    """

    if not isinstance(model, Mapping):
        _fail("invalid_model", "epJSON model must be a mapping", value_type=type(model).__name__)
    _casefold_name(heater_name, description="target WaterHeater:Mixed name")

    heaters = _object_collection(model, _HEATER_OBJECT_TYPE)
    canonical_heater_name, heater = _resolve_named_object(
        heaters,
        heater_name,
        object_type=_HEATER_OBJECT_TYPE,
    )
    candidates = _supply_candidates(model, heater_name=canonical_heater_name)
    if not candidates:
        _fail(
            "missing_supply_reference",
            f"{_HEATER_OBJECT_TYPE} {canonical_heater_name!r} is not referenced by any PlantLoop supply branch",
            heater_name=canonical_heater_name,
        )
    if len(candidates) != 1:
        _fail(
            "ambiguous_supply_reference",
            f"{_HEATER_OBJECT_TYPE} {canonical_heater_name!r} has {len(candidates)} PlantLoop supply references",
            heater_name=canonical_heater_name,
            references=[candidate.source.to_json_data() for candidate in candidates],
        )

    candidate = candidates[0]
    owner = f"{_HEATER_OBJECT_TYPE} {canonical_heater_name!r}"
    before = NodePair(
        inlet=_optional_node(heater, _USE_SIDE_INLET_FIELD, owner=owner),
        outlet=_optional_node(heater, _USE_SIDE_OUTLET_FIELD, owner=owner),
    )
    for field_name, existing, inferred in (
        (_USE_SIDE_INLET_FIELD, before.inlet, candidate.nodes.inlet),
        (_USE_SIDE_OUTLET_FIELD, before.outlet, candidate.nodes.outlet),
    ):
        assert inferred is not None
        if existing is not None and not _nodes_match(existing, inferred):
            _fail(
                "conflicting_node_name",
                f"{owner} field {field_name!r} conflicts with its PlantLoop supply component",
                heater_name=canonical_heater_name,
                field_name=field_name,
                existing_node_name=existing,
                inferred_node_name=inferred,
                source_topology=candidate.source.to_json_data(),
            )

    repaired_model = copy.deepcopy(dict(model))
    repaired_heater = repaired_model[_HEATER_OBJECT_TYPE][canonical_heater_name]
    changed_fields: list[str] = []
    if before.inlet is None:
        repaired_heater[_USE_SIDE_INLET_FIELD] = candidate.nodes.inlet
        changed_fields.append(_USE_SIDE_INLET_FIELD)
    if before.outlet is None:
        repaired_heater[_USE_SIDE_OUTLET_FIELD] = candidate.nodes.outlet
        changed_fields.append(_USE_SIDE_OUTLET_FIELD)

    after = NodePair(
        inlet=repaired_heater[_USE_SIDE_INLET_FIELD],
        outlet=repaired_heater[_USE_SIDE_OUTLET_FIELD],
    )
    report = MixedWaterHeaterUseSideRepairReport(
        heater_name=canonical_heater_name,
        status="repaired" if changed_fields else "already_consistent",
        source=candidate.source,
        before=before,
        inferred=candidate.nodes,
        after=after,
        changed_fields=tuple(changed_fields),
    )
    return repaired_model, report


def _connected_fail(code: str, message: str, **context: Any) -> None:
    raise PlantConnectedMixedWaterHeaterRepairError(code, message, context=context)


def _finite_number(value: Any, *, description: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _connected_fail("invalid_numeric_field", f"{description} must be numeric", value=value)
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        _connected_fail("invalid_numeric_field", f"{description} must be {qualifier}", value=value)
    return number


def _resolve_for_connected_repair(
    model: Mapping[str, Any],
    *,
    object_type: str,
    object_name: str,
) -> tuple[str, Mapping[str, Any]]:
    try:
        return _resolve_named_object(
            _object_collection(model, object_type),
            object_name,
            object_type=object_type,
        )
    except MixedWaterHeaterUseSideRepairError as error:
        _connected_fail(error.code, str(error), **error.context)


def _branch_names(
    model: Mapping[str, Any],
    *,
    branch_list_name: str,
) -> tuple[str, ...]:
    canonical_name, branch_list = _resolve_for_connected_repair(
        model,
        object_type="BranchList",
        object_name=branch_list_name,
    )
    raw_entries = branch_list.get("branches")
    if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
        _connected_fail(
            "malformed_demand_topology",
            f"BranchList {canonical_name!r} field 'branches' must be a sequence",
        )
    names: list[str] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            _connected_fail(
                "malformed_demand_topology",
                f"BranchList {canonical_name!r} entry {index} must be a mapping",
            )
        name = entry.get("branch_name")
        if not isinstance(name, str) or not name.strip():
            _connected_fail(
                "malformed_demand_topology",
                f"BranchList {canonical_name!r} entry {index} needs a branch_name",
            )
        names.append(name)
    if not names:
        _connected_fail("missing_demand_topology", f"BranchList {canonical_name!r} is empty")
    return tuple(names)


def _demand_connection_references(
    model: Mapping[str, Any],
    *,
    plant_loop: Mapping[str, Any],
    connection_name: str,
) -> tuple[tuple[str, int], ...]:
    raw_branch_list_name = plant_loop.get("demand_side_branch_list_name")
    if not isinstance(raw_branch_list_name, str) or not raw_branch_list_name.strip():
        _connected_fail(
            "missing_demand_topology",
            "PlantLoop must declare demand_side_branch_list_name",
        )
    branches = _object_collection(model, "Branch")
    target = connection_name.casefold()
    matches: list[tuple[str, int]] = []
    for branch_reference in _branch_names(model, branch_list_name=raw_branch_list_name):
        branch_name, branch = _resolve_for_connected_repair(
            model,
            object_type="Branch",
            object_name=branch_reference,
        )
        components = branch.get("components")
        if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
            _connected_fail(
                "malformed_demand_topology",
                f"Branch {branch_name!r} field 'components' must be a sequence",
            )
        for index, component in enumerate(components):
            if not isinstance(component, Mapping):
                _connected_fail(
                    "malformed_demand_topology",
                    f"Branch {branch_name!r} component {index} must be a mapping",
                )
            component_type = component.get("component_object_type")
            component_name = component.get("component_name")
            if (
                isinstance(component_type, str)
                and isinstance(component_name, str)
                and component_type.casefold() == "wateruse:connections"
                and component_name.casefold() == target
            ):
                matches.append((branch_name, index))
    # Touch the collection above to reject malformed/missing Branch sections,
    # even though object resolution performs the actual lookups.
    del branches
    return tuple(matches)


def _connected_fixture_profile(
    model: Mapping[str, Any],
    *,
    connection_name: str,
) -> tuple[float, str]:
    _, connection = _resolve_for_connected_repair(
        model,
        object_type="WaterUse:Connections",
        object_name=connection_name,
    )
    raw_connections = connection.get("connections")
    if isinstance(raw_connections, (str, bytes)) or not isinstance(raw_connections, Sequence):
        _connected_fail(
            "malformed_water_use_connection",
            f"WaterUse:Connections {connection_name!r} field 'connections' must be a sequence",
        )
    equipment_names: list[str] = []
    for index, item in enumerate(raw_connections):
        if not isinstance(item, Mapping):
            _connected_fail(
                "malformed_water_use_connection",
                f"WaterUse:Connections {connection_name!r} entry {index} must be a mapping",
            )
        value = item.get("water_use_equipment_name")
        if not isinstance(value, str) or not value.strip():
            _connected_fail(
                "malformed_water_use_connection",
                f"WaterUse:Connections {connection_name!r} entry {index} needs water_use_equipment_name",
            )
        equipment_names.append(value)
    if not equipment_names:
        _connected_fail(
            "missing_water_use_equipment",
            f"WaterUse:Connections {connection_name!r} contains no equipment",
        )

    total_peak_flow = 0.0
    schedule_name: str | None = None
    for equipment_reference in equipment_names:
        equipment_name, equipment = _resolve_for_connected_repair(
            model,
            object_type="WaterUse:Equipment",
            object_name=equipment_reference,
        )
        total_peak_flow += _finite_number(
            equipment.get("peak_flow_rate"),
            description=f"WaterUse:Equipment {equipment_name!r} peak_flow_rate",
            positive=True,
        )
        current_schedule = equipment.get("flow_rate_fraction_schedule_name")
        if not isinstance(current_schedule, str) or not current_schedule.strip():
            _connected_fail(
                "missing_fixture_schedule",
                f"WaterUse:Equipment {equipment_name!r} needs flow_rate_fraction_schedule_name",
            )
        if schedule_name is None:
            schedule_name = current_schedule
        elif schedule_name.casefold() != current_schedule.casefold():
            _connected_fail(
                "ambiguous_fixture_schedule",
                "connected WaterUse:Equipment objects use different flow schedules",
                schedules=[schedule_name, current_schedule],
            )
    assert schedule_name is not None
    return total_peak_flow, schedule_name


def _schedule_maximum(model: Mapping[str, Any], *, schedule_name: str) -> float:
    matches: list[tuple[str, str, Mapping[str, Any]]] = []
    for object_type in ("Schedule:Constant", "Schedule:Compact"):
        collection = model.get(object_type, {})
        if not isinstance(collection, Mapping):
            _connected_fail(
                "malformed_schedule_collection",
                f"epJSON {object_type!r} collection must be a mapping",
            )
        for declared_name, fields in collection.items():
            if isinstance(declared_name, str) and declared_name.casefold() == schedule_name.casefold():
                if not isinstance(fields, Mapping):
                    _connected_fail(
                        "malformed_schedule",
                        f"{object_type} {declared_name!r} must be a mapping",
                    )
                matches.append((object_type, declared_name, fields))
    if len(matches) != 1:
        _connected_fail(
            "ambiguous_setpoint_schedule",
            f"expected exactly one supported schedule named {schedule_name!r}",
            matches=[f"{object_type}:{name}" for object_type, name, _ in matches],
        )
    object_type, declared_name, fields = matches[0]
    if object_type == "Schedule:Constant":
        return _finite_number(
            fields.get("hourly_value"),
            description=f"Schedule:Constant {declared_name!r} hourly_value",
        )
    data = fields.get("data")
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        _connected_fail(
            "malformed_schedule",
            f"Schedule:Compact {declared_name!r} field 'data' must be a sequence",
        )
    values = [
        _finite_number(item.get("field"), description=f"Schedule:Compact {declared_name!r} value")
        for item in data
        if isinstance(item, Mapping)
        and isinstance(item.get("field"), (int, float))
        and not isinstance(item.get("field"), bool)
    ]
    if not values:
        _connected_fail(
            "missing_schedule_values",
            f"Schedule:Compact {declared_name!r} has no numeric values",
        )
    return max(values)


def _loop_setpoint_maximum(model: Mapping[str, Any], *, plant_loop: Mapping[str, Any]) -> float:
    setpoint_node = plant_loop.get("loop_temperature_setpoint_node_name")
    if not isinstance(setpoint_node, str) or not setpoint_node.strip():
        _connected_fail(
            "missing_loop_setpoint_node",
            "PlantLoop must declare loop_temperature_setpoint_node_name",
        )
    managers = model.get("SetpointManager:Scheduled")
    if not isinstance(managers, Mapping):
        _connected_fail(
            "missing_setpoint_manager",
            "epJSON model does not declare SetpointManager:Scheduled objects",
        )
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for manager_name, manager in managers.items():
        if not isinstance(manager, Mapping):
            _connected_fail(
                "malformed_setpoint_manager",
                f"SetpointManager:Scheduled {manager_name!r} must be a mapping",
            )
        target_node = manager.get("setpoint_node_or_nodelist_name")
        if isinstance(target_node, str) and target_node.casefold() == setpoint_node.casefold():
            matches.append((manager_name, manager))
    if len(matches) != 1:
        _connected_fail(
            "ambiguous_setpoint_manager",
            "expected exactly one scheduled setpoint manager for the PlantLoop setpoint node",
            node_name=setpoint_node,
            matches=[name for name, _ in matches],
        )
    manager_name, manager = matches[0]
    schedule_name = manager.get("schedule_name")
    if not isinstance(schedule_name, str) or not schedule_name.strip():
        _connected_fail(
            "missing_setpoint_schedule",
            f"SetpointManager:Scheduled {manager_name!r} needs schedule_name",
        )
    return _schedule_maximum(model, schedule_name=schedule_name)


def repair_plant_connected_mixed_water_heater(
    model: Mapping[str, Any],
    *,
    heater_name: str,
    plant_loop_name: str,
    water_use_connection_name: str,
) -> tuple[dict[str, Any], PlantConnectedMixedWaterHeaterRepairReport]:
    """Restore a legacy mixed water heater that is already wired to a plant.

    The repair is intentionally narrow. It accepts exactly one supply-side
    heater reference and exactly one demand-side ``WaterUse:Connections``
    reference on the requested loop. It verifies that the heater's legacy
    stand-alone draw matches the connected fixture load before removing the
    mutually exclusive stand-alone fields. The connected design flow is made
    explicit as ``Autosize`` and the loop maximum temperature is raised only
    when required by its existing scheduled setpoint.

    No pump power, head, efficiency, temperature schedule, or fixture demand
    is invented. Ambiguous or inconsistent models are rejected without
    mutating the input.
    """

    if not isinstance(model, Mapping):
        _connected_fail("invalid_model", "epJSON model must be a mapping")
    canonical_loop_name, plant_loop = _resolve_for_connected_repair(
        model,
        object_type="PlantLoop",
        object_name=plant_loop_name,
    )
    canonical_heater_name, heater = _resolve_for_connected_repair(
        model,
        object_type=_HEATER_OBJECT_TYPE,
        object_name=heater_name,
    )
    canonical_connection_name, _ = _resolve_for_connected_repair(
        model,
        object_type="WaterUse:Connections",
        object_name=water_use_connection_name,
    )

    try:
        node_model, node_report = repair_mixed_water_heater_use_side_nodes(
            model,
            heater_name=canonical_heater_name,
        )
    except MixedWaterHeaterUseSideRepairError as error:
        _connected_fail(error.code, str(error), **error.context)
    if node_report.source.plant_loop_name.casefold() != canonical_loop_name.casefold():
        _connected_fail(
            "heater_on_different_loop",
            "the heater supply reference is not on the requested PlantLoop",
            requested_loop=canonical_loop_name,
            actual_loop=node_report.source.plant_loop_name,
        )

    demand_matches = _demand_connection_references(
        model,
        plant_loop=plant_loop,
        connection_name=canonical_connection_name,
    )
    if len(demand_matches) != 1:
        _connected_fail(
            "ambiguous_demand_reference",
            "expected exactly one matching WaterUse:Connections component on the loop demand side",
            matches=[{"branch_name": branch, "component_index": index} for branch, index in demand_matches],
        )

    fixture_peak_flow, fixture_schedule = _connected_fixture_profile(
        model,
        connection_name=canonical_connection_name,
    )
    standalone_peak = heater.get("peak_use_flow_rate")
    standalone_schedule = heater.get("use_flow_rate_fraction_schedule_name")
    standalone_present = standalone_peak is not None or standalone_schedule is not None
    if standalone_present:
        if standalone_peak is None or standalone_schedule is None:
            _connected_fail(
                "incomplete_standalone_fields",
                "legacy stand-alone heater fields must either both be present or both be absent",
            )
        standalone_peak_value = _finite_number(
            standalone_peak,
            description=f"{_HEATER_OBJECT_TYPE} {canonical_heater_name!r} peak_use_flow_rate",
            positive=True,
        )
        if not math.isclose(standalone_peak_value, fixture_peak_flow, rel_tol=1e-9, abs_tol=1e-12):
            _connected_fail(
                "standalone_flow_conflict",
                "stand-alone heater flow does not match connected fixture peak flow",
                heater_peak_flow=standalone_peak_value,
                fixture_peak_flow=fixture_peak_flow,
            )
        if not isinstance(standalone_schedule, str) or standalone_schedule.casefold() != fixture_schedule.casefold():
            _connected_fail(
                "standalone_schedule_conflict",
                "stand-alone heater flow schedule does not match connected fixture schedule",
                heater_schedule=standalone_schedule,
                fixture_schedule=fixture_schedule,
            )

    design_flow = heater.get("use_side_design_flow_rate")
    if design_flow is not None and not (isinstance(design_flow, str) and design_flow.casefold() == "autosize"):
        _connected_fail(
            "conflicting_design_flow",
            "existing use_side_design_flow_rate must be absent or Autosize",
            value=design_flow,
        )
    effectiveness = heater.get("use_side_effectiveness")
    if effectiveness is not None and not math.isclose(
        _finite_number(effectiveness, description="use_side_effectiveness"),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _connected_fail(
            "conflicting_effectiveness",
            "existing use_side_effectiveness must be absent or 1.0",
            value=effectiveness,
        )

    sizing_objects = model.get("Sizing:Plant")
    if not isinstance(sizing_objects, Mapping):
        _connected_fail("missing_sizing_plant", "epJSON model does not declare Sizing:Plant objects")
    sizing_matches = [
        name
        for name, fields in sizing_objects.items()
        if isinstance(fields, Mapping)
        and isinstance(fields.get("plant_or_condenser_loop_name"), str)
        and fields["plant_or_condenser_loop_name"].casefold() == canonical_loop_name.casefold()
    ]
    if len(sizing_matches) != 1:
        _connected_fail(
            "ambiguous_sizing_plant",
            "expected exactly one Sizing:Plant object for the requested loop",
            matches=sizing_matches,
        )

    scheduled_maximum = _loop_setpoint_maximum(model, plant_loop=plant_loop)
    loop_maximum_before = _finite_number(
        plant_loop.get("maximum_loop_temperature"),
        description=f"PlantLoop {canonical_loop_name!r} maximum_loop_temperature",
    )
    loop_maximum_after = max(loop_maximum_before, scheduled_maximum)

    repaired = copy.deepcopy(node_model)
    repaired_heater = repaired[_HEATER_OBJECT_TYPE][canonical_heater_name]
    repaired_loop = repaired["PlantLoop"][canonical_loop_name]
    changed_fields = list(node_report.changed_fields)
    if design_flow is None:
        repaired_heater["use_side_design_flow_rate"] = "Autosize"
        changed_fields.append("use_side_design_flow_rate")
    if effectiveness is None:
        repaired_heater["use_side_effectiveness"] = 1.0
        changed_fields.append("use_side_effectiveness")
    removed_fields: list[str] = []
    for field_name in ("peak_use_flow_rate", "use_flow_rate_fraction_schedule_name"):
        if field_name in repaired_heater:
            repaired_heater.pop(field_name)
            removed_fields.append(field_name)
            changed_fields.append(field_name)
    if not math.isclose(loop_maximum_after, loop_maximum_before, rel_tol=0.0, abs_tol=1e-12):
        repaired_loop["maximum_loop_temperature"] = loop_maximum_after
        changed_fields.append("PlantLoop.maximum_loop_temperature")

    report = PlantConnectedMixedWaterHeaterRepairReport(
        heater_name=canonical_heater_name,
        plant_loop_name=canonical_loop_name,
        water_use_connection_name=canonical_connection_name,
        status="repaired" if changed_fields else "already_consistent",
        node_repair=node_report,
        sizing_plant_name=sizing_matches[0],
        scheduled_setpoint_maximum_c=scheduled_maximum,
        loop_maximum_temperature_before_c=loop_maximum_before,
        loop_maximum_temperature_after_c=loop_maximum_after,
        removed_standalone_fields=tuple(removed_fields),
        changed_fields=tuple(changed_fields),
    )
    return repaired, report


def _defrost_fail(code: str, message: str, **context: Any) -> None:
    raise OfficeSmallDefrostEIRCurveRepairError(code, message, context=context)


def _defrost_resolve_collection(
    model: Mapping[str, Any],
    object_type: str,
) -> tuple[str, Mapping[str, Any]]:
    matches = [
        (declared_type, collection)
        for declared_type, collection in model.items()
        if isinstance(declared_type, str) and declared_type.casefold() == object_type.casefold()
    ]
    if not matches:
        _defrost_fail(
            "missing_object_collection",
            f"epJSON model does not declare {object_type!r} objects",
            object_type=object_type,
        )
    if len(matches) != 1:
        _defrost_fail(
            "ambiguous_object_collection",
            f"multiple epJSON collections match {object_type!r} ignoring case",
            object_type=object_type,
            matches=[name for name, _ in matches],
        )
    canonical_type, collection = matches[0]
    if not isinstance(collection, Mapping):
        _defrost_fail(
            "malformed_object_collection",
            f"epJSON {canonical_type!r} collection must be a mapping",
            object_type=canonical_type,
        )
    return canonical_type, collection


def _defrost_resolve_object(
    collection: Mapping[str, Any],
    requested_name: str,
    *,
    object_type: str,
    required: bool = True,
) -> tuple[str, Mapping[str, Any]] | None:
    names: dict[str, str] = {}
    for declared_name in collection:
        if not isinstance(declared_name, str) or not declared_name.strip():
            _defrost_fail(
                "invalid_object_name",
                f"every {object_type} object name must be a non-empty string",
                object_type=object_type,
                value=declared_name,
            )
        folded = declared_name.casefold()
        if folded in names:
            _defrost_fail(
                "ambiguous_object_name",
                f"multiple {object_type} objects share a name ignoring case",
                object_type=object_type,
                matches=[names[folded], declared_name],
            )
        names[folded] = declared_name

    canonical_name = names.get(requested_name.casefold())
    if canonical_name is None:
        if not required:
            return None
        _defrost_fail(
            "object_not_found",
            f"{object_type} object {requested_name!r} was not found",
            object_type=object_type,
            object_name=requested_name,
        )
    parameters = collection[canonical_name]
    if not isinstance(parameters, Mapping):
        _defrost_fail(
            "malformed_object",
            f"{object_type} object {canonical_name!r} must be a mapping",
            object_type=object_type,
            object_name=canonical_name,
        )
    return canonical_name, parameters


def _defrost_curve_snapshot(
    object_name: str,
    fields: Mapping[str, Any],
    *,
    expected_fields: tuple[tuple[str, float], ...],
) -> BiquadraticCurveSnapshot:
    expected = dict(expected_fields)
    if set(fields) != set(expected):
        _defrost_fail(
            "curve_field_mismatch",
            f"{_DEFROST_CURVE_OBJECT_TYPE} {object_name!r} does not have the exact audited fields",
            object_name=object_name,
            expected_fields=list(expected),
            actual_fields=list(fields),
        )
    mismatches: dict[str, dict[str, Any]] = {}
    ordered: list[tuple[str, float]] = []
    for field_name, expected_value in expected_fields:
        actual_value = fields[field_name]
        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, (int, float))
            or not math.isfinite(float(actual_value))
            or actual_value != expected_value
        ):
            mismatches[field_name] = {
                "expected": expected_value,
                "actual": actual_value,
            }
        else:
            ordered.append((field_name, float(actual_value)))
    if mismatches:
        _defrost_fail(
            "curve_value_mismatch",
            f"{_DEFROST_CURVE_OBJECT_TYPE} {object_name!r} does not match the audited signature",
            object_name=object_name,
            mismatches=mismatches,
        )
    return BiquadraticCurveSnapshot(object_name=object_name, fields=tuple(ordered))


def _curve_reference_paths(
    value: Any,
    *,
    curve_name: str,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str | int, ...], ...]:
    matches: list[tuple[str | int, ...]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            matches.extend(
                _curve_reference_paths(
                    nested,
                    curve_name=curve_name,
                    path=(*path, str(key)),
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            matches.extend(
                _curve_reference_paths(
                    nested,
                    curve_name=curve_name,
                    path=(*path, index),
                )
            )
    elif isinstance(value, str) and value.casefold() == curve_name.casefold():
        matches.append(path)
    return tuple(matches)


def repair_officesmall_defrost_eir_curve(
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], OfficeSmallDefrostEIRCurveRepairReport]:
    """Replace one audited OfficeSmall defrost EIR misuse, or reject.

    This opt-in repair is intentionally not a general curve fixer. It accepts
    only the exact ``HPACDefrostCAPFT`` coefficients and bounds observed in the
    audited model and exactly five references from the named single-speed DX
    heating coils. It preserves that legacy curve, adds a constant-one curve
    taken from an EnergyPlus v24.1.0 test/example model, and rewires only those
    five defrost fields. A partially transformed or merely similar model is an
    error. The input mapping is never mutated.
    """

    if not isinstance(model, Mapping):
        raise TypeError("epJSON model must be a mapping")

    curve_type, curves = _defrost_resolve_collection(model, _DEFROST_CURVE_OBJECT_TYPE)
    bad_curve_result = _defrost_resolve_object(
        curves,
        OFFICESMALL_AUDITED_DEFROST_CURVE_NAME,
        object_type=curve_type,
    )
    assert bad_curve_result is not None
    bad_curve_name, bad_curve = bad_curve_result
    bad_curve_snapshot = _defrost_curve_snapshot(
        bad_curve_name,
        bad_curve,
        expected_fields=_AUDITED_DEFROST_CURVE_FIELDS,
    )
    replacement_result = _defrost_resolve_object(
        curves,
        OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME,
        object_type=curve_type,
        required=False,
    )

    coil_type, coils = _defrost_resolve_collection(model, _DEFROST_COIL_OBJECT_TYPE)
    resolved_coils: list[tuple[str, Mapping[str, Any]]] = []
    for requested_name in OFFICESMALL_DEFROST_HEATING_COIL_NAMES:
        result = _defrost_resolve_object(
            coils,
            requested_name,
            object_type=coil_type,
        )
        assert result is not None
        resolved_coils.append(result)

    actual_reference_values: list[str] = []
    for coil_name, coil in resolved_coils:
        curve_reference = coil.get(_DEFROST_CURVE_REFERENCE_FIELD)
        if not isinstance(curve_reference, str) or not curve_reference.strip():
            _defrost_fail(
                "invalid_heating_coil_reference",
                f"{coil_type} {coil_name!r} must declare a defrost EIR curve name",
                object_name=coil_name,
                field=_DEFROST_CURVE_REFERENCE_FIELD,
                value=curve_reference,
            )
        actual_reference_values.append(curve_reference)

    expected_paths = {(coil_type, coil_name, _DEFROST_CURVE_REFERENCE_FIELD) for coil_name, _ in resolved_coils}
    bad_paths = set(_curve_reference_paths(model, curve_name=OFFICESMALL_AUDITED_DEFROST_CURVE_NAME))
    replacement_paths = set(_curve_reference_paths(model, curve_name=OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME))
    all_bad = all(
        value.casefold() == OFFICESMALL_AUDITED_DEFROST_CURVE_NAME.casefold() for value in actual_reference_values
    )
    all_repaired = all(
        value.casefold() == OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME.casefold() for value in actual_reference_values
    )

    if all_bad and bad_paths == expected_paths and not replacement_paths:
        if replacement_result is not None:
            _defrost_fail(
                "conflicting_replacement_curve",
                "the neutral replacement curve already exists before the audited references are repaired",
                object_name=replacement_result[0],
            )
        before_curve = bad_curve_snapshot
        before_references = tuple(
            HeatingCoilDefrostCurveReference(coil_name=name, curve_name=value)
            for (name, _), value in zip(resolved_coils, actual_reference_values, strict=True)
        )
        repaired = copy.deepcopy(dict(model))
        repaired_curves = repaired[curve_type]
        repaired_curves[OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME] = dict(_NEUTRAL_DEFROST_CURVE_FIELDS)
        for coil_name, _ in resolved_coils:
            repaired[coil_type][coil_name][_DEFROST_CURVE_REFERENCE_FIELD] = OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME
        after_curve = BiquadraticCurveSnapshot(
            object_name=OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME,
            fields=_NEUTRAL_DEFROST_CURVE_FIELDS,
        )
        after_references = tuple(
            HeatingCoilDefrostCurveReference(
                coil_name=name,
                curve_name=OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME,
            )
            for name, _ in resolved_coils
        )
        report = OfficeSmallDefrostEIRCurveRepairReport(
            status="repaired",
            before_curve=before_curve,
            after_curve=after_curve,
            before_references=before_references,
            after_references=after_references,
            changed_coils=tuple(name for name, _ in resolved_coils),
        )
        return repaired, report

    if all_repaired and replacement_paths == expected_paths and not bad_paths:
        if replacement_result is None:
            _defrost_fail(
                "missing_replacement_curve",
                "the five heating coils select the neutral curve but that curve is absent",
            )
        replacement_name, replacement_curve = replacement_result
        replacement_snapshot = _defrost_curve_snapshot(
            replacement_name,
            replacement_curve,
            expected_fields=_NEUTRAL_DEFROST_CURVE_FIELDS,
        )
        references = tuple(
            HeatingCoilDefrostCurveReference(coil_name=name, curve_name=value)
            for (name, _), value in zip(resolved_coils, actual_reference_values, strict=True)
        )
        repaired = copy.deepcopy(dict(model))
        report = OfficeSmallDefrostEIRCurveRepairReport(
            status="already_consistent",
            before_curve=replacement_snapshot,
            after_curve=replacement_snapshot,
            before_references=references,
            after_references=references,
            changed_coils=(),
        )
        return repaired, report

    _defrost_fail(
        "unexpected_curve_reference_topology",
        "expected either the exact audited five references or the exact completed repair",
        expected_coils=list(OFFICESMALL_DEFROST_HEATING_COIL_NAMES),
        actual_reference_values=actual_reference_values,
        audited_curve_reference_paths=[list(path) for path in sorted(bad_paths, key=str)],
        replacement_curve_reference_paths=[list(path) for path in sorted(replacement_paths, key=str)],
    )


__all__ = [
    "BiquadraticCurveSnapshot",
    "HeatingCoilDefrostCurveReference",
    "MIXED_WATER_HEATER_USE_SIDE_REPAIR_ID",
    "MIXED_WATER_HEATER_USE_SIDE_REPAIR_VERSION",
    "MODEL_REPAIR_REPORT_SCHEMA_VERSION",
    "MixedWaterHeaterUseSideRepairError",
    "MixedWaterHeaterUseSideRepairReport",
    "NodePair",
    "OFFICESMALL_AUDITED_DEFROST_CURVE_NAME",
    "OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_ID",
    "OFFICESMALL_DEFROST_EIR_CURVE_REPAIR_VERSION",
    "OFFICESMALL_DEFROST_HEATING_COIL_NAMES",
    "OFFICESMALL_NEUTRAL_DEFROST_CURVE_NAME",
    "OfficeSmallDefrostEIRCurveRepairError",
    "OfficeSmallDefrostEIRCurveRepairReport",
    "PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_ID",
    "PLANT_CONNECTED_MIXED_WATER_HEATER_REPAIR_VERSION",
    "PlantConnectedMixedWaterHeaterRepairError",
    "PlantConnectedMixedWaterHeaterRepairReport",
    "PlantSupplyComponentReference",
    "repair_mixed_water_heater_use_side_nodes",
    "repair_officesmall_defrost_eir_curve",
    "repair_plant_connected_mixed_water_heater",
]
