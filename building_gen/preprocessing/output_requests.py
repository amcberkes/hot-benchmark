"""Deterministic, capability-aware EnergyPlus output requests for HOT.

The EnergyPlus API exposes variables and meters independently of whether they
are written to an output file.  These requests therefore serve the narrower
purpose of producing auditable per-timestep artifacts for HOT accounting and
control evaluation.  The builder deliberately avoids speculative requests:
fuel, district-energy, occupancy, thermostat, and ideal-loads outputs are only
requested when the epJSON contains evidence that EnergyPlus can provide them.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


OUTPUT_REQUEST_SCHEMA_VERSION = "hot-output-requests-v1"

_METER_SECTION = "Output:Meter"
_METER_FILE_ONLY_SECTION = "Output:Meter:MeterFileOnly"
_VARIABLE_SECTION = "Output:Variable"
_REQUEST_SECTIONS = (_METER_SECTION, _METER_FILE_ONLY_SECTION, _VARIABLE_SECTION)

_VALID_REPORTING_FREQUENCIES = frozenset(
    {
        "annual",
        "daily",
        "detailed",
        "environment",
        "hourly",
        "monthly",
        "runperiod",
        "timestep",
    }
)

# These are the meter names represented by HOT_ENERGY_RESULT_SCHEMA_V1.  A
# known HOT meter in MeterFileOnly is either promoted to Output:Meter when the
# detected model supports it or removed when it does not.  Unknown, user-owned
# MeterFileOnly requests are preserved.
_KNOWN_HOT_METERS = frozenset(
    {
        "Electricity:Facility",
        "NaturalGas:Facility",
        "DistrictHeatingWater:Facility",
        "DistrictHeatingSteam:Facility",
        "DistrictCooling:Facility",
        "Electricity:HVAC",
        "NaturalGas:HVAC",
        "Heating:Electricity",
        "Cooling:Electricity",
        "Fans:Electricity",
        "Pumps:Electricity",
        "InteriorLights:Electricity",
        "InteriorEquipment:Electricity",
        "Heating:NaturalGas",
    }
)

# EnergyPlus renamed the legacy ``Gas`` resource to ``NaturalGas``.  Treat
# these aliases as managed requests so stale IDFs cannot retain both spellings
# or request a no-longer-valid legacy name.
_LEGACY_HOT_METER_ALIASES = frozenset({"Gas:Facility", "Gas:HVAC", "Heating:Gas"})
_MANAGED_METER_NAMES_CASEFOLDED = frozenset(
    name.casefold() for name in (*_KNOWN_HOT_METERS, *_LEGACY_HOT_METER_ALIASES)
)


class OutputRequestValidationError(ValueError):
    """Raised before mutation when existing output requests are malformed."""


@dataclass(frozen=True)
class ModelOutputCapabilities:
    """Model features that justify conditional HOT output requests."""

    zones: bool
    people: bool
    thermostats: bool
    ideal_loads: bool
    interior_lights: bool
    interior_electric_equipment: bool
    electric_heating: bool
    electric_cooling: bool
    fans: bool
    pumps: bool
    natural_gas_facility: bool
    natural_gas_hvac: bool
    district_heating_water: bool
    district_heating_steam: bool
    district_cooling: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class OutputRequestRecord:
    """One named epJSON request, represented in a JSON-serializable form."""

    section: str
    object_name: str
    fields: tuple[tuple[str, str], ...]

    @property
    def signature(self) -> tuple[str, ...]:
        fields = dict(self.fields)
        if self.section in {_METER_SECTION, _METER_FILE_ONLY_SECTION}:
            return (
                "meter",
                fields["key_name"].casefold(),
                fields["reporting_frequency"].casefold(),
            )
        return (
            "variable",
            fields["key_value"].casefold(),
            fields["variable_name"].casefold(),
            fields["reporting_frequency"].casefold(),
            fields.get("schedule_name", "").casefold(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "object_name": self.object_name,
            "fields": dict(self.fields),
        }


@dataclass(frozen=True)
class DuplicateRequest:
    """A semantic request that occurred under more than one epJSON name."""

    signature: tuple[str, ...]
    occurrences: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": list(self.signature),
            "occurrences": [{"section": section, "object_name": name} for section, name in self.occurrences],
        }


@dataclass(frozen=True)
class OutputRequestDiff:
    """Exact before/after changes made by the output-request builder."""

    added: tuple[OutputRequestRecord, ...]
    removed: tuple[OutputRequestRecord, ...]
    unchanged: tuple[OutputRequestRecord, ...]
    duplicate_requests: tuple[DuplicateRequest, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": [record.to_dict() for record in self.added],
            "removed": [record.to_dict() for record in self.removed],
            "unchanged": [record.to_dict() for record in self.unchanged],
            "duplicate_requests": [duplicate.to_dict() for duplicate in self.duplicate_requests],
        }


@dataclass(frozen=True)
class OutputRequestReport:
    """Auditable summary returned by planning and applying output requests."""

    schema_version: str
    capabilities: ModelOutputCapabilities
    before_counts: tuple[tuple[str, int], ...]
    after_counts: tuple[tuple[str, int], ...]
    request_reasons: tuple[tuple[str, str], ...]
    diff: OutputRequestDiff

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capabilities": self.capabilities.to_dict(),
            "before_counts": dict(self.before_counts),
            "after_counts": dict(self.after_counts),
            "request_reasons": dict(self.request_reasons),
            "diff": self.diff.to_dict(),
        }


@dataclass(frozen=True)
class HotOutputRequestPlan:
    """A validated, deterministic replacement for the managed sections."""

    sections: tuple[tuple[str, tuple[OutputRequestRecord, ...]], ...]
    report: OutputRequestReport

    def section_data(self, section: str) -> dict[str, dict[str, str]]:
        for section_name, records in self.sections:
            if section_name == section:
                return {record.object_name: dict(record.fields) for record in records}
        raise KeyError(section)


def _is_object_section(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(item, Mapping) for item in value.values())


def _object_types(model: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            (name for name, objects in model.items() if _is_object_section(objects) and objects),
            key=str.casefold,
        )
    )


def _has_type(types: Sequence[str], *prefixes: str) -> bool:
    folded_prefixes = tuple(prefix.casefold() for prefix in prefixes)
    return any(object_type.casefold().startswith(folded_prefixes) for object_type in types)


def _has_exact_type(types: Sequence[str], *names: str) -> bool:
    folded_types = {object_type.casefold() for object_type in types}
    return any(name.casefold() in folded_types for name in names)


def _iter_object_fields(model: Mapping[str, Any]):
    for object_type in _object_types(model):
        objects = model[object_type]
        for object_name in sorted(objects, key=str.casefold):
            fields = objects[object_name]
            yield object_type, object_name, fields


def _uses_fuel(
    model: Mapping[str, Any],
    fuel: str,
    *,
    type_prefixes: Sequence[str] | None = None,
) -> bool:
    expected = fuel.casefold().replace(" ", "")
    folded_prefixes = tuple(prefix.casefold() for prefix in type_prefixes or ())
    for object_type, _, fields in _iter_object_fields(model):
        if folded_prefixes and not object_type.casefold().startswith(folded_prefixes):
            continue
        if object_type.casefold() == "gasequipment" and expected == "naturalgas":
            return True
        for field_name, value in fields.items():
            if "fuel_type" not in str(field_name).casefold() or not isinstance(value, str):
                continue
            if value.casefold().replace(" ", "") == expected:
                return True
    return False


def detect_output_capabilities(model: Mapping[str, Any]) -> ModelOutputCapabilities:
    """Infer only capabilities that can be established from epJSON objects."""

    if not isinstance(model, Mapping):
        raise OutputRequestValidationError("epJSON root must be a mapping")
    types = _object_types(model)
    natural_gas_facility = _uses_fuel(model, "NaturalGas")
    natural_gas_hvac = _uses_fuel(
        model,
        "NaturalGas",
        type_prefixes=(
            "AirLoopHVAC:",
            "Boiler:",
            "Coil:Heating:",
            "Furnace:",
            "ZoneHVAC:",
        ),
    )
    return ModelOutputCapabilities(
        zones=_has_exact_type(types, "Zone"),
        people=_has_exact_type(types, "People"),
        thermostats=_has_type(types, "ZoneControl:Thermostat", "ThermostatSetpoint:"),
        ideal_loads=_has_type(types, "ZoneHVAC:IdealLoadsAirSystem"),
        interior_lights=_has_exact_type(types, "Lights"),
        interior_electric_equipment=_has_exact_type(types, "ElectricEquipment"),
        electric_heating=_has_type(
            types,
            "Coil:Heating:Electric",
            "Coil:Heating:DX",
            "ZoneHVAC:Baseboard:Convective:Electric",
            "ZoneHVAC:Baseboard:RadiantConvective:Electric",
        )
        or _uses_fuel(
            model,
            "Electricity",
            type_prefixes=("Boiler:", "Coil:Heating:", "ZoneHVAC:"),
        ),
        electric_cooling=_has_type(
            types,
            "Chiller:Electric",
            "Coil:Cooling:DX",
            "Coil:Cooling:WaterToAirHeatPump",
            "CoolingTower:",
            "EvaporativeCooler:",
            "ZoneHVAC:PackagedTerminalAirConditioner",
        ),
        fans=_has_type(types, "Fan:"),
        pumps=_has_type(types, "Pump:"),
        natural_gas_facility=natural_gas_facility,
        natural_gas_hvac=natural_gas_hvac,
        district_heating_water=_has_type(types, "DistrictHeating:Water"),
        district_heating_steam=_has_type(types, "DistrictHeating:Steam"),
        district_cooling=_has_type(types, "DistrictCooling"),
    )


def _validate_string_field(section: str, object_name: str, fields: Mapping[str, Any], field: str) -> str:
    value = fields.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OutputRequestValidationError(f"{section}[{object_name!r}].{field} must be a non-empty string")
    return value.strip()


def _validated_record(section: str, object_name: Any, fields: Any) -> OutputRequestRecord:
    if not isinstance(object_name, str) or not object_name.strip():
        raise OutputRequestValidationError(f"{section} object names must be non-empty strings")
    if not isinstance(fields, Mapping):
        raise OutputRequestValidationError(f"{section}[{object_name!r}] must be a mapping")

    if section in {_METER_SECTION, _METER_FILE_ONLY_SECTION}:
        allowed = {"key_name", "reporting_frequency"}
        required = ("key_name", "reporting_frequency")
    else:
        allowed = {"key_value", "variable_name", "reporting_frequency", "schedule_name"}
        required = ("key_value", "variable_name", "reporting_frequency")

    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise OutputRequestValidationError(f"{section}[{object_name!r}] has unsupported fields: {', '.join(unknown)}")
    normalized = {field: _validate_string_field(section, object_name, fields, field) for field in required}
    if "schedule_name" in fields:
        normalized["schedule_name"] = _validate_string_field(section, object_name, fields, "schedule_name")
    if normalized["reporting_frequency"].casefold() not in _VALID_REPORTING_FREQUENCIES:
        raise OutputRequestValidationError(
            f"{section}[{object_name!r}].reporting_frequency has unsupported value "
            f"{normalized['reporting_frequency']!r}"
        )
    return OutputRequestRecord(
        section=section,
        object_name=object_name.strip(),
        fields=tuple(sorted(normalized.items())),
    )


def _existing_records(model: Mapping[str, Any]) -> tuple[OutputRequestRecord, ...]:
    records: list[OutputRequestRecord] = []
    for section in _REQUEST_SECTIONS:
        objects = model.get(section, {})
        if not isinstance(objects, Mapping):
            raise OutputRequestValidationError(f"{section} must be a mapping when present")
        for object_name in sorted(objects, key=lambda value: str(value).casefold()):
            records.append(_validated_record(section, object_name, objects[object_name]))
    return tuple(records)


def _request_key(prefix: str, *parts: str) -> str:
    normalized = [
        "".join(character.casefold() if character.isalnum() else "_" for character in part).strip("_") for part in parts
    ]
    return "hot__" + prefix + "__" + "__".join(normalized)


def _meter_record(name: str) -> OutputRequestRecord:
    return OutputRequestRecord(
        section=_METER_SECTION,
        object_name=_request_key("meter", name, "timestep"),
        fields=(("key_name", name), ("reporting_frequency", "Timestep")),
    )


def _variable_record(key: str, name: str) -> OutputRequestRecord:
    return OutputRequestRecord(
        section=_VARIABLE_SECTION,
        object_name=_request_key("variable", name, key, "timestep"),
        fields=(("key_value", key), ("reporting_frequency", "Timestep"), ("variable_name", name)),
    )


def _desired_records(
    capabilities: ModelOutputCapabilities,
) -> tuple[tuple[OutputRequestRecord, ...], tuple[tuple[str, str], ...]]:
    meters: dict[str, str] = {
        "Electricity:Facility": "required facility electricity accounting",
        "Electricity:HVAC": "required HOT HVAC electricity accounting and control reward",
    }
    if capabilities.interior_lights:
        meters["InteriorLights:Electricity"] = "Lights objects detected"
    if capabilities.interior_electric_equipment:
        meters["InteriorEquipment:Electricity"] = "ElectricEquipment objects detected"
    if capabilities.electric_heating:
        meters["Heating:Electricity"] = "electric heating equipment detected"
    if capabilities.electric_cooling:
        meters["Cooling:Electricity"] = "electric cooling equipment detected"
    if capabilities.fans:
        meters["Fans:Electricity"] = "Fan objects detected"
    if capabilities.pumps:
        meters["Pumps:Electricity"] = "Pump objects detected"
    if capabilities.natural_gas_facility:
        meters["NaturalGas:Facility"] = "NaturalGas fuel use detected"
    if capabilities.natural_gas_hvac:
        meters["NaturalGas:HVAC"] = "NaturalGas HVAC equipment detected"
        meters["Heating:NaturalGas"] = "NaturalGas HVAC heating equipment detected"
    if capabilities.district_heating_water:
        meters["DistrictHeatingWater:Facility"] = "DistrictHeating:Water objects detected"
    if capabilities.district_heating_steam:
        meters["DistrictHeatingSteam:Facility"] = "DistrictHeating:Steam objects detected"
    if capabilities.district_cooling:
        meters["DistrictCooling:Facility"] = "DistrictCooling objects detected"

    variables: dict[tuple[str, str], str] = {
        ("Environment", "Site Outdoor Air Drybulb Temperature"): "required HOT weather observation",
        ("Environment", "Site Outdoor Air Relative Humidity"): "required HOT weather observation",
    }
    if capabilities.zones:
        variables[("*", "Zone Mean Air Temperature")] = "required HOT zone-temperature observation"
    if capabilities.people:
        variables[("*", "Zone People Occupant Count")] = "People objects detected; required by reward v2"
    if capabilities.thermostats:
        variables[("*", "Zone Thermostat Heating Setpoint Temperature")] = "thermostat objects detected"
        variables[("*", "Zone Thermostat Cooling Setpoint Temperature")] = "thermostat objects detected"
    if capabilities.ideal_loads:
        variables[("*", "Zone Ideal Loads Zone Sensible Heating Energy")] = "ideal-loads equipment detected"
        variables[("*", "Zone Ideal Loads Zone Sensible Cooling Energy")] = "ideal-loads equipment detected"

    records = [_meter_record(name) for name in sorted(meters, key=str.casefold)]
    records.extend(
        _variable_record(key, name) for key, name in sorted(variables, key=lambda item: (item[1].casefold(), item[0]))
    )
    reasons = {"|".join(_meter_record(name).signature): reason for name, reason in meters.items()}
    reasons.update(
        {"|".join(_variable_record(key, name).signature): reason for (key, name), reason in variables.items()}
    )
    return tuple(records), tuple(sorted(reasons.items()))


def _deduplicated_meter_file_only(
    existing: Sequence[OutputRequestRecord],
) -> tuple[OutputRequestRecord, ...]:
    retained: dict[tuple[str, ...], OutputRequestRecord] = {}
    for record in existing:
        if record.section != _METER_FILE_ONLY_SECTION:
            continue
        fields = dict(record.fields)
        if fields["key_name"].casefold() in _MANAGED_METER_NAMES_CASEFOLDED:
            continue
        retained.setdefault(record.signature, record)
    return tuple(sorted(retained.values(), key=lambda record: record.object_name.casefold()))


def _record_identity(record: OutputRequestRecord) -> tuple[Any, ...]:
    return record.section, record.object_name, record.fields


def _duplicates(existing: Sequence[OutputRequestRecord]) -> tuple[DuplicateRequest, ...]:
    groups: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for record in existing:
        groups.setdefault(record.signature, []).append((record.section, record.object_name))
    return tuple(
        DuplicateRequest(signature, tuple(sorted(occurrences)))
        for signature, occurrences in sorted(groups.items())
        if len(occurrences) > 1
    )


def plan_hot_output_requests(model: Mapping[str, Any]) -> HotOutputRequestPlan:
    """Validate a model and return its deterministic HOT output-request plan.

    Planning does not mutate ``model``.  Malformed existing managed requests
    raise :class:`OutputRequestValidationError` before a plan is returned.
    """

    if not isinstance(model, Mapping):
        raise OutputRequestValidationError("epJSON root must be a mapping")
    existing = _existing_records(model)
    capabilities = detect_output_capabilities(model)
    desired, reasons = _desired_records(capabilities)
    meter_file_only = _deduplicated_meter_file_only(existing)
    final_records = tuple(
        sorted(
            (*desired, *meter_file_only),
            key=lambda record: (record.section.casefold(), record.object_name.casefold()),
        )
    )

    before_by_identity = {_record_identity(record): record for record in existing}
    after_by_identity = {_record_identity(record): record for record in final_records}
    diff = OutputRequestDiff(
        added=tuple(after_by_identity[key] for key in sorted(after_by_identity.keys() - before_by_identity.keys())),
        removed=tuple(before_by_identity[key] for key in sorted(before_by_identity.keys() - after_by_identity.keys())),
        unchanged=tuple(
            before_by_identity[key] for key in sorted(before_by_identity.keys() & after_by_identity.keys())
        ),
        duplicate_requests=_duplicates(existing),
    )
    before_counts = tuple(
        (section, sum(record.section == section for record in existing)) for section in _REQUEST_SECTIONS
    )
    after_counts = tuple(
        (section, sum(record.section == section for record in final_records)) for section in _REQUEST_SECTIONS
    )
    report = OutputRequestReport(
        schema_version=OUTPUT_REQUEST_SCHEMA_VERSION,
        capabilities=capabilities,
        before_counts=before_counts,
        after_counts=after_counts,
        request_reasons=reasons,
        diff=diff,
    )
    records_by_section = tuple(
        (section, tuple(record for record in final_records if record.section == section))
        for section in _REQUEST_SECTIONS
    )
    return HotOutputRequestPlan(records_by_section, report)


def apply_hot_output_requests(
    model: MutableMapping[str, Any],
    *,
    plan: HotOutputRequestPlan | None = None,
) -> OutputRequestReport:
    """Apply a validated plan in place and return its structured diff report."""

    if not isinstance(model, MutableMapping):
        raise OutputRequestValidationError("epJSON root must be a mutable mapping")
    resolved_plan = plan or plan_hot_output_requests(model)
    # Do not accept a stale/cross-model plan.  Re-planning is cheap and makes
    # the mutation boundary fail closed.
    if plan is not None and plan != plan_hot_output_requests(model):
        raise OutputRequestValidationError("output-request plan does not match the current epJSON model")
    for section in _REQUEST_SECTIONS:
        section_data = resolved_plan.section_data(section)
        if section_data:
            model[section] = section_data
        else:
            model.pop(section, None)
    return resolved_plan.report


def configure_hot_output_requests_file(
    epjson_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> OutputRequestReport:
    """Reconcile a file's requests and atomically write valid JSON.

    When ``output_path`` is omitted, the input file is replaced atomically.
    The input is never written if parsing or validation fails.
    """

    source = Path(epjson_path)
    target = Path(output_path) if output_path is not None else source
    with source.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if not isinstance(model, MutableMapping):
        raise OutputRequestValidationError("epJSON root must be a JSON object")
    report = apply_hot_output_requests(model)

    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return report


def copy_with_hot_output_requests(model: Mapping[str, Any]) -> tuple[dict[str, Any], OutputRequestReport]:
    """Return a reconciled deep copy and report without mutating the input."""

    copied = copy.deepcopy(model)
    report = apply_hot_output_requests(copied)
    return copied, report
