"""Dependency-light geometry helpers for EnergyPlus epJSON models.

The control benchmark normalizes energy by the floor area of the zones that
the policy controls.  This module computes that area directly from the model
instead of relying on archetype-name lookup tables or x/y bounding boxes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


_DIRECT_ZONE_EQUIPMENT_TYPES = (
    "ZoneHVAC:IdealLoadsAirSystem",
    "ZoneHVAC:PackagedTerminalAirConditioner",
    "ZoneHVAC:FourPipeFanCoil",
    "ZoneHVAC:UnitHeater",
    "ZoneHVAC:UnitVentilator",
    "ZoneHVAC:PackagedTerminalHeatPump",
    "ZoneHVAC:WindowAirConditioner",
    "ZoneHVAC:Baseboard:Convective:Electric",
    "ZoneHVAC:Baseboard:Convective:Water",
)


def _objects(model: Mapping[str, Any], object_type: str) -> Mapping[str, Any]:
    objects = model.get(object_type, {})
    if not isinstance(objects, Mapping):
        raise ValueError(f"epJSON {object_type!r} collection must be a mapping")
    return objects


def _finite_float(value: Any, *, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a finite number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{description} must be a finite number, got {value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite, got {value!r}")
    return result


def polygon_area_3d(vertices: Sequence[Mapping[str, Any]]) -> float:
    """Return the area in m2 of an EnergyPlus planar polygon in 3D.

    EnergyPlus requires detailed surfaces to be planar.  Newell's vector-area
    formula is valid for horizontal, vertical, tilted, and concave planar
    polygons, unlike an axis-aligned x/y bounding box.
    """

    if isinstance(vertices, (str, bytes)) or not isinstance(vertices, Sequence):
        raise ValueError("surface vertices must be a sequence")
    if len(vertices) < 3:
        raise ValueError("a surface needs at least three vertices")

    coordinates: list[tuple[float, float, float]] = []
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, Mapping):
            raise ValueError(f"vertex {index} must be a mapping")
        coordinates.append(
            (
                _finite_float(
                    vertex.get("vertex_x_coordinate"),
                    description=f"vertex {index} x coordinate",
                ),
                _finite_float(
                    vertex.get("vertex_y_coordinate"),
                    description=f"vertex {index} y coordinate",
                ),
                _finite_float(
                    vertex.get("vertex_z_coordinate"),
                    description=f"vertex {index} z coordinate",
                ),
            )
        )

    area_x = 0.0
    area_y = 0.0
    area_z = 0.0
    for index, point in enumerate(coordinates):
        next_point = coordinates[(index + 1) % len(coordinates)]
        area_x += point[1] * next_point[2] - point[2] * next_point[1]
        area_y += point[2] * next_point[0] - point[0] * next_point[2]
        area_z += point[0] * next_point[1] - point[1] * next_point[0]

    area = 0.5 * math.sqrt(area_x**2 + area_y**2 + area_z**2)
    if area <= 0.0:
        raise ValueError("surface polygon has zero area")
    return area


def _zone_lookup(model: Mapping[str, Any]) -> dict[str, str]:
    zones = _objects(model, "Zone")
    if not zones:
        raise ValueError("epJSON model does not declare any Zone objects")

    lookup: dict[str, str] = {}
    for zone_name in zones:
        if not isinstance(zone_name, str) or not zone_name.strip():
            raise ValueError(f"invalid EnergyPlus zone name: {zone_name!r}")
        key = zone_name.casefold()
        if key in lookup and lookup[key] != zone_name:
            raise ValueError(
                "EnergyPlus zone names must be unique ignoring case: "
                f"{lookup[key]!r} and {zone_name!r}"
            )
        lookup[key] = zone_name
    return lookup


def _zone_list_members(model: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    zone_lookup = _zone_lookup(model)
    result: dict[str, tuple[str, ...]] = {}
    for list_name, parameters in _objects(model, "ZoneList").items():
        if not isinstance(parameters, Mapping):
            raise ValueError(f"ZoneList object {list_name!r} must be a mapping")

        raw_members: list[Any] = []
        zones = parameters.get("zones", [])
        if zones:
            if isinstance(zones, (str, bytes)) or not isinstance(zones, Sequence):
                raise ValueError(f"ZoneList {list_name!r} zones must be a sequence")
            for entry in zones:
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"ZoneList {list_name!r} contains a non-mapping zone entry"
                    )
                raw_members.append(entry.get("zone_name"))
        raw_members.extend(
            value
            for key, value in parameters.items()
            if key.startswith("zone_") and key.endswith("_name")
        )

        members: list[str] = []
        for raw_name in raw_members:
            if not isinstance(raw_name, str) or raw_name.casefold() not in zone_lookup:
                raise ValueError(
                    f"ZoneList {list_name!r} references unknown zone {raw_name!r}"
                )
            canonical_name = zone_lookup[raw_name.casefold()]
            if canonical_name not in members:
                members.append(canonical_name)
        result[str(list_name).casefold()] = tuple(members)
    return result


def controlled_zone_names(model: Mapping[str, Any]) -> tuple[str, ...]:
    """Return model zones connected to HVAC equipment or a thermostat.

    The returned names follow the order of the model's ``Zone`` collection.
    Unconditioned zones such as attics are therefore excluded from the area
    used by the controller unless the caller explicitly selects them.
    """

    zone_lookup = _zone_lookup(model)
    zone_lists = _zone_list_members(model)
    selected: set[str] = set()

    def add_reference(raw_name: Any) -> None:
        if not isinstance(raw_name, str) or not raw_name.strip():
            return
        key = raw_name.casefold()
        if key in zone_lookup:
            selected.add(zone_lookup[key])
        elif key in zone_lists:
            selected.update(zone_lists[key])

    for parameters in _objects(model, "ZoneHVAC:EquipmentConnections").values():
        if isinstance(parameters, Mapping):
            add_reference(parameters.get("zone_name"))

    for equipment_type in _DIRECT_ZONE_EQUIPMENT_TYPES:
        for parameters in _objects(model, equipment_type).values():
            if isinstance(parameters, Mapping):
                add_reference(
                    parameters.get("zone_name")
                    or parameters.get("zone_or_zonelist_name")
                )

    for parameters in _objects(model, "ZoneControl:Thermostat").values():
        if isinstance(parameters, Mapping):
            add_reference(parameters.get("zone_or_zonelist_name"))

    return tuple(name for name in _objects(model, "Zone") if name in selected)


def _selected_zone_names(
    model: Mapping[str, Any], zone_names: Iterable[str] | None
) -> tuple[str, ...]:
    zone_lookup = _zone_lookup(model)
    if zone_names is None:
        selected = tuple(zone_lookup.values())
    else:
        selected_list: list[str] = []
        for raw_name in zone_names:
            if not isinstance(raw_name, str) or raw_name.casefold() not in zone_lookup:
                raise ValueError(f"unknown EnergyPlus zone {raw_name!r}")
            canonical_name = zone_lookup[raw_name.casefold()]
            if canonical_name not in selected_list:
                selected_list.append(canonical_name)
        selected = tuple(selected_list)
    if not selected:
        raise ValueError("at least one zone is required to compute floor area")
    return selected


def _zone_multipliers(model: Mapping[str, Any]) -> dict[str, float]:
    zones = _objects(model, "Zone")
    multipliers: dict[str, float] = {}
    for zone_name, parameters in zones.items():
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Zone object {zone_name!r} must be a mapping")
        multiplier = _finite_float(
            parameters.get("multiplier", 1.0),
            description=f"Zone {zone_name!r} multiplier",
        )
        if multiplier <= 0.0:
            raise ValueError(f"Zone {zone_name!r} multiplier must be positive")
        multipliers[zone_name] = multiplier

    zone_lists = _zone_list_members(model)
    grouped_zones: set[str] = set()
    for group_name, parameters in _objects(model, "ZoneGroup").items():
        if not isinstance(parameters, Mapping):
            raise ValueError(f"ZoneGroup object {group_name!r} must be a mapping")
        list_name = parameters.get("zone_list_name")
        if not isinstance(list_name, str) or list_name.casefold() not in zone_lists:
            raise ValueError(
                f"ZoneGroup {group_name!r} references unknown ZoneList {list_name!r}"
            )
        group_multiplier = _finite_float(
            parameters.get("zone_list_multiplier", 1.0),
            description=f"ZoneGroup {group_name!r} multiplier",
        )
        if group_multiplier <= 0.0:
            raise ValueError(f"ZoneGroup {group_name!r} multiplier must be positive")
        for zone_name in zone_lists[list_name.casefold()]:
            if zone_name in grouped_zones:
                raise ValueError(
                    f"zone {zone_name!r} belongs to more than one ZoneGroup"
                )
            grouped_zones.add(zone_name)
            multipliers[zone_name] *= group_multiplier
    return multipliers


def zone_floor_areas_m2(
    model: Mapping[str, Any], *, zone_names: Iterable[str] | None = None
) -> dict[str, float]:
    """Compute floor area by zone from ``BuildingSurface:Detailed`` polygons."""

    selected = _selected_zone_names(model, zone_names)
    selected_set = set(selected)
    raw_areas = {zone_name: 0.0 for zone_name in selected}
    zone_lookup = _zone_lookup(model)

    for surface_name, parameters in _objects(
        model, "BuildingSurface:Detailed"
    ).items():
        if not isinstance(parameters, Mapping):
            raise ValueError(
                f"BuildingSurface:Detailed object {surface_name!r} must be a mapping"
            )
        if str(parameters.get("surface_type", "")).casefold() != "floor":
            continue
        raw_zone_name = parameters.get("zone_name")
        if not isinstance(raw_zone_name, str):
            raise ValueError(f"floor surface {surface_name!r} has no zone_name")
        zone_name = zone_lookup.get(raw_zone_name.casefold())
        if zone_name is None:
            raise ValueError(
                f"floor surface {surface_name!r} references unknown zone {raw_zone_name!r}"
            )
        if zone_name not in selected_set:
            continue
        try:
            raw_areas[zone_name] += polygon_area_3d(parameters.get("vertices", []))
        except ValueError as error:
            raise ValueError(f"invalid floor surface {surface_name!r}: {error}") from error

    missing = [zone_name for zone_name, area in raw_areas.items() if area <= 0.0]
    if missing:
        raise ValueError(f"selected zones have no positive floor surface area: {missing}")

    multipliers = _zone_multipliers(model)
    return {
        zone_name: raw_areas[zone_name] * multipliers[zone_name]
        for zone_name in selected
    }


def conditioned_floor_area_m2(
    model: Mapping[str, Any], *, zone_names: Iterable[str] | None = None
) -> float:
    """Return the summed floor area of the model's controlled zones."""

    if zone_names is None:
        zone_names = controlled_zone_names(model)
    return math.fsum(zone_floor_areas_m2(model, zone_names=zone_names).values())


def resolve_conditioned_floor_area_m2(
    model: Mapping[str, Any],
    *,
    zone_names: Iterable[str],
    supplied_floor_area_m2: Any = None,
    relative_tolerance: float = 1e-3,
) -> float:
    """Compute the canonical area and validate an optional caller value.

    A small tolerance permits harmless rounding (for example, 511.2 instead of
    511.1574 m2), while rejecting stale archetype constants and gross areas.
    """

    computed = conditioned_floor_area_m2(model, zone_names=zone_names)
    if supplied_floor_area_m2 is None:
        return computed
    supplied = _finite_float(
        supplied_floor_area_m2, description="supplied floor_area_m2"
    )
    if supplied <= 0.0:
        raise ValueError("supplied floor_area_m2 must be positive")
    if not math.isclose(
        supplied, computed, rel_tol=relative_tolerance, abs_tol=1e-6
    ):
        raise ValueError(
            "supplied floor_area_m2 does not match the floor polygons of the "
            f"controlled zones: supplied={supplied:.6g}, computed={computed:.6g}"
        )
    return computed


__all__ = [
    "conditioned_floor_area_m2",
    "controlled_zone_names",
    "polygon_area_3d",
    "resolve_conditioned_floor_area_m2",
    "zone_floor_areas_m2",
]
