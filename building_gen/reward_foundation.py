"""Versioned, simulator-independent reward definitions for HOT control.

The functions in this module operate on physical interval quantities.  They do
not read EnergyPlus handles or infer occupancy, so their numerical semantics
can be tested independently of the simulator integration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any, ClassVar


HOT_LEGACY_ENERGY_COMFORT_V1 = "hot_legacy_energy_comfort_v1"
"""Canonical ID for the objective historically called ``energy_comfort``."""

ENERGY_COMFORT_LEGACY_ALIAS = "energy_comfort"
"""Existing public alias for :data:`HOT_LEGACY_ENERGY_COMFORT_V1`."""

HOT_OCCUPIED_ENERGY_COMFORT_V2 = "hot_occupied_energy_comfort_v2"
"""ID for the occupied, additive energy--comfort objective."""


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class OccupiedEnergyComfortV2Config:
    """All dimensionless weights and physical scales used by reward v2.

    The scales are fixed denominators, not quantities inferred from the current
    batch or building.  This keeps rewards comparable across policies and makes
    every normalization explicit in experiment provenance.
    """

    energy_weight: float
    comfort_weight: float
    energy_scale_kwh_per_m2: float
    comfort_scale_person_celsius_squared_hours_per_m2: float
    comfort_min_c: float
    comfort_max_c: float

    PARAMETER_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "energy_weight",
            "comfort_weight",
            "energy_scale_kwh_per_m2",
            "comfort_scale_person_celsius_squared_hours_per_m2",
            "comfort_min_c",
            "comfort_max_c",
        }
    )

    def __post_init__(self) -> None:
        for name in ("energy_weight", "comfort_weight"):
            value = _finite_real(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "energy_scale_kwh_per_m2",
            "comfort_scale_person_celsius_squared_hours_per_m2",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name=name),
            )
        if self.energy_weight + self.comfort_weight <= 0.0:
            raise ValueError("at least one reward weight must be positive")
        comfort_min = _finite_real(self.comfort_min_c, name="comfort_min_c")
        comfort_max = _finite_real(self.comfort_max_c, name="comfort_max_c")
        if comfort_min >= comfort_max:
            raise ValueError("comfort_min_c must be smaller than comfort_max_c")
        object.__setattr__(self, "comfort_min_c", comfort_min)
        object.__setattr__(self, "comfort_max_c", comfort_max)

    @classmethod
    def from_mapping(cls, parameters: Mapping[str, Any]) -> "OccupiedEnergyComfortV2Config":
        """Validate a pinned benchmark parameter mapping."""

        if not isinstance(parameters, Mapping):
            raise TypeError("reward parameters must be a mapping")
        unknown = sorted(set(parameters) - cls.PARAMETER_NAMES)
        missing = sorted(cls.PARAMETER_NAMES - set(parameters))
        if unknown:
            raise ValueError("hot_occupied_energy_comfort_v2 has unsupported parameter(s): " + ", ".join(unknown))
        if missing:
            raise ValueError("hot_occupied_energy_comfort_v2 requires parameter(s): " + ", ".join(missing))
        return cls(**{name: parameters[name] for name in cls.PARAMETER_NAMES})

    @property
    def parameter_mapping(self) -> dict[str, float]:
        """Return the canonical serializable parameter representation."""

        return {
            "energy_weight": self.energy_weight,
            "comfort_weight": self.comfort_weight,
            "energy_scale_kwh_per_m2": self.energy_scale_kwh_per_m2,
            "comfort_scale_person_celsius_squared_hours_per_m2": (
                self.comfort_scale_person_celsius_squared_hours_per_m2
            ),
            "comfort_min_c": self.comfort_min_c,
            "comfort_max_c": self.comfort_max_c,
        }


@dataclass(frozen=True)
class ZoneComfortSample:
    """One zone's physical state during an interval.

    ``occupant_count`` is the non-negative number of occupants reported for the
    zone.  A value of zero removes the zone's temperature violation from the
    comfort term while retaining its floor area in the energy denominator.
    """

    zone_id: str
    floor_area_m2: float
    temperature_c: float
    occupant_count: float

    def __post_init__(self) -> None:
        if not isinstance(self.zone_id, str) or not self.zone_id.strip():
            raise ValueError("zone_id must be a non-empty string")
        object.__setattr__(self, "zone_id", self.zone_id.strip())
        object.__setattr__(
            self,
            "floor_area_m2",
            _positive_finite(self.floor_area_m2, name=f"{self.zone_id}.floor_area_m2"),
        )
        object.__setattr__(
            self,
            "temperature_c",
            _finite_real(self.temperature_c, name=f"{self.zone_id}.temperature_c"),
        )
        occupants = _finite_real(
            self.occupant_count,
            name=f"{self.zone_id}.occupant_count",
        )
        if occupants < 0.0:
            raise ValueError(f"{self.zone_id}.occupant_count must be non-negative")
        object.__setattr__(self, "occupant_count", occupants)


@dataclass(frozen=True)
class OccupiedEnergyComfortV2Components:
    """Auditable physical and normalized components for one interval."""

    interval_controlled_hvac_site_energy_kwh: float
    timestep_hours: float
    controlled_floor_area_m2: float
    total_occupant_count: float
    controlled_hvac_energy_intensity_kwh_per_m2: float
    comfort_exposure_person_celsius_squared_hours_per_m2: float
    normalized_energy_penalty: float
    normalized_comfort_penalty: float
    weighted_energy_penalty: float
    weighted_comfort_penalty: float
    reward: float

    @property
    def total_penalty(self) -> float:
        return self.weighted_energy_penalty + self.weighted_comfort_penalty


def _temperature_violation_celsius(temperature_c: float, *, minimum_c: float, maximum_c: float) -> float:
    return max(minimum_c - temperature_c, 0.0, temperature_c - maximum_c)


def calculate_hot_occupied_energy_comfort_v2(
    *,
    interval_controlled_hvac_site_energy_kwh: float,
    timestep_hours: float,
    zones: Iterable[ZoneComfortSample],
    config: OccupiedEnergyComfortV2Config,
) -> OccupiedEnergyComfortV2Components:
    """Calculate the v2 reward components for one control interval.

    Energy is controlled-HVAC interval site energy per square metre. Comfort is
    person-weighted squared temperature violation integrated over the interval
    and divided by controlled floor area; it is zero when no occupants are
    present. Consequently, the *undiscounted sum* of returned components over a
    fixed horizon is invariant to subdividing an interval, provided interval
    energy is divided and the physical state is unchanged. A discounted RL
    return is timestep-invariant only when its discount convention is adjusted
    consistently. The components are also invariant to splitting a zone into
    otherwise identical subzones whose areas and occupant counts sum to the
    originals.
    """

    if not isinstance(config, OccupiedEnergyComfortV2Config):
        raise TypeError("config must be an OccupiedEnergyComfortV2Config")
    energy_kwh = _finite_real(
        interval_controlled_hvac_site_energy_kwh,
        name="interval_controlled_hvac_site_energy_kwh",
    )
    if energy_kwh < 0.0:
        raise ValueError("interval_controlled_hvac_site_energy_kwh must be non-negative")
    duration_hours = _positive_finite(timestep_hours, name="timestep_hours")

    zone_samples = tuple(zones)
    if not zone_samples:
        raise ValueError("zones must contain at least one ZoneComfortSample")
    if any(not isinstance(zone, ZoneComfortSample) for zone in zone_samples):
        raise TypeError("zones must contain only ZoneComfortSample values")
    zone_ids = [zone.zone_id for zone in zone_samples]
    if len(set(zone_ids)) != len(zone_ids):
        raise ValueError("zone_id values must be unique")

    total_area_m2 = math.fsum(zone.floor_area_m2 for zone in zone_samples)
    energy_intensity = energy_kwh / total_area_m2
    total_occupants = math.fsum(zone.occupant_count for zone in zone_samples)
    comfort_occupant_sum = math.fsum(
        zone.occupant_count
        * _temperature_violation_celsius(
            zone.temperature_c,
            minimum_c=config.comfort_min_c,
            maximum_c=config.comfort_max_c,
        )
        ** 2
        for zone in zone_samples
    )
    comfort_exposure = duration_hours * comfort_occupant_sum / total_area_m2

    normalized_energy = energy_intensity / config.energy_scale_kwh_per_m2
    normalized_comfort = comfort_exposure / config.comfort_scale_person_celsius_squared_hours_per_m2
    weighted_energy = config.energy_weight * normalized_energy
    weighted_comfort = config.comfort_weight * normalized_comfort
    return OccupiedEnergyComfortV2Components(
        interval_controlled_hvac_site_energy_kwh=energy_kwh,
        timestep_hours=duration_hours,
        controlled_floor_area_m2=total_area_m2,
        total_occupant_count=total_occupants,
        controlled_hvac_energy_intensity_kwh_per_m2=energy_intensity,
        comfort_exposure_person_celsius_squared_hours_per_m2=comfort_exposure,
        normalized_energy_penalty=normalized_energy,
        normalized_comfort_penalty=normalized_comfort,
        weighted_energy_penalty=weighted_energy,
        weighted_comfort_penalty=weighted_comfort,
        reward=-(weighted_energy + weighted_comfort),
    )


__all__ = [
    "ENERGY_COMFORT_LEGACY_ALIAS",
    "HOT_LEGACY_ENERGY_COMFORT_V1",
    "HOT_OCCUPIED_ENERGY_COMFORT_V2",
    "OccupiedEnergyComfortV2Components",
    "OccupiedEnergyComfortV2Config",
    "ZoneComfortSample",
    "calculate_hot_occupied_energy_comfort_v2",
]
