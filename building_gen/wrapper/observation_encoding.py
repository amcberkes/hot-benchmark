"""Versioned, physically defined policy-observation encoding.

EnergyPlus meters report energy accumulated over the current reporting
interval in joules.  Feeding those values directly to a policy makes an
otherwise identical operating point depend on the simulation timestep and the
building size.  HOT observation schema v2 instead represents energy as average
power per conditioned floor area and maps every policy feature to ``[0, 1]``.

The raw EnergyPlus observation remains available in the Gym ``info`` mapping;
clipping here is therefore a policy-input convention, not a data-cleaning
operation.
"""

from __future__ import annotations

import math

from building_gen.accounting import joules_to_kwh


OBSERVATION_SCHEMA_ID = "hot_observation_v2"
ZONE_TEMPERATURE_RANGE_C = (10.0, 40.0)
OUTDOOR_TEMPERATURE_RANGE_C = (-30.0, 50.0)

# 0.05 kW/m2 = 50 W/m2.  The saturating transform maps this reference load to
# 0.5 while retaining a finite, monotonic representation of larger loads.
ENERGY_POWER_DENSITY_SCALE_KW_PER_M2 = 0.05


def _finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def clipped_affine(value: float, lower: float, upper: float) -> float:
    """Map a finite scalar from explicit physical bounds to ``[0, 1]``."""

    value = _finite_real(value, name="value")
    lower = _finite_real(lower, name="lower")
    upper = _finite_real(upper, name="upper")
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    return min(1.0, max(0.0, (value - lower) / (upper - lower)))


def average_power_density_kw_per_m2(
    interval_energy_joules: float,
    *,
    floor_area_m2: float,
    timesteps_per_hour: int,
) -> float:
    """Convert interval energy to average kW per conditioned square metre."""

    floor_area_m2 = _finite_real(floor_area_m2, name="floor_area_m2")
    if floor_area_m2 <= 0.0:
        raise ValueError("floor_area_m2 must be positive")
    if isinstance(timesteps_per_hour, bool) or not isinstance(timesteps_per_hour, int):
        raise TypeError("timesteps_per_hour must be an integer")
    if timesteps_per_hour <= 0:
        raise ValueError("timesteps_per_hour must be positive")
    interval_kwh = joules_to_kwh(interval_energy_joules)
    average_power_kw = interval_kwh * timesteps_per_hour
    return average_power_kw / floor_area_m2


def saturating_nonnegative(value: float, *, scale: float) -> float:
    """Map a nonnegative finite scalar monotonically to ``[0, 1)``."""

    value = _finite_real(value, name="value")
    scale = _finite_real(scale, name="scale")
    if value < 0.0:
        raise ValueError("value must be nonnegative")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return value / (value + scale)


def encode_observation_value(
    key: str,
    value: float,
    *,
    floor_area_m2: float,
    timesteps_per_hour: int,
    energy_scale_kw_per_m2: float = ENERGY_POWER_DENSITY_SCALE_KW_PER_M2,
) -> float:
    """Encode one named HOT observation according to schema v2."""

    if key.endswith("_temperature") and not key.startswith("outdoor"):
        return clipped_affine(value, *ZONE_TEMPERATURE_RANGE_C)
    if key == "outdoor_temperature":
        return clipped_affine(value, *OUTDOOR_TEMPERATURE_RANGE_C)
    if key in {"outdoor_humidity", "hour_of_day", "day_of_week", "day_of_year"}:
        return clipped_affine(value, 0.0, 1.0)
    if key in {"hvac_electricity", "hvac_gas"}:
        power_density = average_power_density_kw_per_m2(
            value,
            floor_area_m2=floor_area_m2,
            timesteps_per_hour=timesteps_per_hour,
        )
        return saturating_nonnegative(power_density, scale=energy_scale_kw_per_m2)
    raise KeyError(f"observation key is not defined by {OBSERVATION_SCHEMA_ID}: {key!r}")


__all__ = [
    "ENERGY_POWER_DENSITY_SCALE_KW_PER_M2",
    "OBSERVATION_SCHEMA_ID",
    "OUTDOOR_TEMPERATURE_RANGE_C",
    "ZONE_TEMPERATURE_RANGE_C",
    "average_power_density_kw_per_m2",
    "clipped_affine",
    "encode_observation_value",
    "saturating_nonnegative",
]
