"""Deterministic Gym action transforms for control-capability manifests.

The policy surface is always a normalized ``Box([0, 1])`` whose dimension is
the manifest's ordered command count.  Expansion produces the complete
physical EnergyPlus endpoint vector, including fixed endpoints such as the
Building2Building VAV cooling setpoint pinned at 40 degrees Celsius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from gymnasium.spaces import Box

from building_gen.control.capabilities import (
    ActionBindingKind,
    ActionCommand,
    ActionCommandUnit,
    ControlCapabilityManifest,
    PhysicalUnit,
    _enum,
    _exact_text,
    _finite_float,
)


class ActionExpansionError(ValueError):
    """Raised when a normalized action cannot be expanded unambiguously."""


@dataclass(frozen=True, slots=True)
class PhysicalEndpointCommand:
    """One fully resolved value for an EnergyPlus actuator endpoint."""

    endpoint_id: str
    component_type: str
    control_type: str
    component_name: str
    unit: PhysicalUnit
    value: float

    def __post_init__(self) -> None:
        for field_name in (
            "endpoint_id",
            "component_type",
            "control_type",
            "component_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _exact_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "unit", _enum(self.unit, PhysicalUnit, field_name="unit"))
        object.__setattr__(self, "value", _finite_float(self.value, field_name="value"))

    @property
    def energyplus_triple(self) -> tuple[str, str, str]:
        return (self.component_type, self.control_type, self.component_name)


@dataclass(frozen=True, slots=True)
class ExpandedControlAction:
    """Normalized policy action plus complete physical endpoint commands."""

    normalized_values: tuple[float, ...]
    command_values: tuple[float, ...]
    endpoint_commands: tuple[PhysicalEndpointCommand, ...]

    @property
    def full_endpoint_values(self) -> tuple[float, ...]:
        return tuple(item.value for item in self.endpoint_commands)

    @property
    def endpoint_values_by_id(self) -> dict[str, float]:
        return {item.endpoint_id: item.value for item in self.endpoint_commands}


def _normalized_array(action: Sequence[float] | np.ndarray, *, dimension: int) -> np.ndarray:
    if isinstance(action, (str, bytes)):
        raise ActionExpansionError("action must be a one-dimensional numeric vector")
    try:
        raw = np.asarray(action, dtype=object)
    except Exception as error:
        raise ActionExpansionError(f"action could not be converted to an array: {error}") from error
    if any(isinstance(value, (bool, np.bool_)) for value in raw.flat):
        raise ActionExpansionError("action must contain only real numeric values, not booleans")
    if raw.ndim != 1 or raw.shape != (dimension,):
        raise ActionExpansionError(f"action must have shape ({dimension},), got {raw.shape}")
    numeric_types = (int, float, np.integer, np.floating)
    if any(not isinstance(value, numeric_types) for value in raw.flat):
        raise ActionExpansionError("action must contain only real numeric values")
    try:
        values = raw.astype(np.float64, copy=False)
    except Exception as error:
        raise ActionExpansionError(f"action could not be converted to finite float64 values: {error}") from error
    if not np.all(np.isfinite(values)):
        raise ActionExpansionError("action values must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ActionExpansionError("normalized action values must lie in [0, 1]")
    return values


def _physical_command_value(command: ActionCommand, normalized_value: float) -> float:
    if not math.isfinite(normalized_value) or not 0.0 <= normalized_value <= 1.0:
        raise ActionExpansionError("normalized command value must be finite and lie in [0, 1]")
    if normalized_value == 0.0:
        return command.lower_bound
    if normalized_value == 1.0:
        return command.upper_bound
    try:
        value = math.fsum(
            (
                (1.0 - normalized_value) * command.lower_bound,
                normalized_value * command.upper_bound,
            )
        )
    except OverflowError as error:
        raise ActionExpansionError("physical command interpolation produced a non-finite result") from error
    if not math.isfinite(value):
        raise ActionExpansionError("physical command interpolation produced a non-finite result")
    if not command.lower_bound <= value <= command.upper_bound:
        raise ActionExpansionError("physical command interpolation escaped its finite bounds")
    return value


@dataclass(frozen=True, slots=True)
class ManifestActionTransform:
    """Expand normalized policy commands using one authenticated capability."""

    manifest: ControlCapabilityManifest
    normalized_space: Box = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.manifest) is not ControlCapabilityManifest:
            raise TypeError("manifest must be an exact ControlCapabilityManifest")
        manifest = ControlCapabilityManifest.from_dict(ControlCapabilityManifest.to_dict(self.manifest))
        object.__setattr__(self, "manifest", manifest)
        dimension = manifest.agent_command_dim
        object.__setattr__(
            self,
            "normalized_space",
            Box(low=0.0, high=1.0, shape=(dimension,), dtype=np.float32),
        )

    @property
    def command_dim(self) -> int:
        return self.manifest.agent_command_dim

    def expand(self, action: Sequence[float] | np.ndarray) -> ExpandedControlAction:
        normalized = _normalized_array(action, dimension=self.command_dim)
        command_values = tuple(
            _physical_command_value(command, float(value))
            for command, value in zip(self.manifest.agent_commands, normalized)
        )
        endpoint_values: dict[str, float] = {}

        for command, value in zip(self.manifest.agent_commands, command_values):
            if command.transform_id == "direct_v1":
                if len(command.target_endpoint_ids) != 1:
                    raise ActionExpansionError("direct_v1 commands must resolve to exactly one endpoint")
                endpoint_id = command.target_endpoint_ids[0]
                if endpoint_id in endpoint_values:
                    raise ActionExpansionError(f"endpoint {endpoint_id!r} is assigned by multiple commands")
                endpoint_values[endpoint_id] = value
                continue

            if command.transform_id == "hot_heating_plus_offset_v1":
                if (
                    self.manifest.task_action_binding.kind is not ActionBindingKind.HOT_THERMOSTAT_OFFSET
                    or command.unit is not ActionCommandUnit.DELTA_CELSIUS
                    or len(command.target_endpoint_ids) != 2
                ):
                    raise ActionExpansionError("invalid HOT thermostat offset command")
                heating_id, cooling_id = command.target_endpoint_ids
                heating_value = endpoint_values.get(heating_id)
                if heating_value is None:
                    raise ActionExpansionError("HOT thermostat offset must follow its heating command")
                if cooling_id in endpoint_values:
                    raise ActionExpansionError(f"endpoint {cooling_id!r} is assigned by multiple commands")
                cooling_value = heating_value + value
                if not math.isfinite(cooling_value):
                    raise ActionExpansionError("HOT thermostat offset produced a non-finite endpoint value")
                endpoint_values[cooling_id] = cooling_value
                continue

            raise ActionExpansionError(f"unsupported action transform {command.transform_id!r}")

        for fixed in self.manifest.task_action_binding.fixed_endpoints:
            if fixed.endpoint_id in endpoint_values:
                raise ActionExpansionError(f"endpoint {fixed.endpoint_id!r} is both commanded and fixed")
            endpoint_values[fixed.endpoint_id] = fixed.value

        expected_ids = {item.endpoint_id for item in self.manifest.full_actuators}
        actual_ids = set(endpoint_values)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ActionExpansionError(
                f"expanded action does not cover the physical interface (missing={missing!r}, extra={extra!r})"
            )

        endpoint_commands: list[PhysicalEndpointCommand] = []
        for actuator in self.manifest.full_actuators:
            value = endpoint_values[actuator.endpoint_id]
            if not math.isfinite(value):
                raise ActionExpansionError(f"expanded endpoint {actuator.endpoint_id!r} is not finite")
            if not actuator.lower_bound <= value <= actuator.upper_bound:
                raise ActionExpansionError(
                    f"expanded endpoint {actuator.endpoint_id!r} lies outside its physical bounds"
                )
            endpoint_commands.append(
                PhysicalEndpointCommand(
                    endpoint_id=actuator.endpoint_id,
                    component_type=actuator.component_type,
                    control_type=actuator.control_type,
                    component_name=actuator.component_name,
                    unit=actuator.unit,
                    value=value,
                )
            )

        return ExpandedControlAction(
            normalized_values=tuple(float(value) for value in normalized),
            command_values=command_values,
            endpoint_commands=tuple(endpoint_commands),
        )


__all__ = [
    "ActionExpansionError",
    "ExpandedControlAction",
    "ManifestActionTransform",
    "PhysicalEndpointCommand",
]
