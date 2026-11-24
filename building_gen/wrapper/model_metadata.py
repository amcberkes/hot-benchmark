"""Dependency-light metadata checks for EnergyPlus epJSON models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_epjson(path: str | Path) -> dict[str, Any]:
    """Load an epJSON object and fail clearly on a non-object root."""

    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"EnergyPlus model not found: {model_path}")
    with model_path.open("r", encoding="utf-8") as model_file:
        model = json.load(model_file)
    if not isinstance(model, dict):
        raise ValueError(f"EnergyPlus model must contain a JSON object: {model_path}")
    return model


def timesteps_per_hour(model: Mapping[str, Any]) -> int:
    """Return the unique declared EnergyPlus zone timesteps per hour."""

    timestep_objects = model.get("Timestep")
    if not isinstance(timestep_objects, Mapping) or not timestep_objects:
        raise ValueError("epJSON model does not declare a Timestep object")

    declared: set[int] = set()
    for object_name, parameters in timestep_objects.items():
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Timestep object {object_name!r} must be a mapping")
        raw_value = parameters.get("number_of_timesteps_per_hour")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(
                f"Timestep object {object_name!r} has invalid "
                f"number_of_timesteps_per_hour={raw_value!r}"
            )
        numeric_value = float(raw_value)
        value = int(numeric_value)
        if numeric_value != value or value <= 0 or 60 % value != 0:
            raise ValueError(
                f"Timestep object {object_name!r} must divide 60 minutes exactly; got {raw_value!r}"
            )
        declared.add(value)

    if len(declared) != 1:
        raise ValueError(f"epJSON model declares conflicting timestep values: {sorted(declared)}")
    return declared.pop()


__all__ = ["load_epjson", "timesteps_per_hour"]
