"""Small deterministic-serialization helpers for benchmark identities.

This module deliberately uses only the Python standard library.  Benchmark
specification and result metadata must remain importable on login nodes and in
analysis environments that do not have EnergyPlus, Gymnasium, or Torch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


class _FrozenMapping(tuple):
    """Tuple-backed mapping marker preserving the empty-object distinction."""


class _FrozenSequence(tuple):
    """Tuple-backed sequence marker preserving the empty-array distinction."""


def canonical_value(value: Any, *, field: str = "value") -> Any:
    """Return a JSON-compatible value with deterministic ordering.

    Non-finite floats and non-string mapping keys are rejected because neither
    has a portable canonical JSON representation.
    """

    if isinstance(value, Enum):
        return canonical_value(value.value, field=field)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: canonical_value(getattr(value, item.name), field=f"{field}.{item.name}")
            for item in fields(value)
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} mapping keys must be strings")
            if key in normalized:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            normalized[key] = canonical_value(item, field=f"{field}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item, field=field) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonical_value(item, field=field) for item in value]
    raise TypeError(f"{field} contains unsupported value of type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to stable, whitespace-free JSON."""

    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_id(namespace: str, payload: Any, *, prefix: str, length: int = 24) -> str:
    """Return a domain-separated deterministic identifier for ``payload``."""

    if not namespace.strip():
        raise ValueError("namespace cannot be empty")
    if not prefix.strip():
        raise ValueError("prefix cannot be empty")
    if not 12 <= length <= 64:
        raise ValueError("length must be between 12 and 64 hexadecimal characters")
    digest = hashlib.sha256(namespace.encode("utf-8") + b"\x00" + canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def freeze_mapping(values: Mapping[str, Any] | None, *, field: str) -> tuple[tuple[str, Any], ...]:
    """Freeze a JSON mapping as a sorted tuple of key/value pairs."""

    if values is None:
        return ()
    canonical = canonical_value(values, field=field)
    if not isinstance(canonical, dict):
        raise TypeError(f"{field} must be a mapping")
    return tuple((key, _freeze_value(value)) for key, value in canonical.items())


def refreeze_mapping(
    values: Mapping[str, Any] | tuple[tuple[str, Any], ...], *, field: str
) -> tuple[tuple[str, Any], ...]:
    """Freeze either a regular mapping or a previously frozen mapping."""

    if isinstance(values, Mapping):
        return freeze_mapping(values, field=field)
    try:
        mapping = {key: _thaw_value(value) for key, value in values}
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be a mapping or key/value tuple") from error
    if len(mapping) != len(values):
        raise ValueError(f"{field} contains duplicate keys")
    return freeze_mapping(mapping, field=field)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenMapping((key, _freeze_value(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenSequence(_freeze_value(item) for item in value)
    return value


def thaw_mapping(values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Convert a mapping frozen by :func:`freeze_mapping` back to JSON values."""

    return {key: _thaw_value(value) for key, value in values}


def _thaw_value(value: Any) -> Any:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_value(item) for key, item in value}
    if isinstance(value, _FrozenSequence):
        return [_thaw_value(item) for item in value]
    if isinstance(value, tuple):
        # Accept legacy/unmarked frozen values produced before the marker
        # subclasses existed. Non-empty tuples are unambiguous; an old empty
        # tuple retains the historical empty-object interpretation.
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw_value(item) for key, item in value}
        return [_thaw_value(item) for item in value]
    return value


__all__ = [
    "canonical_json",
    "canonical_value",
    "content_id",
    "freeze_mapping",
    "refreeze_mapping",
    "thaw_mapping",
]
