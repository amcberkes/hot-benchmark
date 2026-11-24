"""Authenticated, algorithm-neutral scientific training bundles.

This module seals already-written training outputs into one portable manifest.
The top-level manifest is created only after all artifacts and every
preregistered EnergyPlus episode have been independently validated.  The
resulting checkpoint SHA256 is the identity that evaluation must carry.

Checkpoint contents remain an algorithm-adapter responsibility: this module
authenticates opaque bytes and their linkage digest but does not import Torch
or deserialize a model.  Episode ``executed_model_sha256`` and
``source_file_sha256`` fields are recorder assertions only because the model
and original diagnostics-file bytes are not copied into this bundle.  The
embedded diagnostics payload itself is independently re-evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from calendar import monthrange
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from building_gen.benchmark._canonical import (
    canonical_value,
    refreeze_mapping,
    thaw_mapping,
)
from building_gen.benchmark.energyplus_diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    DiagnosticsAllowlist,
    ScientificDiagnosticsError,
    diagnostics_allowlist_from_mapping,
    energyplus_diagnostics_from_mapping,
    require_scientific_diagnostics,
)
from building_gen.benchmark.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    ResolvedRun,
    SoftwareProvenance,
)
from building_gen.benchmark.specs import CalendarWindowSpec, TaskSpec
from building_gen.benchmark.training_evidence import (
    TRAINING_EPISODE_EVIDENCE_SCHEMA_VERSION,
    TRAINING_EVIDENCE_MANIFEST_FILENAME,
    TRAINING_EVIDENCE_MANIFEST_SCHEMA_VERSION,
    ExpectedTrainingEpisode,
    TrainingEpisodeKey,
)
from building_gen.benchmark.warmup_preconditioning import (
    WARMUP_PRECONDITIONING_FILENAME,
    WarmupPreconditioningEvidenceError,
    validate_warmup_preconditioning_payload,
)

TRAINING_BUNDLE_SCHEMA_VERSION = "hot-training-bundle/1.0.0"
TRAINING_BUNDLE_FILENAME = "training_bundle.json"
TRAINING_CONTRACT_SCHEMA_VERSION = "hot-training-contract/1.0.0"
FINAL_CHECKPOINT_SELECTION_RULE_TEMPLATE = "final checkpoint after exactly {completed_transitions} training transitions"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EPISODES_DIRECTORY_NAME = "episodes"
_REQUIRED_DEPENDENCIES = frozenset({"numpy", "torch", "gymnasium", "tensorboard", "PyYAML"})
_SUMMARY_CALENDAR_FIELDS = frozenset(
    {
        "schema_version",
        "strategy_id",
        "schedule_seed",
        "episode_index",
        "calendar_block",
        "month_visit_index",
        "day_cycle",
        "child_seed",
        "calendar_year",
        "start_date",
        "end_date",
        "window",
        "task_name",
        "task_id",
    }
)
_EPISODE_SHA_ASSERTIONS = {
    "executed_model_sha256": "recorder assertion; executed model bytes are not bundled",
    "source_file_sha256": ("recorder assertion; original diagnostics and warmup evidence file bytes are not bundled"),
}
_ARTIFACT_MEDIA_TYPES = {
    "run_config": {"application/json"},
    "training_summary": {"application/json"},
    "final_checkpoint": {"application/octet-stream"},
    "update_metrics": {"application/json"},
    "diagnostics_policy": {"application/json"},
    "training_evidence_manifest": {"application/json"},
    "resolved_experiment": {"application/json"},
    "resolved_trainer": {"application/json"},
}


class ScientificTrainingBundleError(RuntimeError):
    """Raised when a scientific training bundle is absent or inconsistent."""


def final_checkpoint_selection_rule(completed_transitions: int) -> str:
    """Return the sole truthful selection rule for a sealed final checkpoint."""

    if (
        isinstance(completed_transitions, bool)
        or not isinstance(completed_transitions, int)
        or completed_transitions <= 0
    ):
        raise ValueError("completed_transitions must be a positive integer")
    return FINAL_CHECKPOINT_SELECTION_RULE_TEMPLATE.format(
        completed_transitions=completed_transitions,
    )


@dataclass(frozen=True, slots=True)
class ExpectedTrainingSummaryEpisode:
    """Preregistered, non-metric identity of one summary episode record."""

    episode_index: int
    environment_index: int
    completion_global_step: int
    length: int
    training_calendar: tuple[tuple[str, Any], ...] | Mapping[str, Any]
    terminated: bool = True
    truncated: bool = False

    def __post_init__(self) -> None:
        for name in ("episode_index", "environment_index"):
            _nonnegative_int(getattr(self, name), field=name)
        _positive_int(self.completion_global_step, field="completion_global_step")
        _positive_int(self.length, field="length")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise TypeError("terminated and truncated must be booleans")
        if not self.terminated or self.truncated:
            raise ValueError("scientific training summary episodes must terminate without truncation")
        calendar = refreeze_mapping(self.training_calendar, field="training summary calendar")
        if not calendar:
            raise ValueError("training summary calendar cannot be empty")
        object.__setattr__(self, "training_calendar", calendar)

    def expected_fields(self) -> dict[str, Any]:
        payload = {
            "episode_index": self.episode_index,
            "environment_index": self.environment_index,
            "completion_global_step": self.completion_global_step,
            "length": self.length,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "training_calendar": thaw_mapping(self.training_calendar),
        }
        return payload


@dataclass(frozen=True, slots=True)
class ExpectedTrainingUpdate:
    """Expected accounting boundary for one algorithm update."""

    iteration: int
    global_step: int
    episodes_completed: int

    def __post_init__(self) -> None:
        _positive_int(self.iteration, field="iteration")
        _positive_int(self.global_step, field="global_step")
        _nonnegative_int(self.episodes_completed, field="episodes_completed")

    def to_dict(self) -> dict[str, int]:
        return {
            "iteration": self.iteration,
            "global_step": self.global_step,
            "episodes_completed": self.episodes_completed,
        }


@dataclass(frozen=True, slots=True)
class ExpectedTrainingContract:
    """Independent scientific expectations for algorithm-neutral outputs.

    Loss values and episode returns are observed rather than preregistered, so
    the contract fixes their exact field schemas and accounting boundaries and
    requires every observed value to be finite.
    """

    run_config_schema_id: str
    training_summary_schema_id: str
    update_metrics_schema_id: str
    run_config: tuple[tuple[str, Any], ...] | Mapping[str, Any]
    requested_transitions: int
    completed_transitions: int
    iteration_count: int
    summary_episodes: tuple[ExpectedTrainingSummaryEpisode, ...]
    update_boundaries: tuple[ExpectedTrainingUpdate, ...]
    update_record_fields: tuple[str, ...]
    summary_update_metrics_key: str
    summary_evaluation: tuple[tuple[str, Any], ...] | Mapping[str, Any]
    update_aggregation: tuple[tuple[str, Any], ...] | Mapping[str, Any]
    resolved_experiment: tuple[tuple[str, Any], ...] | Mapping[str, Any]
    resolved_trainer: tuple[tuple[str, Any], ...] | Mapping[str, Any]
    schema_version: str = TRAINING_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported training contract schema {self.schema_version!r}")
        for name in (
            "run_config_schema_id",
            "training_summary_schema_id",
            "update_metrics_schema_id",
            "summary_update_metrics_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty, trimmed string")
        requested = _positive_int(self.requested_transitions, field="requested_transitions")
        completed = _positive_int(self.completed_transitions, field="completed_transitions")
        if completed != requested:
            raise ValueError("scientific training must complete exactly its requested transition budget")
        iterations = _positive_int(self.iteration_count, field="iteration_count")
        episodes = tuple(self.summary_episodes)
        boundaries = tuple(self.update_boundaries)
        if not episodes or any(not isinstance(item, ExpectedTrainingSummaryEpisode) for item in episodes):
            raise TypeError("summary_episodes must be a non-empty tuple of ExpectedTrainingSummaryEpisode")
        if len(boundaries) != iterations or any(not isinstance(item, ExpectedTrainingUpdate) for item in boundaries):
            raise ValueError("update_boundaries must contain exactly one ExpectedTrainingUpdate per iteration")
        if tuple(item.episode_index for item in episodes) != tuple(range(len(episodes))):
            raise ValueError("summary episode indices must be consecutive and zero-indexed")
        if any(
            right.completion_global_step < left.completion_global_step for left, right in zip(episodes, episodes[1:])
        ):
            raise ValueError("summary episode completion steps must be non-decreasing")
        if sum(item.length for item in episodes) != completed:
            raise ValueError("summary episode lengths must sum exactly to completed transitions")
        if max(item.completion_global_step for item in episodes) != completed:
            raise ValueError("final summary episode completion step must equal completed transitions")
        if tuple(item.iteration for item in boundaries) != tuple(range(1, iterations + 1)):
            raise ValueError("update iterations must be consecutive and one-indexed")
        if any(right.global_step <= left.global_step for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("update global_step boundaries must increase strictly")
        if any(right.episodes_completed < left.episodes_completed for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("update episode counts must be non-decreasing")
        if boundaries[-1].global_step != completed or boundaries[-1].episodes_completed != len(episodes):
            raise ValueError("final update boundary must equal completed transitions and episodes")
        if any(item.global_step > completed or item.episodes_completed > len(episodes) for item in boundaries):
            raise ValueError("update boundaries cannot exceed final transition or episode counts")
        for boundary in boundaries:
            completed_at_boundary = sum(episode.completion_global_step <= boundary.global_step for episode in episodes)
            if boundary.episodes_completed != completed_at_boundary:
                raise ValueError("each update boundary episode count must equal summary completions at its global_step")
        fields_tuple = tuple(self.update_record_fields)
        required_update_fields = {"iteration", "global_step", "episodes_completed"}
        if (
            not fields_tuple
            or len(fields_tuple) != len(set(fields_tuple))
            or any(not isinstance(item, str) or not item or item.strip() != item for item in fields_tuple)
            or not required_update_fields.issubset(fields_tuple)
        ):
            raise ValueError("update_record_fields must be unique, trimmed, and include accounting fields")
        object.__setattr__(self, "summary_episodes", episodes)
        object.__setattr__(self, "update_boundaries", boundaries)
        object.__setattr__(self, "update_record_fields", tuple(sorted(fields_tuple)))
        for name, label in (
            ("run_config", "run_config"),
            ("summary_evaluation", "summary evaluation"),
            ("update_aggregation", "update aggregation"),
            ("resolved_experiment", "resolved experiment"),
            ("resolved_trainer", "resolved trainer"),
        ):
            frozen = refreeze_mapping(getattr(self, name), field=label)
            if not frozen:
                raise ValueError(f"{label} cannot be empty")
            object.__setattr__(self, name, frozen)

    @property
    def episode_count(self) -> int:
        return len(self.summary_episodes)

    @property
    def run_config_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.run_config)

    @property
    def summary_evaluation_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.summary_evaluation)

    @property
    def update_aggregation_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.update_aggregation)

    @property
    def resolved_experiment_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.resolved_experiment)

    @property
    def resolved_trainer_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.resolved_trainer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_config_schema_id": self.run_config_schema_id,
            "training_summary_schema_id": self.training_summary_schema_id,
            "update_metrics_schema_id": self.update_metrics_schema_id,
            "run_config": self.run_config_mapping,
            "requested_transitions": self.requested_transitions,
            "completed_transitions": self.completed_transitions,
            "iteration_count": self.iteration_count,
            "summary_episodes": [item.expected_fields() for item in self.summary_episodes],
            "update_boundaries": [item.to_dict() for item in self.update_boundaries],
            "update_record_fields": list(self.update_record_fields),
            "summary_update_metrics_key": self.summary_update_metrics_key,
            "summary_evaluation": self.summary_evaluation_mapping,
            "update_aggregation": self.update_aggregation_mapping,
            "resolved_experiment": self.resolved_experiment_mapping,
            "resolved_trainer": self.resolved_trainer_mapping,
        }


@dataclass(frozen=True, slots=True)
class ValidatedTrainingArtifact:
    """One content-authenticated file in a training bundle."""

    role: str
    relative_path: str
    path: Path
    media_type: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedTrainingArtifactReceipt:
    """One authenticated artifact reference from copied manifest bytes.

    Unlike :class:`ValidatedTrainingArtifact`, this value deliberately has no
    filesystem ``path``.  A copied training-bundle receipt authenticates the
    sealed manifest and its references, but must not pretend that the original
    training artifact tree is present beside the copy.
    """

    role: str
    relative_path: str
    media_type: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedTrainingBundle:
    """A complete scientific training bundle safe to link from evaluation."""

    bundle_path: Path
    bundle_sha256: str
    run: ResolvedRun
    task: TaskSpec
    provenance: SoftwareProvenance
    contract: ExpectedTrainingContract
    expected_episodes: tuple[ExpectedTrainingEpisode, ...]
    artifacts: tuple[ValidatedTrainingArtifact, ...]
    episode_artifacts: tuple[ValidatedTrainingArtifact, ...]
    expected_episode_count: int
    training_evidence: Mapping[str, Any]

    def artifact(self, role: str) -> ValidatedTrainingArtifact:
        """Return the unique top-level artifact with ``role``."""

        matches = tuple(item for item in self.artifacts if item.role == role)
        if len(matches) != 1:
            raise ScientificTrainingBundleError(
                f"validated training bundle does not contain exactly one {role!r} artifact"
            )
        return matches[0]

    @property
    def checkpoint_path(self) -> Path:
        return self.artifact("final_checkpoint").path

    @property
    def checkpoint_sha256(self) -> str:
        """Exact opaque checkpoint digest that evaluation must use.

        The bundle deliberately does not import an algorithm runtime or open
        checkpoint internals.  A CLI/algorithm adapter must validate the
        checkpoint schema before calling the writer; this digest then binds
        those validated bytes to later evaluation.
        """

        return self.artifact("final_checkpoint").sha256


@dataclass(frozen=True, slots=True)
class ValidatedTrainingBundleReceipt:
    """Byte-authenticated, portable receipt for a sealed training bundle.

    The receipt carries only facts that can be authenticated from the exact
    copied ``training_bundle.json`` bytes and independently supplied identity
    inputs.  It intentionally does not claim that any referenced training
    artifact remains locally available.
    """

    bundle_sha256: str
    run: ResolvedRun
    task: TaskSpec
    provenance: SoftwareProvenance
    expected_revision: str
    contract: ExpectedTrainingContract
    artifacts: tuple[ValidatedTrainingArtifactReceipt, ...]
    expected_episode_count: int

    def artifact(self, role: str) -> ValidatedTrainingArtifactReceipt:
        """Return the unique copied-manifest artifact reference with ``role``."""

        matches = tuple(item for item in self.artifacts if item.role == role)
        if len(matches) != 1:
            raise ScientificTrainingBundleError(
                f"validated training bundle receipt does not contain exactly one {role!r} artifact"
            )
        return matches[0]

    @property
    def checkpoint_sha256(self) -> str:
        """Exact opaque checkpoint digest linked by the copied manifest."""

        return self.artifact("final_checkpoint").sha256


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise ScientificTrainingBundleError(f"cannot read training artifact {path}: {error}") from error


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ScientificTrainingBundleError(f"{field} must contain exactly 64 lowercase hexadecimal characters")
    return value


def _revision(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ScientificTrainingBundleError(f"{field} must be a hexadecimal Git revision")
    normalized = value.lower()
    if not _REVISION_RE.fullmatch(normalized):
        raise ScientificTrainingBundleError(f"{field} must be a 7--64 character hexadecimal Git revision")
    return normalized


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificTrainingBundleError(f"{field} must be a JSON object")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    result = _nonnegative_int(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _integer(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ScientificTrainingBundleError(f"{field} must be a {qualifier} integer")
    return value


def _strict_json(text: str) -> Any:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if value in {float("inf"), float("-inf")}:
            raise ValueError(f"non-finite number {token}")
        return value

    return json.loads(
        text,
        object_pairs_hook=object_from_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite number {token}")),
        parse_float=finite_float,
    )


def _same_json(actual: Any, expected: Any) -> bool:
    return json.dumps(actual, allow_nan=False, separators=(",", ":"), sort_keys=True) == json.dumps(
        expected,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _read_json_bytes(path: Path, *, field: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        contents = path.read_bytes()
        value = _strict_json(contents.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"cannot read {field} {path}: {error}") from error
    return _mapping(value, field=field), contents


def _portable_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScientificTrainingBundleError(f"{field} must be a non-empty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ScientificTrainingBundleError(f"{field} must be a canonical portable relative path")
    return value


def _assert_no_user_symlinks(path: Path, *, field: str) -> Path:
    """Return an absolute spelling containing no non-platform symlink."""

    absolute = Path(os.path.abspath(path.expanduser()))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        if not cursor.is_symlink():
            continue
        try:
            is_var_alias = cursor == Path("/var") and cursor.resolve(strict=True) == Path("/private/var")
        except OSError:
            is_var_alias = False
        if is_var_alias:
            continue
        raise ScientificTrainingBundleError(f"{field} must not use symbolic links: {cursor}")
    return absolute


def _assert_real_path(root: Path, path: Path, *, field: str) -> tuple[Path, str]:
    candidate = _assert_no_user_symlinks(path, field=field)
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ScientificTrainingBundleError(f"{field} must be an existing file inside {root}: {path}") from error

    logical_root: Path | None = None
    for ancestor in (candidate.parent, *candidate.parents):
        try:
            if ancestor.resolve(strict=True) == root:
                logical_root = ancestor
                break
        except OSError:
            continue
    if logical_root is None:
        raise ScientificTrainingBundleError(f"{field} has no canonical path beneath {root}: {path}")
    lexical_relative = candidate.relative_to(logical_root)
    if relative != lexical_relative:
        raise ScientificTrainingBundleError(f"{field} must use its canonical real path")
    if not resolved.is_file():
        raise ScientificTrainingBundleError(f"{field} must be a regular file: {resolved}")
    portable = _portable_relative_path(relative.as_posix(), field=f"{field} path")
    return resolved, portable


def _assert_distinct_inodes(artifacts: tuple[ValidatedTrainingArtifact, ...]) -> None:
    seen: dict[tuple[int, int], ValidatedTrainingArtifact] = {}
    stats: list[tuple[ValidatedTrainingArtifact, os.stat_result]] = []
    for artifact in artifacts:
        try:
            stat = artifact.path.stat(follow_symlinks=False)
        except OSError as error:
            raise ScientificTrainingBundleError(f"cannot stat training artifact {artifact.path}: {error}") from error
        identity = (stat.st_dev, stat.st_ino)
        previous = seen.get(identity)
        if previous is not None:
            raise ScientificTrainingBundleError(
                "training bundle artifact files must be physically distinct; "
                f"{previous.relative_path!r} and {artifact.relative_path!r} share one inode"
            )
        seen[identity] = artifact
        stats.append((artifact, stat))
    for artifact, stat in stats:
        if stat.st_nlink != 1:
            raise ScientificTrainingBundleError(
                "training bundle artifacts must have exactly one filesystem link; "
                f"{artifact.relative_path!r} has link count {stat.st_nlink}"
            )


def _assert_single_filesystem_link(path: Path, *, field: str) -> None:
    try:
        link_count = path.stat(follow_symlinks=False).st_nlink
    except OSError as error:
        raise ScientificTrainingBundleError(f"cannot stat {field} {path}: {error}") from error
    if link_count != 1:
        raise ScientificTrainingBundleError(f"{field} must have exactly one filesystem link; found {link_count}")


def _root_directory(path: str | Path) -> Path:
    unresolved = Path(path).expanduser()
    _assert_no_user_symlinks(unresolved, field="training bundle directory")
    try:
        root = unresolved.resolve(strict=True)
    except OSError as error:
        raise ScientificTrainingBundleError(f"training bundle directory does not exist: {unresolved}") from error
    if not root.is_dir():
        raise ScientificTrainingBundleError(f"training bundle path is not a directory: {root}")
    return root


def _artifact_from_source(root: Path, source: str | Path, *, role: str) -> ValidatedTrainingArtifact:
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    path, relative_path = _assert_real_path(root, source_path, field=role)
    if path.stat().st_size == 0:
        raise ScientificTrainingBundleError(f"{role} artifact must not be empty: {path}")
    if role == "update_metrics" and path.suffix.casefold() in {".jsonl", ".ndjson"}:
        media_type = "application/x-ndjson"
    elif role == "final_checkpoint":
        media_type = "application/octet-stream"
    else:
        media_type = "application/json"
    return ValidatedTrainingArtifact(role, relative_path, path, media_type, _sha256_file(path))


def _artifact_payload(artifact: ValidatedTrainingArtifact) -> dict[str, str]:
    return {
        "path": artifact.relative_path,
        "media_type": artifact.media_type,
        "sha256": artifact.sha256,
    }


def _artifact_from_reference(
    root: Path,
    value: Any,
    *,
    role: str,
) -> ValidatedTrainingArtifact:
    reference = _mapping(value, field=f"{role} artifact")
    if set(reference) != {"path", "media_type", "sha256"}:
        raise ScientificTrainingBundleError(f"{role} artifact does not match the supported schema")
    relative_path = _portable_relative_path(reference.get("path"), field=f"{role} artifact path")
    media_type = reference.get("media_type")
    if media_type not in _ARTIFACT_MEDIA_TYPES[role]:
        raise ScientificTrainingBundleError(f"{role} artifact has unsupported media_type {media_type!r}")
    path, actual_relative = _assert_real_path(root, root / relative_path, field=role)
    if actual_relative != relative_path:
        raise ScientificTrainingBundleError(f"{role} artifact path is not canonical")
    digest = _sha256(reference.get("sha256"), field=f"{role} artifact sha256")
    if _sha256_file(path) != digest:
        raise ScientificTrainingBundleError(f"{role} artifact checksum mismatch for {path}")
    if path.stat().st_size == 0:
        raise ScientificTrainingBundleError(f"{role} artifact must not be empty: {path}")
    return ValidatedTrainingArtifact(role, relative_path, path, media_type, digest)


def _artifact_receipt_from_reference(
    value: Any,
    *,
    role: str,
) -> ValidatedTrainingArtifactReceipt:
    """Validate a manifest reference without resolving or opening its path."""

    reference = _mapping(value, field=f"{role} artifact")
    if set(reference) != {"path", "media_type", "sha256"}:
        raise ScientificTrainingBundleError(f"{role} artifact does not match the supported schema")
    relative_path = _portable_relative_path(reference.get("path"), field=f"{role} artifact path")
    media_type = reference.get("media_type")
    if media_type not in _ARTIFACT_MEDIA_TYPES[role]:
        raise ScientificTrainingBundleError(f"{role} artifact has unsupported media_type {media_type!r}")
    digest = _sha256(reference.get("sha256"), field=f"{role} artifact sha256")
    return ValidatedTrainingArtifactReceipt(role, relative_path, media_type, digest)


def _task_payload(task: TaskSpec) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "spec": canonical_value(task, field="TaskSpec"),
    }


def _evidence_task_payload(task: TaskSpec) -> dict[str, Any]:
    assert task.building.sha256 is not None and task.weather.sha256 is not None
    return {
        "task_name": task.name,
        "task_id": task.task_id,
        "building_asset_id": task.building.asset_id,
        "building_sha256": task.building.sha256,
        "weather_asset_id": task.weather.asset_id,
        "weather_sha256": task.weather.sha256,
        "warmup_preconditioning": {
            "enabled": task.warmup_preconditioning.enabled,
            "contract_payload": task.warmup_preconditioning.identity_payload(),
            "contract_sha256": task.warmup_preconditioning.sha256,
        },
    }


def _validate_scientific_provenance(provenance: SoftwareProvenance, *, expected_revision: str) -> str:
    if not isinstance(provenance, SoftwareProvenance):
        raise ScientificTrainingBundleError("scientific training bundle requires SoftwareProvenance")
    if provenance.schema_version != PROVENANCE_SCHEMA_VERSION:
        raise ScientificTrainingBundleError("scientific training requires the supported provenance schema")
    if provenance.dirty:
        raise ScientificTrainingBundleError("scientific training bundle refuses dirty Git provenance")
    expected = _revision(expected_revision, field="expected_revision")
    if provenance.revision.lower() != expected:
        raise ScientificTrainingBundleError(
            f"software provenance revision {provenance.revision!r} does not match expected revision {expected!r}"
        )
    if not isinstance(provenance.container_digest, str) or not _CONTAINER_DIGEST_RE.fullmatch(
        provenance.container_digest
    ):
        raise ScientificTrainingBundleError(
            "scientific training provenance requires an exact lowercase sha256 container digest"
        )
    dependencies = provenance.to_dict()["dependencies"]
    missing = _REQUIRED_DEPENDENCIES - set(dependencies)
    if missing:
        raise ScientificTrainingBundleError(
            f"scientific training provenance is missing required dependencies: {sorted(missing)!r}"
        )
    invalid = sorted(
        name
        for name in _REQUIRED_DEPENDENCIES
        if (
            not isinstance(dependencies[name], str)
            or not dependencies[name]
            or dependencies[name].strip() != dependencies[name]
        )
    )
    if invalid:
        raise ScientificTrainingBundleError(
            f"scientific training provenance has invalid dependency versions: {invalid!r}"
        )
    return expected


def _validate_scientific_identity(
    *,
    run: ResolvedRun,
    task: TaskSpec,
    provenance: SoftwareProvenance,
    expected_revision: str,
) -> str:
    if not isinstance(run, ResolvedRun) or run.phase != "train":
        raise ScientificTrainingBundleError("scientific training bundle requires a training ResolvedRun")
    if not isinstance(task, TaskSpec):
        raise ScientificTrainingBundleError("scientific training bundle requires a TaskSpec")
    if run.train_task_ids != (task.task_id,):
        raise ScientificTrainingBundleError(
            "training ResolvedRun must identify exactly the supplied single-task training TaskSpec"
        )
    if not task.publication_ready:
        raise ScientificTrainingBundleError("scientific training bundle requires checksum-pinned task assets")
    return _validate_scientific_provenance(provenance, expected_revision=expected_revision)


def _calendar_date_label(calendar_year: int | None, month: int, day: int) -> str:
    prefix = "" if calendar_year is None else f"{calendar_year:04d}-"
    return f"{prefix}{month:02d}-{day:02d}"


def _validate_summary_calendar_link(
    summary: ExpectedTrainingSummaryEpisode,
    expected: ExpectedTrainingEpisode,
    *,
    task: TaskSpec,
) -> None:
    calendar = thaw_mapping(summary.training_calendar)
    if set(calendar) != _SUMMARY_CALENDAR_FIELDS:
        raise ScientificTrainingBundleError(
            "every training summary episode requires the exact scientific calendar schema"
        )
    for name in ("schedule_seed", "episode_index", "calendar_block", "month_visit_index", "day_cycle", "child_seed"):
        _integer(calendar.get(name), field=f"training summary calendar.{name}")
    if task.episode.episode_length_hours % 24:
        raise ScientificTrainingBundleError(
            "scientific calendar training requires a whole-number-of-days episode horizon"
        )
    duration_days = task.episode.episode_length_hours // 24
    calendar_block = expected.key.episode_index // 12
    latest_start = monthrange(2001, expected.window.start_month)[1] - duration_days + 1
    if latest_start <= 0:
        raise ScientificTrainingBundleError("scientific calendar episode does not fit inside its start month")
    expected_calendar = {
        "schema_version": expected.schedule_schema_version,
        "strategy_id": expected.strategy_id,
        "schedule_seed": expected.key.schedule_seed,
        "episode_index": expected.key.episode_index,
        "calendar_block": calendar_block,
        "month_visit_index": calendar_block,
        "day_cycle": calendar_block // latest_start,
        "child_seed": expected.child_seed,
        "calendar_year": task.calendar_year,
        "start_date": _calendar_date_label(
            task.calendar_year,
            expected.window.start_month,
            expected.window.start_day,
        ),
        "end_date": _calendar_date_label(
            task.calendar_year,
            expected.window.end_month,
            expected.window.end_day,
        ),
        "window": expected.to_dict()["window"],
        "task_name": task.name,
        "task_id": task.task_id,
    }
    if not _same_json(
        {name: calendar.get(name) for name in expected_calendar},
        expected_calendar,
    ):
        raise ScientificTrainingBundleError(
            "training summary calendar does not exactly match its external episode plan and TaskSpec"
        )


def _validate_expected_episodes(
    expected_episodes: tuple[ExpectedTrainingEpisode, ...],
    *,
    task: TaskSpec,
    contract: ExpectedTrainingContract,
) -> tuple[ExpectedTrainingEpisode, ...]:
    if not isinstance(expected_episodes, tuple):
        raise ScientificTrainingBundleError(
            "expected_episodes must be the independently derived exact tuple from the launch plan"
        )
    episodes = expected_episodes
    if not episodes or any(not isinstance(item, ExpectedTrainingEpisode) for item in episodes):
        raise ScientificTrainingBundleError(
            "expected_episodes must be a non-empty tuple of ExpectedTrainingEpisode values"
        )
    if len({item.key for item in episodes}) != len(episodes):
        raise ScientificTrainingBundleError("expected_episodes cannot contain duplicate episode keys")
    if episodes != tuple(sorted(episodes, key=lambda item: item.key.sort_key())):
        raise ScientificTrainingBundleError("expected_episodes must be in canonical key order")
    if len(episodes) != contract.episode_count:
        raise ScientificTrainingBundleError("expected_episodes count does not match the independent training contract")
    if any(item.window.duration_hours != task.episode.episode_length_hours for item in episodes):
        raise ScientificTrainingBundleError("expected_episodes windows do not match the TaskSpec horizon")
    if any(item.length != task.episode.max_steps for item in contract.summary_episodes):
        raise ScientificTrainingBundleError("summary episode lengths do not match the TaskSpec horizon")
    environment_indices = {item.environment_index for item in contract.summary_episodes}
    if environment_indices != set(range(len(environment_indices))):
        raise ScientificTrainingBundleError("training summary environment indices must be contiguous and zero-indexed")
    num_envs = len(environment_indices)
    expected_streams = {item.key.stream_id for item in episodes}
    required_streams = {f"train-env-{index:03d}" for index in range(num_envs)}
    if expected_streams != required_streams or any(item.key.generation != 0 for item in episodes):
        raise ScientificTrainingBundleError(
            "external episodes must exactly cover the zero-generation synchronous training streams"
        )
    if len(episodes) % num_envs:
        raise ScientificTrainingBundleError(
            "synchronous training requires the same complete-episode count for every environment"
        )
    episodes_per_environment = len(episodes) // num_envs
    for environment_index in range(num_envs):
        stream = f"train-env-{environment_index:03d}"
        local_indices = tuple(item.key.episode_index for item in episodes if item.key.stream_id == stream)
        if local_indices != tuple(range(episodes_per_environment)):
            raise ScientificTrainingBundleError(
                "external episode indices must be consecutive within every environment stream"
            )
    expected_by_key = {item.key: item for item in episodes}
    linked_keys: set[TrainingEpisodeKey] = set()
    next_local_episode = [0] * num_envs
    for summary in contract.summary_episodes:
        calendar = thaw_mapping(summary.training_calendar)
        local_episode_index = _integer(
            calendar.get("episode_index"),
            field="training summary calendar.episode_index",
        )
        expected_local_episode = next_local_episode[summary.environment_index]
        if local_episode_index != expected_local_episode:
            raise ScientificTrainingBundleError(
                "training summary local episode indices must be consecutive in completion order"
            )
        expected_summary_index = local_episode_index * num_envs + summary.environment_index
        expected_completion_step = (local_episode_index + 1) * task.episode.max_steps * num_envs
        if (
            summary.episode_index != expected_summary_index
            or summary.completion_global_step != expected_completion_step
        ):
            raise ScientificTrainingBundleError(
                "training summary ordering or completion step violates synchronous equal-horizon accounting"
            )
        key = TrainingEpisodeKey(
            stream_id=f"train-env-{summary.environment_index:03d}",
            generation=0,
            schedule_seed=_integer(
                calendar.get("schedule_seed"),
                field="training summary calendar.schedule_seed",
            ),
            episode_index=local_episode_index,
        )
        expected = expected_by_key.get(key)
        if expected is None or key in linked_keys:
            raise ScientificTrainingBundleError(
                "training summary calendars do not map one-to-one to external environment-stream episodes"
            )
        _validate_summary_calendar_link(summary, expected, task=task)
        linked_keys.add(key)
        next_local_episode[summary.environment_index] += 1
    if next_local_episode != [episodes_per_environment] * num_envs:
        raise ScientificTrainingBundleError(
            "training summary does not contain the same episode count for every environment"
        )
    if linked_keys != set(expected_by_key):
        raise ScientificTrainingBundleError(
            "training summary calendars do not cover every external environment-stream episode"
        )
    return episodes


def _validate_artifact_identity(
    payload: Mapping[str, Any],
    *,
    field: str,
    run: ResolvedRun,
    task: TaskSpec,
) -> None:
    identity = _mapping(payload.get("identity"), field=f"{field}.identity")
    expected_identity = {
        "run_id": run.run_id,
        "task_id": task.task_id,
        "task_fingerprint": task.task_id,
        "building_checksum": task.building.sha256,
        "weather_checksum": task.weather.sha256,
    }
    if not _same_json(identity, expected_identity):
        raise ScientificTrainingBundleError(
            f"{field}.identity does not exactly match the ResolvedRun/TaskSpec training identity"
        )
    if "run" in payload and not _same_json(
        payload["run"],
        canonical_value(run.identity_payload(), field=f"{field}.run"),
    ):
        raise ScientificTrainingBundleError(f"{field}.run does not match the expected ResolvedRun")


def _validate_run_config(
    path: Path,
    *,
    run: ResolvedRun,
    task: TaskSpec,
    contract: ExpectedTrainingContract,
) -> Mapping[str, Any]:
    payload, _ = _read_json_bytes(path, field="run_config")
    if set(payload) != {"schema_id", "identity", "config"}:
        raise ScientificTrainingBundleError("run_config does not match the supported scientific schema")
    if payload.get("schema_id") != contract.run_config_schema_id:
        raise ScientificTrainingBundleError("run_config schema_id does not match the expected training contract")
    _validate_artifact_identity(payload, field="run_config", run=run, task=task)
    if not _same_json(payload.get("config"), contract.run_config_mapping):
        raise ScientificTrainingBundleError("run_config config does not exactly match the expected training contract")
    return payload


def _validate_summary_reference(
    summary_path: Path,
    value: Any,
    expected_path: Path,
    *,
    field: str,
) -> None:
    relative = _portable_relative_path(value, field=field)
    resolved, actual_relative = _assert_real_path(
        summary_path.parent,
        summary_path.parent / relative,
        field=field,
    )
    if actual_relative != relative or resolved != expected_path:
        raise ScientificTrainingBundleError(f"{field} does not reference the authenticated artifact")


def _validate_training_summary(
    path: Path,
    *,
    run: ResolvedRun,
    task: TaskSpec,
    run_config: ValidatedTrainingArtifact,
    checkpoint: ValidatedTrainingArtifact,
    update_metrics: ValidatedTrainingArtifact,
    contract: ExpectedTrainingContract,
) -> Mapping[str, Any]:
    payload, _ = _read_json_bytes(path, field="training_summary")
    if set(payload) != {"schema_id", "status", "identity", "training", "evaluation", "artifacts"}:
        raise ScientificTrainingBundleError("training_summary does not match the supported scientific schema")
    if payload.get("schema_id") != contract.training_summary_schema_id:
        raise ScientificTrainingBundleError("training_summary schema_id does not match the expected training contract")
    _validate_artifact_identity(payload, field="training_summary", run=run, task=task)
    if payload.get("status") != "succeeded":
        raise ScientificTrainingBundleError("scientific training_summary must have status='succeeded'")
    training = _mapping(payload.get("training"), field="training_summary.training")
    expected_training_fields = {
        "requested_transitions",
        "completed_transitions",
        "iterations",
        "episodes_completed",
        "episodes",
    }
    if set(training) != expected_training_fields:
        raise ScientificTrainingBundleError("training_summary.training does not match the supported schema")
    expected_counts = {
        "requested_transitions": contract.requested_transitions,
        "completed_transitions": contract.completed_transitions,
        "iterations": contract.iteration_count,
        "episodes_completed": contract.episode_count,
    }
    for name, expected in expected_counts.items():
        actual = _integer(
            training.get(name),
            field=f"training_summary.training.{name}",
            positive=True,
        )
        if actual != expected:
            raise ScientificTrainingBundleError(
                f"training_summary.training.{name} does not match the expected training contract"
            )
    episodes = training.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != contract.episode_count:
        raise ScientificTrainingBundleError("training_summary episode list is incomplete")
    for index, (actual, expected) in enumerate(zip(episodes, contract.summary_episodes)):
        record = _mapping(actual, field=f"training_summary.training.episodes[{index}]")
        expected_fields = expected.expected_fields()
        if set(record) != set(expected_fields) | {"return"}:
            raise ScientificTrainingBundleError(
                f"training_summary.training.episodes[{index}] does not match the expected record schema"
            )
        if not _same_json({name: record[name] for name in expected_fields}, expected_fields):
            raise ScientificTrainingBundleError(
                f"training_summary.training.episodes[{index}] does not match its expected identity"
            )
        _finite_number(record["return"], field=f"training_summary.training.episodes[{index}].return")
    if not _same_json(payload.get("evaluation"), contract.summary_evaluation_mapping):
        raise ScientificTrainingBundleError(
            "training_summary evaluation does not exactly match the expected training contract"
        )
    artifacts = _mapping(payload.get("artifacts"), field="training_summary.artifacts")
    expected_artifact_fields = {
        "run_config",
        "final_checkpoint",
        contract.summary_update_metrics_key,
    }
    if set(artifacts) != expected_artifact_fields:
        raise ScientificTrainingBundleError("training_summary artifacts do not match the expected schema")
    _validate_summary_reference(
        path,
        artifacts.get("run_config"),
        run_config.path,
        field="training_summary.artifacts.run_config",
    )
    _validate_summary_reference(
        path,
        artifacts.get("final_checkpoint"),
        checkpoint.path,
        field="training_summary.artifacts.final_checkpoint",
    )
    _validate_summary_reference(
        path,
        artifacts.get(contract.summary_update_metrics_key),
        update_metrics.path,
        field=f"training_summary.artifacts.{contract.summary_update_metrics_key}",
    )
    return payload


def _finite_number(value: Any, *, field: str) -> int | float:
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError):
        finite = False
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite:
        raise ScientificTrainingBundleError(f"{field} must be a finite number")
    return value


def _validate_update_metrics(
    artifact: ValidatedTrainingArtifact,
    *,
    run: ResolvedRun,
    task: TaskSpec,
    contract: ExpectedTrainingContract,
) -> Mapping[str, Any]:
    payload, _ = _read_json_bytes(artifact.path, field="update_metrics")
    expected_fields = {
        "schema_id",
        "identity",
        "aggregation",
        "iterations_requested",
        "iterations_recorded",
        "records",
    }
    if set(payload) != expected_fields:
        raise ScientificTrainingBundleError("update_metrics does not match the supported scientific schema")
    if payload.get("schema_id") != contract.update_metrics_schema_id:
        raise ScientificTrainingBundleError("update_metrics schema_id does not match the expected training contract")
    _validate_artifact_identity(payload, field="update_metrics", run=run, task=task)
    if not _same_json(payload.get("aggregation"), contract.update_aggregation_mapping):
        raise ScientificTrainingBundleError(
            "update_metrics aggregation does not exactly match the expected training contract"
        )
    for name in ("iterations_requested", "iterations_recorded"):
        actual = _integer(payload.get(name), field=f"update_metrics.{name}", positive=True)
        if actual != contract.iteration_count:
            raise ScientificTrainingBundleError(f"update_metrics.{name} does not match the expected training contract")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != contract.iteration_count:
        raise ScientificTrainingBundleError("update_metrics records are incomplete")
    expected_record_fields = set(contract.update_record_fields)
    for index, (actual, boundary) in enumerate(zip(records, contract.update_boundaries)):
        record = _mapping(actual, field=f"update_metrics.records[{index}]")
        if set(record) != expected_record_fields:
            raise ScientificTrainingBundleError(
                f"update_metrics.records[{index}] does not match the expected field schema"
            )
        for name, value in record.items():
            _finite_number(value, field=f"update_metrics.records[{index}].{name}")
        for name, expected in boundary.to_dict().items():
            actual_value = _integer(record.get(name), field=f"update_metrics.records[{index}].{name}")
            if actual_value != expected:
                raise ScientificTrainingBundleError(
                    f"update_metrics.records[{index}].{name} does not match its expected boundary"
                )
    return payload


def _validate_resolved_input(
    artifact: ValidatedTrainingArtifact,
    *,
    expected: Mapping[str, Any],
    field: str,
) -> Mapping[str, Any]:
    payload, contents = _read_json_bytes(artifact.path, field=field)
    if not _same_json(payload, expected):
        raise ScientificTrainingBundleError(f"{field} does not exactly match the expected resolved input")
    if contents != _formatted_json_bytes(expected):
        raise ScientificTrainingBundleError(f"{field} is not the canonical resolved-input artifact")
    return payload


def _load_diagnostics_policy(
    artifact: ValidatedTrainingArtifact, *, task: TaskSpec, provenance: SoftwareProvenance
) -> DiagnosticsAllowlist:
    try:
        text = artifact.path.read_text(encoding="utf-8", errors="strict")
        raw = _strict_json(text)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"cannot read diagnostics policy {artifact.path}: {error}") from error
    try:
        policy = diagnostics_allowlist_from_mapping(_mapping(raw, field="diagnostics_policy"))
    except (ValueError, ScientificTrainingBundleError) as error:
        raise ScientificTrainingBundleError(f"invalid diagnostics policy: {error}") from error
    if text != policy.canonical_json() or artifact.sha256 != policy.sha256:
        raise ScientificTrainingBundleError("diagnostics policy is not the exact canonical policy bytes")
    expected_scope = (task.building.sha256, task.weather.sha256, provenance.energyplus_version)
    if (policy.asset_sha256, policy.weather_sha256, policy.energyplus_version) != expected_scope:
        raise ScientificTrainingBundleError(
            "diagnostics policy scope does not match the TaskSpec and SoftwareProvenance"
        )
    return policy


def _episode_key(value: Any, *, field: str) -> TrainingEpisodeKey:
    raw = _mapping(value, field=field)
    if set(raw) != {"stream_id", "generation", "schedule_seed", "episode_index"}:
        raise ScientificTrainingBundleError(f"{field} does not match the supported episode-key schema")
    try:
        return TrainingEpisodeKey(
            stream_id=raw.get("stream_id"),
            generation=raw.get("generation"),
            schedule_seed=raw.get("schedule_seed"),
            episode_index=raw.get("episode_index"),
        )
    except (TypeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"invalid {field}: {error}") from error


def _calendar_window(value: Any, *, field: str) -> CalendarWindowSpec:
    raw = _mapping(value, field=field)
    expected = {"name", "start_month", "start_day", "end_month", "end_day", "duration_hours"}
    if set(raw) != expected:
        raise ScientificTrainingBundleError(f"{field} does not match the supported calendar-window schema")
    for name in ("start_month", "start_day", "end_month", "end_day"):
        _integer(raw.get(name), field=f"{field}.{name}", positive=True)
    try:
        window = CalendarWindowSpec(
            name=raw.get("name"),
            start_month=raw.get("start_month"),
            start_day=raw.get("start_day"),
            end_month=raw.get("end_month"),
            end_day=raw.get("end_day"),
        )
    except (TypeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"invalid {field}: {error}") from error
    if raw.get("duration_hours") != window.duration_hours:
        raise ScientificTrainingBundleError(f"{field}.duration_hours is inconsistent with its dates")
    return window


def _expected_episode(value: Any, *, index: int) -> ExpectedTrainingEpisode:
    field = f"training_evidence.expected_episodes[{index}]"
    raw = _mapping(value, field=field)
    expected_fields = {
        "episode_key",
        "schedule_schema_version",
        "strategy_id",
        "child_seed",
        "window",
    }
    if set(raw) != expected_fields:
        raise ScientificTrainingBundleError(f"{field} does not match the supported expected-episode schema")
    try:
        result = ExpectedTrainingEpisode(
            key=_episode_key(raw.get("episode_key"), field=f"{field}.episode_key"),
            schedule_schema_version=raw.get("schedule_schema_version"),
            strategy_id=raw.get("strategy_id"),
            child_seed=raw.get("child_seed"),
            window=_calendar_window(raw.get("window"), field=f"{field}.window"),
        )
    except (TypeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"invalid {field}: {error}") from error
    if not _same_json(raw, result.to_dict()):
        raise ScientificTrainingBundleError(f"{field} is not canonical")
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _validate_episode_payload(
    payload: Mapping[str, Any],
    *,
    expected: ExpectedTrainingEpisode,
    task: TaskSpec,
    provenance: SoftwareProvenance,
    policy: DiagnosticsAllowlist,
) -> None:
    expected_fields = {
        "schema_version",
        "episode_key",
        "task",
        "calendar",
        "termination",
        "warmup_preconditioning",
        "diagnostics",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != TRAINING_EPISODE_EVIDENCE_SCHEMA_VERSION:
        raise ScientificTrainingBundleError("training episode evidence does not match the supported schema")
    if not _same_json(payload.get("episode_key"), expected.key.to_dict()):
        raise ScientificTrainingBundleError("training episode evidence key does not match its manifest record")
    if payload.get("task") != _evidence_task_payload(task):
        raise ScientificTrainingBundleError("training episode TaskSpec identity does not match the bundle")

    calendar = _mapping(payload.get("calendar"), field="training episode calendar")
    if set(calendar) != {
        "schedule_schema_version",
        "strategy_id",
        "calendar_year",
        "child_seed",
        "window",
    }:
        raise ScientificTrainingBundleError("training episode calendar does not match the supported schema")
    if not _same_json(
        calendar,
        {
            "schedule_schema_version": expected.schedule_schema_version,
            "strategy_id": expected.strategy_id,
            "calendar_year": task.calendar_year,
            "child_seed": expected.child_seed,
            "window": expected.to_dict()["window"],
        },
    ):
        raise ScientificTrainingBundleError("training episode calendar does not match its preregistered episode")

    termination = _mapping(payload.get("termination"), field="training episode termination")
    if not _same_json(
        termination,
        {
            "expected_steps": task.episode.max_steps,
            "step_count": task.episode.max_steps,
            "terminated": True,
            "truncated": False,
            "natural_termination": True,
        },
    ):
        raise ScientificTrainingBundleError("training episode did not terminate naturally at the TaskSpec horizon")

    diagnostics = _mapping(payload.get("diagnostics"), field="training episode diagnostics")
    if set(diagnostics) != {
        "mode",
        "energyplus_version",
        "policy_sha256",
        "executed_model_sha256",
        "source_path",
        "source_file_sha256",
        "canonical_payload_sha256",
        "payload",
    }:
        raise ScientificTrainingBundleError("training episode diagnostics do not match the supported schema")
    if (
        diagnostics.get("mode") != "scientific"
        or diagnostics.get("energyplus_version") != provenance.energyplus_version
    ):
        raise ScientificTrainingBundleError("training episode diagnostics do not identify the scientific runtime")
    if diagnostics.get("policy_sha256") != policy.sha256:
        raise ScientificTrainingBundleError("training episode diagnostics policy does not match the bundle")
    _sha256(diagnostics.get("executed_model_sha256"), field="executed_model_sha256")
    _sha256(diagnostics.get("source_file_sha256"), field="source_file_sha256")
    if diagnostics.get("source_path") != "energyplus/energyplus_diagnostics.json":
        raise ScientificTrainingBundleError("training episode diagnostics source_path is not canonical")
    raw_diagnostics = _mapping(diagnostics.get("payload"), field="training episode diagnostics payload")
    if raw_diagnostics.get("executed_model_sha256") != diagnostics.get("executed_model_sha256"):
        raise ScientificTrainingBundleError("training episode executed-model SHA256 is inconsistent")
    canonical_digest = _sha256_bytes(_canonical_json_bytes(raw_diagnostics))
    if diagnostics.get("canonical_payload_sha256") != canonical_digest:
        raise ScientificTrainingBundleError("training episode diagnostics canonical payload checksum mismatch")
    try:
        parsed = energyplus_diagnostics_from_mapping(raw_diagnostics)
        if parsed.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError(
                f"archived diagnostics schema {parsed.schema_version!r} is non-current; "
                f"scientific training requires {DIAGNOSTICS_SCHEMA_VERSION!r}"
            )
        decision = require_scientific_diagnostics(
            parsed,
            asset_sha256=task.building.sha256,
            weather_sha256=task.weather.sha256,
            allowlist=policy,
        )
        assert decision.canonical_diagnostics_json is not None
        accepted_core = json.loads(decision.canonical_diagnostics_json.decode("utf-8"))
        expected_diagnostic_fields = set(accepted_core) | {"executed_model_sha256"}
        if set(raw_diagnostics) != expected_diagnostic_fields:
            raise ValueError("payload fields do not match its embedded diagnostics schema")
        accepted_payload = dict(accepted_core)
        accepted_payload["executed_model_sha256"] = diagnostics.get("executed_model_sha256")
        if dict(raw_diagnostics) != accepted_payload:
            raise ValueError("persisted diagnostics do not equal the scientifically evaluated snapshot")
    except (TypeError, ValueError, ScientificDiagnosticsError) as error:
        raise ScientificTrainingBundleError(
            f"training episode diagnostics fail the scientific policy: {error}"
        ) from error

    warmup = _mapping(
        payload.get("warmup_preconditioning"),
        field="training episode warmup_preconditioning",
    )
    if set(warmup) != {"enabled", "contract", "evidence"}:
        raise ScientificTrainingBundleError(
            "training episode warmup preconditioning does not match the supported schema"
        )
    expected_contract = {
        "payload": task.warmup_preconditioning.identity_payload(),
        "sha256": task.warmup_preconditioning.sha256,
    }
    if warmup.get("enabled") is not task.warmup_preconditioning.enabled or warmup.get("contract") != expected_contract:
        raise ScientificTrainingBundleError("training episode warmup preconditioning contract does not match TaskSpec")
    if not task.warmup_preconditioning.enabled:
        if warmup.get("evidence") is not None:
            raise ScientificTrainingBundleError(
                "disabled training warmup preconditioning cannot contain runtime evidence"
            )
        return

    evidence = _mapping(
        warmup.get("evidence"),
        field="training episode warmup preconditioning evidence",
    )
    if set(evidence) != {
        "source_path",
        "source_file_sha256",
        "canonical_payload_sha256",
        "payload",
    }:
        raise ScientificTrainingBundleError("training episode warmup preconditioning evidence has an invalid schema")
    if evidence.get("source_path") != (f"energyplus/{WARMUP_PRECONDITIONING_FILENAME}"):
        raise ScientificTrainingBundleError("training episode warmup preconditioning source_path is not canonical")
    _sha256(
        evidence.get("source_file_sha256"),
        field="warmup preconditioning source_file_sha256",
    )
    warmup_payload = _mapping(
        evidence.get("payload"),
        field="training episode warmup preconditioning payload",
    )
    canonical_warmup_digest = _sha256_bytes(_canonical_json_bytes(warmup_payload))
    if evidence.get("canonical_payload_sha256") != canonical_warmup_digest:
        raise ScientificTrainingBundleError(
            "training episode warmup preconditioning canonical payload checksum mismatch"
        )
    try:
        validate_warmup_preconditioning_payload(
            warmup_payload,
            task=task,
            window=expected.window,
            executed_model_sha256=diagnostics["executed_model_sha256"],
        )
    except WarmupPreconditioningEvidenceError as error:
        raise ScientificTrainingBundleError(
            f"training episode warmup preconditioning evidence is invalid: {error}"
        ) from error


def _validate_training_evidence(
    artifact: ValidatedTrainingArtifact,
    *,
    root: Path,
    task: TaskSpec,
    provenance: SoftwareProvenance,
    policy: DiagnosticsAllowlist,
    expected_episodes: tuple[ExpectedTrainingEpisode, ...],
) -> tuple[Mapping[str, Any], tuple[ValidatedTrainingArtifact, ...], tuple[dict[str, Any], ...]]:
    if artifact.path.name != TRAINING_EVIDENCE_MANIFEST_FILENAME:
        raise ScientificTrainingBundleError(
            f"training evidence manifest must be named {TRAINING_EVIDENCE_MANIFEST_FILENAME!r}"
        )
    manifest, _ = _read_json_bytes(artifact.path, field="training_evidence_manifest")
    expected_fields = {
        "schema_version",
        "complete",
        "task",
        "energyplus_version",
        "diagnostics_policy",
        "diagnostics_policy_sha256",
        "expected_episodes",
        "expected_episode_count",
        "record_count",
        "records",
    }
    if set(manifest) != expected_fields:
        raise ScientificTrainingBundleError("training evidence manifest does not match the supported schema")
    if manifest.get("schema_version") != TRAINING_EVIDENCE_MANIFEST_SCHEMA_VERSION:
        raise ScientificTrainingBundleError("unsupported training evidence manifest schema_version")
    if manifest.get("complete") is not True:
        raise ScientificTrainingBundleError("scientific training evidence manifest is not complete")
    if manifest.get("task") != _evidence_task_payload(task):
        raise ScientificTrainingBundleError("training evidence TaskSpec identity does not match the bundle")
    if manifest.get("energyplus_version") != provenance.energyplus_version:
        raise ScientificTrainingBundleError("training evidence EnergyPlus version does not match provenance")
    if (
        manifest.get("diagnostics_policy") != policy.to_dict()
        or manifest.get("diagnostics_policy_sha256") != policy.sha256
    ):
        raise ScientificTrainingBundleError("training evidence diagnostics policy does not match the bundled policy")

    expected_raw = manifest.get("expected_episodes")
    records_raw = manifest.get("records")
    if not isinstance(expected_raw, list) or not isinstance(records_raw, list):
        raise ScientificTrainingBundleError("training evidence expected_episodes and records must be JSON lists")
    expected_count = _integer(
        manifest.get("expected_episode_count"),
        field="training_evidence.expected_episode_count",
        positive=True,
    )
    record_count = _integer(manifest.get("record_count"), field="training_evidence.record_count", positive=True)
    if expected_count != len(expected_raw) or record_count != len(records_raw) or record_count != expected_count:
        raise ScientificTrainingBundleError("training evidence expected/record counts are incomplete or inconsistent")
    if expected_count != len(expected_episodes):
        raise ScientificTrainingBundleError(
            "training evidence count does not match the independently supplied episode plan"
        )
    parsed_expectations = tuple(_expected_episode(value, index=index) for index, value in enumerate(expected_raw))
    if not _same_json(expected_raw, [item.to_dict() for item in expected_episodes]):
        raise ScientificTrainingBundleError(
            "training evidence expectations do not exactly match the independently supplied episode plan"
        )
    if parsed_expectations != expected_episodes:
        raise ScientificTrainingBundleError("training evidence episode plan does not reconstruct exactly")
    expected_by_key = {item.key: item for item in expected_episodes}
    if len(expected_by_key) != expected_count:
        raise ScientificTrainingBundleError("training evidence contains duplicate expected episode keys")
    if expected_episodes != tuple(sorted(expected_episodes, key=lambda item: item.key.sort_key())):
        raise ScientificTrainingBundleError("training evidence expected episodes are not in canonical order")
    if any(item.window.duration_hours != task.episode.episode_length_hours for item in expected_episodes):
        raise ScientificTrainingBundleError(
            "training evidence episode windows do not match the TaskSpec training horizon"
        )

    evidence_root = artifact.path.parent
    episodes_dir = evidence_root / _EPISODES_DIRECTORY_NAME
    if episodes_dir.is_symlink() or not episodes_dir.is_dir():
        raise ScientificTrainingBundleError("training evidence episodes path must be a real directory")
    expected_top_level = {TRAINING_EVIDENCE_MANIFEST_FILENAME, _EPISODES_DIRECTORY_NAME}
    if {path.name for path in evidence_root.iterdir()} != expected_top_level:
        raise ScientificTrainingBundleError("training evidence directory contains missing or stale top-level entries")

    episode_artifacts = []
    episode_references = []
    seen_keys: set[TrainingEpisodeKey] = set()
    seen_paths: set[str] = set()
    for index, value in enumerate(records_raw):
        field = f"training_evidence.records[{index}]"
        record = _mapping(value, field=field)
        if set(record) != {"episode_key", "path", "sha256"}:
            raise ScientificTrainingBundleError(f"{field} does not match the supported record schema")
        key = _episode_key(record.get("episode_key"), field=f"{field}.episode_key")
        if key in seen_keys or key not in expected_by_key:
            raise ScientificTrainingBundleError(f"{field} has a duplicate or unexpected episode key")
        seen_keys.add(key)
        relative_to_evidence = _portable_relative_path(record.get("path"), field=f"{field}.path")
        expected_relative = f"{_EPISODES_DIRECTORY_NAME}/{key.filename}"
        if relative_to_evidence != expected_relative:
            raise ScientificTrainingBundleError(f"{field}.path is not canonical for its episode key")
        episode_path, root_relative = _assert_real_path(
            root,
            evidence_root / relative_to_evidence,
            field=f"training episode {key.filename}",
        )
        if root_relative in seen_paths:
            raise ScientificTrainingBundleError("training evidence contains duplicate episode paths")
        seen_paths.add(root_relative)
        digest = _sha256(record.get("sha256"), field=f"{field}.sha256")
        if _sha256_file(episode_path) != digest:
            raise ScientificTrainingBundleError(f"training episode evidence checksum mismatch for {episode_path}")
        payload, _ = _read_json_bytes(episode_path, field="training episode evidence")
        _validate_episode_payload(
            payload,
            expected=expected_by_key[key],
            task=task,
            provenance=provenance,
            policy=policy,
        )
        episode_artifacts.append(
            ValidatedTrainingArtifact(
                role="training_episode_evidence",
                relative_path=root_relative,
                path=episode_path,
                media_type="application/json",
                sha256=digest,
            )
        )
        episode_references.append(
            {
                "episode_key": key.to_dict(),
                "path": root_relative,
                "media_type": "application/json",
                "sha256": digest,
            }
        )
    if seen_keys != set(expected_by_key):
        raise ScientificTrainingBundleError("training evidence is missing one or more preregistered episodes")
    if tuple(item.relative_path for item in episode_artifacts) != tuple(
        item.relative_path for item in sorted(episode_artifacts, key=lambda item: item.relative_path)
    ):
        raise ScientificTrainingBundleError("training evidence records are not in canonical order")
    expected_filenames = {item.key.filename for item in expected_episodes}
    actual_filenames = {path.name for path in episodes_dir.iterdir()}
    if actual_filenames != expected_filenames:
        raise ScientificTrainingBundleError("training evidence episodes directory contains missing or stale files")
    return manifest, tuple(episode_artifacts), tuple(episode_references)


def _formatted_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_create(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace existing training bundle manifest: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def write_training_bundle(
    output_dir: str | Path,
    *,
    run: ResolvedRun,
    task: TaskSpec,
    provenance: SoftwareProvenance,
    expected_revision: str,
    expected_episodes: tuple[ExpectedTrainingEpisode, ...],
    training_contract: ExpectedTrainingContract,
    run_config_path: str | Path,
    training_summary_path: str | Path,
    final_checkpoint_path: str | Path,
    update_metrics_path: str | Path,
    diagnostics_policy_path: str | Path,
    training_evidence_manifest_path: str | Path,
    resolved_experiment_path: str | Path,
    resolved_trainer_path: str | Path,
    expected_checkpoint_sha256: str,
) -> ValidatedTrainingBundle:
    """Validate all scientific training evidence, then atomically seal it.

    All supplied artifacts must already be real files beneath ``output_dir``.
    No artifact is copied or rewritten.  ``training_bundle.json`` is the last
    file created, and an existing manifest is never replaced.  Every check is
    completed before that atomic creation, so no post-write validation failure
    can leave an invalid top-level manifest.

    The checkpoint is deliberately treated as opaque bytes.  The calling
    algorithm/CLI adapter must validate its internal schema before invoking
    this function and should pass that byte digest through
    ``expected_checkpoint_sha256``.  This bundle then authenticates the exact
    bytes for later evaluation linkage.
    """

    root = _root_directory(output_dir)
    expected_revision = _validate_scientific_identity(
        run=run,
        task=task,
        provenance=provenance,
        expected_revision=expected_revision,
    )
    if not isinstance(training_contract, ExpectedTrainingContract):
        raise ScientificTrainingBundleError("training_contract must be an ExpectedTrainingContract")
    expected_episodes = _validate_expected_episodes(
        expected_episodes,
        task=task,
        contract=training_contract,
    )
    destination = root / TRAINING_BUNDLE_FILENAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing training bundle manifest: {destination}")

    sources = {
        "run_config": run_config_path,
        "training_summary": training_summary_path,
        "final_checkpoint": final_checkpoint_path,
        "update_metrics": update_metrics_path,
        "diagnostics_policy": diagnostics_policy_path,
        "training_evidence_manifest": training_evidence_manifest_path,
        "resolved_experiment": resolved_experiment_path,
        "resolved_trainer": resolved_trainer_path,
    }
    artifacts = {role: _artifact_from_source(root, source, role=role) for role, source in sources.items()}
    artifact_paths = [artifact.relative_path for artifact in artifacts.values()]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ScientificTrainingBundleError("training bundle artifact roles must reference distinct files")
    checkpoint = artifacts["final_checkpoint"]
    expected_checkpoint = _sha256(expected_checkpoint_sha256, field="expected_checkpoint_sha256")
    if checkpoint.sha256 != expected_checkpoint:
        raise ScientificTrainingBundleError("final checkpoint SHA256 does not match the expected linkage digest")

    # Validate the evidence once to derive exact episode references for the
    # manifest.  The complete bundle prevalidation below repeats this check
    # against the finished in-memory manifest before any top-level write.
    policy = _load_diagnostics_policy(artifacts["diagnostics_policy"], task=task, provenance=provenance)
    evidence, episode_artifacts, episode_references = _validate_training_evidence(
        artifacts["training_evidence_manifest"],
        root=root,
        task=task,
        provenance=provenance,
        policy=policy,
        expected_episodes=expected_episodes,
    )

    payload = {
        "schema_version": TRAINING_BUNDLE_SCHEMA_VERSION,
        "run_id": run.run_id,
        "run": canonical_value(run.identity_payload(), field="ResolvedRun"),
        "task": _task_payload(task),
        "provenance": provenance.to_dict(),
        "expected_revision": expected_revision,
        "training_contract": training_contract.to_dict(),
        "checkpoint_sha256": checkpoint.sha256,
        "episode_sha_assertions": _EPISODE_SHA_ASSERTIONS,
        "artifacts": {role: _artifact_payload(artifacts[role]) for role in sorted(artifacts)},
        "training_evidence": {
            "expected_episode_count": evidence["expected_episode_count"],
            "record_count": evidence["record_count"],
            "episodes": list(episode_references),
        },
    }
    contents = _formatted_json_bytes(payload)
    validated = _validate_bundle_payload(
        payload,
        contents=contents,
        root=root,
        bundle_path=destination,
        expected_run=run,
        task=task,
        expected_provenance=provenance,
        expected_revision=expected_revision,
        expected_episodes=expected_episodes,
        training_contract=training_contract,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    _atomic_create(destination, contents)
    return validated


def _provenance_from_payload(value: Any) -> SoftwareProvenance:
    raw = _mapping(value, field="training bundle provenance")
    expected_fields = {
        "schema_version",
        "repository",
        "revision",
        "dirty",
        "python_version",
        "energyplus_version",
        "container_digest",
        "dependencies",
    }
    if set(raw) != expected_fields or raw.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ScientificTrainingBundleError("training bundle provenance does not match the supported schema")
    if not isinstance(raw.get("dirty"), bool):
        raise ScientificTrainingBundleError("training bundle provenance dirty must be a boolean")
    dependencies = raw.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ScientificTrainingBundleError("training bundle provenance dependencies must be a JSON object")
    try:
        return SoftwareProvenance(
            repository=raw.get("repository"),
            revision=raw.get("revision"),
            dirty=raw.get("dirty"),
            python_version=raw.get("python_version"),
            energyplus_version=raw.get("energyplus_version"),
            container_digest=raw.get("container_digest"),
            dependencies=dependencies,
            schema_version=raw.get("schema_version"),
        )
    except (TypeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"invalid training bundle provenance: {error}") from error


def _validate_training_evidence_receipt(
    value: Any,
    *,
    artifacts: tuple[ValidatedTrainingArtifactReceipt, ...],
    contract: ExpectedTrainingContract,
) -> int:
    """Validate portable episode references without opening referenced files."""

    raw = _mapping(value, field="training bundle training_evidence")
    if set(raw) != {"expected_episode_count", "record_count", "episodes"}:
        raise ScientificTrainingBundleError("training bundle training_evidence does not match the supported schema")
    episodes = raw.get("episodes")
    if not isinstance(episodes, list):
        raise ScientificTrainingBundleError("training bundle training_evidence.episodes must be a JSON list")
    expected_count = _integer(
        raw.get("expected_episode_count"),
        field="training bundle training_evidence.expected_episode_count",
        positive=True,
    )
    record_count = _integer(
        raw.get("record_count"),
        field="training bundle training_evidence.record_count",
        positive=True,
    )
    if expected_count != record_count or record_count != len(episodes) or expected_count != contract.episode_count:
        raise ScientificTrainingBundleError(
            "training bundle training_evidence counts do not match its copied training contract"
        )

    artifact_by_role = {item.role: item for item in artifacts}
    evidence_manifest = artifact_by_role["training_evidence_manifest"]
    manifest_path = PurePosixPath(evidence_manifest.relative_path)
    if manifest_path.name != TRAINING_EVIDENCE_MANIFEST_FILENAME:
        raise ScientificTrainingBundleError(
            f"training evidence manifest must be named {TRAINING_EVIDENCE_MANIFEST_FILENAME!r}"
        )
    expected_keys = []
    for summary in contract.summary_episodes:
        calendar = thaw_mapping(summary.training_calendar)
        expected_keys.append(
            TrainingEpisodeKey(
                stream_id=f"train-env-{summary.environment_index:03d}",
                generation=0,
                schedule_seed=_integer(
                    calendar.get("schedule_seed"),
                    field="training bundle contract training_calendar.schedule_seed",
                ),
                episode_index=_integer(
                    calendar.get("episode_index"),
                    field="training bundle contract training_calendar.episode_index",
                ),
            )
        )
    expected_key_tuple = tuple(sorted(expected_keys, key=lambda item: item.sort_key()))
    if len(set(expected_key_tuple)) != len(expected_key_tuple):
        raise ScientificTrainingBundleError("training bundle receipt contains duplicate expected episode keys")

    parsed_keys: list[TrainingEpisodeKey] = []
    paths: list[str] = []
    for index, value in enumerate(episodes):
        field = f"training bundle training_evidence.episodes[{index}]"
        episode = _mapping(value, field=field)
        if set(episode) != {"episode_key", "path", "media_type", "sha256"}:
            raise ScientificTrainingBundleError(f"{field} does not match the supported schema")
        key = _episode_key(episode.get("episode_key"), field=f"{field}.episode_key")
        relative_path = _portable_relative_path(episode.get("path"), field=f"{field}.path")
        expected_path = (manifest_path.parent / _EPISODES_DIRECTORY_NAME / key.filename).as_posix()
        if relative_path != expected_path:
            raise ScientificTrainingBundleError(f"{field}.path is not canonical for its episode key")
        if episode.get("media_type") != "application/json":
            raise ScientificTrainingBundleError(f"{field}.media_type must be 'application/json'")
        _sha256(episode.get("sha256"), field=f"{field}.sha256")
        parsed_keys.append(key)
        paths.append(relative_path)
    if tuple(parsed_keys) != expected_key_tuple:
        raise ScientificTrainingBundleError(
            "training bundle training_evidence episode keys do not match its copied training contract"
        )
    top_level_paths = {item.relative_path for item in artifacts}
    if len(paths) != len(set(paths)) or top_level_paths.intersection(paths):
        raise ScientificTrainingBundleError("training bundle receipt contains duplicate artifact paths")
    return expected_count


def _portable_training_contract_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove only the execution-root path from an otherwise exact contract."""

    normalized = json.loads(json.dumps(value, allow_nan=False))
    run_config = normalized.get("run_config")
    resolved_trainer = normalized.get("resolved_trainer")
    ppo_args = resolved_trainer.get("ppo_args") if isinstance(resolved_trainer, Mapping) else None
    locations = []
    if isinstance(run_config, Mapping) and "results_dir" in run_config:
        locations.append((run_config, "results_dir"))
    if isinstance(ppo_args, Mapping) and "results_dir" in ppo_args:
        locations.append((ppo_args, "results_dir"))
    if not locations:
        return normalized
    if len(locations) != 2:
        raise ScientificTrainingBundleError("training bundle contract contains an incomplete PPO results_dir linkage")
    paths = [location[key] for location, key in locations]
    if any(not isinstance(path, str) or not path.strip() for path in paths) or len(set(paths)) != 1:
        raise ScientificTrainingBundleError(
            "training bundle contract PPO results_dir values are missing or inconsistent"
        )
    for location, key in locations:
        location[key] = "<portable-training-output-root>"
    return normalized


def _validate_bundle_payload(
    payload: Mapping[str, Any],
    *,
    contents: bytes,
    root: Path,
    bundle_path: Path,
    expected_run: ResolvedRun,
    task: TaskSpec,
    expected_provenance: SoftwareProvenance,
    expected_revision: str,
    expected_episodes: tuple[ExpectedTrainingEpisode, ...],
    training_contract: ExpectedTrainingContract,
    expected_checkpoint_sha256: str,
) -> ValidatedTrainingBundle:
    """Validate a complete in-memory manifest and all referenced files."""

    try:
        canonical_contents = _formatted_json_bytes(payload)
    except (TypeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"cannot canonicalize training bundle manifest: {error}") from error
    if contents != canonical_contents:
        raise ScientificTrainingBundleError(
            "training bundle manifest is not the byte-exact canonical training_bundle.json representation"
        )
    expected_revision = _validate_scientific_identity(
        run=expected_run,
        task=task,
        provenance=expected_provenance,
        expected_revision=expected_revision,
    )
    if not isinstance(training_contract, ExpectedTrainingContract):
        raise ScientificTrainingBundleError("training_contract must be an ExpectedTrainingContract")
    expected_episodes = _validate_expected_episodes(
        expected_episodes,
        task=task,
        contract=training_contract,
    )
    expected_fields = {
        "schema_version",
        "run_id",
        "run",
        "task",
        "provenance",
        "expected_revision",
        "training_contract",
        "checkpoint_sha256",
        "episode_sha_assertions",
        "artifacts",
        "training_evidence",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != TRAINING_BUNDLE_SCHEMA_VERSION:
        raise ScientificTrainingBundleError("training bundle manifest does not match the supported schema")
    if payload.get("run_id") != expected_run.run_id:
        raise ScientificTrainingBundleError("training bundle run_id does not match the expected ResolvedRun")
    if not _same_json(payload.get("run"), canonical_value(expected_run.identity_payload(), field="ResolvedRun")):
        raise ScientificTrainingBundleError("training bundle run does not match the expected ResolvedRun")
    if not _same_json(payload.get("task"), _task_payload(task)):
        raise ScientificTrainingBundleError("training bundle TaskSpec does not match the expected TaskSpec")
    if not _same_json(payload.get("training_contract"), training_contract.to_dict()):
        raise ScientificTrainingBundleError(
            "training bundle contract does not match the independently supplied training contract"
        )
    if not _same_json(payload.get("episode_sha_assertions"), _EPISODE_SHA_ASSERTIONS):
        raise ScientificTrainingBundleError("training bundle episode SHA assertion scope is missing or changed")
    provenance = _provenance_from_payload(payload.get("provenance"))
    if not _same_json(payload.get("provenance"), expected_provenance.to_dict()):
        raise ScientificTrainingBundleError("training bundle SoftwareProvenance does not match expected provenance")
    _validate_scientific_provenance(provenance, expected_revision=expected_revision)
    if payload.get("expected_revision") != expected_revision:
        raise ScientificTrainingBundleError("training bundle Git revision does not match the pinned expected revision")

    artifact_payloads = _mapping(payload.get("artifacts"), field="training bundle artifacts")
    if set(artifact_payloads) != set(_ARTIFACT_MEDIA_TYPES):
        raise ScientificTrainingBundleError("training bundle must contain exactly the required artifact roles")
    artifacts = {
        role: _artifact_from_reference(root, artifact_payloads[role], role=role) for role in sorted(artifact_payloads)
    }
    artifact_paths = [item.relative_path for item in artifacts.values()]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ScientificTrainingBundleError("training bundle artifact roles reference duplicate files")
    checkpoint = artifacts["final_checkpoint"]
    manifest_checkpoint = _sha256(payload.get("checkpoint_sha256"), field="checkpoint_sha256")
    if manifest_checkpoint != checkpoint.sha256:
        raise ScientificTrainingBundleError("training bundle checkpoint linkage SHA256 is inconsistent")
    expected_checkpoint = _sha256(expected_checkpoint_sha256, field="expected_checkpoint_sha256")
    if checkpoint.sha256 != expected_checkpoint:
        raise ScientificTrainingBundleError("training checkpoint SHA256 does not match evaluation linkage")

    _validate_run_config(
        artifacts["run_config"].path,
        run=expected_run,
        task=task,
        contract=training_contract,
    )
    _validate_update_metrics(
        artifacts["update_metrics"],
        run=expected_run,
        task=task,
        contract=training_contract,
    )
    _validate_resolved_input(
        artifacts["resolved_experiment"],
        expected=training_contract.resolved_experiment_mapping,
        field="resolved_experiment",
    )
    _validate_resolved_input(
        artifacts["resolved_trainer"],
        expected=training_contract.resolved_trainer_mapping,
        field="resolved_trainer",
    )
    policy = _load_diagnostics_policy(artifacts["diagnostics_policy"], task=task, provenance=provenance)
    evidence, episode_artifacts, expected_episode_references = _validate_training_evidence(
        artifacts["training_evidence_manifest"],
        root=root,
        task=task,
        provenance=provenance,
        policy=policy,
        expected_episodes=expected_episodes,
    )
    training_evidence = _mapping(payload.get("training_evidence"), field="training bundle training_evidence")
    if set(training_evidence) != {"expected_episode_count", "record_count", "episodes"}:
        raise ScientificTrainingBundleError("training bundle training_evidence does not match the supported schema")
    if not _same_json(
        training_evidence,
        {
            "expected_episode_count": evidence["expected_episode_count"],
            "record_count": evidence["record_count"],
            "episodes": list(expected_episode_references),
        },
    ):
        raise ScientificTrainingBundleError(
            "training bundle episode references do not exactly authenticate the training evidence manifest"
        )
    _validate_training_summary(
        artifacts["training_summary"].path,
        run=expected_run,
        task=task,
        run_config=artifacts["run_config"],
        checkpoint=checkpoint,
        update_metrics=artifacts["update_metrics"],
        contract=training_contract,
    )
    all_paths = artifact_paths + [item.relative_path for item in episode_artifacts]
    if len(all_paths) != len(set(all_paths)):
        raise ScientificTrainingBundleError("training bundle contains duplicate artifact paths")
    _assert_distinct_inodes(tuple(artifacts.values()) + episode_artifacts)

    return ValidatedTrainingBundle(
        bundle_path=bundle_path,
        bundle_sha256=_sha256_bytes(contents),
        run=expected_run,
        task=task,
        provenance=provenance,
        contract=training_contract,
        expected_episodes=expected_episodes,
        artifacts=tuple(artifacts[role] for role in sorted(artifacts)),
        episode_artifacts=episode_artifacts,
        expected_episode_count=evidence["expected_episode_count"],
        training_evidence=evidence,
    )


def validate_training_bundle_receipt(
    contents: bytes,
    *,
    expected_run: ResolvedRun,
    task: TaskSpec,
    expected_provenance: SoftwareProvenance,
    expected_revision: str,
    expected_checkpoint_sha256: str,
    expected_training_contract: ExpectedTrainingContract,
) -> ValidatedTrainingBundleReceipt:
    """Authenticate byte-exact copied ``training_bundle.json`` contents.

    This is the reporting-side counterpart to :func:`validate_training_bundle`.
    It requires independently derived run, source-task, provenance, revision,
    and checkpoint identities, while validating all portable references and
    the complete manifest schema.  It deliberately does not resolve, open, or
    checksum the original referenced files: only the copied manifest is
    expected to travel with an evaluation bundle.
    """

    normalized_revision = _validate_scientific_identity(
        run=expected_run,
        task=task,
        provenance=expected_provenance,
        expected_revision=expected_revision,
    )
    expected_checkpoint = _sha256(expected_checkpoint_sha256, field="expected_checkpoint_sha256")
    if not isinstance(expected_training_contract, ExpectedTrainingContract):
        raise ScientificTrainingBundleError(
            "expected_training_contract must be an independently derived ExpectedTrainingContract"
        )
    if not isinstance(contents, bytes):
        raise ScientificTrainingBundleError("training bundle receipt contents must be exact bytes")
    try:
        decoded = contents.decode("utf-8", errors="strict")
        payload = _mapping(_strict_json(decoded), field="training bundle receipt")
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"cannot decode training bundle receipt: {error}") from error
    try:
        canonical_contents = _formatted_json_bytes(payload)
    except (TypeError, ValueError) as error:
        raise ScientificTrainingBundleError(f"cannot canonicalize training bundle receipt: {error}") from error
    if contents != canonical_contents:
        raise ScientificTrainingBundleError(
            "training bundle receipt is not the byte-exact canonical training_bundle.json representation"
        )

    expected_fields = {
        "schema_version",
        "run_id",
        "run",
        "task",
        "provenance",
        "expected_revision",
        "training_contract",
        "checkpoint_sha256",
        "episode_sha_assertions",
        "artifacts",
        "training_evidence",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != TRAINING_BUNDLE_SCHEMA_VERSION:
        raise ScientificTrainingBundleError("training bundle receipt does not match the supported schema")
    if payload.get("run_id") != expected_run.run_id:
        raise ScientificTrainingBundleError("training bundle receipt run_id does not match the expected ResolvedRun")
    if not _same_json(payload.get("run"), canonical_value(expected_run.identity_payload(), field="ResolvedRun")):
        raise ScientificTrainingBundleError("training bundle receipt run does not match the expected ResolvedRun")
    if not _same_json(payload.get("task"), _task_payload(task)):
        raise ScientificTrainingBundleError("training bundle receipt TaskSpec does not match the expected TaskSpec")

    provenance = _provenance_from_payload(payload.get("provenance"))
    if not _same_json(payload.get("provenance"), expected_provenance.to_dict()):
        raise ScientificTrainingBundleError(
            "training bundle receipt SoftwareProvenance does not match expected provenance"
        )
    _validate_scientific_provenance(provenance, expected_revision=normalized_revision)
    if payload.get("expected_revision") != normalized_revision:
        raise ScientificTrainingBundleError("training bundle receipt Git revision does not match expected revision")
    if not _same_json(payload.get("episode_sha_assertions"), _EPISODE_SHA_ASSERTIONS):
        raise ScientificTrainingBundleError("training bundle receipt episode SHA assertion scope is missing or changed")

    contract_payload = _mapping(payload.get("training_contract"), field="training bundle contract")
    if not _same_json(
        _portable_training_contract_identity(contract_payload),
        _portable_training_contract_identity(expected_training_contract.to_dict()),
    ):
        raise ScientificTrainingBundleError(
            "training bundle receipt contract does not match the independently derived trainer plan"
        )
    contract = expected_training_contract
    artifact_payloads = _mapping(payload.get("artifacts"), field="training bundle receipt artifacts")
    if set(artifact_payloads) != set(_ARTIFACT_MEDIA_TYPES):
        raise ScientificTrainingBundleError("training bundle receipt must contain exactly the required artifact roles")
    artifacts = tuple(
        _artifact_receipt_from_reference(artifact_payloads[role], role=role) for role in sorted(artifact_payloads)
    )
    artifact_paths = [item.relative_path for item in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ScientificTrainingBundleError("training bundle receipt artifact roles reference duplicate paths")
    checkpoint = tuple(item for item in artifacts if item.role == "final_checkpoint")[0]
    manifest_checkpoint = _sha256(payload.get("checkpoint_sha256"), field="checkpoint_sha256")
    if manifest_checkpoint != checkpoint.sha256:
        raise ScientificTrainingBundleError("training bundle receipt checkpoint linkage SHA256 is inconsistent")
    if checkpoint.sha256 != expected_checkpoint:
        raise ScientificTrainingBundleError(
            "training bundle receipt checkpoint SHA256 does not match evaluation linkage"
        )
    expected_episode_count = _validate_training_evidence_receipt(
        payload.get("training_evidence"),
        artifacts=artifacts,
        contract=contract,
    )
    return ValidatedTrainingBundleReceipt(
        bundle_sha256=_sha256_bytes(contents),
        run=expected_run,
        task=task,
        provenance=provenance,
        expected_revision=normalized_revision,
        contract=contract,
        artifacts=artifacts,
        expected_episode_count=expected_episode_count,
    )


def validate_training_bundle(
    bundle_path: str | Path,
    *,
    expected_run: ResolvedRun,
    task: TaskSpec,
    expected_provenance: SoftwareProvenance,
    expected_revision: str,
    expected_episodes: tuple[ExpectedTrainingEpisode, ...],
    training_contract: ExpectedTrainingContract,
    expected_checkpoint_sha256: str,
) -> ValidatedTrainingBundle:
    """Authenticate a sealed bundle against independently derived inputs.

    ``expected_episodes`` and ``training_contract`` must come from the launch
    plan, not from the bundle being checked.  Checkpoint internals remain an
    explicit algorithm-adapter boundary; this function authenticates their
    exact byte digest only.
    """

    path = Path(bundle_path).expanduser()
    if path.name != TRAINING_BUNDLE_FILENAME:
        raise ScientificTrainingBundleError(f"training bundle must be named {TRAINING_BUNDLE_FILENAME!r}")
    root = _root_directory(path.parent)
    resolved_path, relative = _assert_real_path(root, path, field="training bundle manifest")
    if relative != TRAINING_BUNDLE_FILENAME:
        raise ScientificTrainingBundleError("training bundle manifest must be a top-level sibling artifact")
    _assert_single_filesystem_link(resolved_path, field="training bundle manifest")
    payload, contents = _read_json_bytes(resolved_path, field="training bundle manifest")
    return _validate_bundle_payload(
        payload,
        contents=contents,
        root=root,
        bundle_path=resolved_path,
        expected_run=expected_run,
        task=task,
        expected_provenance=expected_provenance,
        expected_revision=expected_revision,
        expected_episodes=expected_episodes,
        training_contract=training_contract,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )


__all__ = [
    "ExpectedTrainingContract",
    "ExpectedTrainingSummaryEpisode",
    "ExpectedTrainingUpdate",
    "ScientificTrainingBundleError",
    "FINAL_CHECKPOINT_SELECTION_RULE_TEMPLATE",
    "TRAINING_BUNDLE_FILENAME",
    "TRAINING_BUNDLE_SCHEMA_VERSION",
    "TRAINING_CONTRACT_SCHEMA_VERSION",
    "ValidatedTrainingArtifact",
    "ValidatedTrainingBundle",
    "ValidatedTrainingBundleReceipt",
    "final_checkpoint_selection_rule",
    "validate_training_bundle",
    "validate_training_bundle_receipt",
    "write_training_bundle",
]
