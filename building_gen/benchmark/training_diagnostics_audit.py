"""Engineering-only audit of archived calendar-training diagnostics.

This module deliberately depends only on the Python standard library and the
simulator-independent EnergyPlus diagnostics schema.  It summarizes warning
signatures observed during a calendar training run, but it never approves
those warnings and never creates a scientific diagnostics allowlist.

Archived diagnostics do not embed the immutable source building and weather
checksums.  The audit therefore verifies caller-supplied building and weather
files against required SHA256 values, while explicitly remaining
non-publication evidence.  Scientific training evidence must establish the
episode-to-task linkage separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from building_gen.benchmark.energyplus_diagnostics import (
    DIAGNOSTICS_FILENAME,
    DIAGNOSTICS_SCHEMA_VERSION,
    EnergyPlusDiagnostics,
    canonical_energyplus_diagnostics_snapshot,
    energyplus_diagnostics_from_mapping,
    parse_energyplus_diagnostics,
)

TRAINING_DIAGNOSTICS_AUDIT_SCHEMA_VERSION = "hot-training-diagnostics-engineering-audit/1.0.0"
TRAINING_DIAGNOSTICS_AUDIT_JSON = "training_diagnostics_audit.json"
TRAINING_DIAGNOSTICS_AUDIT_MARKDOWN = "training_diagnostics_audit.md"
ENGINEERING_DISCLAIMER = (
    "ENGINEERING-ONLY / NON-PUBLICATION: this audit inventories warnings; "
    "it does not approve any warning, create a scientific allowlist, or prove "
    "per-episode building/weather lineage."
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CALENDAR_RE = re.compile(r"^(?P<task>.+)__calendar-seed-(?P<seed>[0-9]+)$")
_GENERATION_RE = re.compile(r"^generation-(?P<generation>[0-9]{4,})$")
_EPISODE_RE = re.compile(r"^episode-(?P<episode>[0-9]{8,})__(?P<window>[A-Za-z0-9][A-Za-z0-9_.-]*)$")
_ENERGYPLUS_OUTPUT_FILENAMES = (
    "eplusout.err",
    "eplusout.end",
    DIAGNOSTICS_FILENAME,
)


class TrainingDiagnosticsAuditInputError(ValueError):
    """Raised when an audit input cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class ScopedAsset:
    """One byte-verified source asset supplied to the engineering audit."""

    role: str
    filename: str
    expected_sha256: str
    actual_sha256: str
    size_bytes: int
    matches_expected: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodePathIdentity:
    """Logical calendar episode encoded by the archived directory suffix."""

    task_name: str
    schedule_seed: int
    generation: int
    episode_index: int
    window_name: str

    @property
    def duplicate_key(self) -> tuple[str, int, int, int]:
        # Window is deliberately excluded: two paths claiming different
        # windows for the same task/seed/generation/index are also duplicates.
        return (
            self.task_name,
            self.schedule_seed,
            self.generation,
            self.episode_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArchivedOutputPresence:
    """Presence and content identity of one expected EnergyPlus output."""

    filename: str
    present: bool
    file_sha256: str | None
    size_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuditedDiagnosticsRecord:
    """One complete, incomplete, or rejected EnergyPlus episode directory."""

    path: str
    classification: str
    violations: tuple[str, ...]
    file_sha256: str | None = None
    episode: EpisodePathIdentity | None = None
    energyplus_version: str | None = None
    executed_model_sha256: str | None = None
    total_warning_count: int | None = None
    marked_warning_count: int | None = None
    recurring_warning_count: int | None = None
    archived_outputs: tuple[ArchivedOutputPresence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.episode is not None:
            payload["episode"] = self.episode.to_dict()
        payload["archived_outputs"] = [item.to_dict() for item in self.archived_outputs]
        return payload


@dataclass(frozen=True, slots=True)
class WarningAggregate:
    """Per-signature counts across complete calendar episodes only."""

    signature: str
    maximum_occurrences_per_episode: int
    total_occurrences: int
    episodes_containing_signature: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PathViolation:
    """A path that prevents complete, unambiguous archive discovery."""

    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticsAudit:
    """Deterministic result of one engineering-only diagnostics audit."""

    building: ScopedAsset
    weather: ScopedAsset
    expected_energyplus_version: str
    expected_complete_diagnostics_count: int
    allow_extra_incomplete_requested: bool
    engineering_integrity_checks_passed: bool
    input_scope_verified: bool
    warning_review_required: bool
    complete_total_warning_count: int
    complete_maximum_warning_count_per_episode: int
    complete_diagnostics: tuple[AuditedDiagnosticsRecord, ...]
    extra_incomplete_diagnostics: tuple[AuditedDiagnosticsRecord, ...]
    rejected_diagnostics: tuple[AuditedDiagnosticsRecord, ...]
    path_violations: tuple[PathViolation, ...]
    marked_warning_aggregates: tuple[WarningAggregate, ...]
    recurring_warning_aggregates: tuple[WarningAggregate, ...]
    violations: tuple[str, ...]
    engineering_exceptions: tuple[str, ...]
    schema_version: str = TRAINING_DIAGNOSTICS_AUDIT_SCHEMA_VERSION

    @property
    def publication_ready(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_mode": "engineering_non_publication",
            "disclaimer": ENGINEERING_DISCLAIMER,
            "publication_ready": False,
            "warnings_approved": False,
            "scientific_allowlist_written": False,
            "engineering_integrity_checks_passed": self.engineering_integrity_checks_passed,
            "input_scope_verified": self.input_scope_verified,
            "warning_review_required": self.warning_review_required,
            "scope": {
                "building": self.building.to_dict(),
                "weather": self.weather.to_dict(),
                "expected_energyplus_version": self.expected_energyplus_version,
                "scope_limitation": (
                    "The source files are checksum-verified audit inputs, but archived "
                    "diagnostics do not establish per-episode source-asset lineage."
                ),
            },
            "coverage": {
                "expected_complete_diagnostics_count": self.expected_complete_diagnostics_count,
                "energyplus_episode_directories_count": (
                    len(self.complete_diagnostics)
                    + len(self.extra_incomplete_diagnostics)
                    + len(self.rejected_diagnostics)
                ),
                "complete_diagnostics_count": len(self.complete_diagnostics),
                "extra_incomplete_diagnostics_count": len(self.extra_incomplete_diagnostics),
                "rejected_diagnostics_count": len(self.rejected_diagnostics),
                "allow_extra_incomplete_requested": self.allow_extra_incomplete_requested,
                "required_archived_outputs": list(_ENERGYPLUS_OUTPUT_FILENAMES),
            },
            "warning_summary_complete_episodes_only": {
                "total_warning_count": self.complete_total_warning_count,
                "maximum_warning_count_per_episode": self.complete_maximum_warning_count_per_episode,
                "marked_signatures": [item.to_dict() for item in self.marked_warning_aggregates],
                "recurring_signatures": [item.to_dict() for item in self.recurring_warning_aggregates],
            },
            "complete_diagnostics": [item.to_dict() for item in self.complete_diagnostics],
            "extra_incomplete_diagnostics": [item.to_dict() for item in self.extra_incomplete_diagnostics],
            "rejected_diagnostics": [item.to_dict() for item in self.rejected_diagnostics],
            "path_violations": [item.to_dict() for item in self.path_violations],
            "violations": list(self.violations),
            "engineering_exceptions": list(self.engineering_exceptions),
        }

    def to_json(self) -> str:
        """Return stable, human-readable JSON with no timestamps or host paths."""

        return json.dumps(self.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _ArchivedOutputCandidate:
    filename: str
    path: Path
    relative_path: str
    present: bool
    file_sha256: str | None
    size_bytes: int | None
    filesystem_identity: tuple[int, int] | None
    violations: tuple[str, ...]

    def public_presence(self) -> ArchivedOutputPresence:
        return ArchivedOutputPresence(
            filename=self.filename,
            present=self.present,
            file_sha256=self.file_sha256,
            size_bytes=self.size_bytes,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    energyplus_dir: Path
    relative_directory: str
    path: Path
    relative_path: str
    identity: EpisodePathIdentity | None
    file_sha256: str | None
    preliminary_violations: tuple[str, ...]
    outputs: tuple[_ArchivedOutputCandidate, ...]

    def output(self, filename: str) -> _ArchivedOutputCandidate:
        for item in self.outputs:
            if item.filename == filename:
                return item
        raise KeyError(filename)

    @property
    def output_inventory(self) -> tuple[ArchivedOutputPresence, ...]:
        return tuple(item.public_presence() for item in self.outputs)


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _expected_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    digest = value.casefold()
    if not _SHA256_RE.fullmatch(digest):
        raise TrainingDiagnosticsAuditInputError(f"{field} must contain exactly 64 hexadecimal characters")
    return digest


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _real_file(path: str | Path, *, role: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise TrainingDiagnosticsAuditInputError(f"{role} must not be a symbolic link: {unresolved}")
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise TrainingDiagnosticsAuditInputError(f"cannot resolve {role} {unresolved}: {error}") from error
    if not resolved.is_file():
        raise TrainingDiagnosticsAuditInputError(f"{role} must be a regular file: {resolved}")
    return resolved


def _scoped_asset(
    path: str | Path,
    *,
    role: str,
    expected_sha256: str,
) -> ScopedAsset:
    resolved = _real_file(path, role=role)
    expected = _expected_sha256(expected_sha256, field=f"expected_{role}_sha256")
    actual = _sha256_file(resolved)
    return ScopedAsset(
        role=role,
        filename=resolved.name,
        expected_sha256=expected,
        actual_sha256=actual,
        size_bytes=resolved.stat().st_size,
        matches_expected=actual == expected,
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise TrainingDiagnosticsAuditInputError(f"diagnostics candidate escaped archive root: {path}") from error
    value = relative.as_posix()
    if not value or value.startswith("/") or ".." in relative.parts:
        raise TrainingDiagnosticsAuditInputError(f"invalid archive-relative path: {value!r}")
    return value


def _episode_identity(relative_path: str) -> EpisodePathIdentity:
    parts = Path(relative_path).parts
    if len(parts) < 5 or parts[-1] != DIAGNOSTICS_FILENAME or parts[-2] != "energyplus":
        raise TrainingDiagnosticsAuditInputError(
            "diagnostics path must end in "
            "<task>__calendar-seed-N/generation-NNNN/"
            f"episode-NNNNNNNN__<window>/energyplus/{DIAGNOSTICS_FILENAME}"
        )
    calendar_match = _CALENDAR_RE.fullmatch(parts[-5])
    generation_match = _GENERATION_RE.fullmatch(parts[-4])
    episode_match = _EPISODE_RE.fullmatch(parts[-3])
    if calendar_match is None or generation_match is None or episode_match is None:
        raise TrainingDiagnosticsAuditInputError(
            "diagnostics path has a malformed calendar episode suffix: " f"{relative_path}"
        )
    return EpisodePathIdentity(
        task_name=calendar_match.group("task"),
        schedule_seed=int(calendar_match.group("seed")),
        generation=int(generation_match.group("generation")),
        episode_index=int(episode_match.group("episode")),
        window_name=episode_match.group("window"),
    )


def _discover_candidates(root: Path) -> tuple[list[_Candidate], tuple[PathViolation, ...]]:
    discovered_directories: list[Path] = []
    path_violations: list[PathViolation] = []

    def walk_error(error: OSError) -> None:
        filename = getattr(error, "filename", None)
        if filename:
            try:
                label = _relative_path(Path(filename), root)
            except TrainingDiagnosticsAuditInputError:
                label = str(filename)
        else:
            label = "."
        path_violations.append(PathViolation(label, f"archive traversal failed: {error}"))

    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
        current_path = Path(current)
        directories.sort()
        filenames.sort()
        if current_path.name == "energyplus":
            discovered_directories.append(current_path)
        elif any(name in _ENERGYPLUS_OUTPUT_FILENAMES for name in filenames):
            for name in filenames:
                if name in _ENERGYPLUS_OUTPUT_FILENAMES:
                    path_violations.append(
                        PathViolation(
                            _relative_path(current_path / name, root),
                            "EnergyPlus output is outside an episode energyplus directory",
                        )
                    )
        retained_directories: list[str] = []
        for name in directories:
            child = current_path / name
            if child.is_symlink():
                path_violations.append(
                    PathViolation(
                        _relative_path(child, root),
                        "symbolic-link directories are not traversed in a fail-closed audit",
                    )
                )
            else:
                retained_directories.append(name)
        directories[:] = retained_directories

    candidates: list[_Candidate] = []
    for energyplus_dir in sorted(discovered_directories, key=lambda item: _relative_path(item, root)):
        relative_directory = _relative_path(energyplus_dir, root)
        path = energyplus_dir / DIAGNOSTICS_FILENAME
        relative = _relative_path(path, root)
        violations: list[str] = []
        identity: EpisodePathIdentity | None = None
        outputs: list[_ArchivedOutputCandidate] = []
        for filename in _ENERGYPLUS_OUTPUT_FILENAMES:
            output_path = energyplus_dir / filename
            output_relative = _relative_path(output_path, root)
            output_violations: list[str] = []
            present = output_path.is_symlink() or output_path.exists()
            digest: str | None = None
            size_bytes: int | None = None
            filesystem_identity: tuple[int, int] | None = None
            if present:
                if output_path.is_symlink():
                    output_violations.append(f"{filename} is a symbolic link")
                else:
                    try:
                        status = output_path.stat(follow_symlinks=False)
                    except OSError as error:
                        output_violations.append(f"cannot stat {filename}: {error}")
                    else:
                        if not output_path.is_file():
                            output_violations.append(f"{filename} is not a regular file")
                        else:
                            size_bytes = status.st_size
                            filesystem_identity = (status.st_dev, status.st_ino)
                            if status.st_nlink != 1:
                                output_violations.append(
                                    f"{filename} has multiple hard links and is not a unique archive path"
                                )
                            if status.st_size == 0:
                                output_violations.append(f"{filename} is empty")
                            try:
                                digest = _sha256_file(output_path)
                            except OSError as error:
                                output_violations.append(f"cannot hash {filename}: {error}")
            outputs.append(
                _ArchivedOutputCandidate(
                    filename=filename,
                    path=output_path,
                    relative_path=output_relative,
                    present=present,
                    file_sha256=digest,
                    size_bytes=size_bytes,
                    filesystem_identity=filesystem_identity,
                    violations=tuple(output_violations),
                )
            )
            violations.extend(output_violations)
        try:
            identity = _episode_identity(relative)
        except TrainingDiagnosticsAuditInputError as error:
            violations.append(str(error))
        candidates.append(
            _Candidate(
                energyplus_dir=energyplus_dir,
                relative_directory=relative_directory,
                path=path,
                relative_path=relative,
                identity=identity,
                file_sha256=next(item.file_sha256 for item in outputs if item.filename == DIAGNOSTICS_FILENAME),
                preliminary_violations=tuple(violations),
                outputs=tuple(outputs),
            )
        )

    logical_paths: defaultdict[tuple[str, int, int, int], list[str]] = defaultdict(list)
    filesystem_paths: defaultdict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for candidate in candidates:
        if candidate.identity is not None:
            logical_paths[candidate.identity.duplicate_key].append(candidate.relative_path)
        for output in candidate.outputs:
            if output.filesystem_identity is not None:
                filesystem_paths[output.filesystem_identity].append((candidate.relative_path, output.relative_path))

    duplicate_reasons: defaultdict[str, list[str]] = defaultdict(list)
    for paths in logical_paths.values():
        if len(paths) > 1:
            labels = sorted(paths)
            reason = "duplicate logical calendar episode paths: " + ", ".join(labels)
            for path in labels:
                duplicate_reasons[path].append(reason)
    for owners in filesystem_paths.values():
        if len(owners) > 1:
            labels = sorted(output_path for _, output_path in owners)
            reason = "multiple archive outputs alias the same filesystem object: " + ", ".join(labels)
            for candidate_path, _output_path in owners:
                duplicate_reasons[candidate_path].append(reason)

    final_candidates = [
        _Candidate(
            energyplus_dir=item.energyplus_dir,
            relative_directory=item.relative_directory,
            path=item.path,
            relative_path=item.relative_path,
            identity=item.identity,
            file_sha256=item.file_sha256,
            preliminary_violations=(item.preliminary_violations + tuple(sorted(duplicate_reasons[item.relative_path]))),
            outputs=item.outputs,
        )
        for item in candidates
    ]
    return final_candidates, tuple(sorted(path_violations, key=lambda item: (item.path, item.reason)))


def _strict_json_object(raw_bytes: bytes, *, path: str) -> Mapping[str, Any]:
    def object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise TrainingDiagnosticsAuditInputError(f"invalid diagnostics JSON at {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TrainingDiagnosticsAuditInputError(f"diagnostics JSON at {path} must be an object")
    return payload


def _diagnostics_from_candidate(candidate: _Candidate) -> tuple[EnergyPlusDiagnostics, str]:
    try:
        raw_bytes = candidate.path.read_bytes()
    except OSError as error:
        raise TrainingDiagnosticsAuditInputError(
            f"cannot read diagnostics JSON at {candidate.relative_path}: {error}"
        ) from error
    if candidate.file_sha256 is None or _sha256_bytes(raw_bytes) != candidate.file_sha256:
        raise TrainingDiagnosticsAuditInputError(
            f"diagnostics file changed while it was being audited: {candidate.relative_path}"
        )
    payload = _strict_json_object(raw_bytes, path=candidate.relative_path)
    executed_model_sha256 = payload.get("executed_model_sha256")
    if not isinstance(executed_model_sha256, str) or not _SHA256_RE.fullmatch(executed_model_sha256):
        raise TrainingDiagnosticsAuditInputError(
            f"executed_model_sha256 at {candidate.relative_path} must be 64 lowercase hexadecimal characters"
        )
    try:
        diagnostics = energyplus_diagnostics_from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise TrainingDiagnosticsAuditInputError(
            f"invalid diagnostics payload at {candidate.relative_path}: {error}"
        ) from error
    snapshot_payload = json.loads(canonical_energyplus_diagnostics_snapshot(diagnostics).canonical_json.decode("utf-8"))
    expected_fields = set(snapshot_payload) | {"executed_model_sha256"}
    if set(payload) != expected_fields:
        raise TrainingDiagnosticsAuditInputError(
            f"diagnostics fields at {candidate.relative_path} do not match their embedded exact schema: "
            f"missing={sorted(expected_fields - set(payload))!r}, "
            f"unexpected={sorted(set(payload) - expected_fields)!r}"
        )
    persisted_core = dict(payload)
    persisted_core.pop("executed_model_sha256")
    if persisted_core != snapshot_payload:
        raise TrainingDiagnosticsAuditInputError(f"diagnostics fields at {candidate.relative_path} are not canonical")
    return energyplus_diagnostics_from_mapping(snapshot_payload), executed_model_sha256


def _completion_violations(diagnostics: EnergyPlusDiagnostics) -> list[str]:
    violations: list[str] = []
    if not diagnostics.err_completion_present:
        violations.append("EnergyPlus err log has no completion record")
    if not diagnostics.end_completion_present:
        violations.append("EnergyPlus end file has no completion record")
    if not diagnostics.completion_records_agree:
        violations.append("EnergyPlus err/end completion records do not agree")
    if not diagnostics.completed_successfully:
        violations.append("EnergyPlus did not complete successfully")
    return violations


def _safe_zero_transition_incomplete(
    diagnostics: EnergyPlusDiagnostics,
    *,
    expected_energyplus_version: str,
) -> bool:
    """Recognize only diagnostics with no evidence of a consumed episode."""

    return (
        bool(_completion_violations(diagnostics))
        and not diagnostics.err_completion_present
        and not diagnostics.end_completion_present
        and diagnostics.schema_version == DIAGNOSTICS_SCHEMA_VERSION
        and diagnostics.energyplus_version in {None, expected_energyplus_version}
        and diagnostics.severe_error_count == 0
        and diagnostics.fatal_error_count == 0
        and not diagnostics.warning_occurrences
        and not diagnostics.recurring_warning_occurrences
        and not diagnostics.severe_occurrences
        and not diagnostics.fatal_occurrences
        and diagnostics.accounted_warning_count == 0
        and diagnostics.total_warning_count in {None, 0}
        and diagnostics.unattributed_warning_count in {None, 0}
    )


def _record_from_diagnostics(
    candidate: _Candidate,
    diagnostics: EnergyPlusDiagnostics,
    executed_model_sha256: str | None,
    *,
    expected_energyplus_version: str,
) -> AuditedDiagnosticsRecord:
    completion_violations = _completion_violations(diagnostics)
    violations: list[str] = []
    if diagnostics.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
        violations.append(
            f"unsupported diagnostics schema {diagnostics.schema_version!r}; "
            f"expected {DIAGNOSTICS_SCHEMA_VERSION!r}"
        )
    if diagnostics.energyplus_version != expected_energyplus_version:
        # A version-less safe incomplete child is retained as an explicit
        # extra below.  A complete file, or any asserted wrong version,
        # remains a hard failure.
        if not completion_violations or diagnostics.energyplus_version is not None:
            violations.append(
                f"EnergyPlus version {diagnostics.energyplus_version!r} does not match "
                f"expected scope {expected_energyplus_version!r}"
            )
    if diagnostics.severe_error_count:
        violations.append(f"EnergyPlus reported {diagnostics.severe_error_count} severe error(s)")
    if diagnostics.fatal_error_count:
        violations.append(f"EnergyPlus reported {diagnostics.fatal_error_count} fatal error(s)")
    if diagnostics.total_warning_count is None and not completion_violations:
        violations.append("completed diagnostics have no EnergyPlus warning summary")
    if diagnostics.unattributed_warning_count not in {0, None}:
        violations.append(f"EnergyPlus has {diagnostics.unattributed_warning_count} unattributed warning occurrence(s)")

    marked_count = sum(item.count for item in diagnostics.warning_occurrences)
    recurring_count = sum(item.count for item in diagnostics.recurring_warning_occurrences)
    if completion_violations and marked_count + recurring_count:
        violations.append(
            "incomplete EnergyPlus outputs contain " f"{marked_count + recurring_count} warning occurrence(s)"
        )
    common = {
        "path": candidate.relative_path,
        "file_sha256": candidate.file_sha256,
        "episode": candidate.identity,
        "energyplus_version": diagnostics.energyplus_version,
        "executed_model_sha256": executed_model_sha256,
        "total_warning_count": diagnostics.total_warning_count,
        "marked_warning_count": marked_count,
        "recurring_warning_count": recurring_count,
        "archived_outputs": candidate.output_inventory,
    }
    if not completion_violations and not violations:
        return AuditedDiagnosticsRecord(
            classification="complete",
            violations=(),
            **common,
        )
    if not violations and _safe_zero_transition_incomplete(
        diagnostics,
        expected_energyplus_version=expected_energyplus_version,
    ):
        return AuditedDiagnosticsRecord(
            classification="extra_incomplete",
            violations=tuple(completion_violations),
            **common,
        )
    return AuditedDiagnosticsRecord(
        classification="rejected",
        violations=tuple(violations + completion_violations),
        **common,
    )


def _rejected_candidate(candidate: _Candidate, violations: Sequence[str]) -> AuditedDiagnosticsRecord:
    return AuditedDiagnosticsRecord(
        path=candidate.relative_path,
        classification="rejected",
        violations=tuple(violations),
        file_sha256=candidate.file_sha256,
        episode=candidate.identity,
        archived_outputs=candidate.output_inventory,
    )


def _read_archived_text(output: _ArchivedOutputCandidate) -> str:
    if not output.present:
        return ""
    try:
        raw_bytes = output.path.read_bytes()
    except OSError as error:
        raise TrainingDiagnosticsAuditInputError(
            f"cannot read archived EnergyPlus output {output.relative_path}: {error}"
        ) from error
    if output.file_sha256 is None or _sha256_bytes(raw_bytes) != output.file_sha256:
        raise TrainingDiagnosticsAuditInputError(
            f"archived EnergyPlus output changed while being audited: {output.relative_path}"
        )
    return raw_bytes.decode("utf-8", errors="replace")


def _record_without_persisted_diagnostics(
    candidate: _Candidate,
    *,
    expected_energyplus_version: str,
) -> tuple[AuditedDiagnosticsRecord, EnergyPlusDiagnostics | None]:
    missing = tuple(item.filename for item in candidate.outputs if not item.present)
    missing_reason = "missing archived EnergyPlus output(s): " + ", ".join(missing)
    err_output = candidate.output("eplusout.err")
    end_output = candidate.output("eplusout.end")
    if not err_output.present:
        return (
            _rejected_candidate(
                candidate,
                (missing_reason, "an incomplete extra requires a real eplusout.err for review"),
            ),
            None,
        )
    try:
        diagnostics = parse_energyplus_diagnostics(
            _read_archived_text(err_output),
            _read_archived_text(end_output),
        )
    except (OSError, TypeError, ValueError, TrainingDiagnosticsAuditInputError) as error:
        return (
            _rejected_candidate(
                candidate,
                (missing_reason, f"cannot parse incomplete EnergyPlus outputs: {error}"),
            ),
            None,
        )
    record = _record_from_diagnostics(
        candidate,
        diagnostics,
        None,
        expected_energyplus_version=expected_energyplus_version,
    )
    if record.classification == "extra_incomplete":
        return (
            AuditedDiagnosticsRecord(
                path=record.path,
                classification=record.classification,
                violations=record.violations + (missing_reason,),
                file_sha256=record.file_sha256,
                episode=record.episode,
                energyplus_version=record.energyplus_version,
                executed_model_sha256=record.executed_model_sha256,
                total_warning_count=record.total_warning_count,
                marked_warning_count=record.marked_warning_count,
                recurring_warning_count=record.recurring_warning_count,
                archived_outputs=record.archived_outputs,
            ),
            diagnostics,
        )
    return (
        AuditedDiagnosticsRecord(
            path=record.path,
            classification="rejected",
            violations=record.violations + (missing_reason,),
            file_sha256=record.file_sha256,
            episode=record.episode,
            energyplus_version=record.energyplus_version,
            executed_model_sha256=record.executed_model_sha256,
            total_warning_count=record.total_warning_count,
            marked_warning_count=record.marked_warning_count,
            recurring_warning_count=record.recurring_warning_count,
            archived_outputs=record.archived_outputs,
        ),
        diagnostics,
    )


def _aggregate_warnings(
    records_and_diagnostics: Sequence[tuple[AuditedDiagnosticsRecord, EnergyPlusDiagnostics]],
    *,
    recurring: bool,
) -> tuple[WarningAggregate, ...]:
    totals: defaultdict[str, int] = defaultdict(int)
    maxima: defaultdict[str, int] = defaultdict(int)
    episode_counts: defaultdict[str, int] = defaultdict(int)
    for _record, diagnostics in records_and_diagnostics:
        occurrences = diagnostics.recurring_warning_occurrences if recurring else diagnostics.warning_occurrences
        for occurrence in occurrences:
            totals[occurrence.signature] += occurrence.count
            maxima[occurrence.signature] = max(maxima[occurrence.signature], occurrence.count)
            if occurrence.count > 0:
                episode_counts[occurrence.signature] += 1
    return tuple(
        WarningAggregate(
            signature=signature,
            maximum_occurrences_per_episode=maxima[signature],
            total_occurrences=totals[signature],
            episodes_containing_signature=episode_counts[signature],
        )
        for signature in sorted(totals)
    )


def audit_training_diagnostics(
    diagnostics_root: str | Path,
    *,
    building_path: str | Path,
    expected_building_sha256: str,
    weather_path: str | Path,
    expected_weather_sha256: str,
    expected_energyplus_version: str,
    expected_complete_count: int,
    allow_extra_incomplete: bool = False,
) -> TrainingDiagnosticsAudit:
    """Audit one archive and inventory warnings without approving them.

    Exactly ``expected_complete_count`` complete diagnostics are required.
    Every discovered calendar episode ``energyplus`` directory is inventoried.
    Safe, warning-free incomplete directories are never counted.  They fail
    the audit by default and may only be acknowledged for engineering
    inspection through ``allow_extra_incomplete=True``.  The result remains
    permanently non-publication in either case.
    """

    expected_count = _positive_integer(expected_complete_count, field="expected_complete_count")
    if not isinstance(allow_extra_incomplete, bool):
        raise TypeError("allow_extra_incomplete must be a boolean")
    if (
        not isinstance(expected_energyplus_version, str)
        or not expected_energyplus_version
        or expected_energyplus_version.strip() != expected_energyplus_version
    ):
        raise TrainingDiagnosticsAuditInputError("expected_energyplus_version must be a non-empty, trimmed string")
    building = _scoped_asset(
        building_path,
        role="building",
        expected_sha256=expected_building_sha256,
    )
    weather = _scoped_asset(
        weather_path,
        role="weather",
        expected_sha256=expected_weather_sha256,
    )
    if Path(building_path).expanduser().resolve() == Path(weather_path).expanduser().resolve():
        raise TrainingDiagnosticsAuditInputError("building_path and weather_path must identify distinct files")

    unresolved_root = Path(diagnostics_root).expanduser()
    if unresolved_root.is_symlink():
        raise TrainingDiagnosticsAuditInputError(f"diagnostics_root must not be a symbolic link: {unresolved_root}")
    try:
        root = unresolved_root.resolve(strict=True)
    except OSError as error:
        raise TrainingDiagnosticsAuditInputError(
            f"cannot resolve diagnostics_root {unresolved_root}: {error}"
        ) from error
    if not root.is_dir():
        raise TrainingDiagnosticsAuditInputError(f"diagnostics_root must be a directory: {root}")

    candidates, path_violations = _discover_candidates(root)
    complete: list[AuditedDiagnosticsRecord] = []
    incomplete: list[AuditedDiagnosticsRecord] = []
    rejected: list[AuditedDiagnosticsRecord] = []
    complete_with_payload: list[tuple[AuditedDiagnosticsRecord, EnergyPlusDiagnostics]] = []

    for candidate in candidates:
        if candidate.preliminary_violations:
            rejected.append(_rejected_candidate(candidate, candidate.preliminary_violations))
            continue
        diagnostics_output = candidate.output(DIAGNOSTICS_FILENAME)
        if not diagnostics_output.present:
            record, _partial_diagnostics = _record_without_persisted_diagnostics(
                candidate,
                expected_energyplus_version=expected_energyplus_version,
            )
            if record.classification == "extra_incomplete":
                incomplete.append(record)
            else:
                rejected.append(record)
            continue
        try:
            diagnostics, executed_model_sha256 = _diagnostics_from_candidate(candidate)
        except TrainingDiagnosticsAuditInputError as error:
            rejected.append(_rejected_candidate(candidate, (str(error),)))
            continue
        record = _record_from_diagnostics(
            candidate,
            diagnostics,
            executed_model_sha256,
            expected_energyplus_version=expected_energyplus_version,
        )
        missing_outputs = tuple(item.filename for item in candidate.outputs if not item.present)
        if missing_outputs:
            missing_reason = "missing archived EnergyPlus output(s): " + ", ".join(missing_outputs)
            err_present = candidate.output("eplusout.err").present
            if record.classification == "extra_incomplete" and err_present:
                record = replace(record, violations=record.violations + (missing_reason,))
            else:
                record = replace(
                    record,
                    classification="rejected",
                    violations=record.violations + (missing_reason,),
                )
        if record.classification == "complete":
            complete.append(record)
            complete_with_payload.append((record, diagnostics))
        elif record.classification == "extra_incomplete":
            incomplete.append(record)
        else:
            rejected.append(record)

    complete.sort(key=lambda item: item.path)
    incomplete.sort(key=lambda item: item.path)
    rejected.sort(key=lambda item: item.path)
    complete_with_payload.sort(key=lambda item: item[0].path)

    violations: list[str] = []
    input_scope_verified = building.matches_expected and weather.matches_expected
    if not building.matches_expected:
        violations.append(
            "building SHA256 mismatch: " f"expected={building.expected_sha256}, actual={building.actual_sha256}"
        )
    if not weather.matches_expected:
        violations.append(
            "weather SHA256 mismatch: " f"expected={weather.expected_sha256}, actual={weather.actual_sha256}"
        )
    if path_violations:
        violations.append(f"archive traversal found {len(path_violations)} unsafe or unreadable path(s)")
    if rejected:
        violations.append(f"{len(rejected)} EnergyPlus episode directory/directories were malformed or rejected")
    if len(complete) != expected_count:
        violations.append("complete diagnostics count mismatch: " f"expected={expected_count}, actual={len(complete)}")

    engineering_exceptions: list[str] = []
    if incomplete:
        message = (
            f"{len(incomplete)} warning-free incomplete diagnostics file(s) were excluded "
            "from the complete episode count"
        )
        if allow_extra_incomplete:
            engineering_exceptions.append(message + " under the explicit engineering-only override")
        else:
            violations.append(message + "; rerun with an explicit engineering-only acknowledgement if appropriate")

    marked = _aggregate_warnings(complete_with_payload, recurring=False)
    recurring = _aggregate_warnings(complete_with_payload, recurring=True)
    complete_warning_counts = [diagnostics.total_warning_count or 0 for _, diagnostics in complete_with_payload]
    total_warning_count = sum(complete_warning_counts)
    maximum_warning_count = max(complete_warning_counts, default=0)

    return TrainingDiagnosticsAudit(
        building=building,
        weather=weather,
        expected_energyplus_version=expected_energyplus_version,
        expected_complete_diagnostics_count=expected_count,
        allow_extra_incomplete_requested=allow_extra_incomplete,
        engineering_integrity_checks_passed=not violations,
        input_scope_verified=input_scope_verified,
        warning_review_required=bool(marked or recurring),
        complete_total_warning_count=total_warning_count,
        complete_maximum_warning_count_per_episode=maximum_warning_count,
        complete_diagnostics=tuple(complete),
        extra_incomplete_diagnostics=tuple(incomplete),
        rejected_diagnostics=tuple(rejected),
        path_violations=path_violations,
        marked_warning_aggregates=marked,
        recurring_warning_aggregates=recurring,
        violations=tuple(violations),
        engineering_exceptions=tuple(engineering_exceptions),
    )


def _markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_training_diagnostics_markdown(audit: TrainingDiagnosticsAudit) -> str:
    """Render a concise deterministic review report."""

    status = "PASS (engineering integrity only)" if audit.engineering_integrity_checks_passed else "FAIL"
    lines = [
        "# ENGINEERING-ONLY training diagnostics audit",
        "",
        f"> **{ENGINEERING_DISCLAIMER}**",
        "",
        f"**Status:** {status}",
        "",
        "## Exact input scope",
        "",
        "| Input | File | Expected SHA256 | Actual SHA256 | Match |",
        "|---|---|---|---|---|",
    ]
    for asset in (audit.building, audit.weather):
        lines.append(
            f"| {asset.role} | `{_markdown_escape(asset.filename)}` | `{asset.expected_sha256}` | "
            f"`{asset.actual_sha256}` | {'yes' if asset.matches_expected else 'NO'} |"
        )
    lines.extend(
        [
            "",
            f"Expected EnergyPlus version: `{_markdown_escape(audit.expected_energyplus_version)}`.",
            "",
            "The archive itself does not embed the immutable source building/weather hashes; "
            "scientific episode-to-task lineage must be established separately.",
            "",
            "## Coverage",
            "",
            "| Episode directories | Complete expected | Complete found | Extra incomplete | Rejected | Unsafe paths |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {len(audit.complete_diagnostics) + len(audit.extra_incomplete_diagnostics) + len(audit.rejected_diagnostics)} | "
            f"{audit.expected_complete_diagnostics_count} | {len(audit.complete_diagnostics)} | "
            f"{len(audit.extra_incomplete_diagnostics)} | {len(audit.rejected_diagnostics)} | "
            f"{len(audit.path_violations)} |",
            "",
            "## Warning inventory (complete episodes only)",
            "",
            f"EnergyPlus reported **{audit.complete_total_warning_count}** total warning occurrences; "
            f"the per-episode maximum was **{audit.complete_maximum_warning_count_per_episode}**.",
            "These counts are an inventory for manual review, not an approval.",
            "",
            "| Kind | Exact normalized signature | Max / episode | Total | Episodes containing |",
            "|---|---|---:|---:|---:|",
        ]
    )
    aggregates = [("marked", item) for item in audit.marked_warning_aggregates] + [
        ("recurring", item) for item in audit.recurring_warning_aggregates
    ]
    if aggregates:
        for kind, item in aggregates:
            lines.append(
                f"| {kind} | `{_markdown_escape(item.signature)}` | "
                f"{item.maximum_occurrences_per_episode} | {item.total_occurrences} | "
                f"{item.episodes_containing_signature} |"
            )
    else:
        lines.append("| — | No warning signatures in complete diagnostics | 0 | 0 | 0 |")

    lines.extend(["", "## Audit findings", ""])
    if audit.violations:
        lines.append("Violations:")
        lines.append("")
        lines.extend(f"- {_markdown_escape(item)}" for item in audit.violations)
    else:
        lines.append("No engineering integrity violation was detected.")
    if audit.engineering_exceptions:
        lines.extend(["", "Explicit engineering-only exceptions:", ""])
        lines.extend(f"- {_markdown_escape(item)}" for item in audit.engineering_exceptions)

    if audit.extra_incomplete_diagnostics:
        lines.extend(["", "Extra incomplete diagnostics (never counted as episodes):", ""])
        for record in audit.extra_incomplete_diagnostics:
            reasons = "; ".join(record.violations)
            lines.append(f"- `{_markdown_escape(record.path)}` — {_markdown_escape(reasons)}")
    if audit.rejected_diagnostics:
        lines.extend(["", "Rejected diagnostics:", ""])
        for record in audit.rejected_diagnostics:
            reasons = "; ".join(record.violations)
            lines.append(f"- `{_markdown_escape(record.path)}` — {_markdown_escape(reasons)}")
    if audit.path_violations:
        lines.extend(["", "Unsafe or unreadable archive paths:", ""])
        for item in audit.path_violations:
            lines.append(f"- `{_markdown_escape(item.path)}` — {_markdown_escape(item.reason)}")
    lines.extend(
        [
            "",
            "No scientific allowlist was generated. Any warning allowance requires a separate, "
            "reviewed and checksum-scoped scientific policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_create(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(contents)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace existing audit artifact {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def write_training_diagnostics_audit(
    audit: TrainingDiagnosticsAudit,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown artifacts into a new directory."""

    if not isinstance(audit, TrainingDiagnosticsAudit):
        raise TypeError("audit must be a TrainingDiagnosticsAudit")
    unresolved = Path(output_dir).expanduser()
    if unresolved.is_symlink():
        raise TrainingDiagnosticsAuditInputError(f"audit output_dir must not be a symbolic link: {unresolved}")
    destination = unresolved.resolve()
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"audit output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(f"audit output directory is not empty: {destination}; use a new directory")
    else:
        destination.mkdir(parents=True)
    json_path = destination / TRAINING_DIAGNOSTICS_AUDIT_JSON
    markdown_path = destination / TRAINING_DIAGNOSTICS_AUDIT_MARKDOWN
    _atomic_create(json_path, audit.to_json().encode("utf-8"))
    _atomic_create(
        markdown_path,
        render_training_diagnostics_markdown(audit).encode("utf-8"),
    )
    return json_path, markdown_path


__all__ = [
    "ENGINEERING_DISCLAIMER",
    "TRAINING_DIAGNOSTICS_AUDIT_JSON",
    "TRAINING_DIAGNOSTICS_AUDIT_MARKDOWN",
    "TRAINING_DIAGNOSTICS_AUDIT_SCHEMA_VERSION",
    "ArchivedOutputPresence",
    "AuditedDiagnosticsRecord",
    "EpisodePathIdentity",
    "PathViolation",
    "ScopedAsset",
    "TrainingDiagnosticsAudit",
    "TrainingDiagnosticsAuditInputError",
    "WarningAggregate",
    "audit_training_diagnostics",
    "render_training_diagnostics_markdown",
    "write_training_diagnostics_audit",
]
