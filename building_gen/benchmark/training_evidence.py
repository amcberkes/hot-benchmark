"""Fail-closed scientific evidence for EnergyPlus training episodes.

The recorder in this module is independent of the learning algorithm.  It
consumes episode-completion events from a training environment, validates the
same checksum-scoped EnergyPlus diagnostics contract used for scientific
evaluation, and writes deterministic evidence records.  A bundle is complete
only after every preregistered episode has been recorded exactly once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from building_gen.benchmark.energyplus_diagnostics import (
    DIAGNOSTICS_FILENAME,
    DIAGNOSTICS_SCHEMA_VERSION,
    DiagnosticsGate,
    EnergyPlusDiagnostics,
    ScientificDiagnosticsError,
    canonical_energyplus_diagnostics_snapshot,
    energyplus_diagnostics_from_mapping,
    require_scientific_diagnostics,
)
from building_gen.benchmark.specs import (
    LEGACY_TASK_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    CalendarWindowSpec,
    TaskSpec,
)
from building_gen.benchmark.warmup_preconditioning import (
    WARMUP_PRECONDITIONING_FILENAME,
    WarmupPreconditioningEvidenceError,
    validate_warmup_preconditioning_evidence,
)

TRAINING_EPISODE_EVIDENCE_SCHEMA_VERSION = "hot-training-episode-evidence/1.1.0"
TRAINING_EVIDENCE_MANIFEST_SCHEMA_VERSION = "hot-training-evidence-manifest/1.1.0"
TRAINING_EVIDENCE_MANIFEST_FILENAME = "training_evidence_manifest.json"
_EPISODES_DIRECTORY_NAME = "episodes"
_STREAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ScientificTrainingEvidenceError(RuntimeError):
    """Raised when scientific training evidence is absent or inconsistent."""


def _nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _positive_integer(value: Any, *, field_name: str) -> int:
    result = _nonnegative_integer(value, field_name=field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _formatted_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_formatted_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> bytes:
    """Create ``path`` atomically without permitting an existing record."""

    contents = _formatted_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(contents)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ScientificTrainingEvidenceError(
                f"refusing to replace stale training evidence record {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return contents


@dataclass(frozen=True, slots=True)
class TrainingEpisodeKey:
    """Stable identity of one episode in one deterministic training stream."""

    stream_id: str
    generation: int
    schedule_seed: int
    episode_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not _STREAM_ID_RE.fullmatch(self.stream_id):
            raise ValueError(
                "stream_id must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-'"
            )
        for field_name in ("generation", "schedule_seed", "episode_index"):
            _nonnegative_integer(getattr(self, field_name), field_name=field_name)

    @property
    def filename(self) -> str:
        return (
            f"{self.stream_id}__generation-{self.generation:04d}__"
            f"seed-{self.schedule_seed:010d}__episode-{self.episode_index:08d}.json"
        )

    def sort_key(self) -> tuple[str, int, int, int]:
        return (self.stream_id, self.generation, self.schedule_seed, self.episode_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "generation": self.generation,
            "schedule_seed": self.schedule_seed,
            "episode_index": self.episode_index,
        }


@dataclass(frozen=True, slots=True)
class ExpectedTrainingEpisode:
    """Preregistered calendar and seed contract for one training episode."""

    key: TrainingEpisodeKey
    schedule_schema_version: str
    strategy_id: str
    child_seed: int
    window: CalendarWindowSpec

    def __post_init__(self) -> None:
        if not isinstance(self.key, TrainingEpisodeKey):
            raise TypeError("key must be a TrainingEpisodeKey")
        for field_name in ("schedule_schema_version", "strategy_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty, trimmed string")
        _nonnegative_integer(self.child_seed, field_name="child_seed")
        if not isinstance(self.window, CalendarWindowSpec):
            raise TypeError("window must be a CalendarWindowSpec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_key": self.key.to_dict(),
            "schedule_schema_version": self.schedule_schema_version,
            "strategy_id": self.strategy_id,
            "child_seed": self.child_seed,
            "window": {
                "name": self.window.name,
                "start_month": self.window.start_month,
                "start_day": self.window.start_day,
                "end_month": self.window.end_month,
                "end_day": self.window.end_day,
                "duration_hours": self.window.duration_hours,
            },
        }


@dataclass(frozen=True, slots=True)
class TrainingEpisodeCompletion:
    """Environment-owned facts available before a training child is closed."""

    key: TrainingEpisodeKey
    schedule_schema_version: str
    strategy_id: str
    task_name: str
    task_id: str
    building_sha256: str | None
    weather_sha256: str | None
    calendar_year: int | None
    child_seed: int
    window: CalendarWindowSpec
    episode_work_dir: Path
    expected_steps: int
    step_count: int
    terminated: bool
    truncated: bool
    diagnostics_path: Path | None
    executed_model_sha256: str | None
    warmup_preconditioning_evidence_path: Path | None = None
    warmup_preconditioning_contract_payload: Mapping[str, Any] | None = None
    warmup_preconditioning_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, TrainingEpisodeKey):
            raise TypeError("key must be a TrainingEpisodeKey")
        for field_name in ("schedule_schema_version", "strategy_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty, trimmed string")
        if not isinstance(self.window, CalendarWindowSpec):
            raise TypeError("window must be a CalendarWindowSpec")
        if not isinstance(self.task_name, str) or not self.task_name:
            raise ValueError("task_name cannot be empty")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id cannot be empty")
        _nonnegative_integer(self.child_seed, field_name="child_seed")
        _positive_integer(self.expected_steps, field_name="expected_steps")
        _nonnegative_integer(self.step_count, field_name="step_count")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise TypeError("terminated and truncated must be booleans")
        object.__setattr__(self, "episode_work_dir", Path(self.episode_work_dir).resolve())
        if self.diagnostics_path is not None:
            object.__setattr__(self, "diagnostics_path", Path(self.diagnostics_path))
        if self.warmup_preconditioning_evidence_path is not None:
            object.__setattr__(
                self,
                "warmup_preconditioning_evidence_path",
                Path(self.warmup_preconditioning_evidence_path),
            )
        if self.warmup_preconditioning_contract_payload is not None and not isinstance(
            self.warmup_preconditioning_contract_payload,
            Mapping,
        ):
            raise TypeError("warmup_preconditioning_contract_payload must be a mapping or None")
        if self.warmup_preconditioning_contract_sha256 is not None and not isinstance(
            self.warmup_preconditioning_contract_sha256, str
        ):
            raise TypeError("warmup_preconditioning_contract_sha256 must be a string or None")

    @property
    def natural_termination(self) -> bool:
        """Whether the task reached its declared horizon without truncation."""

        return self.terminated and not self.truncated and self.step_count == self.expected_steps


EpisodeCompletionCallback = Callable[[TrainingEpisodeCompletion], None]


@dataclass(frozen=True, slots=True)
class _RecordedEpisode:
    key: TrainingEpisodeKey
    relative_path: str
    sha256: str


class ScientificTrainingEvidenceRecorder:
    """Validate and persist a preregistered set of scientific episodes.

    ``output_dir`` is a dedicated evidence directory and must be absent or
    empty.  This deliberately rejects implicit resume/overwrite behavior: a
    previous partial bundle is stale evidence and requires an explicit new run
    directory.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        task: TaskSpec,
        diagnostics_gate: DiagnosticsGate,
        energyplus_version: str,
        expected_episodes: Iterable[ExpectedTrainingEpisode],
    ) -> None:
        if not diagnostics_gate.scientific:
            raise ValueError("scientific training evidence requires a scientific DiagnosticsGate")
        if task.building.sha256 is None or task.weather.sha256 is None:
            raise ScientificTrainingEvidenceError(
                "scientific training evidence requires checksum-pinned building and weather assets"
            )
        if not isinstance(energyplus_version, str) or not energyplus_version.strip():
            raise ValueError("energyplus_version cannot be empty")

        expected = tuple(expected_episodes)
        if not expected:
            raise ValueError("expected_episodes cannot be empty")
        if any(not isinstance(item, ExpectedTrainingEpisode) for item in expected):
            raise TypeError("expected_episodes must contain only ExpectedTrainingEpisode values")
        keys = [item.key for item in expected]
        if len(keys) != len(set(keys)):
            raise ValueError("expected_episodes cannot contain duplicate episode keys")
        expected = tuple(sorted(expected, key=lambda item: item.key.sort_key()))
        filenames = [item.key.filename for item in expected]
        if len(filenames) != len(set(filenames)):
            raise ValueError("expected episode keys produce duplicate record filenames")

        unresolved_output_dir = Path(output_dir).expanduser()
        if unresolved_output_dir.is_symlink():
            raise ScientificTrainingEvidenceError("training evidence output_dir must not be a symbolic link")
        self.output_dir = unresolved_output_dir.resolve()
        self.episodes_dir = self.output_dir / _EPISODES_DIRECTORY_NAME
        self.manifest_path = self.output_dir / TRAINING_EVIDENCE_MANIFEST_FILENAME
        self.task = task
        self.diagnostics_gate = diagnostics_gate
        self.energyplus_version = energyplus_version.strip()
        self.expected_episodes = expected
        self._expected_by_key = {item.key: item for item in expected}
        self._expected_keys = frozenset(self._expected_by_key)
        self._records: dict[TrainingEpisodeKey, _RecordedEpisode] = {}
        self._finalized = False
        self._lock = threading.RLock()
        try:
            self.policy = diagnostics_gate.policy_for(
                asset_sha256=task.building.sha256,
                weather_sha256=task.weather.sha256,
                energyplus_version=self.energyplus_version,
            )
        except ScientificDiagnosticsError as error:
            raise ScientificTrainingEvidenceError(str(error)) from error
        assert self.policy is not None

        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise NotADirectoryError(f"training evidence path is not a directory: {self.output_dir}")
            if any(self.output_dir.iterdir()):
                raise FileExistsError(
                    f"training evidence directory is not empty: {self.output_dir}; "
                    "use a new directory instead of reusing stale evidence"
                )
        else:
            self.output_dir.mkdir(parents=True)
        self.episodes_dir.mkdir()
        _atomic_replace_json(self.manifest_path, self._manifest_payload(complete=False))

    def _task_payload(self) -> dict[str, Any]:
        assert self.task.building.sha256 is not None and self.task.weather.sha256 is not None
        return {
            "task_name": self.task.name,
            "task_id": self.task.task_id,
            "building_asset_id": self.task.building.asset_id,
            "building_sha256": self.task.building.sha256,
            "weather_asset_id": self.task.weather.asset_id,
            "weather_sha256": self.task.weather.sha256,
            "warmup_preconditioning": {
                "enabled": self.task.warmup_preconditioning.enabled,
                "contract_payload": self.task.warmup_preconditioning.identity_payload(),
                "contract_sha256": self.task.warmup_preconditioning.sha256,
            },
        }

    def _manifest_payload(self, *, complete: bool) -> dict[str, Any]:
        records = [
            {
                "episode_key": record.key.to_dict(),
                "path": record.relative_path,
                "sha256": record.sha256,
            }
            for record in sorted(self._records.values(), key=lambda item: item.key.sort_key())
        ]
        return {
            "schema_version": TRAINING_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "complete": complete,
            "task": self._task_payload(),
            "energyplus_version": self.energyplus_version,
            "diagnostics_policy": self.policy.to_dict(),
            "diagnostics_policy_sha256": self.policy.sha256,
            "expected_episodes": [item.to_dict() for item in self.expected_episodes],
            "expected_episode_count": len(self.expected_episodes),
            "record_count": len(records),
            "records": records,
        }

    def _assert_manifest(self, *, complete: bool) -> None:
        try:
            actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScientificTrainingEvidenceError(
                f"cannot validate in-progress training evidence manifest {self.manifest_path}: {error}"
            ) from error
        expected = self._manifest_payload(complete=complete)
        if actual != expected:
            state = "sealed" if complete else "in-progress"
            raise ScientificTrainingEvidenceError(f"{state} training evidence manifest was modified or is stale")

    def _assert_layout(self) -> None:
        expected_top_level = {self.manifest_path.name, self.episodes_dir.name}
        actual_top_level = {path.name for path in self.output_dir.iterdir()}
        if actual_top_level != expected_top_level:
            raise ScientificTrainingEvidenceError(
                "training evidence directory contains missing or stale top-level entries: "
                f"expected={sorted(expected_top_level)!r}, actual={sorted(actual_top_level)!r}"
            )
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise ScientificTrainingEvidenceError("training evidence manifest must be a real file")
        if not self.episodes_dir.is_dir() or self.episodes_dir.is_symlink():
            raise ScientificTrainingEvidenceError("training evidence episodes path must be a real directory")
        expected_files = {Path(record.relative_path).name for record in self._records.values()}
        actual_files = {path.name for path in self.episodes_dir.iterdir()}
        if actual_files != expected_files:
            raise ScientificTrainingEvidenceError(
                "training evidence contains missing or stale episode records: "
                f"expected={sorted(expected_files)!r}, actual={sorted(actual_files)!r}"
            )

    def _read_diagnostics(
        self,
        completion: TrainingEpisodeCompletion,
    ) -> tuple[EnergyPlusDiagnostics, Mapping[str, Any], bytes]:
        if completion.diagnostics_path is None:
            raise ScientificTrainingEvidenceError("completed episode has no EnergyPlus diagnostics path")
        expected_path = completion.episode_work_dir / "energyplus" / DIAGNOSTICS_FILENAME
        path = completion.diagnostics_path
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as error:
            raise ScientificTrainingEvidenceError(
                f"cannot resolve EnergyPlus diagnostics for completed episode: {path}"
            ) from error
        if path.is_symlink() or resolved_path != expected_path.resolve():
            raise ScientificTrainingEvidenceError(
                "completed episode diagnostics do not belong to its exact work directory: "
                f"expected {expected_path}, found {resolved_path}"
            )
        if not resolved_path.is_file():
            raise ScientificTrainingEvidenceError(
                f"completed episode diagnostics are not a regular file: {resolved_path}"
            )
        try:
            raw_bytes = resolved_path.read_bytes()
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScientificTrainingEvidenceError(
                f"cannot read EnergyPlus diagnostics {resolved_path}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ScientificTrainingEvidenceError("EnergyPlus diagnostics payload must be a JSON object")
        try:
            diagnostics = energyplus_diagnostics_from_mapping(payload)
        except (TypeError, ValueError) as error:
            raise ScientificTrainingEvidenceError(f"invalid EnergyPlus diagnostics payload: {error}") from error
        snapshot_payload = json.loads(
            canonical_energyplus_diagnostics_snapshot(diagnostics).canonical_json.decode("utf-8")
        )
        expected_fields = set(snapshot_payload) | {"executed_model_sha256"}
        if set(payload) != expected_fields:
            raise ScientificTrainingEvidenceError(
                "EnergyPlus diagnostics fields do not match their embedded schema: "
                f"missing={sorted(expected_fields - set(payload))!r}, "
                f"unexpected={sorted(set(payload) - expected_fields)!r}"
            )
        if snapshot_payload["schema_version"] != DIAGNOSTICS_SCHEMA_VERSION:
            raise ScientificTrainingEvidenceError(
                f"archived diagnostics schema {snapshot_payload['schema_version']!r} is non-current; "
                f"scientific training requires {DIAGNOSTICS_SCHEMA_VERSION!r}"
            )
        return diagnostics, payload, raw_bytes

    def _verify_record_files(self) -> None:
        for record in self._records.values():
            path = self.output_dir / record.relative_path
            if path.is_symlink() or not path.is_file():
                raise ScientificTrainingEvidenceError(
                    f"training episode evidence is missing or not a regular file: {path}"
                )
            if _sha256(path.read_bytes()) != record.sha256:
                raise ScientificTrainingEvidenceError(f"training episode evidence was modified after recording: {path}")

    def _warmup_preconditioning_payload(
        self,
        completion: TrainingEpisodeCompletion,
        *,
        executed_model_sha256: str,
    ) -> dict[str, Any]:
        """Authenticate an enabled episode intervention, or state it is disabled."""

        contract = {
            "payload": self.task.warmup_preconditioning.identity_payload(),
            "sha256": self.task.warmup_preconditioning.sha256,
        }
        runtime_contract_payload = completion.warmup_preconditioning_contract_payload
        runtime_contract_sha256 = completion.warmup_preconditioning_contract_sha256
        runtime_contract_absent = runtime_contract_payload is None and runtime_contract_sha256 is None
        if (runtime_contract_payload is None) != (runtime_contract_sha256 is None):
            raise ScientificTrainingEvidenceError(
                "training episode advertises an incomplete runtime warmup preconditioning contract"
            )
        if self.task.schema_version == TASK_SCHEMA_VERSION:
            if runtime_contract_absent:
                raise ScientificTrainingEvidenceError(
                    "current training episode has no complete runtime warmup preconditioning contract"
                )
        elif self.task.schema_version != LEGACY_TASK_SCHEMA_VERSION:
            raise ScientificTrainingEvidenceError(f"unsupported TaskSpec schema {self.task.schema_version!r}")
        if not runtime_contract_absent and (
            runtime_contract_payload != contract["payload"] or runtime_contract_sha256 != contract["sha256"]
        ):
            raise ScientificTrainingEvidenceError(
                "training episode runtime warmup preconditioning contract does not match TaskSpec"
            )
        source = completion.warmup_preconditioning_evidence_path
        if not self.task.warmup_preconditioning.enabled:
            if source is not None:
                raise ScientificTrainingEvidenceError(
                    "disabled warmup preconditioning unexpectedly produced runtime evidence"
                )
            return {
                "enabled": False,
                "contract": contract,
                "evidence": None,
            }
        if source is None:
            raise ScientificTrainingEvidenceError("enabled warmup preconditioning produced no runtime evidence")
        expected_path = completion.episode_work_dir / "energyplus" / WARMUP_PRECONDITIONING_FILENAME
        try:
            resolved_path = source.resolve(strict=True)
        except OSError as error:
            raise ScientificTrainingEvidenceError(
                f"cannot resolve warmup preconditioning evidence for completed episode: {source}"
            ) from error
        if source.is_symlink() or resolved_path != expected_path.resolve():
            raise ScientificTrainingEvidenceError(
                "completed episode warmup preconditioning evidence does not belong to its exact "
                f"work directory: expected {expected_path}, found {resolved_path}"
            )
        try:
            payload, raw_bytes = validate_warmup_preconditioning_evidence(
                resolved_path,
                task=self.task,
                window=completion.window,
                executed_model_sha256=executed_model_sha256,
            )
        except WarmupPreconditioningEvidenceError as error:
            raise ScientificTrainingEvidenceError(str(error)) from error
        return {
            "enabled": True,
            "contract": contract,
            "evidence": {
                "source_path": f"energyplus/{WARMUP_PRECONDITIONING_FILENAME}",
                "source_file_sha256": _sha256(raw_bytes),
                "canonical_payload_sha256": _sha256(_canonical_json_bytes(payload)),
                "payload": dict(payload),
            },
        }

    def _episode_payload(self, completion: TrainingEpisodeCompletion) -> dict[str, Any]:
        expected_episode = self._expected_by_key.get(completion.key)
        if expected_episode is None:
            raise ScientificTrainingEvidenceError(f"unexpected training episode {completion.key.to_dict()!r}")
        if completion.schedule_schema_version != expected_episode.schedule_schema_version:
            raise ScientificTrainingEvidenceError(
                "episode schedule schema does not match the preregistered training contract"
            )
        if completion.strategy_id != expected_episode.strategy_id:
            raise ScientificTrainingEvidenceError(
                "episode calendar strategy does not match the preregistered training contract"
            )
        if completion.child_seed != expected_episode.child_seed:
            raise ScientificTrainingEvidenceError(
                "episode child seed does not match the preregistered training contract"
            )
        if completion.window != expected_episode.window:
            raise ScientificTrainingEvidenceError(
                "episode calendar window does not match the preregistered training contract"
            )
        assert self.task.building.sha256 is not None and self.task.weather.sha256 is not None
        expected_task_fields = (
            self.task.name,
            self.task.task_id,
            self.task.building.sha256,
            self.task.weather.sha256,
        )
        actual_task_fields = (
            completion.task_name,
            completion.task_id,
            completion.building_sha256,
            completion.weather_sha256,
        )
        if actual_task_fields != expected_task_fields:
            raise ScientificTrainingEvidenceError(
                "episode task/building/weather identity does not match the scientific training contract"
            )
        if completion.calendar_year != self.task.calendar_year:
            raise ScientificTrainingEvidenceError(
                "episode calendar year does not match the scientific training TaskSpec"
            )
        if completion.window.duration_hours != self.task.episode.episode_length_hours:
            raise ScientificTrainingEvidenceError(
                "episode calendar window does not match the TaskSpec training horizon"
            )
        if completion.expected_steps != self.task.episode.max_steps:
            raise ScientificTrainingEvidenceError(
                f"episode expected_steps {completion.expected_steps} does not match TaskSpec "
                f"max_steps {self.task.episode.max_steps}"
            )
        if not completion.natural_termination:
            raise ScientificTrainingEvidenceError(
                "scientific training episodes must terminate naturally at the declared horizon: "
                f"steps={completion.step_count}/{completion.expected_steps}, "
                f"terminated={completion.terminated}, truncated={completion.truncated}"
            )
        if not isinstance(completion.executed_model_sha256, str):
            raise ScientificTrainingEvidenceError("completed episode has no executed-model SHA256")
        executed_model_sha256 = completion.executed_model_sha256.lower()
        if not _SHA256_RE.fullmatch(executed_model_sha256):
            raise ScientificTrainingEvidenceError("executed-model SHA256 is malformed")

        diagnostics, diagnostics_payload, diagnostics_bytes = self._read_diagnostics(completion)
        persisted_model_sha256 = diagnostics_payload["executed_model_sha256"]
        if not isinstance(persisted_model_sha256, str) or not _SHA256_RE.fullmatch(persisted_model_sha256.lower()):
            raise ScientificTrainingEvidenceError("EnergyPlus diagnostics contain a malformed executed-model SHA256")
        if persisted_model_sha256.lower() != executed_model_sha256:
            raise ScientificTrainingEvidenceError(
                "EnergyPlus diagnostics executed-model SHA256 does not match the model " "materialized for this episode"
            )
        warmup_preconditioning = self._warmup_preconditioning_payload(
            completion,
            executed_model_sha256=executed_model_sha256,
        )
        try:
            decision = require_scientific_diagnostics(
                diagnostics,
                asset_sha256=self.task.building.sha256,
                weather_sha256=self.task.weather.sha256,
                allowlist=self.policy,
            )
        except ScientificDiagnosticsError as error:
            raise ScientificTrainingEvidenceError(str(error)) from error
        assert decision.canonical_diagnostics_json is not None
        accepted_core = json.loads(decision.canonical_diagnostics_json.decode("utf-8"))
        persisted_core = dict(diagnostics_payload)
        persisted_core.pop("executed_model_sha256")
        if persisted_core != accepted_core:
            raise ScientificTrainingEvidenceError(
                "EnergyPlus diagnostics changed between loading and scientific acceptance"
            )
        accepted_payload = dict(accepted_core)
        accepted_payload["executed_model_sha256"] = executed_model_sha256

        return {
            "schema_version": TRAINING_EPISODE_EVIDENCE_SCHEMA_VERSION,
            "episode_key": completion.key.to_dict(),
            "task": self._task_payload(),
            "calendar": {
                "schedule_schema_version": completion.schedule_schema_version,
                "strategy_id": completion.strategy_id,
                "calendar_year": completion.calendar_year,
                "child_seed": completion.child_seed,
                "window": {
                    "name": completion.window.name,
                    "start_month": completion.window.start_month,
                    "start_day": completion.window.start_day,
                    "end_month": completion.window.end_month,
                    "end_day": completion.window.end_day,
                    "duration_hours": completion.window.duration_hours,
                },
            },
            "termination": {
                "expected_steps": completion.expected_steps,
                "step_count": completion.step_count,
                "terminated": completion.terminated,
                "truncated": completion.truncated,
                "natural_termination": completion.natural_termination,
            },
            "warmup_preconditioning": warmup_preconditioning,
            "diagnostics": {
                "mode": "scientific",
                "energyplus_version": accepted_core["energyplus_version"],
                "policy_sha256": self.policy.sha256,
                "executed_model_sha256": executed_model_sha256,
                "source_path": f"energyplus/{DIAGNOSTICS_FILENAME}",
                "source_file_sha256": _sha256(diagnostics_bytes),
                "canonical_payload_sha256": _sha256(_canonical_json_bytes(accepted_payload)),
                "payload": accepted_payload,
            },
        }

    def __call__(self, completion: TrainingEpisodeCompletion) -> None:
        """Validate and record one completion event exactly once."""

        self.record(completion)

    def record(self, completion: TrainingEpisodeCompletion) -> Path:
        """Validate and atomically persist one preregistered episode."""

        if not isinstance(completion, TrainingEpisodeCompletion):
            raise TypeError("completion must be a TrainingEpisodeCompletion")
        with self._lock:
            if self._finalized:
                raise ScientificTrainingEvidenceError("cannot add episodes to a finalized training evidence bundle")
            self._assert_manifest(complete=False)
            self._assert_layout()
            if completion.key in self._records:
                raise ScientificTrainingEvidenceError(f"duplicate training episode {completion.key.to_dict()!r}")
            payload = self._episode_payload(completion)
            destination = self.episodes_dir / completion.key.filename
            contents = _atomic_create_json(destination, payload)
            record = _RecordedEpisode(
                key=completion.key,
                relative_path=f"{_EPISODES_DIRECTORY_NAME}/{completion.key.filename}",
                sha256=_sha256(contents),
            )
            self._records[completion.key] = record
            _atomic_replace_json(self.manifest_path, self._manifest_payload(complete=False))
            return destination

    def finalize(self) -> Path:
        """Seal the manifest only when every expected episode is exact and present."""

        with self._lock:
            if self._finalized:
                self._assert_manifest(complete=True)
                self._assert_layout()
                self._verify_record_files()
                return self.manifest_path
            self._assert_manifest(complete=False)
            self._assert_layout()
            missing = self._expected_keys - set(self._records)
            if missing:
                labels = [key.to_dict() for key in sorted(missing, key=TrainingEpisodeKey.sort_key)]
                raise ScientificTrainingEvidenceError(
                    f"scientific training evidence is incomplete; missing episode(s): {labels!r}"
                )
            self._verify_record_files()
            _atomic_replace_json(self.manifest_path, self._manifest_payload(complete=True))
            self._finalized = True
            return self.manifest_path


__all__ = [
    "EpisodeCompletionCallback",
    "ExpectedTrainingEpisode",
    "ScientificTrainingEvidenceError",
    "ScientificTrainingEvidenceRecorder",
    "TRAINING_EPISODE_EVIDENCE_SCHEMA_VERSION",
    "TRAINING_EVIDENCE_MANIFEST_FILENAME",
    "TRAINING_EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "TrainingEpisodeCompletion",
    "TrainingEpisodeKey",
]
