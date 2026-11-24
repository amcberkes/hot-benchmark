"""Fail-closed EnergyPlus runtime binding for control-capability manifests.

The structural capability manifest is deliberately independent from the
EnergyPlus Python API.  This module is the narrow runtime boundary that binds
that authenticated structure to one live EnergyPlus state: it resolves every
declared handle before the first policy action, expands normalized actions,
sets the complete physical endpoint vector, and records actuator/effect
read-backs in canonical order.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from building_gen.control.action_space import (
    ExpandedControlAction,
    ManifestActionTransform,
)
from building_gen.control.capabilities import ControlCapabilityManifest

MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION = "hot-manifest-control-runtime/0.1.0-draft"
MANIFEST_ACTION_BINDING_HASH_NAMESPACE = "HOT manifest runtime action binding"
MANIFEST_ACTION_SUBMISSION_HASH_NAMESPACE = "HOT manifest runtime action submission"
MANIFEST_ACTION_EVIDENCE_HASH_NAMESPACE = "HOT manifest runtime action evidence"
MANIFEST_EXECUTION_SNAPSHOT_SCHEMA_VERSION = "hot-manifest-execution-snapshot/1.0.0"
_EXECUTION_SNAPSHOT_DIRECTORY_MODE = 0o500
_EXECUTION_SNAPSHOT_FILE_MODE = 0o400
_EXECUTION_SNAPSHOT_FILENAME = "prepared-model.epJSON"


class ManifestControlRuntimeError(RuntimeError):
    """Raised when an authenticated control interface cannot run exactly."""


class ManifestExecutionSnapshotCleanupError(ManifestControlRuntimeError):
    """Preserve both a primary failure and its snapshot-cleanup failure."""

    def __init__(self, primary_error: Exception, cleanup_error: Exception) -> None:
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error
        super().__init__(
            "manifest execution failed and its transient snapshot cleanup also failed: "
            f"primary={primary_error!r}; cleanup={cleanup_error!r}"
        )
        self.__cause__ = primary_error


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _domain_sha256(namespace: str, payload: Any) -> str:
    return hashlib.sha256(namespace.encode("utf-8") + b"\x00" + _canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_manifest(value: Any) -> ControlCapabilityManifest:
    if type(value) is not ControlCapabilityManifest:
        raise TypeError("control manifest must be an exact ControlCapabilityManifest")
    try:
        return ControlCapabilityManifest.from_json(ControlCapabilityManifest.to_json(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise ManifestControlRuntimeError(f"control manifest is malformed or tampered: {error}") from error


def task_action_binding_sha256(manifest: ControlCapabilityManifest) -> str:
    """Hash the exact policy-command binding independently of the full manifest."""

    canonical = _canonical_manifest(manifest)
    return _domain_sha256(
        MANIFEST_ACTION_BINDING_HASH_NAMESPACE,
        canonical.task_action_binding.to_dict(),
    )


def _stable_regular_file_read(
    path: str | os.PathLike[str],
) -> tuple[bytes, os.stat_result]:
    file_path = Path(path)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(file_path, flags)
    except OSError as error:
        raise ManifestControlRuntimeError(
            f"cannot open prepared model as a regular non-symlink file: {file_path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ManifestControlRuntimeError("prepared model must be a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if identity_before != identity_after:
            raise ManifestControlRuntimeError("prepared model changed while its identity was being verified")
        try:
            path_after = os.stat(file_path, follow_symlinks=False)
        except OSError as error:
            raise ManifestControlRuntimeError("prepared model path changed while it was being verified") from error
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_uid,
            path_after.st_gid,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
            path_after.st_nlink,
        )
        if not stat.S_ISREG(path_after.st_mode) or path_identity != identity_after:
            raise ManifestControlRuntimeError("prepared model path changed while it was being verified")
        contents = b"".join(chunks)
        if len(contents) != before.st_size:
            raise ManifestControlRuntimeError("prepared model size changed while it was being read")
        return contents, after
    finally:
        os.close(descriptor)


def _stable_regular_file_bytes(path: str | os.PathLike[str]) -> bytes:
    return _stable_regular_file_read(path)[0]


def verify_manifest_prepared_model(
    manifest: ControlCapabilityManifest,
    path: str | os.PathLike[str],
) -> str:
    """Verify and return the prepared model hash bound by ``manifest``."""

    contents = verified_manifest_prepared_model_bytes(manifest, path)
    return hashlib.sha256(contents).hexdigest()


def verified_manifest_prepared_model_bytes(
    manifest: ControlCapabilityManifest,
    path: str | os.PathLike[str],
) -> bytes:
    """Own one stable read of the exact prepared bytes bound by ``manifest``."""

    canonical = _canonical_manifest(manifest)
    contents = _stable_regular_file_bytes(path)
    actual = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(actual, canonical.prepared_model_sha256):
        raise ManifestControlRuntimeError(
            "prepared model SHA-256 does not match the control capability manifest: "
            f"expected {canonical.prepared_model_sha256}, found {actual}"
        )
    return contents


@dataclass(frozen=True, slots=True)
class ManifestExecutionSnapshot:
    """One immutable per-run copy of the exact prepared model bytes."""

    source_path: str
    source_sha256: str
    source_device: int
    source_inode: int
    source_size: int
    directory_path: str
    directory_device: int
    directory_inode: int
    model_path: str
    model_sha256: str
    model_device: int
    model_inode: int
    model_size: int
    schema_version: str = MANIFEST_EXECUTION_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != MANIFEST_EXECUTION_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("manifest execution snapshot has an unknown schema")
        for field_name in ("source_path", "directory_path", "model_path"):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise TypeError(f"manifest execution snapshot {field_name} must be a nonempty exact string")
        for field_name in ("source_sha256", "model_sha256"):
            object.__setattr__(
                self,
                field_name,
                _claimed_sha256(getattr(self, field_name), field_name=field_name),
            )
        if self.source_sha256 != self.model_sha256:
            raise ValueError("manifest execution snapshot source and model hashes differ")
        for field_name in (
            "source_device",
            "source_inode",
            "source_size",
            "directory_device",
            "directory_inode",
            "model_device",
            "model_inode",
            "model_size",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise TypeError(f"manifest execution snapshot {field_name} must be a nonnegative exact integer")
        if self.source_size != self.model_size:
            raise ValueError("manifest execution snapshot source and model sizes differ")
        directory = Path(self.directory_path)
        model = Path(self.model_path)
        if model.parent != directory or model.name != _EXECUTION_SNAPSHOT_FILENAME:
            raise ValueError("manifest execution snapshot model path is not canonical")


def _snapshot_directory_metadata(path: Path) -> os.stat_result:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ManifestControlRuntimeError(f"cannot inspect manifest execution snapshot directory: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ManifestControlRuntimeError("manifest execution snapshot parent must be a non-symlink directory")
    if stat.S_IMODE(metadata.st_mode) != _EXECUTION_SNAPSHOT_DIRECTORY_MODE:
        raise ManifestControlRuntimeError("manifest execution snapshot directory must have mode 0500")
    if metadata.st_uid != os.geteuid():
        raise ManifestControlRuntimeError("manifest execution snapshot directory must be owned by the current user")
    return metadata


def _created_snapshot_directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_nlink,
    )


def _revalidate_created_snapshot_directory(
    path: Path,
    descriptor: int,
    expected_identity: tuple[int, int, int, int],
) -> os.stat_result:
    descriptor_metadata = os.fstat(descriptor)
    try:
        path_metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ManifestControlRuntimeError("cannot revalidate manifest execution snapshot directory") from error
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or not stat.S_ISDIR(path_metadata.st_mode)
        or descriptor_metadata.st_uid != os.geteuid()
        or path_metadata.st_uid != os.geteuid()
        or _created_snapshot_directory_identity(descriptor_metadata) != expected_identity
        or _created_snapshot_directory_identity(path_metadata) != expected_identity
    ):
        raise ManifestControlRuntimeError("manifest execution snapshot directory identity changed during publication")
    return descriptor_metadata


def _open_created_snapshot_directory(
    path: Path,
    created_metadata: os.stat_result,
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ManifestControlRuntimeError(
            "cannot open manifest execution snapshot directory for publication"
        ) from error
    try:
        _revalidate_created_snapshot_directory(
            path,
            descriptor,
            _created_snapshot_directory_identity(created_metadata),
        )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _canonical_execution_snapshot(value: Any) -> ManifestExecutionSnapshot:
    if type(value) is not ManifestExecutionSnapshot:
        raise TypeError("snapshot must be an exact ManifestExecutionSnapshot")
    return ManifestExecutionSnapshot(
        source_path=value.source_path,
        source_sha256=value.source_sha256,
        source_device=value.source_device,
        source_inode=value.source_inode,
        source_size=value.source_size,
        directory_path=value.directory_path,
        directory_device=value.directory_device,
        directory_inode=value.directory_inode,
        model_path=value.model_path,
        model_sha256=value.model_sha256,
        model_device=value.model_device,
        model_inode=value.model_inode,
        model_size=value.model_size,
        schema_version=value.schema_version,
    )


def verify_manifest_execution_snapshot(
    manifest: ControlCapabilityManifest,
    snapshot: ManifestExecutionSnapshot,
) -> str:
    """Reopen and verify the immutable model path passed to EnergyPlus."""

    canonical = _canonical_manifest(manifest)
    snapshot = _canonical_execution_snapshot(snapshot)
    if snapshot.model_sha256 != canonical.prepared_model_sha256:
        raise ManifestControlRuntimeError("execution snapshot identity differs from the capability prepared-model hash")
    directory = Path(snapshot.directory_path)
    directory_metadata = _snapshot_directory_metadata(directory)
    if (directory_metadata.st_dev, directory_metadata.st_ino) != (
        snapshot.directory_device,
        snapshot.directory_inode,
    ):
        raise ManifestControlRuntimeError("manifest execution snapshot directory identity changed")
    model_path = Path(snapshot.model_path)
    try:
        metadata = os.stat(model_path, follow_symlinks=False)
    except OSError as error:
        raise ManifestControlRuntimeError("manifest execution snapshot path is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != _EXECUTION_SNAPSHOT_FILE_MODE
        or metadata.st_uid != os.geteuid()
        or (metadata.st_dev, metadata.st_ino, metadata.st_size)
        != (snapshot.model_device, snapshot.model_inode, snapshot.model_size)
    ):
        raise ManifestControlRuntimeError(
            "manifest execution snapshot must be a current-user single-link regular file with mode 0400"
        )
    actual = hashlib.sha256(_stable_regular_file_bytes(model_path)).hexdigest()
    if not hmac.compare_digest(actual, snapshot.model_sha256):
        raise ManifestControlRuntimeError("manifest execution snapshot bytes changed")
    return actual


def _open_snapshot_directory_for_cleanup(snapshot: ManifestExecutionSnapshot) -> int:
    directory = Path(snapshot.directory_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        raise ManifestControlRuntimeError("cannot open manifest execution snapshot directory for cleanup") from error
    try:
        metadata = os.fstat(descriptor)
        path_metadata = os.stat(directory, follow_symlinks=False)
        expected_identity = (snapshot.directory_device, snapshot.directory_inode)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or (metadata.st_dev, metadata.st_ino) != expected_identity
            or (path_metadata.st_dev, path_metadata.st_ino) != expected_identity
            or not stat.S_ISDIR(path_metadata.st_mode)
        ):
            raise ManifestControlRuntimeError("manifest execution snapshot directory identity changed before cleanup")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _unlink_snapshot_entries(
    descriptor: int,
    snapshot: ManifestExecutionSnapshot,
) -> list[str]:
    os.fchmod(descriptor, 0o700)
    findings: list[str] = []
    expected_descriptor = None
    try:
        for entry_name in os.listdir(descriptor):
            metadata = os.stat(entry_name, dir_fd=descriptor, follow_symlinks=False)
            if entry_name != _EXECUTION_SNAPSHOT_FILENAME:
                findings.append(f"unexpected snapshot-directory entry {entry_name!r}")
                continue
            if not stat.S_ISREG(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != (
                snapshot.model_device,
                snapshot.model_inode,
            ):
                findings.append("canonical snapshot model path has a foreign identity")
                continue
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            expected_descriptor = os.open(entry_name, flags, dir_fd=descriptor)
            opened_metadata = os.fstat(expected_descriptor)
            if not stat.S_ISREG(opened_metadata.st_mode) or (
                opened_metadata.st_dev,
                opened_metadata.st_ino,
            ) != (snapshot.model_device, snapshot.model_inode):
                raise ManifestControlRuntimeError("manifest execution snapshot model identity changed during cleanup")
            os.unlink(entry_name, dir_fd=descriptor)
        if expected_descriptor is None:
            findings.append("expected snapshot model inode was absent during cleanup")
        elif os.fstat(expected_descriptor).st_nlink != 0:
            findings.append("snapshot model inode retains a link outside its private directory")
        os.fsync(descriptor)
        return findings
    finally:
        if expected_descriptor is not None:
            os.close(expected_descriptor)


def remove_manifest_execution_snapshot(snapshot: ManifestExecutionSnapshot) -> None:
    """Remove one transient snapshot without following or deleting foreign paths."""

    snapshot = _canonical_execution_snapshot(snapshot)
    directory = Path(snapshot.directory_path)
    descriptor = _open_snapshot_directory_for_cleanup(snapshot)
    try:
        findings = _unlink_snapshot_entries(descriptor, snapshot)
        path_metadata = os.stat(directory, follow_symlinks=False)
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            snapshot.directory_device,
            snapshot.directory_inode,
        ):
            raise ManifestControlRuntimeError("manifest execution snapshot directory path changed during cleanup")
    finally:
        os.close(descriptor)
    try:
        directory.rmdir()
    except OSError as error:
        detail = "; ".join(findings) if findings else "directory is not empty"
        raise ManifestControlRuntimeError(f"manifest execution snapshot cleanup was incomplete: {detail}") from error
    if os.path.lexists(directory):
        raise ManifestControlRuntimeError("manifest execution snapshot directory remains after cleanup")
    if findings:
        raise ManifestControlRuntimeError("manifest execution snapshot cleanup found: " + "; ".join(findings))


def _cleanup_failed_snapshot(
    directory: Path,
    *,
    directory_identity: tuple[int, int],
    model_identity: tuple[int, int] | None,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    model_descriptor = None
    try:
        metadata = os.fstat(descriptor)
        path_metadata = os.stat(directory, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or (metadata.st_dev, metadata.st_ino) != directory_identity
            or (path_metadata.st_dev, path_metadata.st_ino) != directory_identity
        ):
            raise ManifestControlRuntimeError("failed snapshot directory identity changed before cleanup")
        os.fchmod(descriptor, 0o700)
        entries = os.listdir(descriptor)
        if entries:
            if entries != [_EXECUTION_SNAPSHOT_FILENAME] or model_identity is None:
                raise ManifestControlRuntimeError("failed snapshot cleanup found a foreign directory entry")
            model_metadata = os.stat(entries[0], dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(model_metadata.st_mode)
                or (
                    model_metadata.st_dev,
                    model_metadata.st_ino,
                )
                != model_identity
            ):
                raise ManifestControlRuntimeError("failed snapshot model path has a foreign identity")
            model_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            model_descriptor = os.open(entries[0], model_flags, dir_fd=descriptor)
            opened_metadata = os.fstat(model_descriptor)
            if (opened_metadata.st_dev, opened_metadata.st_ino) != model_identity:
                raise ManifestControlRuntimeError("failed snapshot model identity changed during cleanup")
            os.unlink(entries[0], dir_fd=descriptor)
            if os.fstat(model_descriptor).st_nlink != 0:
                raise ManifestControlRuntimeError("failed snapshot model retains an external link")
        os.fsync(descriptor)
    finally:
        if model_descriptor is not None:
            os.close(model_descriptor)
        os.close(descriptor)
    path_metadata = os.stat(directory, follow_symlinks=False)
    if (path_metadata.st_dev, path_metadata.st_ino) != directory_identity:
        raise ManifestControlRuntimeError("failed snapshot directory path changed during cleanup")
    directory.rmdir()


def materialize_manifest_execution_snapshot(
    manifest: ControlCapabilityManifest,
    source_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> ManifestExecutionSnapshot:
    """Own exact prepared bytes in one private, non-writable per-run directory."""

    canonical = _canonical_manifest(manifest)
    absolute_source = Path(os.path.abspath(os.fspath(source_path)))
    source_bytes, source_metadata = _stable_regular_file_read(absolute_source)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if not hmac.compare_digest(source_sha256, canonical.prepared_model_sha256):
        raise ManifestControlRuntimeError(
            "prepared model SHA-256 does not match the control capability manifest: "
            f"expected {canonical.prepared_model_sha256}, found {source_sha256}"
        )

    root = Path(os.path.abspath(os.fspath(output_root)))
    try:
        root_metadata = os.stat(root, follow_symlinks=False)
    except OSError as error:
        raise ManifestControlRuntimeError("manifest execution snapshot root must be an existing directory") from error
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_uid != os.geteuid():
        raise ManifestControlRuntimeError(
            "manifest execution snapshot root must be a current-user non-symlink directory"
        )
    directory = Path(tempfile.mkdtemp(prefix=".hot-manifest-input-", dir=root))
    created_directory_metadata = os.stat(directory, follow_symlinks=False)
    directory_identity = (created_directory_metadata.st_dev, created_directory_metadata.st_ino)
    model_path = directory / _EXECUTION_SNAPSHOT_FILENAME
    directory_descriptor = None
    descriptor = None
    model_metadata = None
    try:
        directory_descriptor = _open_created_snapshot_directory(
            directory,
            created_directory_metadata,
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            _EXECUTION_SNAPSHOT_FILENAME,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        view = memoryview(source_bytes)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise ManifestControlRuntimeError("could not completely write manifest execution snapshot")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, _EXECUTION_SNAPSHOT_FILE_MODE)
        model_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(model_metadata.st_mode)
            or model_metadata.st_nlink != 1
            or model_metadata.st_size != len(source_bytes)
        ):
            raise ManifestControlRuntimeError("manifest execution snapshot publication is incomplete")
        os.close(descriptor)
        descriptor = None
        published_directory_metadata = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(published_directory_metadata.st_mode)
            or (
                published_directory_metadata.st_dev,
                published_directory_metadata.st_ino,
                published_directory_metadata.st_uid,
            )
            != (
                created_directory_metadata.st_dev,
                created_directory_metadata.st_ino,
                created_directory_metadata.st_uid,
            )
            or os.listdir(directory_descriptor) != [_EXECUTION_SNAPSHOT_FILENAME]
        ):
            raise ManifestControlRuntimeError(
                "manifest execution snapshot directory changed while publishing its model"
            )
        published_directory_identity = _created_snapshot_directory_identity(published_directory_metadata)
        _revalidate_created_snapshot_directory(
            directory,
            directory_descriptor,
            published_directory_identity,
        )
        os.fchmod(directory_descriptor, _EXECUTION_SNAPSHOT_DIRECTORY_MODE)
        os.fsync(directory_descriptor)
        _revalidate_created_snapshot_directory(
            directory,
            directory_descriptor,
            published_directory_identity,
        )
        os.close(directory_descriptor)
        directory_descriptor = None
        directory_metadata = _snapshot_directory_metadata(directory)
        if _created_snapshot_directory_identity(directory_metadata) != published_directory_identity:
            raise ManifestControlRuntimeError(
                "manifest execution snapshot directory identity changed after publication"
            )
        snapshot = ManifestExecutionSnapshot(
            source_path=str(absolute_source),
            source_sha256=source_sha256,
            source_device=source_metadata.st_dev,
            source_inode=source_metadata.st_ino,
            source_size=source_metadata.st_size,
            directory_path=str(directory),
            directory_device=directory_metadata.st_dev,
            directory_inode=directory_metadata.st_ino,
            model_path=str(model_path),
            model_sha256=source_sha256,
            model_device=model_metadata.st_dev,
            model_inode=model_metadata.st_ino,
            model_size=model_metadata.st_size,
        )
        verify_manifest_execution_snapshot(canonical, snapshot)
        return snapshot
    except Exception as error:
        cleanup_model_identity = None
        if descriptor is not None:
            descriptor_metadata = os.fstat(descriptor)
            cleanup_model_identity = (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
            os.close(descriptor)
        elif model_metadata is not None:
            cleanup_model_identity = (model_metadata.st_dev, model_metadata.st_ino)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        try:
            _cleanup_failed_snapshot(
                directory,
                directory_identity=directory_identity,
                model_identity=cleanup_model_identity,
            )
        except Exception as cleanup_error:
            raise ManifestExecutionSnapshotCleanupError(error, cleanup_error) from error
        raise


def _native_handle(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise ManifestControlRuntimeError(f"EnergyPlus returned a non-native handle for {label}")
    if value < 0:
        raise ManifestControlRuntimeError(f"EnergyPlus could not resolve {label}")
    return value


def _finite_readback(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestControlRuntimeError(f"EnergyPlus returned a nonnumeric read-back for {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ManifestControlRuntimeError(f"EnergyPlus returned a non-finite read-back for {label}")
    if result == 0.0:
        result = 0.0
    return result


def _owned_json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(value))


def _claimed_sha256(value: Any, *, field_name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ManifestControlRuntimeError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(slots=True)
class ManifestControlRuntime:
    """Bind one canonical capability manifest to one EnergyPlus run at a time."""

    manifest: ControlCapabilityManifest
    action_transform: ManifestActionTransform = field(init=False)
    action_binding_sha256: str = field(init=False)
    actuator_handles: dict[str, int] = field(default_factory=dict, init=False)
    effect_handles: dict[str, int] = field(default_factory=dict, init=False)
    _pending_submission: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _completed_evidence: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        canonical = _canonical_manifest(self.manifest)
        object.__setattr__(self, "manifest", canonical)
        object.__setattr__(self, "action_transform", ManifestActionTransform(canonical))
        object.__setattr__(
            self,
            "action_binding_sha256",
            task_action_binding_sha256(canonical),
        )

    @property
    def capability_sha256(self) -> str:
        return self.manifest.capability_sha256

    @property
    def command_dim(self) -> int:
        return self.action_transform.command_dim

    @property
    def effect_variable_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(item.energyplus_pair for item in self.manifest.effect_variables)

    def reset_run(self) -> None:
        """Discard handles and interval evidence owned by a prior E+ state."""

        self.actuator_handles = {}
        self.effect_handles = {}
        self._pending_submission = None
        self._completed_evidence = None

    def request_effect_variables(self, exchange: Any, state: int) -> None:
        """Request every effect variable before EnergyPlus begins the run."""

        for variable in self.manifest.effect_variables:
            exchange.request_variable(state, variable.variable_name, variable.key_value)

    def resolve_handles(self, exchange: Any, state: int) -> None:
        """Resolve the complete interface once and reject missing/aliased handles."""

        if self.actuator_handles or self.effect_handles:
            raise ManifestControlRuntimeError("manifest control handles were already resolved for this run")

        actuator_handles: dict[str, int] = {}
        actuator_owner_by_handle: dict[int, str] = {}
        for actuator in self.manifest.full_actuators:
            handle = _native_handle(
                exchange.get_actuator_handle(
                    state,
                    actuator.component_type,
                    actuator.control_type,
                    actuator.component_name,
                ),
                label=f"actuator {actuator.endpoint_id!r}",
            )
            previous = actuator_owner_by_handle.get(handle)
            if previous is not None:
                raise ManifestControlRuntimeError(
                    "distinct manifest actuator endpoints resolved to one ambiguous handle: "
                    f"{previous!r} and {actuator.endpoint_id!r}"
                )
            actuator_owner_by_handle[handle] = actuator.endpoint_id
            actuator_handles[actuator.endpoint_id] = handle

        effect_handles: dict[str, int] = {}
        effect_owner_by_handle: dict[int, str] = {}
        for variable in self.manifest.effect_variables:
            handle = _native_handle(
                exchange.get_variable_handle(state, variable.variable_name, variable.key_value),
                label=f"effect output {variable.variable_id!r}",
            )
            previous = effect_owner_by_handle.get(handle)
            if previous is not None:
                raise ManifestControlRuntimeError(
                    "distinct manifest effect outputs resolved to one ambiguous handle: "
                    f"{previous!r} and {variable.variable_id!r}"
                )
            effect_owner_by_handle[handle] = variable.variable_id
            effect_handles[variable.variable_id] = handle

        if len(actuator_handles) != len(self.manifest.full_actuators):
            raise ManifestControlRuntimeError("manifest actuator handle coverage is incomplete")
        if len(effect_handles) != len(self.manifest.effect_variables):
            raise ManifestControlRuntimeError("manifest effect-output handle coverage is incomplete")
        self.actuator_handles = actuator_handles
        self.effect_handles = effect_handles

    def _canonical_expanded_action(self, value: Any) -> ExpandedControlAction:
        if type(value) is not ExpandedControlAction:
            raise TypeError("manifest runtime action must be an exact ExpandedControlAction")
        try:
            recomputed = self.action_transform.expand(value.normalized_values)
        except (AttributeError, TypeError, ValueError) as error:
            raise ManifestControlRuntimeError(f"expanded control action is malformed or tampered: {error}") from error
        if recomputed != value:
            raise ManifestControlRuntimeError("expanded control action does not match its normalized command vector")
        return recomputed

    def apply_action(self, exchange: Any, state: int, action: Any) -> None:
        """Set and immediately read back every physical endpoint exactly once."""

        if self._pending_submission is not None:
            raise ManifestControlRuntimeError("cannot overwrite an incomplete manifest control interval")
        expected_endpoint_ids = tuple(item.endpoint_id for item in self.manifest.full_actuators)
        if tuple(self.actuator_handles) != expected_endpoint_ids:
            raise ManifestControlRuntimeError("manifest actuator handles were not completely preflighted")
        expanded = self._canonical_expanded_action(action)

        for command in expanded.endpoint_commands:
            exchange.set_actuator_value(
                state,
                self.actuator_handles[command.endpoint_id],
                command.value,
            )

        submitted_records: list[dict[str, Any]] = []
        readback_records: list[dict[str, Any]] = []
        for order, (actuator, command) in enumerate(zip(self.manifest.full_actuators, expanded.endpoint_commands)):
            if command.endpoint_id != actuator.endpoint_id or command.energyplus_triple != actuator.energyplus_triple:
                raise ManifestControlRuntimeError("expanded endpoint order differs from the authenticated manifest")
            base_record = {
                "order": order,
                "endpoint_id": actuator.endpoint_id,
                "channel": actuator.channel.value,
                "component_type": actuator.component_type,
                "control_type": actuator.control_type,
                "component_name": actuator.component_name,
                "unit": actuator.unit.value,
                "role": actuator.role.value,
            }
            submitted_records.append({**base_record, "value": command.value})
            readback = _finite_readback(
                exchange.get_actuator_value(state, self.actuator_handles[actuator.endpoint_id]),
                label=f"actuator {actuator.endpoint_id!r}",
            )
            if readback != command.value:
                raise ManifestControlRuntimeError(
                    f"actuator {actuator.endpoint_id!r} read-back mismatch: "
                    f"submitted {command.value}, found {readback}"
                )
            readback_records.append(
                {
                    **base_record,
                    "submitted_value": command.value,
                    "readback_value": readback,
                }
            )

        normalized_records = [
            {
                "order": command.order,
                "name": command.name,
                "unit": command.unit.value,
                "normalized_value": expanded.normalized_values[index],
                "physical_value": expanded.command_values[index],
                "target_endpoint_ids": list(command.target_endpoint_ids),
            }
            for index, command in enumerate(self.manifest.agent_commands)
        ]
        submission_payload = {
            "normalized_commands": normalized_records,
            "submitted_endpoints": submitted_records,
            "actuator_readbacks": readback_records,
        }
        self._pending_submission = {
            **submission_payload,
            "submission_sha256": _domain_sha256(
                MANIFEST_ACTION_SUBMISSION_HASH_NAMESPACE,
                submission_payload,
            ),
        }
        self._completed_evidence = None

    def complete_interval(self, exchange: Any, state: int) -> dict[str, Any]:
        """Read all declared effects after zone reporting and seal the evidence."""

        if self._pending_submission is None:
            raise ManifestControlRuntimeError("cannot read effects before a manifest action was submitted")
        expected_variable_ids = tuple(item.variable_id for item in self.manifest.effect_variables)
        if tuple(self.effect_handles) != expected_variable_ids:
            raise ManifestControlRuntimeError("manifest effect-output handles were not completely preflighted")

        effect_records: list[dict[str, Any]] = []
        for order, variable in enumerate(self.manifest.effect_variables):
            value = _finite_readback(
                exchange.get_variable_value(state, self.effect_handles[variable.variable_id]),
                label=f"effect output {variable.variable_id!r}",
            )
            if not variable.lower_bound <= value <= variable.upper_bound:
                raise ManifestControlRuntimeError(
                    f"effect output {variable.variable_id!r} lies outside its authenticated bounds"
                )
            effect_records.append(
                {
                    "order": order,
                    "variable_id": variable.variable_id,
                    "semantic": variable.semantic.value,
                    "variable_name": variable.variable_name,
                    "key_value": variable.key_value,
                    "unit": variable.unit.value,
                    "actuator_endpoint_ids": list(variable.actuator_endpoint_ids),
                    "value": value,
                }
            )

        payload = {
            "schema_version": MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION,
            "control_capability_sha256": self.capability_sha256,
            "task_action_binding_sha256": self.action_binding_sha256,
            **self._pending_submission,
            "effect_readbacks": effect_records,
        }
        evidence = {
            **payload,
            "evidence_sha256": _domain_sha256(
                MANIFEST_ACTION_EVIDENCE_HASH_NAMESPACE,
                payload,
            ),
        }
        self._pending_submission = None
        verified = self.verify_evidence(evidence)
        self._completed_evidence = verified
        return _owned_json_mapping(verified)

    def _verify_evidence_identity(self, evidence: dict[str, Any]) -> None:
        expected_keys = {
            "schema_version",
            "control_capability_sha256",
            "task_action_binding_sha256",
            "normalized_commands",
            "submitted_endpoints",
            "actuator_readbacks",
            "submission_sha256",
            "effect_readbacks",
            "evidence_sha256",
        }
        if set(evidence) != expected_keys:
            raise ManifestControlRuntimeError("manifest action evidence has missing or unknown fields")
        if evidence["schema_version"] != MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION:
            raise ManifestControlRuntimeError("manifest action evidence has an unknown schema")
        if evidence["control_capability_sha256"] != self.capability_sha256:
            raise ManifestControlRuntimeError("manifest action evidence has the wrong capability identity")
        if evidence["task_action_binding_sha256"] != self.action_binding_sha256:
            raise ManifestControlRuntimeError("manifest action evidence has the wrong action-binding identity")

    def _evidence_record_lists(
        self,
        evidence: dict[str, Any],
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        normalized = evidence["normalized_commands"]
        submitted = evidence["submitted_endpoints"]
        readbacks = evidence["actuator_readbacks"]
        effects = evidence["effect_readbacks"]
        if not isinstance(normalized, list) or len(normalized) != len(self.manifest.agent_commands):
            raise ManifestControlRuntimeError("manifest action evidence has incomplete normalized commands")
        if not isinstance(submitted, list) or len(submitted) != len(self.manifest.full_actuators):
            raise ManifestControlRuntimeError("manifest action evidence has incomplete endpoint submissions")
        if not isinstance(readbacks, list) or len(readbacks) != len(self.manifest.full_actuators):
            raise ManifestControlRuntimeError("manifest action evidence has incomplete actuator read-backs")
        if not isinstance(effects, list) or len(effects) != len(self.manifest.effect_variables):
            raise ManifestControlRuntimeError("manifest action evidence has incomplete effect read-backs")
        return normalized, submitted, readbacks, effects

    def _expand_evidence_action(self, normalized: list[Any]) -> ExpandedControlAction:
        try:
            return self.action_transform.expand([item["normalized_value"] for item in normalized])
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestControlRuntimeError(
                f"manifest action evidence has invalid normalized commands: {error}"
            ) from error

    def _verify_normalized_evidence(
        self,
        normalized: list[Any],
        expanded: ExpandedControlAction,
    ) -> None:
        for index, (record, command) in enumerate(zip(normalized, self.manifest.agent_commands)):
            expected = {
                "order": command.order,
                "name": command.name,
                "unit": command.unit.value,
                "normalized_value": expanded.normalized_values[index],
                "physical_value": expanded.command_values[index],
                "target_endpoint_ids": list(command.target_endpoint_ids),
            }
            if record != expected:
                raise ManifestControlRuntimeError("manifest normalized-command evidence differs from its binding")

    @staticmethod
    def _endpoint_evidence_base(order: int, actuator: Any) -> dict[str, Any]:
        return {
            "order": order,
            "endpoint_id": actuator.endpoint_id,
            "channel": actuator.channel.value,
            "component_type": actuator.component_type,
            "control_type": actuator.control_type,
            "component_name": actuator.component_name,
            "unit": actuator.unit.value,
            "role": actuator.role.value,
        }

    def _verify_endpoint_evidence(
        self,
        submitted: list[Any],
        readbacks: list[Any],
        expanded: ExpandedControlAction,
    ) -> None:
        for order, (actuator, endpoint, submitted_record, readback_record) in enumerate(
            zip(self.manifest.full_actuators, expanded.endpoint_commands, submitted, readbacks)
        ):
            base = self._endpoint_evidence_base(order, actuator)
            if submitted_record != {**base, "value": endpoint.value}:
                raise ManifestControlRuntimeError("manifest endpoint submission differs from its expanded action")
            if readback_record != {
                **base,
                "submitted_value": endpoint.value,
                "readback_value": endpoint.value,
            }:
                raise ManifestControlRuntimeError("manifest actuator read-back differs from its submission")

    def _verify_effect_evidence(self, effects: list[Any]) -> None:
        for order, (variable, record) in enumerate(zip(self.manifest.effect_variables, effects)):
            if not isinstance(record, dict):
                raise ManifestControlRuntimeError("manifest effect read-back must be a record")
            effect_value = _finite_readback(record.get("value"), label=f"effect output {variable.variable_id!r}")
            expected = {
                "order": order,
                "variable_id": variable.variable_id,
                "semantic": variable.semantic.value,
                "variable_name": variable.variable_name,
                "key_value": variable.key_value,
                "unit": variable.unit.value,
                "actuator_endpoint_ids": list(variable.actuator_endpoint_ids),
                "value": effect_value,
            }
            if record != expected or not variable.lower_bound <= effect_value <= variable.upper_bound:
                raise ManifestControlRuntimeError("manifest effect read-back differs from its descriptor")

    def _verify_evidence_hashes(
        self,
        evidence: dict[str, Any],
        normalized: list[Any],
        submitted: list[Any],
        readbacks: list[Any],
    ) -> None:
        submission_payload = {
            "normalized_commands": normalized,
            "submitted_endpoints": submitted,
            "actuator_readbacks": readbacks,
        }
        claimed_submission = _claimed_sha256(evidence["submission_sha256"], field_name="submission_sha256")
        expected_submission = _domain_sha256(MANIFEST_ACTION_SUBMISSION_HASH_NAMESPACE, submission_payload)
        if not hmac.compare_digest(claimed_submission, expected_submission):
            raise ManifestControlRuntimeError("manifest action submission hash does not match its records")
        payload = {key: item for key, item in evidence.items() if key != "evidence_sha256"}
        claimed_evidence = _claimed_sha256(evidence["evidence_sha256"], field_name="evidence_sha256")
        expected_evidence = _domain_sha256(MANIFEST_ACTION_EVIDENCE_HASH_NAMESPACE, payload)
        if not hmac.compare_digest(claimed_evidence, expected_evidence):
            raise ManifestControlRuntimeError("manifest action evidence hash does not match its records")

    def verify_evidence(self, value: Any) -> dict[str, Any]:
        """Recompute one evidence record against the bound manifest and action."""

        if not isinstance(value, Mapping):
            raise TypeError("manifest action evidence must be a mapping")
        try:
            evidence = _owned_json_mapping(value)
        except (TypeError, ValueError) as error:
            raise ManifestControlRuntimeError(
                f"manifest action evidence is not canonical JSON data: {error}"
            ) from error
        self._verify_evidence_identity(evidence)
        normalized, submitted, readbacks, effects = self._evidence_record_lists(evidence)
        expanded = self._expand_evidence_action(normalized)
        self._verify_normalized_evidence(normalized, expanded)
        self._verify_endpoint_evidence(submitted, readbacks, expanded)
        self._verify_effect_evidence(effects)
        self._verify_evidence_hashes(evidence, normalized, submitted, readbacks)
        return evidence

    @property
    def completed_evidence(self) -> dict[str, Any] | None:
        if self._completed_evidence is None:
            return None
        return _owned_json_mapping(self._completed_evidence)


__all__ = [
    "MANIFEST_ACTION_BINDING_HASH_NAMESPACE",
    "MANIFEST_ACTION_EVIDENCE_HASH_NAMESPACE",
    "MANIFEST_ACTION_SUBMISSION_HASH_NAMESPACE",
    "MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION",
    "MANIFEST_EXECUTION_SNAPSHOT_SCHEMA_VERSION",
    "ManifestControlRuntime",
    "ManifestControlRuntimeError",
    "ManifestExecutionSnapshot",
    "ManifestExecutionSnapshotCleanupError",
    "materialize_manifest_execution_snapshot",
    "remove_manifest_execution_snapshot",
    "task_action_binding_sha256",
    "verified_manifest_prepared_model_bytes",
    "verify_manifest_execution_snapshot",
    "verify_manifest_prepared_model",
]
