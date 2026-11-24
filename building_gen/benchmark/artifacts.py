"""Versioned JSON schemas for benchmark results and generated reports."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from building_gen.benchmark._canonical import canonical_json, content_id
from building_gen.benchmark.provenance import ResolvedRun, SoftwareProvenance


RESULT_SCHEMA_VERSION = "hot-control-result/1.0.0"
REPORT_SCHEMA_VERSION = "hot-control-report/1.0.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class MetricValue:
    """One scalar metric with enough semantics for safe aggregation."""

    name: str
    value: float
    unit: str
    aggregation: str
    direction: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.unit.strip():
            raise ValueError("metric name and unit cannot be empty")
        if not math.isfinite(self.value):
            raise ValueError(f"metric {self.name!r} must be finite")
        if self.aggregation not in {"sum", "mean", "max", "min", "final", "count"}:
            raise ValueError(f"unsupported aggregation {self.aggregation!r}")
        if self.direction not in {"minimize", "maximize", "neutral"}:
            raise ValueError(f"unsupported metric direction {self.direction!r}")


@dataclass(frozen=True)
class ArtifactRef:
    """Content-verifiable reference to a checkpoint, table, plot, or log."""

    role: str
    uri: str
    media_type: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.uri.strip() or not self.media_type.strip():
            raise ValueError("artifact role, uri, and media_type cannot be empty")
        if self.sha256 is not None:
            digest = self.sha256.lower().removeprefix("sha256:")
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError("artifact sha256 must contain exactly 64 hexadecimal characters")
            object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class RunResult:
    """Portable result emitted by one :class:`ResolvedRun`."""

    run: ResolvedRun
    status: str
    provenance: SoftwareProvenance
    metrics: tuple[MetricValue, ...] = field(default_factory=tuple)
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    failure: str | None = None
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("status must be 'succeeded', 'failed', or 'cancelled'")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("metric names must be unique within one result")
        if self.status == "succeeded" and self.failure is not None:
            raise ValueError("a succeeded result cannot contain a failure message")
        if self.status == "failed" and (self.failure is None or not self.failure.strip()):
            raise ValueError("a failed result requires a failure message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run.run_id,
            "run": self.run.identity_payload(),
            "status": self.status,
            "provenance": self.provenance.to_dict(),
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "aggregation": metric.aggregation,
                    "direction": metric.direction,
                }
                for metric in self.metrics
            ],
            "artifacts": [
                {
                    "role": artifact.role,
                    "uri": artifact.uri,
                    "media_type": artifact.media_type,
                    "sha256": artifact.sha256,
                }
                for artifact in self.artifacts
            ],
            "failure": self.failure,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class ReportArtifact:
    """Machine-readable manifest for a benchmark report and its evidence."""

    title: str
    result_run_ids: tuple[str, ...]
    summary_metrics: tuple[MetricValue, ...]
    artifacts: tuple[ArtifactRef, ...]
    provenance: SoftwareProvenance
    schema_version: str = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("report title cannot be empty")
        object.__setattr__(self, "result_run_ids", tuple(self.result_run_ids))
        object.__setattr__(self, "summary_metrics", tuple(self.summary_metrics))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if not self.result_run_ids or len(set(self.result_run_ids)) != len(self.result_run_ids):
            raise ValueError("result_run_ids must be non-empty and unique")
        if len({metric.name for metric in self.summary_metrics}) != len(self.summary_metrics):
            raise ValueError("summary metric names must be unique")

    @property
    def report_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "result_run_ids": sorted(self.result_run_ids),
            "summary_metrics": self.summary_metrics,
            "artifacts": self.artifacts,
            "provenance": self.provenance.to_dict(),
        }
        return content_id("HOT control report", payload, prefix="report")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "title": self.title,
            "result_run_ids": list(self.result_run_ids),
            "summary_metrics": [metric.__dict__ for metric in self.summary_metrics],
            "artifacts": [artifact.__dict__ for artifact in self.artifacts],
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_json(self.to_dict())
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, allow_nan=False)


__all__ = ["ArtifactRef", "MetricValue", "ReportArtifact", "RunResult"]
