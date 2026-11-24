"""Deterministic run identities and explicit software/input provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from building_gen.benchmark._canonical import content_id, refreeze_mapping, thaw_mapping
from building_gen.benchmark.registry import (
    AlgorithmDescriptor,
    AlgorithmRegistry,
    assert_compatible,
)
from building_gen.benchmark.specs import ExperimentSpec, TaskSpec

RUN_SCHEMA_VERSION = "hot-control-run/1.0.0"
PROVENANCE_SCHEMA_VERSION = "hot-control-provenance/1.0.0"
SCIENTIFIC_REPOSITORY = "github.com/amcberkes/hot"
LEGACY_SCIENTIFIC_REPOSITORY = "github.com/amcberkes/HOT_benchmark"
SCIENTIFIC_DEPENDENCIES = ("numpy", "torch", "gymnasium", "tensorboard", "PyYAML")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SCIENTIFIC_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SoftwareProvenance:
    """Software versions recorded with every result artifact."""

    repository: str
    revision: str
    dirty: bool
    python_version: str
    energyplus_version: str
    container_digest: str | None = None
    dependencies: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("repository", "python_version", "energyplus_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        if not _REVISION_RE.fullmatch(self.revision):
            raise ValueError("revision must be a 7--64 character hexadecimal version-control revision")
        if self.container_digest is not None and not self.container_digest.strip():
            raise ValueError("container_digest cannot be empty when supplied")
        object.__setattr__(
            self,
            "dependencies",
            refreeze_mapping(self.dependencies, field="software dependencies"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "revision": self.revision.lower(),
            "dirty": self.dirty,
            "python_version": self.python_version,
            "energyplus_version": self.energyplus_version,
            "container_digest": self.container_digest,
            "dependencies": thaw_mapping(self.dependencies),
        }


def validate_scientific_software_provenance(
    provenance: SoftwareProvenance,
    *,
    expected_revision: str | None = None,
    context: str = "scientific execution",
) -> None:
    """Fail closed on the software identity required for publication evidence.

    Engineering results intentionally retain the more permissive
    :class:`SoftwareProvenance` schema.  Scientific execution and publication
    artifact validation share this stricter boundary so a diagnostics-clean
    result cannot be promoted when its checkout, container, or dependency
    identity is incomplete.
    """

    if not isinstance(provenance, SoftwareProvenance):
        raise TypeError("provenance must be a SoftwareProvenance")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    context = context.strip()
    if provenance.dirty:
        raise ValueError(f"{context} requires clean software provenance")
    # Preserve authentic historical artifacts from the predecessor repository.
    # New scientific execution separately authenticates the current checkout.
    if provenance.repository not in {SCIENTIFIC_REPOSITORY, LEGACY_SCIENTIFIC_REPOSITORY}:
        raise ValueError(f"{context} provenance must identify {SCIENTIFIC_REPOSITORY!r} or its predecessor")
    if not _SCIENTIFIC_REVISION_RE.fullmatch(provenance.revision):
        raise ValueError(f"{context} provenance revision must be a full lowercase Git object ID")
    if expected_revision is not None:
        if not isinstance(expected_revision, str) or not _SCIENTIFIC_REVISION_RE.fullmatch(expected_revision):
            raise ValueError("expected scientific revision must be a full lowercase Git object ID")
        if provenance.revision != expected_revision:
            raise ValueError(f"{context} provenance revision does not match checkout HEAD")
    if not isinstance(provenance.container_digest, str) or not _CONTAINER_DIGEST_RE.fullmatch(
        provenance.container_digest
    ):
        raise ValueError(f"{context} requires an exact lowercase sha256 container digest")
    dependencies = provenance.to_dict()["dependencies"]
    missing = [
        name
        for name in SCIENTIFIC_DEPENDENCIES
        if not isinstance(dependencies.get(name), str)
        or not dependencies[name]
        or dependencies[name].strip() != dependencies[name]
    ]
    if missing:
        raise ValueError(f"{context} provenance is missing dependency versions: {missing!r}")


@dataclass(frozen=True)
class ResolvedRun:
    """One deterministic training or evaluation unit in an experiment matrix."""

    experiment_id: str
    algorithm_id: str
    algorithm_version: str
    phase: str
    seed: int
    train_task_ids: tuple[str, ...]
    evaluation_task_id: str | None = None
    parent_training_run_id: str | None = None
    schema_version: str = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.phase not in {"train", "evaluate"}:
            raise ValueError("phase must be 'train' or 'evaluate'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "train_task_ids", tuple(self.train_task_ids))
        if self.phase == "train":
            if not self.train_task_ids:
                raise ValueError("training runs require at least one train_task_id")
            if self.evaluation_task_id is not None or self.parent_training_run_id is not None:
                raise ValueError("training runs cannot declare an evaluation task or parent run")
        else:
            if self.evaluation_task_id is None:
                raise ValueError("evaluation runs require evaluation_task_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "phase": self.phase,
            "seed": self.seed,
            "train_task_ids": self.train_task_ids,
            "evaluation_task_id": self.evaluation_task_id,
            "parent_training_run_id": self.parent_training_run_id,
        }

    @property
    def run_id(self) -> str:
        return content_id("HOT control run", self.identity_payload(), prefix="run")


def _training_run(
    experiment_id: str,
    descriptor: AlgorithmDescriptor,
    seed: int,
    train_task_ids: tuple[str, ...],
) -> ResolvedRun:
    return ResolvedRun(
        experiment_id=experiment_id,
        algorithm_id=descriptor.algorithm_id,
        algorithm_version=descriptor.version,
        phase="train",
        seed=seed,
        train_task_ids=train_task_ids,
    )


def resolve_experiment_runs(
    experiment: ExperimentSpec,
    tasks: dict[str, TaskSpec],
    registry: AlgorithmRegistry,
    *,
    require_available: bool = True,
    require_checksums: bool = False,
    allow_disabled: bool = False,
) -> tuple[ResolvedRun, ...]:
    """Validate and expand an experiment into deterministic run records."""

    if not experiment.enabled and not allow_disabled:
        raise ValueError(
            f"experiment {experiment.name!r} is disabled; resolve its placeholders and set enabled=true before launch"
        )
    assert_compatible(
        experiment,
        tasks,
        registry,
        require_available=require_available,
        require_checksums=require_checksums,
    )
    descriptor = registry.get(experiment.algorithm_id)
    task_ids = {name: task.task_id for name, task in tasks.items()}
    experiment_id = experiment.experiment_id(task_ids=task_ids, algorithm_version=descriptor.version)
    train_task_ids = tuple(task_ids[name] for name in experiment.train_tasks)
    runs: list[ResolvedRun] = []
    for seed in experiment.seeds:
        training: ResolvedRun | None = None
        if descriptor.requires_training:
            training = _training_run(experiment_id, descriptor, seed, train_task_ids)
            runs.append(training)
        for task_name in experiment.evaluation_tasks:
            runs.append(
                ResolvedRun(
                    experiment_id=experiment_id,
                    algorithm_id=descriptor.algorithm_id,
                    algorithm_version=descriptor.version,
                    phase="evaluate",
                    seed=seed,
                    train_task_ids=train_task_ids,
                    evaluation_task_id=task_ids[task_name],
                    parent_training_run_id=training.run_id if training is not None else None,
                )
            )
    return tuple(runs)


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "SCIENTIFIC_DEPENDENCIES",
    "SCIENTIFIC_REPOSITORY",
    "ResolvedRun",
    "SoftwareProvenance",
    "resolve_experiment_runs",
    "validate_scientific_software_provenance",
]
