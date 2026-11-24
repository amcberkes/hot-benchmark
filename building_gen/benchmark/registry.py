"""Algorithm descriptors and framework-neutral compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from building_gen.benchmark._canonical import content_id
from building_gen.benchmark.specs import ExperimentSpec, SpaceSpec, TaskSpec


class AlgorithmKind(str, Enum):
    BUILTIN_POLICY = "builtin_policy"
    EXTERNAL_TRAINER = "external_trainer"


class AlgorithmStatus(str, Enum):
    AVAILABLE = "available"
    PLANNED = "planned"


class TransferShapePolicy(str, Enum):
    """How one trained model relates to task-space signatures."""

    PER_TASK = "per_task"
    EXACT = "exact"


@dataclass(frozen=True)
class AlgorithmDescriptor:
    """Static metadata about a policy or trainer.

    ``entrypoint`` is a string, not an imported callable.  Listing or
    validating PPO therefore never imports Torch or an RL framework.
    """

    algorithm_id: str
    version: str
    kind: AlgorithmKind
    status: AlgorithmStatus
    entrypoint: str
    requires_training: bool
    supported_action_profiles: frozenset[str]
    supported_observation_kinds: frozenset[str] = frozenset({"box"})
    supported_action_kinds: frozenset[str] = frozenset({"box"})
    transfer_shape_policy: TransferShapePolicy = TransferShapePolicy.EXACT
    description: str = ""

    def __post_init__(self) -> None:
        if not self.algorithm_id.strip() or not self.version.strip():
            raise ValueError("algorithm_id and version cannot be empty")
        if ":" not in self.entrypoint:
            raise ValueError("entrypoint must use 'module:object' notation")
        if not self.supported_action_profiles:
            raise ValueError("supported_action_profiles cannot be empty")

    @property
    def descriptor_id(self) -> str:
        return content_id("HOT algorithm descriptor", self, prefix="algorithm")


class AlgorithmRegistry:
    """Explicit registry with duplicate protection and helpful lookups."""

    def __init__(self, descriptors: tuple[AlgorithmDescriptor, ...] = ()) -> None:
        self._descriptors: dict[str, AlgorithmDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: AlgorithmDescriptor) -> None:
        if descriptor.algorithm_id in self._descriptors:
            raise ValueError(f"algorithm {descriptor.algorithm_id!r} is already registered")
        self._descriptors[descriptor.algorithm_id] = descriptor

    def get(self, algorithm_id: str) -> AlgorithmDescriptor:
        try:
            return self._descriptors[algorithm_id]
        except KeyError as error:
            available = ", ".join(sorted(self._descriptors)) or "none"
            raise KeyError(f"unknown algorithm {algorithm_id!r}; registered algorithms: {available}") from error

    def list(self, *, include_planned: bool = True) -> tuple[AlgorithmDescriptor, ...]:
        descriptors = self._descriptors.values()
        if not include_planned:
            descriptors = (item for item in descriptors if item.status is AlgorithmStatus.AVAILABLE)
        return tuple(sorted(descriptors, key=lambda item: item.algorithm_id))


@dataclass(frozen=True)
class CompatibilityIssue:
    task_name: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"task {self.task_name!r}: {self.field}: {self.message}"


class BenchmarkCompatibilityError(ValueError):
    """Raised when an experiment cannot be executed with its declared tasks."""

    def __init__(self, experiment_name: str, issues: tuple[CompatibilityIssue, ...]) -> None:
        self.experiment_name = experiment_name
        self.issues = issues
        details = "\n  - ".join(str(issue) for issue in issues)
        super().__init__(f"experiment {experiment_name!r} is incompatible:\n  - {details}")


def _space_difference(source: SpaceSpec, target: SpaceSpec, label: str) -> str:
    differences = []
    for field_name in ("kind", "shape", "dtype", "encoding", "lower_bound", "upper_bound"):
        source_value = getattr(source, field_name)
        target_value = getattr(target, field_name)
        if source_value != target_value:
            differences.append(f"{field_name} {source_value!r} != {target_value!r}")
    return f"{label} signature differs from training task ({'; '.join(differences)})"


def compatibility_issues(
    experiment: ExperimentSpec,
    tasks: dict[str, TaskSpec],
    registry: AlgorithmRegistry,
    *,
    require_available: bool = True,
    require_checksums: bool = False,
) -> tuple[CompatibilityIssue, ...]:
    """Return every actionable incompatibility instead of failing at the first."""

    descriptor = registry.get(experiment.algorithm_id)
    issues: list[CompatibilityIssue] = []
    referenced_names = experiment.train_tasks + experiment.evaluation_tasks
    missing = [name for name in referenced_names if name not in tasks]
    for name in missing:
        issues.append(CompatibilityIssue(name, "task", "is not defined in the benchmark suite"))
    if missing:
        return tuple(issues)

    if require_available and descriptor.status is not AlgorithmStatus.AVAILABLE:
        issues.append(
            CompatibilityIssue(
                "<algorithm>",
                "status",
                f"algorithm {descriptor.algorithm_id!r} is {descriptor.status.value}; no runnable trainer is registered",
            )
        )
    if descriptor.requires_training and not experiment.train_tasks:
        issues.append(
            CompatibilityIssue("<experiment>", "train_tasks", "the selected algorithm requires at least one training task")
        )
    if not descriptor.requires_training and experiment.train_tasks:
        issues.append(
            CompatibilityIssue("<experiment>", "train_tasks", "a built-in baseline must not declare training tasks")
        )

    referenced_tasks = [tasks[name] for name in dict.fromkeys(referenced_names)]
    for task in referenced_tasks:
        if task.observation_space.kind not in descriptor.supported_observation_kinds:
            issues.append(
                CompatibilityIssue(
                    task.name,
                    "observation_space.kind",
                    f"{task.observation_space.kind!r} is not supported by {descriptor.algorithm_id!r}",
                )
            )
        if task.action_space.kind not in descriptor.supported_action_kinds:
            issues.append(
                CompatibilityIssue(
                    task.name,
                    "action_space.kind",
                    f"{task.action_space.kind!r} is not supported by {descriptor.algorithm_id!r}",
                )
            )
        if task.action_profile.profile_id not in descriptor.supported_action_profiles:
            supported = ", ".join(sorted(descriptor.supported_action_profiles))
            issues.append(
                CompatibilityIssue(
                    task.name,
                    "action_profile",
                    f"{task.action_profile.profile_id!r} is unsupported; expected one of: {supported}",
                )
            )
        if require_checksums and not task.publication_ready:
            issues.append(
                CompatibilityIssue(
                    task.name,
                    "assets",
                    "building and weather SHA256 checksums are required for a publication run",
                )
            )

    if descriptor.transfer_shape_policy is TransferShapePolicy.EXACT and experiment.train_tasks:
        source = tasks[experiment.train_tasks[0]]
        for task_name in experiment.train_tasks[1:] + experiment.evaluation_tasks:
            target = tasks[task_name]
            if target.observation_space.signature() != source.observation_space.signature():
                issues.append(
                    CompatibilityIssue(
                        target.name,
                        "observation_space",
                        _space_difference(source.observation_space, target.observation_space, "observation"),
                    )
                )
            if target.action_space.signature() != source.action_space.signature():
                issues.append(
                    CompatibilityIssue(
                        target.name,
                        "action_space",
                        _space_difference(source.action_space, target.action_space, "action"),
                    )
                )
            if target.action_profile.transfer_signature() != source.action_profile.transfer_signature():
                issues.append(
                    CompatibilityIssue(
                        target.name,
                        "action_profile",
                        "channel layout, transform version, units, physical bounds, or entity count differs "
                        "from the training task "
                        f"profile {source.action_profile.profile_id!r}",
                    )
                )
    return tuple(issues)


def assert_compatible(
    experiment: ExperimentSpec,
    tasks: dict[str, TaskSpec],
    registry: AlgorithmRegistry,
    *,
    require_available: bool = True,
    require_checksums: bool = False,
) -> None:
    issues = compatibility_issues(
        experiment,
        tasks,
        registry,
        require_available=require_available,
        require_checksums=require_checksums,
    )
    if issues:
        raise BenchmarkCompatibilityError(experiment.name, issues)


def default_algorithm_registry() -> AlgorithmRegistry:
    """Return built-in baselines and dependency-free trainer descriptors."""

    thermostat = frozenset({"thermostat_v1"})
    return AlgorithmRegistry(
        (
            AlgorithmDescriptor(
                algorithm_id="native_incumbent",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.evaluation:native_incumbent_policy",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Engineering evaluation of native schedules with thermostat overrides released.",
            ),
            AlgorithmDescriptor(
                algorithm_id="occupancy_reactive_thermostat",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:occupancy_reactive_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Measured per-zone occupancy thermostat without preconditioning.",
            ),
            AlgorithmDescriptor(
                algorithm_id="holiday_aware_thermostat",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:holiday_aware_thermostat_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Weekly preconditioning with measured-occupancy holiday control.",
            ),
            AlgorithmDescriptor(
                algorithm_id="scheduled_thermostat",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:scheduled_thermostat_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Explicit occupied/unoccupied weekly calendar thermostat schedule.",
            ),
            AlgorithmDescriptor(
                algorithm_id="seasonal_thermostat",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:seasonal_thermostat_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Explicit calendar-season heating and cooling pairs.",
            ),
            AlgorithmDescriptor(
                algorithm_id="fixed_thermostat",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:fixed_thermostat_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Explicit heating and cooling setpoints in every controlled zone.",
            ),
            AlgorithmDescriptor(
                algorithm_id="comfort_band_20_24",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:comfort_band_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Fixed 20 C heating and 24 C cooling in every zone.",
            ),
            AlgorithmDescriptor(
                algorithm_id="static_midpoint",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:static_midpoint_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Set every normalized control channel to its midpoint.",
            ),
            AlgorithmDescriptor(
                algorithm_id="ashrae_seasonal_setpoints",
                version="1.0.0",
                kind=AlgorithmKind.BUILTIN_POLICY,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.benchmark.policies:ashrae_seasonal_action",
                requires_training=False,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.PER_TASK,
                description="Repeat the declared winter or summer thermostat pair in every zone.",
            ),
            AlgorithmDescriptor(
                algorithm_id="ppo",
                version="1.0.0",
                kind=AlgorithmKind.EXTERNAL_TRAINER,
                status=AlgorithmStatus.AVAILABLE,
                entrypoint="building_gen.wrapper.ppo_trainer:main",
                requires_training=True,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.EXACT,
                description="External PPO trainer descriptor; resolving it is the launcher's responsibility.",
            ),
            AlgorithmDescriptor(
                algorithm_id="sac",
                version="planned-v1",
                kind=AlgorithmKind.EXTERNAL_TRAINER,
                status=AlgorithmStatus.PLANNED,
                entrypoint="building_gen.benchmark.external_trainers:sac",
                requires_training=True,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.EXACT,
                description="Scientific benchmark integration pending; Gym adapter exists in wrapper.rl_baselines.",
            ),
            AlgorithmDescriptor(
                algorithm_id="td3",
                version="planned-v1",
                kind=AlgorithmKind.EXTERNAL_TRAINER,
                status=AlgorithmStatus.PLANNED,
                entrypoint="building_gen.benchmark.external_trainers:td3",
                requires_training=True,
                supported_action_profiles=thermostat,
                transfer_shape_policy=TransferShapePolicy.EXACT,
                description="Scientific benchmark integration pending; Gym adapter exists in wrapper.rl_baselines.",
            ),
        )
    )


__all__ = [
    "AlgorithmDescriptor",
    "AlgorithmKind",
    "AlgorithmRegistry",
    "AlgorithmStatus",
    "BenchmarkCompatibilityError",
    "CompatibilityIssue",
    "TransferShapePolicy",
    "assert_compatible",
    "compatibility_issues",
    "default_algorithm_registry",
]
