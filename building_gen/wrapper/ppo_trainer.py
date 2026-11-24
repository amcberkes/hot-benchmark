"""Reproducible PPO training for HOT control tasks.

Production runs construct :func:`create_simulator` directly instead of relying
on Gym's global registry. Tests and alternative simulators can inject an
environment-factory builder without importing EnergyPlus.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import stat
import time
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from building_gen.control.capabilities import ControlCapabilityManifest
from building_gen.reward_foundation import HOT_OCCUPIED_ENERGY_COMFORT_V2
from building_gen.wrapper.manifest_control import (
    MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION,
    ManifestControlRuntime,
    task_action_binding_sha256,
)
from building_gen.wrapper.ppo_energyplus import Agent

CHECKPOINT_SCHEMA_ID = "hot-ppo-checkpoint/1.0.0"
RUN_CONFIG_SCHEMA_ID = "hot-ppo-run-config/1.0.0"
TRAINING_SUMMARY_SCHEMA_ID = "hot-ppo-training-summary/1.0.0"
PPO_UPDATE_METRICS_SCHEMA_ID = "hot-ppo-update-metrics/1.0.0"
MANIFEST_ACTION_EVIDENCE_CHAIN_SCHEMA_ID = "hot-ppo-manifest-action-evidence-chain/1.0.0"
PPO_UPDATE_RECORD_FIELDS = (
    "iteration",
    "global_step",
    "episodes_completed",
    "learning_rate",
    "total_loss",
    "value_loss",
    "policy_loss",
    "entropy",
    "old_approx_kl",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "epochs_processed",
    "minibatches_processed",
)
PPO_UPDATE_AGGREGATION = {
    "minibatch_statistics": "arithmetic mean over all processed minibatches and epochs",
    "explained_variance": "one rollout-level statistic per iteration before the policy update",
}
DEFAULT_OBSERVATION_SCHEMA_ID = "hot_observation_v2"
DEFAULT_ACTION_SCHEMA_ID = "thermostat_v1"
POLICY_DTYPE = torch.float32
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

EnvironmentThunk = Callable[[], gym.Env]
EnvironmentFactoryBuilder = Callable[["PPOArgs", Path, str, int], EnvironmentThunk]
SimulatorFactory = Callable[..., gym.Env]


@dataclass
class PPOArgs:
    """Complete configuration for one PPO training run."""

    exp_name: str = "energyplus_ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = False
    track: bool = False
    wandb_project: str = "hot-control"
    wandb_entity: str | None = None
    wandb_tags: list[str] | None = None
    save_model: bool = True
    results_dir: str = "results"

    # Immutable benchmark/provenance identifiers. Publication runners should
    # provide these; ad-hoc runs may leave them unset.
    run_id: str | None = None
    task_id: str | None = None
    task_fingerprint: str | None = None
    building_checksum: str | None = None
    weather_checksum: str | None = None
    observation_schema_id: str = DEFAULT_OBSERVATION_SCHEMA_ID
    action_schema_id: str = DEFAULT_ACTION_SCHEMA_ID
    control_capability_file: str | None = None
    control_capability_file_sha256: str | None = None
    control_capability_sha256: str | None = None
    control_action_binding_sha256: str | None = None
    prepared_model_sha256: str | None = None

    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    reward_type: str = "energy_comfort"
    energy_weight: float = 1.0
    comfort_weight: float = 1.0
    stability_weight: float = 0.0
    comfort_temp_range: tuple[float, float] = (20.0, 24.0)
    energy_scale_kwh_per_m2: float | None = None
    comfort_scale_person_celsius_squared_hours_per_m2: float | None = None
    num_envs: int = 1
    num_steps: int = 96
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    eval_frequency: int = 100_000
    eval_episodes: int = 3
    run_final_evaluation: bool = True
    checkpoint_every_episodes: int = 1
    episode_length_hours: int = 24
    timesteps_per_hour: int = 4

    building_file: str | None = None
    weather_file: str | None = None
    weather_validation: str | None = None
    building_characteristics: dict[str, Any] | None = None
    building_floor_area: float | None = None
    eplus_output_dir: str | None = None

    # Computed by ``validate_args``.
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    def __post_init__(self) -> None:
        if self.wandb_tags is None:
            self.wandb_tags = []
        if self.building_characteristics is None:
            self.building_characteristics = {}


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    run_dir: Path
    global_step: int
    iterations: int
    episodes_completed: int
    final_checkpoint: Path | None
    evaluation_returns: tuple[float, ...]
    summary_path: Path | None = None
    update_metrics_path: Path | None = None


def _finite_real_argument(name: str, value: Any) -> float:
    """Return a finite real-valued argument while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _exact_sha256(value: Any, *, field_name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field_name} must be an exact lowercase SHA-256 digest")
    return value


def _stable_regular_file_bytes(path_value: str, *, label: str) -> bytes:
    """Own one unchanged single-link regular file without following symlinks."""

    if type(path_value) is not str or not path_value:
        raise ValueError(f"{label} path must be a nonempty exact string")
    path = Path(path_value)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label} as a regular non-symlink file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)

        def identity(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_uid,
                item.st_gid,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
                item.st_nlink,
            )

        if identity(before) != identity(after):
            raise ValueError(f"{label} changed while it was being read")
        path_after = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(path_after.st_mode) or identity(path_after) != identity(after):
            raise ValueError(f"{label} path changed while it was being read")
        contents = b"".join(chunks)
        if len(contents) != after.st_size:
            raise ValueError(f"{label} size changed while it was being read")
        return contents
    finally:
        os.close(descriptor)


def _load_control_manifest(args: PPOArgs) -> tuple[ControlCapabilityManifest, bytes] | None:
    fields = (
        "control_capability_file",
        "control_capability_file_sha256",
        "control_capability_sha256",
        "control_action_binding_sha256",
        "prepared_model_sha256",
    )
    supplied = tuple(getattr(args, name) is not None for name in fields)
    if not any(supplied):
        if type(args.action_schema_id) is not str or args.action_schema_id != DEFAULT_ACTION_SCHEMA_ID:
            raise ValueError(
                "PPO without a control capability manifest requires " f"action_schema_id={DEFAULT_ACTION_SCHEMA_ID!r}"
            )
        return None
    if not all(supplied):
        missing = [name for name in fields if getattr(args, name) is None]
        raise ValueError(f"manifest PPO control identity is all-or-none; missing {missing!r}")
    assert args.control_capability_file is not None
    contents = _stable_regular_file_bytes(args.control_capability_file, label="control capability file")
    expected_file_sha = _exact_sha256(
        args.control_capability_file_sha256,
        field_name="control_capability_file_sha256",
    )
    measured_file_sha = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(measured_file_sha, expected_file_sha):
        raise ValueError(
            "control capability file SHA-256 mismatch: " f"expected {expected_file_sha}, found {measured_file_sha}"
        )
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("control capability file is not UTF-8") from error
    manifest = ControlCapabilityManifest.from_json(text)
    if manifest.to_json().encode("utf-8") != contents:
        raise ValueError("control capability file is not the sole canonical manifest encoding")
    expected_capability = _exact_sha256(
        args.control_capability_sha256,
        field_name="control_capability_sha256",
    )
    if not hmac.compare_digest(manifest.capability_sha256, expected_capability):
        raise ValueError("control capability semantic SHA-256 does not match the PPO identity")
    expected_binding = _exact_sha256(
        args.control_action_binding_sha256,
        field_name="control_action_binding_sha256",
    )
    measured_binding = task_action_binding_sha256(manifest)
    if not hmac.compare_digest(measured_binding, expected_binding):
        raise ValueError("control action-binding SHA-256 does not match the PPO identity")
    expected_model = _exact_sha256(args.prepared_model_sha256, field_name="prepared_model_sha256")
    if not hmac.compare_digest(manifest.prepared_model_sha256, expected_model):
        raise ValueError("prepared-model SHA-256 does not match the PPO control identity")
    if args.building_checksum != expected_model:
        raise ValueError("building_checksum must equal the manifest prepared-model SHA-256")
    if type(args.action_schema_id) is not str or args.action_schema_id != MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION:
        raise ValueError("manifest PPO action_schema_id must be " f"{MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION!r}")
    return manifest, contents


def _control_identity(args: PPOArgs) -> dict[str, str] | None:
    if args.control_capability_file is None:
        return None
    assert args.control_capability_file_sha256 is not None
    assert args.control_capability_sha256 is not None
    assert args.control_action_binding_sha256 is not None
    assert args.prepared_model_sha256 is not None
    return {
        "capability_file_sha256": args.control_capability_file_sha256,
        "capability_sha256": args.control_capability_sha256,
        "action_binding_sha256": args.control_action_binding_sha256,
        "prepared_model_sha256": args.prepared_model_sha256,
    }


def _run_identity(args: PPOArgs) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "run_id": args.run_id,
        "task_id": args.task_id,
        "task_fingerprint": args.task_fingerprint,
        "building_checksum": args.building_checksum,
        "weather_checksum": args.weather_checksum,
    }
    control_identity = _control_identity(args)
    if control_identity is not None:
        identity["manifest_control"] = control_identity
    return identity


def validate_args(args: PPOArgs) -> PPOArgs:
    """Validate rollout dimensions before allocating tensors."""

    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {args.seed!r}")
    for name in (
        "total_timesteps",
        "num_envs",
        "num_steps",
        "num_minibatches",
        "update_epochs",
        "episode_length_hours",
        "timesteps_per_hour",
        "eval_episodes",
        "checkpoint_every_episodes",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(args.eval_frequency, bool) or not isinstance(args.eval_frequency, int) or args.eval_frequency < 0:
        raise ValueError("eval_frequency must be a non-negative integer; zero disables periodic evaluation")
    if not isinstance(args.run_final_evaluation, bool):
        raise ValueError("run_final_evaluation must be a boolean")

    args.batch_size = args.num_envs * args.num_steps
    if args.num_minibatches > args.batch_size:
        raise ValueError(
            f"num_minibatches ({args.num_minibatches}) cannot exceed rollout batch size ({args.batch_size})"
        )
    if args.batch_size % args.num_minibatches:
        raise ValueError(
            "num_envs * num_steps must be divisible by num_minibatches: "
            f"{args.batch_size} % {args.num_minibatches} != 0"
        )
    if args.total_timesteps % args.batch_size:
        raise ValueError(
            "total_timesteps must be divisible by num_envs * num_steps so the budget is exact: "
            f"{args.total_timesteps} % {args.batch_size} != 0"
        )
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    learning_rate = _finite_real_argument("learning_rate", args.learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    gamma = _finite_real_argument("gamma", args.gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    gae_lambda = _finite_real_argument("gae_lambda", args.gae_lambda)
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1]")
    clip_coef = _finite_real_argument("clip_coef", args.clip_coef)
    if not 0.0 < clip_coef < 1.0:
        raise ValueError("clip_coef must lie strictly between 0 and 1")
    for name in ("ent_coef", "vf_coef"):
        if _finite_real_argument(name, getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if _finite_real_argument("max_grad_norm", args.max_grad_norm) <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if args.target_kl is not None and _finite_real_argument("target_kl", args.target_kl) <= 0.0:
        raise ValueError("target_kl must be positive when provided")

    for name in ("energy_weight", "comfort_weight", "stability_weight"):
        if _finite_real_argument(name, getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if not isinstance(args.comfort_temp_range, (tuple, list)) or len(args.comfort_temp_range) != 2:
        raise ValueError("comfort_temp_range must contain exactly two finite real numbers")
    comfort_lower = _finite_real_argument("comfort_temp_range lower bound", args.comfort_temp_range[0])
    comfort_upper = _finite_real_argument("comfort_temp_range upper bound", args.comfort_temp_range[1])
    if comfort_lower >= comfort_upper:
        raise ValueError("comfort_temp_range lower bound must be smaller than upper bound")
    for name in (
        "energy_scale_kwh_per_m2",
        "comfort_scale_person_celsius_squared_hours_per_m2",
    ):
        value = getattr(args, name)
        if value is None:
            if args.reward_type == HOT_OCCUPIED_ENERGY_COMFORT_V2:
                raise ValueError(f"{name} must be positive and finite for reward v2")
            continue
        if _finite_real_argument(name, value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if args.building_floor_area is not None:
        if _finite_real_argument("building_floor_area", args.building_floor_area) <= 0.0:
            raise ValueError("building_floor_area must be positive when provided")
    for name in ("run_id", "task_id"):
        value = getattr(args, name)
        if value is not None and not _SAFE_ID.fullmatch(value):
            raise ValueError(f"{name} contains characters unsafe for artifact paths: {value!r}")
    _load_control_manifest(args)
    return args


def _resolved_run_id(args: PPOArgs) -> str:
    if args.run_id:
        return args.run_id
    safe_exp = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.exp_name).strip("-.") or "ppo"
    return f"{safe_exp}-seed{args.seed}-{int(time.time())}"


def _require_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all().item())
    else:
        finite = bool(np.isfinite(np.asarray(value)).all())
    if not finite:
        raise FloatingPointError(f"{name} contains NaN or infinite values")


def _require_unit_interval(name: str, value: Any, *, tolerance: float = 1e-7) -> None:
    if isinstance(value, torch.Tensor):
        minimum = float(value.detach().min().item())
        maximum = float(value.detach().max().item())
    else:
        array = np.asarray(value)
        minimum = float(array.min())
        maximum = float(array.max())
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(f"{name} must lie in [0,1], observed range [{minimum}, {maximum}]")


def _require_finite_parameters(agent: Agent) -> None:
    for name, parameter in agent.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"policy parameter {name!r} became non-finite")


def _space_metadata(space: gym.Space, schema_id: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_id": schema_id,
        "type": type(space).__name__,
        "shape": list(space.shape or ()),
        "dtype": str(getattr(space, "dtype", "unknown")),
    }
    if isinstance(space, gym.spaces.Box):
        metadata["low"] = np.asarray(space.low).tolist()
        metadata["high"] = np.asarray(space.high).tolist()
    return metadata


def _validate_checkpoint_space(
    *,
    label: str,
    saved: dict[str, Any],
    expected_space: gym.Space,
    expected_schema_id: str,
) -> None:
    expected = _space_metadata(expected_space, expected_schema_id)
    for key in ("schema_id", "type", "shape", "dtype"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"checkpoint {label} {key} mismatch: saved {saved.get(key)!r}, expected {expected[key]!r}")
    if isinstance(expected_space, gym.spaces.Box):
        for key in ("low", "high"):
            saved_bound = np.asarray(saved.get(key))
            expected_bound = np.asarray(expected[key])
            if saved_bound.shape != expected_bound.shape or not np.allclose(
                saved_bound, expected_bound, rtol=0.0, atol=0.0
            ):
                raise ValueError(f"checkpoint {label} {key} bounds do not match the target environment")


def make_env(
    args: PPOArgs,
    output_dir: str | Path,
    *,
    weather_file: str | None = None,
    simulator_factory: SimulatorFactory | None = None,
) -> EnvironmentThunk:
    """Build one simulator stream with immutable per-episode output paths."""

    output_dir = Path(output_dir)

    def build_episode(episode_index: int) -> gym.Env:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer")
        episode_output_dir = output_dir / f"episode-{episode_index:06d}"
        episode_output_dir.parent.mkdir(parents=True, exist_ok=True)
        episode_output_dir.mkdir(mode=0o700, exist_ok=False)
        factory = simulator_factory
        if factory is None:
            from building_gen.wrapper.create_simulator import create_simulator

            factory = create_simulator
        characteristics = dict(args.building_characteristics or {})
        if args.building_floor_area is not None:
            characteristics["floor_area_m2"] = args.building_floor_area
        manifest_binding = _load_control_manifest(args)
        control_manifest = manifest_binding[0] if manifest_binding is not None else None
        factory_kwargs = dict(
            path_to_building=args.building_file,
            path_to_weather=weather_file or args.weather_file,
            building_characteristics=characteristics,
            reward_type=args.reward_type,
            energy_weight=args.energy_weight,
            comfort_weight=args.comfort_weight,
            stability_weight=args.stability_weight,
            comfort_temp_range=args.comfort_temp_range,
            energy_scale_kwh_per_m2=args.energy_scale_kwh_per_m2,
            comfort_scale_person_celsius_squared_hours_per_m2=(args.comfort_scale_person_celsius_squared_hours_per_m2),
            episode_length_hours=args.episode_length_hours,
            timesteps_per_hour=args.timesteps_per_hour,
            eplus_output_dir=str(episode_output_dir),
        )
        if control_manifest is not None:
            factory_kwargs["control_manifest"] = control_manifest
        env = factory(**factory_kwargs)
        # HOT now emits the versioned [0,1] observation and action encodings.
        # Re-normalizing or clipping would change the policy's declared schema.
        return gym.wrappers.RecordEpisodeStatistics(env)

    def thunk() -> gym.Env:
        return _EpisodeOutputIsolationWrapper(build_episode)

    return thunk


class _EpisodeOutputIsolationWrapper(gym.Wrapper):
    """Own one simulator per episode without reusing its output directory."""

    def __init__(self, episode_factory: Callable[[int], gym.Env]) -> None:
        self._episode_factory = episode_factory
        self._episode_index = 0
        self._has_reset = False
        self._closed = False
        super().__init__(episode_factory(self._episode_index))

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("cannot reset a closed episode-output wrapper")
        if self._has_reset:
            expected_observation_space = self.observation_space
            expected_action_space = self.action_space
            self.env.close()
            replacement = self._episode_factory(self._episode_index + 1)
            if replacement.observation_space != expected_observation_space:
                replacement.close()
                raise ValueError("replacement episode changed the observation space")
            if replacement.action_space != expected_action_space:
                replacement.close()
                raise ValueError("replacement episode changed the action space")
            self._episode_index += 1
            self.env = replacement
        observation, info = self.env.reset(**kwargs)
        self._has_reset = True
        return observation, info

    def close(self) -> None:
        if self._closed:
            return
        self.env.close()
        self._closed = True


def energyplus_env_factory_builder(args: PPOArgs, run_dir: Path, phase: str, env_index: int) -> EnvironmentThunk:
    """Create a uniquely isolated EnergyPlus environment thunk."""

    root = Path(args.eplus_output_dir) if args.eplus_output_dir else run_dir / "eplus_outputs"
    weather = args.weather_validation if phase.startswith("eval") and args.weather_validation else args.weather_file
    return make_env(args, root / phase / f"env-{env_index:03d}", weather_file=weather)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _checkpoint_payload(
    *,
    args: PPOArgs,
    agent: Agent,
    optimizer: optim.Optimizer,
    global_step: int,
    iteration: int,
    episodes_completed: int,
    observation_space: gym.Space,
    action_space: gym.Space,
    manifest_action_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_id": CHECKPOINT_SCHEMA_ID,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "iteration": iteration,
        "episodes_completed": episodes_completed,
        "rng_state": _rng_state(),
        "config": asdict(args),
        "identity": _run_identity(args),
        "observation_space": _space_metadata(observation_space, args.observation_schema_id),
        "action_space": _space_metadata(action_space, args.action_schema_id),
    }
    if manifest_action_evidence is not None:
        payload["manifest_action_evidence"] = manifest_action_evidence
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically persist a structured PPO checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    agent: Agent,
    optimizer: optim.Optimizer | None = None,
    *,
    expected_observation_space: gym.Space,
    expected_action_space: gym.Space,
    observation_schema_id: str = DEFAULT_OBSERVATION_SCHEMA_ID,
    action_schema_id: str = DEFAULT_ACTION_SCHEMA_ID,
    expected_control_capability_file_sha256: str | None = None,
    expected_control_capability_sha256: str | None = None,
    expected_control_action_binding_sha256: str | None = None,
    expected_prepared_model_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load a structured checkpoint and optionally restore its optimizer."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_id") != CHECKPOINT_SCHEMA_ID:
        raise ValueError(f"unsupported PPO checkpoint schema: {payload.get('schema_id')!r}")
    expected_control_fields = (
        expected_control_capability_file_sha256,
        expected_control_capability_sha256,
        expected_control_action_binding_sha256,
        expected_prepared_model_sha256,
    )
    if any(value is not None for value in expected_control_fields) and not all(
        value is not None for value in expected_control_fields
    ):
        raise ValueError("expected checkpoint manifest-control identity must be all-or-none")
    expected_control_identity = None
    if all(value is not None for value in expected_control_fields):
        expected_control_identity = {
            "capability_file_sha256": _exact_sha256(
                expected_control_capability_file_sha256,
                field_name="expected_control_capability_file_sha256",
            ),
            "capability_sha256": _exact_sha256(
                expected_control_capability_sha256,
                field_name="expected_control_capability_sha256",
            ),
            "action_binding_sha256": _exact_sha256(
                expected_control_action_binding_sha256,
                field_name="expected_control_action_binding_sha256",
            ),
            "prepared_model_sha256": _exact_sha256(
                expected_prepared_model_sha256,
                field_name="expected_prepared_model_sha256",
            ),
        }
    saved_identity = payload.get("identity")
    if not isinstance(saved_identity, dict):
        raise ValueError("checkpoint identity is missing or malformed")
    if saved_identity.get("manifest_control") != expected_control_identity:
        raise ValueError("checkpoint manifest-control identity does not match the target environment")
    if expected_control_identity is not None:
        _validate_checkpoint_manifest_evidence(
            payload.get("manifest_action_evidence"),
            global_step=payload.get("global_step"),
            capability_sha256=expected_control_identity["capability_sha256"],
            action_binding_sha256=expected_control_identity["action_binding_sha256"],
        )
    _validate_checkpoint_space(
        label="observation space",
        saved=payload.get("observation_space", {}),
        expected_space=expected_observation_space,
        expected_schema_id=observation_schema_id,
    )
    _validate_checkpoint_space(
        label="action space",
        saved=payload.get("action_space", {}),
        expected_space=expected_action_space,
        expected_schema_id=action_schema_id,
    )
    agent.load_state_dict(payload["agent_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        state = payload["rng_state"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    return payload


def _validate_checkpoint_manifest_evidence(
    value: Any,
    *,
    global_step: Any,
    capability_sha256: str,
    action_binding_sha256: str,
) -> None:
    expected_fields = {
        "schema_id",
        "verified_transition_count",
        "evidence_chain_sha256",
        "control_capability_sha256",
        "task_action_binding_sha256",
    }
    if type(value) is not dict or any(type(key) is not str for key in value) or set(value) != expected_fields:
        raise ValueError("checkpoint manifest action-evidence chain is missing or malformed")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("checkpoint global_step must be a non-negative exact integer")
    if type(value["schema_id"]) is not str or value["schema_id"] != MANIFEST_ACTION_EVIDENCE_CHAIN_SCHEMA_ID:
        raise ValueError("checkpoint manifest action-evidence chain has an unknown schema")
    count = value["verified_transition_count"]
    if type(count) is not int or count != global_step:
        raise ValueError("checkpoint manifest action-evidence count does not match global_step")
    _exact_sha256(value["evidence_chain_sha256"], field_name="checkpoint evidence_chain_sha256")
    claimed_capability = _exact_sha256(
        value["control_capability_sha256"],
        field_name="checkpoint evidence control_capability_sha256",
    )
    if not hmac.compare_digest(claimed_capability, capability_sha256):
        raise ValueError("checkpoint manifest action-evidence capability identity does not match")
    claimed_binding = _exact_sha256(
        value["task_action_binding_sha256"],
        field_name="checkpoint evidence task_action_binding_sha256",
    )
    if not hmac.compare_digest(claimed_binding, action_binding_sha256):
        raise ValueError("checkpoint manifest action-evidence binding identity does not match")


def _json_safe(value: Any) -> Any:
    """Convert vector-info values to stable JSON-native values."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _unbatch_vector_info(mapping: dict[str, Any], env_index: int) -> dict[str, Any]:
    """Extract one environment's item from Gymnasium's dictionary-of-arrays info."""

    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if key.startswith("_"):
            continue
        mask = mapping.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask).reshape(-1)[env_index]):
            continue
        if isinstance(value, dict):
            result[key] = _unbatch_vector_info(value, env_index)
        else:
            array = np.asarray(value, dtype=object)
            if array.ndim == 0:
                result[key] = _json_safe(array.item())
            else:
                result[key] = _json_safe(array[env_index])
    return result


def _terminal_info(infos: dict[str, Any], env_index: int) -> dict[str, Any]:
    """Return terminal-step info across Gymnasium vector-info representations."""

    final_info = infos.get("final_info")
    if isinstance(final_info, dict):
        return _unbatch_vector_info(final_info, env_index)
    if final_info is not None:
        array = np.asarray(final_info, dtype=object).reshape(-1)
        if env_index < len(array) and isinstance(array[env_index], dict):
            return _json_safe(array[env_index])
    return _unbatch_vector_info(infos, env_index)


def _validate_manifest_reset_info(runtime: ManifestControlRuntime, info: Any) -> None:
    if type(info) is not dict:
        raise ValueError("manifest PPO reset info must be a mapping")
    schema = info.get("manifest_control_schema_version")
    if type(schema) is not str or schema != MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION:
        raise ValueError("manifest PPO reset info has an unknown control schema")
    capability = _exact_sha256(
        info.get("control_capability_sha256"),
        field_name="manifest PPO reset control_capability_sha256",
    )
    if not hmac.compare_digest(capability, runtime.capability_sha256):
        raise ValueError("manifest PPO reset info has the wrong capability identity")
    binding = _exact_sha256(
        info.get("task_action_binding_sha256"),
        field_name="manifest PPO reset task_action_binding_sha256",
    )
    if not hmac.compare_digest(binding, runtime.action_binding_sha256):
        raise ValueError("manifest PPO reset info has the wrong action-binding identity")
    if info.get("manifest_action_evidence") is not None:
        raise ValueError("manifest PPO reset info unexpectedly contains completed action evidence")


def _validate_manifest_step_info(
    runtime: ManifestControlRuntime,
    action: Any,
    info: Any,
) -> dict[str, Any]:
    if type(info) is not dict:
        raise ValueError("manifest PPO step info must be a mapping")
    schema = info.get("manifest_control_schema_version")
    if type(schema) is not str or schema != MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION:
        raise ValueError("manifest PPO step info has an unknown control schema")
    submitted_schema = info.get("submitted_action_schema_version")
    if type(submitted_schema) is not str or submitted_schema != MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION:
        raise ValueError("manifest PPO step info has an unknown submitted-action schema")
    capability = _exact_sha256(
        info.get("control_capability_sha256"),
        field_name="manifest PPO step control_capability_sha256",
    )
    if not hmac.compare_digest(capability, runtime.capability_sha256):
        raise ValueError("manifest PPO step info has the wrong capability identity")
    binding = _exact_sha256(
        info.get("task_action_binding_sha256"),
        field_name="manifest PPO step task_action_binding_sha256",
    )
    if not hmac.compare_digest(binding, runtime.action_binding_sha256):
        raise ValueError("manifest PPO step info has the wrong action-binding identity")
    evidence = runtime.verify_evidence(info.get("manifest_action_evidence"))
    expected = np.asarray(action, dtype=np.float64)
    observed = np.asarray(
        [record["normalized_value"] for record in evidence["normalized_commands"]],
        dtype=np.float64,
    )
    if expected.shape != (runtime.command_dim,) or not np.array_equal(observed, expected):
        raise ValueError("manifest PPO action evidence differs from the policy action")
    if info.get("submitted_action_physical") != evidence["submitted_endpoints"]:
        raise ValueError("manifest PPO submitted physical actions differ from authenticated evidence")
    if info.get("actuator_readback") != {"available": True, "records": evidence["actuator_readbacks"]}:
        raise ValueError("manifest PPO actuator read-backs differ from authenticated evidence")
    if info.get("actuator_effect_readback") != {"available": True, "records": evidence["effect_readbacks"]}:
        raise ValueError("manifest PPO effect read-backs differ from authenticated evidence")
    return evidence


class _ManifestEvidenceAccumulator:
    """Commit to every verified policy transition in deterministic vector order."""

    _DOMAIN = b"HOT PPO manifest action evidence chain v1\x00"

    def __init__(self, runtime: ManifestControlRuntime | None) -> None:
        self.runtime = runtime
        self.count = 0
        self._state = hashlib.sha256(self._DOMAIN + b"seed\x00").digest()

    def add(self, *, global_step: int, environment_index: int, action: Any, info: Any) -> None:
        if self.runtime is None:
            return
        evidence = _validate_manifest_step_info(self.runtime, action, info)
        payload = json.dumps(
            {
                "environment_index": environment_index,
                "evidence_sha256": evidence["evidence_sha256"],
                "global_step": global_step,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._state = hashlib.sha256(
            self._DOMAIN + b"step\x00" + self.count.to_bytes(8, "big") + self._state + payload
        ).digest()
        self.count += 1

    def summary(self) -> dict[str, Any] | None:
        if self.runtime is None:
            return None
        return {
            "schema_id": MANIFEST_ACTION_EVIDENCE_CHAIN_SCHEMA_ID,
            "verified_transition_count": self.count,
            "evidence_chain_sha256": self._state.hex(),
            "control_capability_sha256": self.runtime.capability_sha256,
            "task_action_binding_sha256": self.runtime.action_binding_sha256,
        }


def _validate_vector_reset_manifest(
    runtime: ManifestControlRuntime | None,
    infos: Any,
    *,
    num_envs: int,
    reset_mask: np.ndarray | None = None,
) -> None:
    if runtime is None:
        return
    if not isinstance(infos, dict):
        raise ValueError("manifest PPO vector reset info must be a mapping")
    if reset_mask is None:
        environment_indices = list(range(num_envs))
    else:
        if reset_mask.shape != (num_envs,) or reset_mask.dtype != np.bool_:
            raise ValueError("manifest PPO reset mask must be a boolean vector matching num_envs")
        environment_indices = [int(index) for index in np.flatnonzero(reset_mask)]
    for environment_index in environment_indices:
        _validate_manifest_reset_info(runtime, _unbatch_vector_info(infos, environment_index))


def _record_vector_manifest_evidence(
    accumulator: _ManifestEvidenceAccumulator,
    *,
    actions: torch.Tensor,
    infos: Any,
    global_step: int,
    num_envs: int,
) -> None:
    if accumulator.runtime is None:
        return
    if not isinstance(infos, dict):
        raise ValueError("manifest PPO vector step info must be a mapping")
    for environment_index in range(num_envs):
        accumulator.add(
            global_step=global_step - num_envs + environment_index + 1,
            environment_index=environment_index,
            action=actions[environment_index].cpu().numpy(),
            info=_unbatch_vector_info(infos, environment_index),
        )


def _validate_ppo_spaces(
    observation_space: gym.Space,
    action_space: gym.Space,
    args: PPOArgs,
    manifest_runtime: ManifestControlRuntime | None,
) -> None:
    if not isinstance(observation_space, gym.spaces.Box):
        raise ValueError("PPO supports only Box observation spaces")
    if not (np.allclose(observation_space.low, 0.0) and np.allclose(observation_space.high, 1.0)):
        raise ValueError(
            f"{args.observation_schema_id} must emit observations in [0,1]; received {observation_space!r}"
        )
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("PPO supports only Box action spaces")
    if not (np.allclose(action_space.low, 0.0) and np.allclose(action_space.high, 1.0)):
        raise ValueError(f"{args.action_schema_id} must accept actions in [0,1]; received {action_space!r}")
    if manifest_runtime is not None and action_space.shape != (manifest_runtime.command_dim,):
        raise ValueError("manifest PPO action-space dimension differs from its authenticated command binding")


class _EpisodeAccounting:
    """Track episodes directly from rewards and done flags.

    Episode completion is an environment transition, not optional metadata.
    Keeping this state in the trainer makes accounting independent of whether
    an injected environment happens to use ``RecordEpisodeStatistics``.
    """

    def __init__(self, num_envs: int) -> None:
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._lengths = np.zeros(num_envs, dtype=np.int64)
        self.records: list[dict[str, Any]] = []

    def step(
        self,
        *,
        rewards: Any,
        terminations: Any,
        truncations: Any,
        infos: dict[str, Any],
        global_step: int,
    ) -> list[dict[str, Any]]:
        rewards_array = np.asarray(rewards, dtype=np.float64).reshape(-1)
        terminations_array = np.asarray(terminations, dtype=bool).reshape(-1)
        truncations_array = np.asarray(truncations, dtype=bool).reshape(-1)
        if not (rewards_array.shape == terminations_array.shape == truncations_array.shape == self._returns.shape):
            raise ValueError("episode-accounting arrays must match num_envs")

        self._returns += rewards_array
        self._lengths += 1
        completed: list[dict[str, Any]] = []
        for env_index in np.flatnonzero(terminations_array | truncations_array):
            terminal_info = _terminal_info(infos, int(env_index))
            record: dict[str, Any] = {
                "episode_index": len(self.records),
                "environment_index": int(env_index),
                "completion_global_step": int(global_step),
                "return": float(self._returns[env_index]),
                "length": int(self._lengths[env_index]),
                "terminated": bool(terminations_array[env_index]),
                "truncated": bool(truncations_array[env_index]),
            }
            calendar = terminal_info.get("training_calendar")
            if isinstance(calendar, dict):
                record["training_calendar"] = _json_safe(calendar)
            self.records.append(record)
            completed.append(record)
            self._returns[env_index] = 0.0
            self._lengths[env_index] = 0
        return completed


def evaluate_agent(
    agent: Agent,
    env_factory: EnvironmentThunk,
    *,
    eval_episodes: int,
    device: torch.device,
    seed: int,
    manifest_runtime: ManifestControlRuntime | None = None,
) -> list[float]:
    """Evaluate deterministically using exactly one owned environment."""

    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    env = env_factory()
    try:
        return _evaluate_in_env(
            agent,
            env,
            eval_episodes=eval_episodes,
            device=device,
            seed=seed,
            manifest_runtime=manifest_runtime,
        )
    finally:
        env.close()


def _evaluate_in_env(
    agent: Agent,
    env: gym.Env,
    *,
    eval_episodes: int,
    device: torch.device,
    seed: int,
    manifest_runtime: ManifestControlRuntime | None = None,
) -> list[float]:
    """Evaluate in a caller-owned environment without creating another one."""

    returns: list[float] = []
    for episode in range(eval_episodes):
        observation, reset_info = env.reset(seed=seed + episode)
        if manifest_runtime is not None:
            _validate_manifest_reset_info(manifest_runtime, reset_info)
        _require_finite("evaluation reset observation", observation)
        _require_unit_interval("evaluation reset observation", observation)
        terminated = truncated = False
        episode_return = 0.0
        while not (terminated or truncated):
            tensor_observation = torch.as_tensor(observation, dtype=POLICY_DTYPE, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(tensor_observation, deterministic=True)
            _require_finite("evaluation action", action)
            _require_unit_interval("evaluation action", action)
            submitted_action = action.squeeze(0).cpu().numpy()
            observation, reward, terminated, truncated, step_info = env.step(submitted_action)
            if manifest_runtime is not None:
                _validate_manifest_step_info(manifest_runtime, submitted_action, step_info)
            _require_finite("evaluation step observation", observation)
            _require_unit_interval("evaluation step observation", observation)
            _require_finite("evaluation reward", reward)
            episode_return += float(reward)
        returns.append(episode_return)
    return returns


def _write_run_config(run_dir: Path, args: PPOArgs) -> None:
    payload = {
        "schema_id": RUN_CONFIG_SCHEMA_ID,
        "identity": _run_identity(args),
        "config": asdict(args),
    }
    destination = run_dir / "run_config.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _write_resolved_control_capability(run_dir: Path, args: PPOArgs) -> Path | None:
    binding = _load_control_manifest(args)
    if binding is None:
        return None
    _, contents = binding
    destination = run_dir / "resolved_control_capability.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"resolved control capability already exists: {destination}")
    temporary = run_dir / f".resolved_control_capability.json.tmp-{os.getpid()}"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            view = memoryview(contents)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while retaining the control capability")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    retained = _stable_regular_file_bytes(str(destination), label="retained control capability")
    if retained != contents:
        raise RuntimeError("retained control capability bytes differ from the authenticated input")
    return destination


def _write_training_summary(
    run_dir: Path,
    args: PPOArgs,
    *,
    global_step: int,
    iterations: int,
    episode_records: list[dict[str, Any]],
    final_checkpoint: Path | None,
    update_metrics_path: Path,
    evaluation_returns: list[float],
    manifest_action_evidence: dict[str, Any] | None = None,
) -> Path:
    """Atomically persist the outcome and episode-level training provenance."""

    training_payload = {
        "requested_transitions": args.total_timesteps,
        "completed_transitions": global_step,
        "iterations": iterations,
        "episodes_completed": len(episode_records),
        "episodes": episode_records,
    }
    if manifest_action_evidence is not None:
        training_payload["manifest_action_evidence"] = manifest_action_evidence
    artifacts = {
        "run_config": "run_config.json",
        "final_checkpoint": (str(final_checkpoint.relative_to(run_dir)) if final_checkpoint is not None else None),
        "ppo_update_metrics": str(update_metrics_path.relative_to(run_dir)),
    }
    if args.control_capability_file is not None:
        artifacts["control_capability"] = "resolved_control_capability.json"
    payload = {
        "schema_id": TRAINING_SUMMARY_SCHEMA_ID,
        "status": "succeeded",
        "identity": _run_identity(args),
        "training": training_payload,
        "evaluation": {
            "episodes": len(evaluation_returns),
            "returns": [float(value) for value in evaluation_returns],
            "mean_return": (float(np.mean(evaluation_returns)) if evaluation_returns else None),
        },
        "artifacts": artifacts,
    }
    destination = run_dir / "training_summary.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _write_ppo_update_metrics(
    run_dir: Path,
    args: PPOArgs,
    records: list[dict[str, int | float]],
) -> Path:
    """Atomically persist one complete, finite metrics record per PPO update."""

    if len(records) != args.num_iterations:
        raise ValueError(
            "PPO update metrics must contain exactly one record per iteration: "
            f"expected {args.num_iterations}, received {len(records)}"
        )
    for expected_iteration, record in enumerate(records, start=1):
        if record.get("iteration") != expected_iteration:
            raise ValueError("PPO update metric iterations must be consecutive and one-indexed")
        _require_finite(f"PPO update metrics iteration {expected_iteration}", list(record.values()))

    payload = {
        "schema_id": PPO_UPDATE_METRICS_SCHEMA_ID,
        "identity": _run_identity(args),
        "aggregation": dict(PPO_UPDATE_AGGREGATION),
        "iterations_requested": args.num_iterations,
        "iterations_recorded": len(records),
        "records": records,
    }
    destination = run_dir / "ppo_update_metrics.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def _update_policy(
    *,
    args: PPOArgs,
    agent: Agent,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
) -> dict[str, int | float]:
    batch_obs = obs.reshape((-1,) + obs.shape[2:])
    batch_actions = actions.reshape((-1,) + actions.shape[2:])
    batch_logprobs = old_logprobs.reshape(-1)
    batch_advantages = advantages.reshape(-1)
    batch_returns = returns.reshape(-1)
    batch_values = old_values.reshape(-1)
    indices = np.arange(args.batch_size)
    minibatch_metrics: dict[str, list[float]] = {
        "total_loss": [],
        "value_loss": [],
        "policy_loss": [],
        "entropy": [],
        "old_approx_kl": [],
        "approx_kl": [],
        "clipfrac": [],
    }
    epochs_processed = 0

    for epoch in range(args.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, args.batch_size, args.minibatch_size):
            minibatch = indices[start : start + args.minibatch_size]
            _, new_logprob, entropy, new_value = agent.get_action_and_value(
                batch_obs[minibatch], batch_actions[minibatch]
            )
            log_ratio = new_logprob - batch_logprobs[minibatch]
            ratio = log_ratio.exp()
            with torch.no_grad():
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
            minibatch_advantages = batch_advantages[minibatch]
            if args.norm_adv and args.minibatch_size > 1:
                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (
                    minibatch_advantages.std() + 1e-8
                )
            policy_loss = torch.maximum(
                -minibatch_advantages * ratio,
                -minibatch_advantages * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef),
            ).mean()
            new_value = new_value.view(-1)
            if args.clip_vloss:
                value_unclipped = (new_value - batch_returns[minibatch]) ** 2
                value_clipped_prediction = batch_values[minibatch] + (new_value - batch_values[minibatch]).clamp(
                    -args.clip_coef, args.clip_coef
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        value_unclipped,
                        (value_clipped_prediction - batch_returns[minibatch]) ** 2,
                    ).mean()
                )
            else:
                value_loss = 0.5 * ((new_value - batch_returns[minibatch]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
            current_metrics = {
                "total_loss": float(loss.item()),
                "value_loss": float(value_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "entropy": float(entropy_loss.item()),
                "old_approx_kl": float(old_approx_kl.item()),
                "approx_kl": float(approx_kl.item()),
                "clipfrac": float(clip_fraction.item()),
            }
            _require_finite("PPO minibatch update metrics", list(current_metrics.values()))
            for name, value in current_metrics.items():
                minibatch_metrics[name].append(value)
            optimizer.zero_grad()
            loss.backward()
            for name, parameter in agent.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"policy gradient {name!r} became non-finite")
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            _require_finite_parameters(agent)
        epochs_processed = epoch + 1
        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    predictions = batch_values.cpu().numpy()
    targets = batch_returns.cpu().numpy()
    target_variance = np.var(targets)
    explained_variance = 0.0 if target_variance == 0 else 1.0 - np.var(targets - predictions) / target_variance
    metrics: dict[str, int | float] = {name: float(np.mean(values)) for name, values in minibatch_metrics.items()}
    metrics.update(
        {
            "explained_variance": float(np.asarray(explained_variance).item()),
            "epochs_processed": epochs_processed,
            "minibatches_processed": len(minibatch_metrics["total_loss"]),
        }
    )
    _require_finite("PPO update metrics", list(metrics.values()))
    return metrics


def main(
    args: PPOArgs,
    *,
    env_factory_builder: EnvironmentFactoryBuilder | None = None,
) -> TrainingResult:
    """Train PPO for exactly ``total_timesteps`` environment transitions."""

    validate_args(args)
    manifest_binding = _load_control_manifest(args)
    manifest_runtime = ManifestControlRuntime(manifest_binding[0]) if manifest_binding is not None else None
    args.run_id = _resolved_run_id(args)
    run_dir = Path(args.results_dir) / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}; choose a new run_id")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_control_capability(run_dir, args)
    _write_run_config(run_dir, args)
    builder = env_factory_builder or energyplus_env_factory_builder

    if args.track:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_id,
                save_code=True,
                tags=args.wandb_tags,
            )
        except ImportError:
            logging.getLogger(__name__).warning("wandb is unavailable; continuing without tracking")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    writer = SummaryWriter(str(run_dir / "logs"))
    logger = logging.getLogger(f"ppo.{args.run_id}")
    logger.setLevel(logging.INFO)
    log_handler = logging.FileHandler(run_dir / "training.log")
    logger.addHandler(log_handler)

    envs: gym.vector.SyncVectorEnv | None = None
    final_checkpoint: Path | None = None
    evaluation_returns: list[float] = []
    update_metric_records: list[dict[str, int | float]] = []
    global_step = 0
    episodes_completed = 0
    episode_accounting = _EpisodeAccounting(args.num_envs)
    manifest_evidence = _ManifestEvidenceAccumulator(manifest_runtime)
    last_checkpoint_bucket = 0
    last_eval_step = -1
    start_time = time.time()

    try:
        environment_thunks = [builder(args, run_dir, "train", index) for index in range(args.num_envs)]
        envs = gym.vector.SyncVectorEnv(
            environment_thunks,
            autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        )
        _validate_ppo_spaces(
            envs.single_observation_space,
            envs.single_action_space,
            args,
            manifest_runtime,
        )
        agent = Agent(envs).to(device=device, dtype=POLICY_DTYPE)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        dtype = POLICY_DTYPE
        obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=dtype, device=device
        )
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=dtype, device=device
        )
        logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=dtype, device=device)
        rewards = torch.zeros((args.num_steps, args.num_envs), dtype=dtype, device=device)
        dones = torch.zeros((args.num_steps, args.num_envs), dtype=dtype, device=device)
        values = torch.zeros((args.num_steps, args.num_envs), dtype=dtype, device=device)

        next_obs_array, reset_infos = envs.reset(seed=args.seed)
        _validate_vector_reset_manifest(manifest_runtime, reset_infos, num_envs=args.num_envs)
        _require_finite("training reset observations", next_obs_array)
        _require_unit_interval("training reset observations", next_obs_array)
        next_obs = torch.as_tensor(next_obs_array, dtype=dtype, device=device)
        next_done = torch.zeros(args.num_envs, dtype=dtype, device=device)
        next_eval_step = args.eval_frequency if args.eval_frequency else None

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = args.learning_rate * (1.0 - (iteration - 1.0) / args.num_iterations)
            new_episode_boundary = False
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                _require_finite("sampled actions", action)
                _require_unit_interval("sampled actions", action)
                _require_finite("sampled action log probabilities", logprob)
                _require_finite("value predictions", value)
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value.flatten()
                next_obs_array, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                _require_finite("training step observations", next_obs_array)
                _require_unit_interval("training step observations", next_obs_array)
                _require_finite("training rewards", reward)
                next_done_array = np.logical_or(terminations, truncations)
                _record_vector_manifest_evidence(
                    manifest_evidence,
                    actions=action,
                    infos=infos,
                    global_step=global_step,
                    num_envs=args.num_envs,
                )
                rewards[step] = torch.as_tensor(reward, dtype=dtype, device=device)
                records = episode_accounting.step(
                    rewards=reward,
                    terminations=terminations,
                    truncations=truncations,
                    infos=infos,
                    global_step=global_step,
                )
                if records:
                    new_episode_boundary = True
                    episodes_completed += len(records)
                    for record in records:
                        writer.add_scalar("charts/episodic_return", record["return"], global_step)
                        writer.add_scalar("charts/episodic_length", record["length"], global_step)
                if np.any(next_done_array) and global_step < args.total_timesteps:
                    next_obs_array, autoreset_infos = envs.reset(
                        options={"reset_mask": next_done_array.copy()},
                    )
                    _validate_vector_reset_manifest(
                        manifest_runtime,
                        autoreset_infos,
                        num_envs=args.num_envs,
                        reset_mask=next_done_array,
                    )
                    _require_finite("training autoreset observations", next_obs_array)
                    _require_unit_interval("training autoreset observations", next_obs_array)
                next_obs = torch.as_tensor(next_obs_array, dtype=dtype, device=device)
                next_done = torch.as_tensor(next_done_array, dtype=dtype, device=device)

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                last_gae: torch.Tensor | float = 0.0
                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[step + 1]
                        next_values = values[step + 1]
                    delta = rewards[step] + args.gamma * next_values * next_nonterminal - values[step]
                    last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                    advantages[step] = last_gae
                returns = advantages + values
                _require_finite("advantages", advantages)
                _require_finite("returns", returns)

            metrics = _update_policy(
                args=args,
                agent=agent,
                optimizer=optimizer,
                obs=obs,
                actions=actions,
                old_logprobs=logprobs,
                advantages=advantages,
                returns=returns,
                old_values=values,
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            update_record: dict[str, int | float] = {
                "iteration": iteration,
                "global_step": global_step,
                "episodes_completed": episodes_completed,
                "learning_rate": learning_rate,
                **metrics,
            }
            _require_finite(f"PPO update record {iteration}", list(update_record.values()))
            update_metric_records.append(update_record)
            writer.add_scalar("charts/learning_rate", learning_rate, global_step)
            writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-9)), global_step)
            for name, value in metrics.items():
                writer.add_scalar(f"losses/{name}", value, global_step)

            checkpoint_bucket = episodes_completed // args.checkpoint_every_episodes
            if args.save_model and new_episode_boundary and checkpoint_bucket > last_checkpoint_bucket:
                payload = _checkpoint_payload(
                    args=args,
                    agent=agent,
                    optimizer=optimizer,
                    global_step=global_step,
                    iteration=iteration,
                    episodes_completed=episodes_completed,
                    observation_space=envs.single_observation_space,
                    action_space=envs.single_action_space,
                    manifest_action_evidence=manifest_evidence.summary(),
                )
                save_checkpoint(
                    run_dir / "checkpoints" / f"episode-{episodes_completed:06d}-step-{global_step:012d}.pt",
                    payload,
                )
                last_checkpoint_bucket = checkpoint_bucket

            if next_eval_step is not None and global_step >= next_eval_step:
                evaluation_returns = evaluate_agent(
                    agent,
                    builder(args, run_dir, f"eval-step-{global_step:012d}", 0),
                    eval_episodes=args.eval_episodes,
                    device=device,
                    seed=args.seed + 1_000_000 + global_step,
                    manifest_runtime=manifest_runtime,
                )
                writer.add_scalar("validation/mean_episodic_return", np.mean(evaluation_returns), global_step)
                last_eval_step = global_step
                while next_eval_step <= global_step:
                    next_eval_step += args.eval_frequency

        if args.save_model:
            final_checkpoint = save_checkpoint(
                run_dir / "checkpoints" / "final.pt",
                _checkpoint_payload(
                    args=args,
                    agent=agent,
                    optimizer=optimizer,
                    global_step=global_step,
                    iteration=args.num_iterations,
                    episodes_completed=episodes_completed,
                    observation_space=envs.single_observation_space,
                    action_space=envs.single_action_space,
                    manifest_action_evidence=manifest_evidence.summary(),
                ),
            )
        if args.run_final_evaluation and last_eval_step != global_step:
            evaluation_returns = evaluate_agent(
                agent,
                builder(args, run_dir, "eval-final", 0),
                eval_episodes=args.eval_episodes,
                device=device,
                seed=args.seed + 2_000_000,
                manifest_runtime=manifest_runtime,
            )
            writer.add_scalar("eval/mean_episodic_return", np.mean(evaluation_returns), global_step)
        update_metrics_path = _write_ppo_update_metrics(run_dir, args, update_metric_records)
        summary_path = _write_training_summary(
            run_dir,
            args,
            global_step=global_step,
            iterations=args.num_iterations,
            episode_records=episode_accounting.records,
            final_checkpoint=final_checkpoint,
            update_metrics_path=update_metrics_path,
            evaluation_returns=evaluation_returns,
            manifest_action_evidence=manifest_evidence.summary(),
        )
        return TrainingResult(
            run_id=args.run_id,
            run_dir=run_dir,
            global_step=global_step,
            iterations=args.num_iterations,
            episodes_completed=episodes_completed,
            final_checkpoint=final_checkpoint,
            evaluation_returns=tuple(evaluation_returns),
            summary_path=summary_path,
            update_metrics_path=update_metrics_path,
        )
    finally:
        if envs is not None:
            envs.close()
        writer.close()
        logger.removeHandler(log_handler)
        log_handler.close()
        if args.track:
            try:
                import wandb

                wandb.finish()
            except ImportError:
                pass


def ppo_evaluate(
    *,
    model_path: str | Path,
    env_factory: EnvironmentThunk,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    control_manifest: ControlCapabilityManifest | None = None,
) -> list[float]:
    """Load a structured checkpoint and evaluate it in one environment."""

    env = env_factory()
    try:
        spaces = SimpleNamespace(
            single_observation_space=env.observation_space,
            single_action_space=env.action_space,
        )
        agent = Agent(spaces).to(device=device, dtype=POLICY_DTYPE)
        manifest_runtime = ManifestControlRuntime(control_manifest) if control_manifest is not None else None
        capability_file_sha256 = (
            hashlib.sha256(manifest_runtime.manifest.to_json().encode("utf-8")).hexdigest()
            if manifest_runtime is not None
            else None
        )
        load_checkpoint(
            model_path,
            agent,
            expected_observation_space=env.observation_space,
            expected_action_space=env.action_space,
            action_schema_id=(
                MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION if manifest_runtime is not None else DEFAULT_ACTION_SCHEMA_ID
            ),
            expected_control_capability_file_sha256=capability_file_sha256,
            expected_control_capability_sha256=(
                manifest_runtime.capability_sha256 if manifest_runtime is not None else None
            ),
            expected_control_action_binding_sha256=(
                manifest_runtime.action_binding_sha256 if manifest_runtime is not None else None
            ),
            expected_prepared_model_sha256=(
                manifest_runtime.manifest.prepared_model_sha256 if manifest_runtime is not None else None
            ),
            map_location=device,
        )
        return _evaluate_in_env(
            agent,
            env,
            eval_episodes=eval_episodes,
            device=device,
            seed=seed,
            manifest_runtime=manifest_runtime,
        )
    finally:
        env.close()


__all__ = [
    "CHECKPOINT_SCHEMA_ID",
    "RUN_CONFIG_SCHEMA_ID",
    "PPO_UPDATE_AGGREGATION",
    "PPO_UPDATE_RECORD_FIELDS",
    "PPO_UPDATE_METRICS_SCHEMA_ID",
    "TRAINING_SUMMARY_SCHEMA_ID",
    "PPOArgs",
    "TrainingResult",
    "energyplus_env_factory_builder",
    "evaluate_agent",
    "load_checkpoint",
    "main",
    "make_env",
    "ppo_evaluate",
    "save_checkpoint",
    "validate_args",
]
