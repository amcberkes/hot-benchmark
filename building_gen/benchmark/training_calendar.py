"""Seeded, calendar-stratified training episodes for HOT control tasks.

EnergyPlus fixes a ``RunPeriod`` when an environment is constructed.  Reusing
one 24-hour environment therefore restarts every PPO episode on the same day.
This module presents a Gym-compatible proxy which constructs a fresh child
environment for each reset and delegates the actual materialization to
``create_benchmark_environment``.

The module imports Gym but not EnergyPlus at import time.  A real simulator is
loaded only when the configured environment builder is called; unit tests can
inject a lightweight fake builder.
"""

from __future__ import annotations

import hashlib
import operator
from calendar import monthrange
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import gymnasium as gym

from building_gen.benchmark.environment import (
    create_benchmark_environment,
    validate_environment,
)
from building_gen.benchmark.specs import CalendarWindowSpec, TaskSpec
from building_gen.benchmark.training_evidence import (
    EpisodeCompletionCallback,
    ExpectedTrainingEpisode,
    TrainingEpisodeCompletion,
    TrainingEpisodeKey,
)

CALENDAR_TRAINING_SCHEMA_VERSION = "hot-calendar-training/1.0.0"
STRATEGY_ID = "month-stratified-days-without-replacement-v1"
_MONTHS_PER_BLOCK = 12


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _digest(seed: int, namespace: str, *parts: int) -> bytes:
    payload = ":".join([CALENDAR_TRAINING_SCHEMA_VERSION, str(seed), namespace, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("ascii")).digest()


def _permutation(values: range, *, seed: int, namespace: str, parts: tuple[int, ...]) -> tuple[int, ...]:
    """Return a stable hash-keyed permutation without mutable RNG state."""

    return tuple(
        sorted(
            values,
            key=lambda value: (_digest(seed, namespace, *parts, value), value),
        )
    )


@dataclass(frozen=True, slots=True)
class CalendarTrainingSample:
    """One deterministic training window selected by the schedule."""

    schedule_seed: int
    episode_index: int
    calendar_block: int
    month_visit_index: int
    day_cycle: int
    child_seed: int
    window: CalendarWindowSpec
    strategy_id: str = STRATEGY_ID
    pool_sha256: str | None = None

    def evidence_expectation(
        self,
        *,
        stream_id: str,
        generation: int = 0,
    ) -> ExpectedTrainingEpisode:
        """Return the exact scientific evidence contract for this sample."""

        return ExpectedTrainingEpisode(
            key=TrainingEpisodeKey(
                stream_id=stream_id,
                generation=generation,
                schedule_seed=self.schedule_seed,
                episode_index=self.episode_index,
            ),
            schedule_schema_version=CALENDAR_TRAINING_SCHEMA_VERSION,
            strategy_id=self.strategy_id,
            child_seed=self.child_seed,
            window=self.window,
        )

    def to_dict(self, *, calendar_year: int | None = None) -> dict[str, Any]:
        """Return JSON-safe episode provenance suitable for Gym ``info``."""

        start = date(calendar_year or 2001, self.window.start_month, self.window.start_day)
        end = date(calendar_year or 2001, self.window.end_month, self.window.end_day)
        if calendar_year is None:
            start_label = f"{start.month:02d}-{start.day:02d}"
            end_label = f"{end.month:02d}-{end.day:02d}"
        else:
            start_label = f"{calendar_year:04d}-{start.month:02d}-{start.day:02d}"
            end_label = f"{calendar_year:04d}-{end.month:02d}-{end.day:02d}"
        return {
            "schema_version": CALENDAR_TRAINING_SCHEMA_VERSION,
            "strategy_id": self.strategy_id,
            **({"pool_sha256": self.pool_sha256} if self.pool_sha256 is not None else {}),
            "schedule_seed": self.schedule_seed,
            "episode_index": self.episode_index,
            "calendar_block": self.calendar_block,
            "month_visit_index": self.month_visit_index,
            "day_cycle": self.day_cycle,
            "child_seed": self.child_seed,
            "calendar_year": calendar_year,
            "start_date": start_label,
            "end_date": end_label,
            "window": {
                "name": self.window.name,
                "start_month": self.window.start_month,
                "start_day": self.window.start_day,
                "end_month": self.window.end_month,
                "end_day": self.window.end_day,
                "duration_hours": self.window.duration_hours_for_calendar_year(calendar_year),
            },
        }


class StratifiedCalendarSchedule:
    """Random-access, seed-deterministic schedule of calendar windows.

    Every consecutive block of twelve episodes contains one episode from each
    month in a seed-dependent order.  Within a month, valid start days are
    visited in a seed-dependent permutation without replacement before any day
    is reused.  The schedule is therefore reproducible while avoiding a fixed
    January-1 training distribution.
    """

    def __init__(self, *, seed: int, episode_length_hours: int) -> None:
        self.seed = _nonnegative_integer(seed, name="seed")
        self.episode_length_hours = _positive_integer(episode_length_hours, name="episode_length_hours")
        if self.episode_length_hours % 24:
            raise ValueError("calendar training currently requires episode_length_hours to be a multiple of 24")
        self.duration_days = self.episode_length_hours // 24
        # Requiring each sampled window to stay inside its month makes the
        # monthly strata unambiguous.  February is the shortest canonical month.
        if self.duration_days > 28:
            raise ValueError("calendar-stratified episodes cannot be longer than 28 days")

    def sample(self, episode_index: int) -> CalendarTrainingSample:
        """Select an episode without depending on calls made before it."""

        episode_index = _nonnegative_integer(episode_index, name="episode_index")
        block, position = divmod(episode_index, _MONTHS_PER_BLOCK)
        month_order = _permutation(
            range(1, 13),
            seed=self.seed,
            namespace="month-order",
            parts=(block,),
        )
        month = month_order[position]

        # Each month occurs exactly once per block, so ``block`` is also the
        # zero-based visit count for the selected month.
        month_visit = block
        latest_start = monthrange(2001, month)[1] - self.duration_days + 1
        day_cycle, day_position = divmod(month_visit, latest_start)
        day_order = _permutation(
            range(1, latest_start + 1),
            seed=self.seed,
            namespace="day-order",
            parts=(month, day_cycle),
        )
        start_day = day_order[day_position]
        start = date(2001, month, start_day)
        end = start + timedelta(days=self.duration_days - 1)
        window = CalendarWindowSpec(
            name=(f"train-e{episode_index:08d}-" f"m{start.month:02d}d{start.day:02d}-{self.duration_days}d"),
            start_month=start.month,
            start_day=start.day,
            end_month=end.month,
            end_day=end.day,
        )
        child_seed = int.from_bytes(_digest(self.seed, "child-seed", episode_index), "big") % (2**32)
        return CalendarTrainingSample(
            schedule_seed=self.seed,
            episode_index=episode_index,
            calendar_block=block,
            month_visit_index=month_visit,
            day_cycle=day_cycle,
            child_seed=child_seed,
            window=window,
        )


def expected_calendar_training_episodes(
    task: TaskSpec,
    *,
    seed: int,
    total_timesteps: int,
    num_envs: int,
) -> tuple[ExpectedTrainingEpisode, ...]:
    """Preregister the exact complete episodes implied by a training budget."""

    seed = _nonnegative_integer(seed, name="seed")
    total_timesteps = _positive_integer(total_timesteps, name="total_timesteps")
    num_envs = _positive_integer(num_envs, name="num_envs")
    if total_timesteps % num_envs:
        raise ValueError("total_timesteps must be divisible by num_envs")
    transitions_per_environment = total_timesteps // num_envs
    if transitions_per_environment % task.episode.max_steps:
        raise ValueError(
            "scientific calendar training requires a complete-episode budget per environment: "
            f"{transitions_per_environment} transitions is not divisible by "
            f"{task.episode.max_steps} steps per episode"
        )
    episodes_per_environment = transitions_per_environment // task.episode.max_steps
    expected: list[ExpectedTrainingEpisode] = []
    for env_index in range(num_envs):
        schedule = StratifiedCalendarSchedule(
            seed=seed + env_index,
            episode_length_hours=task.episode.episode_length_hours,
        )
        stream_id = f"train-env-{env_index:03d}"
        expected.extend(
            schedule.sample(episode_index).evidence_expectation(stream_id=stream_id)
            for episode_index in range(episodes_per_environment)
        )
    return tuple(expected)


EnvironmentBuilder = Callable[..., Any]
SimulatorFactory = Callable[..., Any]


class CalendarTrainingEnvironment(gym.Env):
    """Gym-compatible proxy that runs each episode on a sampled calendar day.

    A child is prepared during construction so vector-environment libraries can
    inspect its spaces before the first reset.  The first reset uses that child;
    every later reset closes the old child before constructing the next one.
    """

    def __init__(
        self,
        task: TaskSpec,
        *,
        data_root: str | Path,
        work_dir: str | Path,
        seed: int,
        start_episode_index: int = 0,
        environment_builder: EnvironmentBuilder = create_benchmark_environment,
        simulator_factory: SimulatorFactory | None = None,
        stream_id: str = "default",
        episode_completion_callback: EpisodeCompletionCallback | None = None,
        max_episodes: int | None = None,
        reserved_windows: tuple[CalendarWindowSpec, ...] | None = None,
    ) -> None:
        super().__init__()
        if task.episode.training_end_behavior == "truncate" and episode_completion_callback is not None:
            raise ValueError("continuing-task truncation is not qualified for the scientific completion recorder")
        self.task = task
        self.data_root = Path(data_root).expanduser().resolve()
        self.work_dir = Path(work_dir).expanduser().resolve()
        self._weekly_reward = None
        if task.weekly_training_reward is not None:
            if episode_completion_callback is not None:
                raise ValueError('weekly reward normalization is not qualified for the scientific recorder')
            from building_gen.weekly_reward import WeeklyTrainingReward
            self._weekly_reward = WeeklyTrainingReward(task, self.data_root)
        self._start_episode_index = _nonnegative_integer(start_episode_index, name="start_episode_index")
        self._max_episodes = None if max_episodes is None else _positive_integer(max_episodes, name="max_episodes")
        self._episode_limit_exclusive = (
            None if self._max_episodes is None else self._start_episode_index + self._max_episodes
        )
        self._schedule = StratifiedCalendarSchedule(
            seed=seed,
            episode_length_hours=task.episode.episode_length_hours,
        )
        self._sample_at = self._schedule.sample
        self._partition_identity = None
        if reserved_windows is not None:
            from building_gen.benchmark.training_partitions import PartitionedCalendarSchedule
            self._schedule = PartitionedCalendarSchedule(seed=seed,
                episode_length_hours=task.episode.episode_length_hours,
                reserved_windows=reserved_windows, calendar_year=task.calendar_year)
            self._sample_at = self._schedule.training_sample
            self._partition_identity = self._schedule.identity_payload()
        self._environment_builder = environment_builder
        self._simulator_factory = simulator_factory
        # Validate the public stream identifier through the evidence key even
        # when no callback is installed.  A scientific recorder can therefore
        # rely on the same identity contract for every emitted event.
        TrainingEpisodeKey(
            stream_id=stream_id,
            generation=0,
            schedule_seed=self._schedule.seed,
            episode_index=self._start_episode_index,
        )
        if episode_completion_callback is not None and not callable(episode_completion_callback):
            raise TypeError("episode_completion_callback must be callable")
        self.stream_id = stream_id
        self._episode_completion_callback = episode_completion_callback
        self._generation = 0
        self._has_reset = False
        self._closed = False
        self._history: list[dict[str, Any]] = []
        self._episode_reset_succeeded = False
        self._episode_step_count = 0
        self._episode_terminated = False
        self._episode_truncated = False
        self._episode_completion_emitted = False
        self._exhausted = False
        self._last_observation: Any | None = None
        self._sample = self._sample_at(self._start_episode_index)
        self._child = self._build_child(self._sample)

        # Vectorized Gym implementations inspect these before calling reset.
        self.observation_space = self._child.observation_space
        self.action_space = self._child.action_space
        self.metadata = dict(getattr(self._child, "metadata", {}))
        self.render_mode = getattr(self._child, "render_mode", None)
        self.controlled_zones = list(task.action_profile.controlled_entities)
        self.observation_names = getattr(self._child, "observation_names", None)
        self.observation_encoding_id = getattr(
            self._child,
            "observation_encoding_id",
            None,
        )
        self.action_names = getattr(self._child, "action_names", None)
        self.action_endpoint_mapping = getattr(
            self._child,
            "action_endpoint_mapping",
            None,
        )
        self.action_transform_id = getattr(
            self._child,
            "action_transform_id",
            None,
        )

    @property
    def unwrapped(self) -> "CalendarTrainingEnvironment":
        """Match Gym's access convention while retaining calendar metadata."""

        return self

    @property
    def current_sample(self) -> CalendarTrainingSample:
        return self._sample

    @property
    def episode_provenance(self) -> dict[str, Any]:
        """Describe the currently prepared or active calendar episode."""

        provenance = self._sample.to_dict(calendar_year=self.task.calendar_year)
        provenance.update({"task_name": self.task.name, "task_id": self.task.task_id})
        return provenance

    @property
    def calendar_history(self) -> tuple[dict[str, Any], ...]:
        """Return provenance for episodes whose child reset succeeded."""

        return tuple(dict(item) for item in self._history)

    def _episode_work_dir(self, sample: CalendarTrainingSample) -> Path:
        return (
            self.work_dir
            / f"{self.task.name}__calendar-seed-{sample.schedule_seed}"
            / f"generation-{self._generation:04d}"
            / f"episode-{sample.episode_index:08d}__{sample.window.name}"
        )

    def _build_child(self, sample: CalendarTrainingSample) -> Any:
        child = self._environment_builder(
            self.task,
            sample.window,
            data_root=self.data_root,
            work_dir=self._episode_work_dir(sample),
            simulator_factory=self._simulator_factory,
        )
        if child is None:
            raise RuntimeError("environment_builder returned None")
        validate_environment(child, self.task)
        return child

    def _reset_episode_tracking(self) -> None:
        self._episode_reset_succeeded = False
        self._episode_step_count = 0
        self._episode_terminated = False
        self._episode_truncated = False
        self._episode_completion_emitted = False

    def _completion_event(self, child: Any) -> TrainingEpisodeCompletion:
        diagnostics_path = getattr(child, "energyplus_diagnostics_path", None)
        warmup_preconditioning_evidence_path = getattr(
            child,
            "warmup_preconditioning_evidence_path",
            None,
        )
        warmup_preconditioning_contract_payload = getattr(
            child,
            "warmup_preconditioning_contract_payload",
            None,
        )
        warmup_preconditioning_contract_sha256 = getattr(
            child,
            "warmup_preconditioning_contract_sha256",
            None,
        )
        executed_model_sha256 = getattr(
            child,
            "energyplus_executed_model_sha256",
            None,
        )
        return TrainingEpisodeCompletion(
            key=TrainingEpisodeKey(
                stream_id=self.stream_id,
                generation=self._generation,
                schedule_seed=self._sample.schedule_seed,
                episode_index=self._sample.episode_index,
            ),
            schedule_schema_version=CALENDAR_TRAINING_SCHEMA_VERSION,
            strategy_id=self._sample.strategy_id,
            task_name=self.task.name,
            task_id=self.task.task_id,
            building_sha256=self.task.building.sha256,
            weather_sha256=self.task.weather.sha256,
            calendar_year=self.task.calendar_year,
            child_seed=self._sample.child_seed,
            window=self._sample.window,
            episode_work_dir=self._episode_work_dir(self._sample),
            expected_steps=self.task.episode.max_steps,
            step_count=self._episode_step_count,
            terminated=self._episode_terminated,
            truncated=self._episode_truncated,
            diagnostics_path=(None if diagnostics_path is None else Path(diagnostics_path)),
            executed_model_sha256=executed_model_sha256,
            warmup_preconditioning_evidence_path=(
                None if warmup_preconditioning_evidence_path is None else Path(warmup_preconditioning_evidence_path)
            ),
            warmup_preconditioning_contract_payload=(
                None
                if warmup_preconditioning_contract_payload is None
                else dict(warmup_preconditioning_contract_payload)
            ),
            warmup_preconditioning_contract_sha256=(warmup_preconditioning_contract_sha256),
        )

    def _emit_episode_completion(self, child: Any) -> None:
        callback = self._episode_completion_callback
        if (
            callback is None
            or self._episode_completion_emitted
            or not self._episode_reset_succeeded
            or self._episode_step_count == 0
        ):
            return
        # Mark before invoking user code so a failing fail-closed recorder is
        # never retried as a duplicate during final environment cleanup.
        self._episode_completion_emitted = True
        callback(self._completion_event(child))

    def _close_child(self) -> None:
        child = self._child
        self._child = None
        if child is not None:
            try:
                # Terminal callbacks normally run in ``step``.  This fallback
                # surfaces a partially consumed episode when a trainer closes
                # or rotates a child before its natural horizon.
                self._emit_episode_completion(child)
            finally:
                close = getattr(child, "close", None)
                if callable(close):
                    close()

    def _replace_child(self, *, seed: int | None, episode_index: int) -> None:
        self._close_child()
        if seed is not None:
            self._schedule = StratifiedCalendarSchedule(
                seed=seed,
                episode_length_hours=self.task.episode.episode_length_hours,
            )
            self._generation += 1
            self._exhausted = False
        elif self._episode_limit_exclusive is not None and episode_index >= self._episode_limit_exclusive:
            self._exhausted = True
            return
        self._sample = self._sample_at(episode_index)
        self._reset_episode_tracking()
        self._child = self._build_child(self._sample)

    def _annotate_info(self, info: Mapping[str, Any] | None) -> dict[str, Any]:
        if info is None:
            result: dict[str, Any] = {}
        elif isinstance(info, Mapping):
            result = dict(info)
        else:
            raise TypeError("child environment info must be a mapping")
        if "training_calendar" in result:
            raise ValueError("child environment used reserved info key 'training_calendar'")
        result["training_calendar"] = self.episode_provenance
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("cannot reset a closed CalendarTrainingEnvironment")
        if seed is not None:
            seed = _nonnegative_integer(seed, name="seed")
        super().reset(seed=seed)

        if not self._has_reset:
            if seed is not None and seed != self._schedule.seed:
                self._replace_child(seed=seed, episode_index=self._start_episode_index)
        elif seed is None:
            self._replace_child(seed=None, episode_index=self._sample.episode_index + 1)
        else:
            # As in Gym, an explicit seed restarts the random sequence.
            self._replace_child(seed=seed, episode_index=self._start_episode_index)

        if self._child is None:
            if not self._exhausted or self._last_observation is None:
                raise RuntimeError("calendar child environment is unavailable")
            exhausted_info = {
                "training_calendar_exhausted": {
                    "episode_count": self._max_episodes,
                    "last_episode_index": self._sample.episode_index,
                }
            }
            return deepcopy(self._last_observation), self._annotate_info(exhausted_info)
        if self._weekly_reward is not None:
            self._weekly_reward.begin(self._sample.window)
        try:
            observation, info = self._child.reset(
                seed=self._sample.child_seed,
                options=options,
            )
        except Exception:
            self._close_child()
            raise
        self._has_reset = True
        self._episode_reset_succeeded = True
        self._last_observation = deepcopy(observation)
        provenance = self.episode_provenance
        self._history.append(provenance)
        return observation, self._annotate_info(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._closed or self._child is None:
            raise RuntimeError("cannot step a closed CalendarTrainingEnvironment")
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        observation, reward, terminated, truncated, info = self._child.step(action)
        if self._weekly_reward is not None:
            normalized, evidence = self._weekly_reward.score(info, terminated=terminated, truncated=truncated)
            info = dict(info, base_reward=reward, training_reward_components=evidence)
            reward = normalized
        self._last_observation = deepcopy(observation)
        self._episode_step_count += 1
        if self.task.episode.training_end_behavior == "truncate" and (terminated or truncated):
            if (not terminated or truncated
                    or self._episode_step_count != self.task.episode.max_steps
                    or type(info.get("energyplus_exit_code")) is not int
                    or info["energyplus_exit_code"] != 0):
                raise RuntimeError("continuing-task cutoff requires a successful full simulator episode")
            # Keep the actual final observation. SB3's vector wrapper records it
            # as terminal_observation before resetting onto another calendar day.
            terminated, truncated = False, True
            info = dict(info, training_time_limit=True)
        self._episode_terminated = bool(terminated)
        self._episode_truncated = bool(truncated)
        if self._episode_terminated or self._episode_truncated:
            # EnergyPlus' terminal transition waits for its final process exit
            # status, so persisted diagnostics are available at this point.
            self._emit_episode_completion(self._child)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._annotate_info(info),
        )

    def continuation_state(self) -> dict[str, Any]:
        """Describe the next fresh episode, rejecting unfinished simulator state."""
        if self._closed or self._exhausted:
            raise ValueError("cannot checkpoint a closed or exhausted calendar environment")
        if self._episode_step_count == 0:
            next_index = self._sample.episode_index
        elif (((self._episode_terminated and not self._episode_truncated)
               or (self.task.episode.training_end_behavior == "truncate"
                   and self._episode_truncated and not self._episode_terminated))
              and self._episode_step_count == self.task.episode.max_steps):
            next_index = self._sample.episode_index + 1
        else:
            raise ValueError("calendar continuation requires a complete episode boundary")
        if self._episode_limit_exclusive is not None and next_index >= self._episode_limit_exclusive:
            raise ValueError("calendar continuation would exceed the declared episode limit")
        return {"schema_version": "hot-calendar-continuation/1.0.0",
                "task_id": self.task.task_id, "strategy_id": self._sample.strategy_id,
                **({"calendar_partition": self._partition_identity} if self._partition_identity is not None else {}),
                "schedule_seed": self._schedule.seed, "next_episode_index": next_index,
                "next_sample": self._sample_at(next_index).to_dict(calendar_year=self.task.calendar_year)}

    def render(self) -> Any:
        if self._child is None:
            raise RuntimeError("cannot render a closed CalendarTrainingEnvironment")
        render = getattr(self._child, "render", None)
        if not callable(render):
            raise NotImplementedError("child environment does not implement render")
        return render()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_child()


def make_calendar_training_environment(
    task: TaskSpec,
    *,
    data_root: str | Path,
    work_dir: str | Path,
    seed: int,
    start_episode_index: int = 0,
    environment_builder: EnvironmentBuilder = create_benchmark_environment,
    simulator_factory: SimulatorFactory | None = None,
    stream_id: str = "default",
    episode_completion_callback: EpisodeCompletionCallback | None = None,
    max_episodes: int | None = None,
    reserved_windows: tuple[CalendarWindowSpec, ...] | None = None,
) -> Callable[[], CalendarTrainingEnvironment]:
    """Return a no-argument thunk suitable for ``gym.vector.SyncVectorEnv``."""

    def thunk() -> CalendarTrainingEnvironment:
        return CalendarTrainingEnvironment(
            task,
            data_root=data_root,
            work_dir=work_dir,
            seed=seed,
            start_episode_index=start_episode_index,
            environment_builder=environment_builder,
            simulator_factory=simulator_factory,
            stream_id=stream_id,
            episode_completion_callback=episode_completion_callback,
            max_episodes=max_episodes,
            reserved_windows=reserved_windows,
        )

    return thunk


__all__ = [
    "CALENDAR_TRAINING_SCHEMA_VERSION",
    "STRATEGY_ID",
    "CalendarTrainingEnvironment",
    "CalendarTrainingSample",
    "StratifiedCalendarSchedule",
    "expected_calendar_training_episodes",
    "make_calendar_training_environment",
]
