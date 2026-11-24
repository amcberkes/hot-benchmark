"""Engineering training CLI for the optional Gym RL baseline adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from building_gen.wrapper.rl_baselines import ALGORITHMS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hot train-baseline", description=__doc__, allow_abbrev=False)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--task', required=True, help='Training task name in the suite; includes its reward contract.')
    parser.add_argument('--algorithm', required=True, choices=ALGORITHMS)
    parser.add_argument('--parameters', type=Path, help='JSON object of algorithm parameters.')
    parser.add_argument('--comfort-constraint', type=Path, help='Pinned training-only comfort reference and rollout penalty configuration (PPO only).')
    parser.add_argument('--reserve-window', action='append', help='Exclude inclusive MM-DD:MM-DD dates from training; repeat for multiple windows. Repeat the same partition when resuming.')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--steps', required=True, type=int)
    parser.add_argument('--checkpoint-interval', type=int,
                        help='Save evaluation snapshots after optimizer updates at approximately this transition interval; does not select a checkpoint.')
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--resume-from', type=Path, help='Parent checkpoint; steps are additional transitions, written to a new output directory.')
    parser.add_argument('--transfer-from', type=Path, help='Authenticated source PPO bundle; copies shared actor/distribution with fresh target critic and optimizer.')
    parser.add_argument('--transfer-source-task', help='Source task name in the same suite.')
    parser.add_argument('--actor-initialization', type=Path, help='Audited supervised actor fit for fresh PPO; copies actor weights only.')
    parser.add_argument('--actor-source-suite', type=Path, help='Original actor task suite; permits only a weekly energy-weight change.')
    parser.add_argument('--actor-source-task', help='Original actor task name in --actor-source-suite.')
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    from building_gen.benchmark._canonical import canonical_value
    from building_gen.benchmark.config import load_benchmark_suite
    from building_gen.benchmark.environment import resolve_task_assets
    from building_gen.benchmark.training_calendar import make_calendar_training_environment
    from building_gen.wrapper.rl_baselines import _implementation, train_agent

    if args.seed < 0 or args.steps <= 0:
        raise ValueError('seed must be nonnegative and steps must be positive')
    interval = getattr(args, 'checkpoint_interval', None)
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError('checkpoint interval must be a positive integer')
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite run: {args.output_dir}')
    suite = load_benchmark_suite(args.suite)
    if args.task not in suite.task_map:
        raise ValueError(f'unknown task {args.task!r}; choose {sorted(suite.task_map)}')
    task = suite.task_map[args.task]
    if task.action_profile.profile_id != 'thermostat_v1':
        raise ValueError('this task factory currently supports thermostat_v1; manifest profiles are not yet integrated')
    if args.algorithm == 'dqn':
        raise ValueError('DQN needs a discrete task interface; the current benchmark task factory is continuous')
    parameters = json.loads(args.parameters.read_text(encoding='utf-8')) if args.parameters else None
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError('algorithm parameter file must be a JSON object')
    _implementation(args.algorithm)  # Check optional dependency/version before constructing a simulator.
    signature = {"task_id": task.task_id, "definition": canonical_value(task.identity_payload())}
    resume_from = getattr(args, 'resume_from', None)
    comfort_penalty_config = None
    constraint_path = getattr(args, 'comfort_constraint', None)
    if constraint_path is not None:
        from building_gen.wrapper.comfort_penalty import load_comfort_penalty_config
        if args.algorithm != 'ppo' or resume_from is not None:
            raise ValueError('comfort constraint supports fresh PPO training only')
        comfort_penalty_config = load_comfort_penalty_config(constraint_path, task, args.data_root)
        rollout_steps = comfort_penalty_config.steps_per_week * 12
        if (parameters or {}).get('n_steps', 2048) != rollout_steps or args.steps % rollout_steps:
            raise ValueError('comfort constraint requires complete twelve-week rollouts')
        if interval is not None and interval % rollout_steps:
            raise ValueError('comfort checkpoints must align with rollouts')
    start_episode_index = 0
    if resume_from is not None:
        import importlib.metadata
        from building_gen.wrapper.rl_baselines import _parameters
        parent = json.loads((resume_from/'training.json').read_text(encoding='utf-8'))
        if parent.get('status') != 'complete':
            raise ValueError('resume requires a complete parent checkpoint')
        if parent.get('algorithm') != args.algorithm or parent.get('seed') != args.seed:
            raise ValueError('resume algorithm and seed must match the parent checkpoint')
        if parent.get('task_signature') != signature:
            raise ValueError('resume task semantics must match the parent checkpoint')
        if parameters is not None and _parameters(args.algorithm, parameters) != parent['parameters']:
            raise ValueError('resume cannot change algorithm parameters')
        if parent.get('dependencies') != {name: importlib.metadata.version(name) for name in ('torch', 'gymnasium', 'numpy')}:
            raise ValueError('resume dependency versions must match the parent checkpoint')
        continuation = parent.get('continuation', {})
        if continuation.get('available') is not True or continuation.get('mode') != 'fresh_episode_boundary':
            raise ValueError('parent checkpoint has no supported episode-boundary continuation')
        calendar = continuation['calendar']
        if calendar.get('task_id') != task.task_id or calendar.get('schedule_seed') != args.seed:
            raise ValueError('resume calendar task and seed must match')
        start_episode_index = calendar['next_episode_index']
        if type(start_episode_index) is not int or start_episode_index < 0:
            raise ValueError('resume calendar index must be a nonnegative integer')
    reserved_windows = None
    reserved_values = getattr(args, 'reserve_window', None)
    if reserved_values is not None:
        import re
        from building_gen.benchmark.specs import CalendarWindowSpec
        from building_gen.benchmark.training_partitions import PartitionedCalendarSchedule
        reserved_windows = []
        for index, value in enumerate(reserved_values):
            match = re.fullmatch(r'(\d{2})-(\d{2}):(\d{2})-(\d{2})', value)
            if match is None:
                raise ValueError('reserved windows must use MM-DD:MM-DD')
            reserved_windows.append(CalendarWindowSpec(f'reserved-{index}', *(int(v) for v in match.groups())))
        reserved_windows = tuple(reserved_windows)
        partition = PartitionedCalendarSchedule(seed=args.seed,
            episode_length_hours=task.episode.episode_length_hours,
            reserved_windows=reserved_windows, calendar_year=task.calendar_year).identity_payload()
    else:
        partition = None
    if resume_from is not None and calendar.get('calendar_partition') != partition:
        raise ValueError('resume reserved-date partition must match the parent checkpoint')
    actor_reward_tasks = None
    source_suite = getattr(args, 'actor_source_suite', None)
    source_name = getattr(args, 'actor_source_task', None)
    if source_suite is not None or source_name is not None:
        if source_suite is None or source_name is None or getattr(args, 'actor_initialization', None) is None:
            raise ValueError('actor source suite/task require each other and actor initialization')
        from building_gen.benchmark.config import load_benchmark_suite
        from building_gen.wrapper.actor_initialization import reward_only_compatibility
        source_task = load_benchmark_suite(source_suite).task_map[source_name]
        reward_only_compatibility(source_task, task)
        actor_reward_tasks = (source_task, task)
    transfer_initialization = None
    transfer_from = getattr(args, 'transfer_from', None)
    transfer_source = getattr(args, 'transfer_source_task', None)
    if transfer_from is not None or transfer_source is not None:
        if transfer_from is None or transfer_source is None or args.algorithm != 'ppo' or resume_from is not None or getattr(args, 'actor_initialization', None) is not None:
            raise ValueError('transfer requires source bundle/task and fresh PPO without other initialization')
        from building_gen.wrapper.shared_zone_transfer import validate_shared_targets
        source_task = suite.task_map[transfer_source]
        validate_shared_targets(source_task, [task])
        transfer_initialization = (transfer_from, source_task, task)
    resolve_task_assets(task, args.data_root)
    environment = make_calendar_training_environment(
        task, data_root=args.data_root, work_dir=args.work_dir, seed=args.seed,
        start_episode_index=start_episode_index,
        **({"reserved_windows": reserved_windows} if reserved_windows is not None else {}),
    )()
    try:
        receipt = train_agent(
            args.algorithm, environment, seed=args.seed, total_timesteps=args.steps,
            task_signature=signature,
            output_dir=args.output_dir, parameters=parameters, resume_from=resume_from,
            checkpoint_interval=interval,
            **({"transfer_initialization": transfer_initialization} if transfer_initialization is not None else {}),
            **({'actor_reward_tasks': actor_reward_tasks} if actor_reward_tasks is not None else {}),
            **({'actor_initialization': args.actor_initialization} if getattr(args, 'actor_initialization', None) is not None else {}),
            **({"comfort_penalty_config": comfort_penalty_config} if comfort_penalty_config is not None else {}),
        )
    finally:
        training_error = sys.exc_info()[1]
        try:
            environment.close()
        except Exception as cleanup_error:
            status_path = args.output_dir / 'training.json'
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding='utf-8'))
                status.update(status='failed', cleanup_error=repr(cleanup_error))
                status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
            if training_error is None:
                raise
            print(f'Environment cleanup also failed: {cleanup_error}', file=sys.stderr)
    print(f'Wrote {receipt}')
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    """Evaluate a final adapter checkpoint with the existing HOT rollout accounting."""
    from building_gen.benchmark.cli import _add_provenance_arguments
    parser = argparse.ArgumentParser(prog="hot evaluate-baseline", description=evaluate_main.__doc__, allow_abbrev=False)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--source-task', required=True)
    parser.add_argument('--target-task', required=True, action='append')
    parser.add_argument('--protocol-from-experiment', required=True,
                        help='Use only the evaluation protocol from this suite experiment.')
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stochastic-diagnostic-seed',type=int,help='Shared-zone Gaussian sampling diagnostic; excluded from deterministic source qualification.')
    _add_provenance_arguments(parser)
    parser.set_defaults(repository='github.com/amcberkes/hot')
    args = parser.parse_args(argv)
    try:
        return run_evaluation(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


def run_evaluation(args: argparse.Namespace) -> int:
    import importlib.metadata
    from dataclasses import replace
    from building_gen.benchmark.cli import _rollout_environment_factory, _software_provenance
    from building_gen.benchmark.config import load_benchmark_suite
    from building_gen.benchmark.environment import resolve_task_assets
    from building_gen.benchmark.rl_evaluation import evaluate_baseline

    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite evaluation: {args.output_dir}')
    suite = load_benchmark_suite(args.suite)
    source = suite.task_map[args.source_task]
    targets = [suite.task_map[name] for name in args.target_task]
    protocol = suite.experiment_map[args.protocol_from_experiment].evaluation_protocol
    if protocol is None:
        raise ValueError('selected experiment has no evaluation protocol')
    for task in [source, *targets]:
        resolve_task_assets(task, args.data_root)
    provenance = replace(_software_provenance(args), dependencies={
        name: importlib.metadata.version(name) for name in ('stable-baselines3', 'torch', 'gymnasium', 'numpy')
    })
    paths = evaluate_baseline(run_dir=args.run_dir, source=source, targets=targets, protocol=protocol,
        environment_factory=_rollout_environment_factory(data_root=args.data_root, work_dir=args.work_dir),
        provenance=provenance, output_dir=args.output_dir,
        selection_rule=getattr(args, "selection_rule", None),
        stochastic_diagnostic_seed=getattr(args,"stochastic_diagnostic_seed",None))
    for path in paths:
        print(f'Wrote {path}')
    return 0
