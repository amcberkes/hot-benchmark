"""Engineering evaluation of Gym baseline checkpoints using HOT accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from building_gen.benchmark._canonical import canonical_value, content_id
from building_gen.benchmark.energyplus_diagnostics import DiagnosticsGate
from building_gen.benchmark.evaluation import evaluate_run, learned_callable_policy
from building_gen.benchmark.provenance import ResolvedRun, SoftwareProvenance
from building_gen.benchmark.specs import EvaluationProtocolSpec, TaskSpec


def validate_targets(source: TaskSpec, targets: Sequence[TaskSpec]) -> None:
    """Validate fixed-vector transfer compatibility before training or evaluation."""
    if not targets or len({task.task_id for task in targets}) != len(targets):
        raise ValueError('evaluation targets must be nonempty and unique')
    for target in targets:
        if (target.observation_space.signature() != source.observation_space.signature()
                or target.action_space.signature() != source.action_space.signature()
                or target.action_profile.transfer_signature() != source.action_profile.transfer_signature()):
            raise ValueError(f'target {target.name} has incompatible observation or action semantics')
        if target.action_profile.profile_id != 'thermostat_v1':
            raise ValueError('manifest-profile evaluation is not integrated')


def evaluate_baseline(*, run_dir: Path, source: TaskSpec, targets: Sequence[TaskSpec],
                      protocol: EvaluationProtocolSpec, environment_factory: Any,
                      provenance: SoftwareProvenance, output_dir: Path, selection_rule: str | None = None,
                      stochastic_diagnostic_seed: int | None = None) -> tuple[Path, ...]:
    """Evaluate frozen checkpoints; seeded Gaussian sampling is a separate engineering diagnostic."""
    from building_gen.wrapper.rl_baselines import SB3_VERSION, load_agent

    signature = {'task_id': source.task_id, 'definition': canonical_value(source.identity_payload())}
    agent = load_agent(run_dir, None, task_signature=signature)
    receipt = json.loads((Path(run_dir)/'training.json').read_text(encoding='utf-8'))
    from building_gen.wrapper.shared_zone_policy import SharedZonePolicy
    shared = isinstance(agent.policy, SharedZonePolicy)
    if shared:
        from building_gen.wrapper.shared_zone_transfer import SharedZoneActorAdapter, validate_shared_targets
        from building_gen.benchmark.environment import _expected_observation_names
        if receipt['parameters'].get('policy_architecture') != 'shared_zone_v1':
            raise ValueError('shared policy/receipt architecture mismatch')
        validate_shared_targets(source, targets)
    else:
        validate_targets(source, targets)
    if agent.num_timesteps != receipt['completed_timesteps'] or agent.seed != receipt['seed']:
        raise ValueError('training receipt counters/seed disagree with the loaded model')
    if stochastic_diagnostic_seed is not None:
        if (type(stochastic_diagnostic_seed) is not int or not 0<=stochastic_diagnostic_seed<2**63
                or not shared or receipt['algorithm']!='ppo'):
            raise ValueError('Stochastic diagnostics require shared-zone PPO and a valid explicit seed')
        from dataclasses import replace
        protocol=replace(protocol,deterministic_actions=False)
    algorithm_id = f"sb3_{receipt['algorithm']}"
    parent_id = content_id('HOT SB3 engineering training', receipt, prefix='training')
    identity={'training': parent_id, 'targets': [task.task_id for task in targets], 'protocol': protocol}
    if stochastic_diagnostic_seed is not None:identity['stochastic_diagnostic_seed']=stochastic_diagnostic_seed
    experiment_id = content_id('HOT SB3 engineering evaluation',identity,prefix='experiment')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    status = {'schema_version': 'hot-sb3-evaluation/1.0.0', 'mode': 'engineering',
              'status': 'running', 'training_id': parent_id, 'experiment_id': experiment_id,
              'expected_target_ids': [task.task_id for task in targets], 'results': [],
              'policy_adapter': 'shared_zone_actor_v1' if shared else 'fixed_vector_v1'}

    if stochastic_diagnostic_seed is not None:
        status.update(schema_version='hot-sb3-stochastic-diagnostic/1.0.0',mode='engineering_stochastic_diagnostic',
            sampling_seed=stochastic_diagnostic_seed,protocol=canonical_value(protocol),policy_adapter='shared_zone_gaussian_sampling_v1')

    def record():
        temporary = destination/'.evaluation_status.tmp'
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        temporary.replace(destination/'evaluation_status.json')

    record()
    paths = []
    try:
        for target in targets:
            predictor = SharedZoneActorAdapter(agent.policy, _expected_observation_names(source), _expected_observation_names(target),sampling_seed=stochastic_diagnostic_seed) if shared else lambda observation: agent.predict(observation, deterministic=True)[0]
            policy = learned_callable_policy(
                predictor,
                algorithm_id=algorithm_id, parent_training_run_id=parent_id,
                checkpoint_path=Path(run_dir)/'model.zip', checkpoint_sha256=receipt['artifacts']['model.zip'],
                training_step=agent.num_timesteps,
                selection_rule=((f'stochastic Gaussian diagnostic with sampling seed {stochastic_diagnostic_seed}; ' if stochastic_diagnostic_seed is not None else '')+(selection_rule or f'final SB3 checkpoint after {agent.num_timesteps} transitions; no target-based selection')),
            )
            status['current_target_id'] = target.task_id
            record()
            run = ResolvedRun(experiment_id=experiment_id, algorithm_id=algorithm_id,
                              algorithm_version=SB3_VERSION, phase='evaluate', seed=receipt['seed'],
                              train_task_ids=(source.task_id,), evaluation_task_id=target.task_id,
                              parent_training_run_id=parent_id)
            result = evaluate_run(run=run, task=target, protocol=protocol,
                environment_factory=environment_factory, policy=policy, provenance=provenance,
                output_dir=destination, overwrite=False, diagnostics_gate=DiagnosticsGate(mode='engineering'),
                stochastic_diagnostic_seed=stochastic_diagnostic_seed)
            paths.append(result.result_path)
            status['results'].append(result.result_path.name)
            record()
        status['status'] = 'complete'
        record()
        return tuple(paths)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        record()
        raise
