"""Deterministic checkpoint selection using validation outcomes only."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping, Sequence


class NoFeasibleCheckpointError(ValueError):
    """All valid candidates exceed the declared validation comfort bound."""


@dataclass(frozen=True)
class ValidationOutcome:
    """One task's aggregated validation energy intensity and occupied exposure."""
    hvac_energy_kwh_per_m2: float
    comfort_exposure_person_celsius_squared_hours: float
    occupied_person_hours: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.occupied_person_hours == 0:
            raise ValueError('zero occupied exposure cannot qualify occupied-comfort validation')

    @property
    def comfort_mse_celsius_squared(self) -> float:
        return self.comfort_exposure_person_celsius_squared_hours / self.occupied_person_hours


@dataclass(frozen=True)
class ValidationCheckpoint:
    training_step: int
    checkpoint_sha256: str
    outcomes: Mapping[str, ValidationOutcome]

    def __post_init__(self):
        if type(self.training_step) is not int or self.training_step < 0:
            raise ValueError('training_step must be a nonnegative integer (zero denotes initialization)')
        if not isinstance(self.checkpoint_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', self.checkpoint_sha256):
            raise ValueError('checkpoint_sha256 must be a lowercase SHA-256 digest')
        if not isinstance(self.outcomes, Mapping) or not self.outcomes:
            raise ValueError('checkpoint requires validation outcomes')
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, ValidationOutcome)
               for k, v in self.outcomes.items()):
            raise ValueError('validation outcomes must map task identifiers to ValidationOutcome values')
        from types import MappingProxyType
        object.__setattr__(self, 'outcomes', MappingProxyType(dict(self.outcomes)))


def select_validation_checkpoint(candidates: Sequence[ValidationCheckpoint], *,
                                 validation_task_ids: Sequence[str],
                                 comfort_mse_bound: float | None = None,
                                 objective: str = "energy_under_comfort_bound") -> ValidationCheckpoint:
    """Minimize equal-task mean HVAC kWh/m² subject to every task's comfort bound.

    Comfort-only mode instead minimizes equal-task mean occupied MSE, with
    energy as its first tie-breaker and no absolute comfort gate. Remaining
    ties choose the earlier training step, then the checkpoint digest. A missing
    task or undefined occupied comfort invalidates the comparison; it is never
    silently dropped. The caller must authenticate the checkpoints and enforce
    disjoint validation/test data before supplying these validation-only inputs.
    """
    tasks = tuple(validation_task_ids)
    if not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('validation task identifiers must be nonempty and unique')
    if objective not in ('energy_under_comfort_bound', 'comfort_only'):
        raise ValueError('unknown validation selection objective')
    if objective == 'comfort_only':
        if comfort_mse_bound is not None:
            raise ValueError('comfort-only selection does not use a comfort bound')
    elif (isinstance(comfort_mse_bound, bool) or not isinstance(comfort_mse_bound, (int, float))
            or not math.isfinite(comfort_mse_bound) or comfort_mse_bound < 0):
        raise ValueError('comfort bound must be finite and nonnegative')
    if not candidates:
        raise ValueError('at least one validation checkpoint is required')
    identities = set()
    feasible = []
    for candidate in candidates:
        if not isinstance(candidate, ValidationCheckpoint):
            raise TypeError('candidates must be ValidationCheckpoint values')
        if set(candidate.outcomes) != set(tasks):
            raise ValueError('every checkpoint must cover exactly the declared validation tasks')
        identity = (candidate.training_step, candidate.checkpoint_sha256)
        if identity in identities:
            raise ValueError('duplicate validation checkpoint')
        identities.add(identity)
        if objective == 'comfort_only' or all(candidate.outcomes[t].comfort_mse_celsius_squared <= comfort_mse_bound for t in tasks):
            energy = math.fsum(candidate.outcomes[t].hvac_energy_kwh_per_m2 / len(tasks) for t in tasks)
            score = ((math.fsum(candidate.outcomes[t].comfort_mse_celsius_squared / len(tasks) for t in tasks), energy)
                     if objective == 'comfort_only' else (energy,))
            feasible.append((*score, *identity, candidate))
    if not feasible:
        raise NoFeasibleCheckpointError('no checkpoint satisfies the comfort bound on every validation task')
    return min(feasible, key=lambda item: item[:-1])[-1]


def _selection_ids(task_ids, protocol, aggregation):
    import json
    if aggregation == 'pooled':
        return tuple(task_ids)
    if aggregation != 'weekly':
        raise ValueError('validation aggregation must be pooled or weekly')
    if any(w.duration_hours_for_calendar_year(None) != 168 for w in protocol.windows):
        raise ValueError('weekly validation requires seven-day windows')
    return tuple(json.dumps([task, w.name, episode], separators=(',', ':'))
                 for task in task_ids for w in protocol.windows
                 for episode in range(protocol.episodes_per_window))


def weekly_validation_outcomes(rows, *, task_id, protocol):
    """Aggregate authenticated interval rows into separate seven-day outcomes.

    Each returned key identifies one task/window/repeat. The selector gives
    these keys equal weight, never weights weeks by their occupancy totals.
    """
    import json
    from building_gen.accounting import EnergyCarrier
    expected = _selection_ids([task_id], protocol, 'weekly')
    groups = {key: [] for key in expected}
    windows = {w.name: w for w in protocol.windows}
    for row in rows:
        key = json.dumps([row['task_id'], row['window'], int(row['episode_index'])], separators=(',', ':'))
        if key not in groups:
            raise ValueError('unexpected weekly validation row identity')
        groups[key].append(row)
    outcomes = {}
    common_area = None
    for key, values in groups.items():
        if not values:
            raise ValueError('missing weekly validation rows')
        frequency = int(values[0]['timesteps_per_hour'])
        if frequency <= 0 or 60 % frequency:
            raise ValueError('invalid weekly validation timestep')
        steps = 168 * frequency
        if len(values) != steps:
            raise ValueError('incomplete weekly validation trajectory')
        energies, exposures, occupants = [], [], []
        window = windows[values[0]['window']]
        for index, row in enumerate(values):
            if (int(row['step_index']) != index or int(row['expected_steps']) != steps
                    or int(row['timesteps_per_hour']) != frequency
                    or int(row['terminated']) != int(index == steps - 1)
                    or int(row['episodes_per_window']) != protocol.episodes_per_window):
                raise ValueError('weekly validation step sequence or termination mismatch')
            for field in ('start_month', 'start_day', 'end_month', 'end_day'):
                if int(row['window_' + field]) != getattr(window, field):
                    raise ValueError('weekly validation dates differ from protocol')
            area = float(row['controlled_floor_area_m2'])
            duration = float(row['interval_hours'])
            exposure = float(row['occupied_comfort_exposure_person_celsius_squared_hours_per_m2'])
            people = float(row['occupied_person_hours'])
            if any(not math.isfinite(x) or x < 0 for x in (area, duration, exposure, people)) or area == 0:
                raise ValueError('invalid weekly validation physical quantities')
            if not math.isclose(duration, 1 / frequency, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('weekly validation interval duration mismatch')
            if common_area is None:
                common_area = area
            if area != common_area:
                raise ValueError('weekly validation floor area changes')
            carrier_values = [float(row['controlled_hvac_energy_' + carrier.value + '_kwh'])
                              for carrier in EnergyCarrier
                              if row['controlled_hvac_energy_' + carrier.value + '_kwh'] != '']
            if not carrier_values or any(not math.isfinite(x) or x < 0 for x in carrier_values):
                raise ValueError('invalid weekly validation energy')
            energies.append(math.fsum(carrier_values) / area)
            exposures.append(exposure * area)
            occupants.append(people)
        outcomes[key] = ValidationOutcome(math.fsum(energies), math.fsum(exposures), math.fsum(occupants))
    return outcomes


def load_validation_checkpoint(training_dir, evaluation_dir, *, validation_task_ids: Sequence[str],
                               validation_protocol, aggregation='pooled') -> ValidationCheckpoint:
    """Load a complete engineering evaluation, checking identities and artifacts.

    The caller must supply the frozen validation-only task list and enforce data
    separation. This does not upgrade engineering diagnostics to scientific
    qualification or independently certify the simulator's physical accuracy.
    """
    import hashlib
    import json
    from pathlib import Path
    from building_gen.benchmark.reporting import _artifact_payload
    from building_gen.benchmark.artifacts import MetricValue
    from building_gen.benchmark._canonical import content_id
    from building_gen.benchmark.specs import EvaluationProtocolSpec

    if not isinstance(validation_protocol, EvaluationProtocolSpec) or not validation_protocol.deterministic_actions:
        raise ValueError('validation requires an explicit deterministic evaluation protocol')

    training_dir, evaluation_dir = Path(training_dir), Path(evaluation_dir)
    receipt = json.loads((training_dir / 'training.json').read_text())
    if receipt['status'] != 'complete':
        raise ValueError('validation requires a completed training checkpoint')
    for name, digest in {**receipt['artifacts'], **receipt['diagnostics']}.items():
        if Path(name).name != name or hashlib.sha256((training_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError('training artifact checksum mismatch')
    status = json.loads((evaluation_dir / 'evaluation_status.json').read_text())
    tasks = tuple(validation_task_ids)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('validation tasks must be nonempty and unique')
    parent_id = content_id('HOT SB3 engineering training', receipt, prefix='training')
    expected_experiment = content_id('HOT SB3 engineering evaluation',
        {'training': parent_id, 'targets': list(tasks), 'protocol': validation_protocol}, prefix='experiment')
    if (status['status'] != 'complete' or status['expected_target_ids'] != list(tasks)
            or status['training_id'] != parent_id or status['experiment_id'] != expected_experiment):
        raise ValueError('validation status does not match the declared tasks, protocol and checkpoint')
    selection_ids = _selection_ids(tasks, validation_protocol, aggregation)
    outcomes = {}
    seen_tasks = set()
    for name in status['results']:
        if Path(name).name != name:
            raise ValueError('validation result must be a sibling file')
        path = evaluation_dir / name
        result = json.loads(path.read_text())
        run = result['run']
        if (result['status'] != 'succeeded' or result['failure'] is not None
                or run['seed'] != receipt['seed'] or run['algorithm_id'] != f"sb3_{receipt['algorithm']}"
                or run['parent_training_run_id'] != parent_id or run['experiment_id'] != status['experiment_id']):
            raise ValueError('validation result identity or status mismatch')
        task_id = run['evaluation_task_id']
        if task_id not in tasks or task_id in seen_tasks:
            raise ValueError('validation has extra or duplicate task results')
        seen_tasks.add(task_id)
        artifacts = result['artifacts']
        roles = [a['role'] for a in artifacts]
        if len(set(roles)) != len(roles) or not {'step_trajectory', 'policy_checkpoint', 'evaluation_evidence'} <= set(roles):
            raise ValueError('validation artifact roles are missing or duplicated')
        for artifact in artifacts:
            _artifact_payload(result, path, role=artifact['role'], media_type=artifact['media_type'])
        checkpoint = next(a for a in artifacts if a['role'] == 'policy_checkpoint')
        if checkpoint['sha256'] != receipt['artifacts']['model.zip']:
            raise ValueError('validation evaluated a different checkpoint')
        metrics = {}
        for raw in result['metrics']:
            value = MetricValue(**raw)
            if value.name in metrics:
                raise ValueError('duplicate validation metric')
            metrics[value.name] = value
        names_units = (
            ('controlled_hvac_energy_intensity_kwh_per_m2', 'kWh/m2'),
            ('occupied_comfort_exposure_person_celsius_squared_hours', 'person degC2 h'),
            ('occupied_person_hours', 'person h'),
        )
        values = []
        for metric_name, unit in names_units:
            metric = metrics.get(metric_name)
            if metric is None or metric.unit != unit or metric.aggregation != 'sum':
                raise ValueError(f'missing or incompatible validation metric: {metric_name}')
            values.append(metric.value)
        pooled = ValidationOutcome(*values)
        if aggregation == 'weekly':
            import csv
            import io
            artifact = next(a for a in artifacts if a['role'] == 'step_trajectory')
            payload = _artifact_payload(result, path, role='step_trajectory', media_type=artifact['media_type']).payload
            weekly = weekly_validation_outcomes(csv.DictReader(io.StringIO(payload.decode('utf-8'))),
                task_id=task_id, protocol=validation_protocol)
            for attribute in ('hvac_energy_kwh_per_m2', 'comfort_exposure_person_celsius_squared_hours', 'occupied_person_hours'):
                if not math.isclose(math.fsum(getattr(o, attribute) for o in weekly.values()),
                                    getattr(pooled, attribute), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('weekly trajectory totals differ from reported metrics')
            outcomes.update(weekly)
        else:
            outcomes[task_id] = pooled
    if set(outcomes) != set(selection_ids):
        raise ValueError('validation results do not cover the declared tasks')
    return ValidationCheckpoint(receipt['completed_timesteps'], receipt['artifacts']['model.zip'], outcomes)


def validate_validation_test_separation(*, validation_tasks, validation_protocol,
                                        test_tasks, test_protocol) -> None:
    """Reject shared building/weather bytes on overlapping evaluation dates.

    Task names, rewards, asset paths and calendar-year labels cannot make the
    same weather samples held out. Nonoverlapping month/day windows on identical
    assets are allowed. This checks evaluation data reuse only: training calendar
    exclusion and scientific lineage/family holdouts require separate checks.
    """
    from building_gen.benchmark.specs import TaskSpec, EvaluationProtocolSpec

    def intervals(tasks, protocol):
        if not isinstance(protocol, EvaluationProtocolSpec):
            raise TypeError('evaluation partitions require explicit protocols')
        tasks = tuple(tasks)
        if not tasks or any(not isinstance(task, TaskSpec) for task in tasks):
            raise ValueError('evaluation partitions require nonempty TaskSpec lists')
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError('evaluation partition contains duplicate tasks')
        records = []
        for task in tasks:
            for window in protocol.windows:
                window.date_bounds_for_calendar_year(task.calendar_year)
                records.append((task.building.sha256, task.weather.sha256,
                                (window.start_month, window.start_day),
                                (window.end_month, window.end_day), task.name))
        return records

    validation = intervals(validation_tasks, validation_protocol)
    test = intervals(test_tasks, test_protocol)
    for building, weather, start, end, name in validation:
        for other_building, other_weather, other_start, other_end, other_name in test:
            if (building == other_building and weather == other_weather
                    and max(start, other_start) <= min(end, other_end)):
                raise ValueError(f'validation/test asset-date overlap: {name!r} and {other_name!r}')


def evaluate_and_select_checkpoints(*, training_dir, output_dir, validation_task_ids,
                                    validation_protocol, comfort_mse_bound, evaluate,
                                    objective="energy_under_comfort_bound", aggregation="pooled"):
    """Evaluate all recorded snapshots and save a reproducible selection receipt.

    ``evaluate(checkpoint_dir, evaluation_dir)`` must produce the standard HOT
    engineering evaluation bundle. Callers enforce train/validation/test splits
    before invoking this operation. Existing outputs are never overwritten.
    """
    import hashlib
    import json
    from pathlib import Path
    if objective not in ('energy_under_comfort_bound', 'comfort_only'):
        raise ValueError('unknown validation selection objective')
    if objective == 'comfort_only':
        if comfort_mse_bound is not None:
            raise ValueError('comfort-only selection does not use a comfort bound')
    elif (isinstance(comfort_mse_bound, bool) or not isinstance(comfort_mse_bound, (int, float))
            or not math.isfinite(comfort_mse_bound) or comfort_mse_bound < 0):
        raise ValueError('comfort bound must be finite and nonnegative')
    from building_gen.benchmark.specs import EvaluationProtocolSpec
    if not isinstance(validation_protocol, EvaluationProtocolSpec) or not validation_protocol.deterministic_actions:
        raise ValueError('validation requires an explicit deterministic evaluation protocol')
    validation_task_ids = tuple(validation_task_ids)
    if (not validation_task_ids or len(set(validation_task_ids)) != len(validation_task_ids)
            or any(not isinstance(task, str) or not task.strip() for task in validation_task_ids)):
        raise ValueError('validation tasks must be nonempty and unique')
    selection_ids = _selection_ids(validation_task_ids, validation_protocol, aggregation)
    training_dir, output_dir = Path(training_dir), Path(output_dir)
    training_path = training_dir/'training.json'
    training_bytes = training_path.read_bytes()
    training = json.loads(training_bytes)
    records = training.get('validation_checkpoints', [])
    if training.get('status') != 'complete' or not records:
        raise ValueError('selection requires completed training with recorded snapshots')
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite validation selection: {output_dir}')
    candidates = []
    directories = []
    seen = set()
    for record in records:
        relative = Path(record['directory'])
        if (relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != 'checkpoints'
                or relative.parts[1] in {'.', '..'} or str(relative) in seen):
            raise ValueError('snapshot directory must be a unique checkpoints child')
        seen.add(str(relative))
        directory = training_dir/relative
        if directory.is_symlink() or directory.resolve().parent != (training_dir/'checkpoints').resolve():
            raise ValueError('snapshot directory escapes the training checkpoint tree')
        payload = (directory/'training.json').read_bytes()
        if hashlib.sha256(payload).hexdigest() != record['training_receipt_sha256']:
            raise ValueError('snapshot receipt checksum mismatch')
        snapshot = json.loads(payload)
        if (snapshot['completed_timesteps'] != record['training_step']
                or snapshot['task_signature'] != training['task_signature']
                or snapshot['seed'] != training['seed'] or snapshot['algorithm'] != training['algorithm']):
            raise ValueError('snapshot identity differs from training')
        directories.append(directory)
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence = []
    for index, (record, directory) in enumerate(zip(records, directories)):
        destination = output_dir/f'candidate-{index}'
        if hashlib.sha256((directory/'training.json').read_bytes()).hexdigest() != record['training_receipt_sha256']:
            raise ValueError('snapshot receipt changed before evaluation')
        evaluate(directory, destination)
        if hashlib.sha256((directory/'training.json').read_bytes()).hexdigest() != record['training_receipt_sha256']:
            raise ValueError('snapshot receipt changed during evaluation')
        candidate = load_validation_checkpoint(directory, destination,
            validation_task_ids=validation_task_ids, validation_protocol=validation_protocol, aggregation=aggregation)
        candidates.append(candidate)
        evidence.append({'training_step': candidate.training_step, 'checkpoint_sha256': candidate.checkpoint_sha256,
            'checkpoint_directory': record['directory'], 'training_receipt_sha256': record['training_receipt_sha256'],
            'evaluation_directory': destination.name,
            'evaluation_status_sha256': hashlib.sha256((destination/'evaluation_status.json').read_bytes()).hexdigest(),
            'outcomes': {task: {'hvac_energy_kwh_per_m2': outcome.hvac_energy_kwh_per_m2,
                               'comfort_mse_celsius_squared': outcome.comfort_mse_celsius_squared,
                               'occupied_person_hours': outcome.occupied_person_hours}
                         for task, outcome in candidate.outcomes.items()}})
    failure = None
    selected_index = None
    try:
        chosen = select_validation_checkpoint(candidates, validation_task_ids=selection_ids,
                                              comfort_mse_bound=comfort_mse_bound, objective=objective)
        selected_index = next(index for index, candidate in enumerate(candidates) if candidate is chosen)
    except NoFeasibleCheckpointError as error:
        failure = error
    if training_path.read_bytes() != training_bytes:
        raise ValueError('training receipt changed during validation')
    receipt = {'schema_version': 'hot-validation-selection/1.0.0', 'mode': 'engineering',
        'objective': objective,
        'rule': ('minimum equal-task mean occupied comfort MSE; ties energy, earliest step, digest'
                 if objective == 'comfort_only' else
                 'minimum equal-task mean HVAC kWh/m2 subject to each task comfort MSE bound; ties earliest step then digest'),
        'comfort_mse_bound_celsius_squared': comfort_mse_bound,
        'validation_task_ids': list(selection_ids), 'source_validation_task_ids': list(validation_task_ids),
        'aggregation': aggregation,
        'training_receipt_sha256': hashlib.sha256(training_bytes).hexdigest(),
        'candidates': evidence, 'selected_candidate_index': selected_index,
        'status': 'no_feasible_checkpoint' if failure is not None else 'selected',
        'failure_reason': str(failure) if failure is not None else None}
    (output_dir/'selection.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
    if failure is not None:
        raise failure
    return directories[selected_index]
