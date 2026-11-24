"""Build a pinned weekly reward catalog from verified reference-run artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from building_gen.benchmark.config import load_benchmark_suite
from building_gen.benchmark.registry import default_algorithm_registry
from building_gen.benchmark.reporting import _artifact_payload, _portable_sibling_uri, _sha256
from building_gen.benchmark.checkpoint_selection import weekly_validation_outcomes
from building_gen.weekly_reward import build_weekly_calibration

MONTH_TO_SEASON = {m: ('winter' if m in (12,1,2) else 'spring' if m in (3,4,5)
                       else 'summer' if m in (6,7,8) else 'autumn') for m in range(1,13)}


def _fixed_actions_match(actual, expected):
    """Allow float32 action conversion roundoff, at most 1e-6 degrees C."""
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for value, reference in zip(actual, expected):
        if not isinstance(value, dict) or set(value) != set(reference) or value['zone'] != reference['zone']:
            return False
        for key in ('heating_setpoint_c', 'cooling_setpoint_c'):
            number = value[key]
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number)
                    or not math.isclose(number, reference[key], rel_tol=0, abs_tol=1e-6)):
                return False
    return True


def _verified_trajectory_rows(artifact, result_path):
    """Stream a sibling CSV, checking its checksum before and after parsing."""
    if set(artifact) != {'role', 'uri', 'media_type', 'sha256'} or artifact['media_type'] != 'text/csv':
        raise ValueError('invalid trajectory artifact schema')
    uri = _portable_sibling_uri(artifact['uri'], field='step_trajectory uri')
    path = result_path.parent / uri
    if path.is_symlink() or not path.is_file():
        raise ValueError('trajectory must be a regular sibling file')
    expected = _sha256(artifact['sha256'], field='step_trajectory sha256')
    def verify():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError('step_trajectory checksum mismatch')
    verify()
    with path.open(newline='', encoding='utf-8') as stream:
        yield from csv.DictReader(stream)
    verify()


def catalog_from_results(suite, *, task_name, result_paths, training_experiment, validation_experiment, reference_algorithm="fixed_thermostat"):
    if reference_algorithm not in ('fixed_thermostat','native_incumbent'):
        raise ValueError('unsupported calibration reference controller')
    expected_parameters = {'heating_c':20,'cooling_c':24} if reference_algorithm=='fixed_thermostat' else {}
    task = suite.task_map[task_name]
    task_ids = {t.name:t.task_id for t in suite.tasks}
    registry = default_algorithm_registry()
    expected = {}
    for role, name in [('training', training_experiment), ('validation', validation_experiment)]:
        experiment = suite.experiment_map[name]
        if (experiment.algorithm_id != reference_algorithm or experiment.parameter_mapping != expected_parameters
                or experiment.evaluation_tasks != (task_name,) or experiment.seeds != (0,)
                or experiment.evaluation_protocol.episodes_per_window != 1):
            raise ValueError('calibration requires one seed-0 declared reference per week and task')
        key = experiment.experiment_id(task_ids=task_ids,algorithm_version=registry.get(reference_algorithm).version)
        if key in expected:
            raise ValueError('training and validation reference experiments must differ')
        expected[key] = (role, experiment.evaluation_protocol)
    measurements, sources, seen = [], [], set()
    for path in map(Path,result_paths):
        payload=path.read_bytes()
        result=json.loads(payload)
        run=result['run']
        key=run['experiment_id']
        if key not in expected or key in seen:
            raise ValueError('extra or duplicate calibration result')
        seen.add(key)
        role, protocol = expected[key]
        if (result['status'] != 'succeeded' or result['failure'] is not None or run['evaluation_task_id'] != task.task_id
                or run['algorithm_id'] != reference_algorithm or run['algorithm_version'] != registry.get(reference_algorithm).version
                or run['seed'] != 0):
            raise ValueError('calibration result identity or success status mismatch')
        artifacts = {}
        for artifact in result['artifacts']:
            if artifact['role'] in artifacts:
                raise ValueError('duplicate calibration artifact role')
            artifacts[artifact['role']] = (artifact if artifact['role']=='step_trajectory' else
                _artifact_payload(result,path,role=artifact['role'],media_type=artifact['media_type']).payload)
        diagnostics=json.loads(artifacts['energyplus_diagnostics'])
        if (len(diagnostics['episodes']) != len(protocol.windows) or
                {(e['window'],e['episode_index']) for e in diagnostics['episodes']} != {(w.name,0) for w in protocol.windows}):
            raise ValueError('calibration diagnostics episode coverage mismatch')
        for episode in diagnostics['episodes']:
            d=episode['diagnostics']
            if not d['completed_successfully'] or d['severe_error_count'] or d['fatal_error_count']:
                raise ValueError('calibration includes unsuccessful simulator diagnostics')
        reader=_verified_trajectory_rows(artifacts['step_trajectory'], path)
        rows=[]
        weekly_fields={'task_id','window','episode_index','timesteps_per_hour','step_index','expected_steps','terminated','episodes_per_window','controlled_floor_area_m2','interval_hours','occupied_comfort_exposure_person_celsius_squared_hours_per_m2','occupied_person_hours'}
        expected_actions=[dict(zone=z,heating_setpoint_c=20.0,cooling_setpoint_c=24.0)
                          for z in task.action_profile.controlled_entities]
        for row in reader:
            rows.append({k:v for k,v in row.items() if k in weekly_fields or k.startswith("window_") or (k.startswith("controlled_hvac_energy_") and k.endswith("_kwh"))})
            if int(row['timesteps_per_hour']) != task.episode.timesteps_per_hour:
                raise ValueError('calibration timestep differs from task')
            if reference_algorithm=='native_incumbent':
                native=json.loads(row['native_control_json'])
                if native['thermostat_overrides']!='released' or native['submitted_policy_setpoints']:
                    raise ValueError('native reference has thermostat overrides')
                continue
            if (row['actuator_readback_status'] != 'available' or row['actuator_readback_phase'] != 'end_zone_timestep'
                    or not _fixed_actions_match(json.loads(row['submitted_action_physical_json']), expected_actions)
                    or not _fixed_actions_match(json.loads(row['actuator_readback_physical_json']), expected_actions)):
                raise ValueError('calibration requires verified fixed 20–24°C actuator values')
        outcomes=weekly_validation_outcomes(rows,task_id=task.task_id,protocol=protocol)
        area=float(rows[0]['controlled_floor_area_m2'])
        metrics={m['name']:m for m in result['metrics']}
        if len(metrics)!=len(result['metrics']):
            raise ValueError('duplicate calibration result metrics')
        for metric,attribute,unit in (
            ('controlled_hvac_energy_intensity_kwh_per_m2','hvac_energy_kwh_per_m2','kWh/m2'),
            ('occupied_comfort_exposure_person_celsius_squared_hours','comfort_exposure_person_celsius_squared_hours','person degC2 h'),
            ('occupied_person_hours','occupied_person_hours','person h')):
            reported=metrics[metric]
            if (reported['unit']!=unit or reported['aggregation']!='sum' or
                    not math.isclose(math.fsum(getattr(o,attribute) for o in outcomes.values()),reported['value'],rel_tol=1e-9,abs_tol=1e-9)):
                raise ValueError('calibration interval totals differ from reported metrics')
        for window in protocol.windows:
            outcome=outcomes[json.dumps([task.task_id,window.name,0],separators=(',',':'))]
            measurements.append({k:getattr(window,k) for k in ('start_month','start_day','end_month','end_day')} | dict(
                role=role,controlled_floor_area_m2=area,occupied_person_hours=outcome.occupied_person_hours,
                energy_kwh=outcome.hvac_energy_kwh_per_m2*area))
        sources.append(dict(role=role,result_sha256=hashlib.sha256(payload).hexdigest(),run_id=result['run_id'],
                            artifacts=result['artifacts']))
    if seen != set(expected):
        raise ValueError('missing training or validation reference result')
    return build_weekly_calibration(task, measurements, month_to_season=MONTH_TO_SEASON,
        provenance=dict(reference_controller=({'heating_c':20,'cooling_c':24} if reference_algorithm=='fixed_thermostat' else {'algorithm_id':'native_incumbent','thermostat_overrides':'released'}),sources=sources))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--training-experiment',required=True)
    parser.add_argument('--validation-experiment',required=True)
    parser.add_argument('--result',type=Path,action='append',required=True)
    parser.add_argument('--reference-algorithm',choices=('fixed_thermostat','native_incumbent'),default='fixed_thermostat')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    suite_payload = args.suite.read_bytes()
    catalog=catalog_from_results(load_benchmark_suite(args.suite), task_name=args.task,result_paths=args.result,
        training_experiment=args.training_experiment,validation_experiment=args.validation_experiment,reference_algorithm=args.reference_algorithm)
    if args.suite.read_bytes() != suite_payload:
        raise ValueError('suite changed while building calibration')
    catalog['provenance']['suite_sha256']=hashlib.sha256(suite_payload).hexdigest()
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(catalog,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'Wrote {len(catalog["weeks"])} calibrated weeks to {args.output}')


if __name__=='__main__':
    main()
