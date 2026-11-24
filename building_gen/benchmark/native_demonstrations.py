"""Native thermostat output evidence, separate from policy actuator commands."""
import math

SCHEMA = 'hot-native-thermostat-targets/1'
TYPED_SCHEMA = 'hot-native-thermostat-targets/2'


def native_setpoint_evidence(info):
    """Read completed-interval targets, never next-boundary observations.

    Empty evidence remains valid for older interfaces, but cannot supply labels
    for imitation. Native outputs are not proof of actuator command agreement.
    """
    raw = info.get('energy_interval_observation', {})
    suffix = '_thermostat_heating_setpoint_c'
    zones = sorted(key[:-len(suffix)] for key in raw if key.endswith(suffix))
    if not zones:
        return {}
    typed=any(f'{zone}_thermostat_control_type' in raw for zone in zones)
    records = []
    for zone in zones:
        record = {'zone': zone}
        for channel in ('heating', 'cooling'):
            value = raw.get(f'{zone}_thermostat_{channel}_setpoint_c')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('native thermostat output must be present and finite')
            record[f'{channel}_setpoint_c'] = value
        if typed:
            control_type=raw.get(f'{zone}_thermostat_control_type')
            if (type(control_type) not in (int,float) or not math.isfinite(control_type)
                    or control_type not in (0,1,2,3,4)):
                raise ValueError('native thermostat control type must be reported as an integer 0 through 4')
            control_type=int(control_type)
            record.update(control_type=control_type,heating_active=control_type in (1,3,4),cooling_active=control_type in (2,3,4))
            if control_type==3 and record['heating_setpoint_c']!=record['cooling_setpoint_c']:
                raise ValueError('single heating/cooling thermostat must report one shared target')
            dual=control_type==4
        else:
            dual=True  # Legacy evidence never guessed an inactive mode.
        if dual and record['heating_setpoint_c'] >= record['cooling_setpoint_c']:
            raise ValueError(f'native dual thermostat targets must have a positive gap: {zone}')
        records.append(record)
    return dict(schema=TYPED_SCHEMA if typed else SCHEMA, phase='end_zone_timestep',
                clock=info['energy_interval_clock'], records=records)


def encode_native_targets(evidence, task, decision_clock):
    """Encode exact native targets; refuse unsupported targets rather than clip."""
    from building_gen.benchmark.policies import fixed_thermostat_action
    if (evidence.get('schema') not in (SCHEMA,TYPED_SCHEMA) or evidence.get('phase') != 'end_zone_timestep'
            or evidence.get('clock') != decision_clock):
        raise ValueError('native target evidence is missing or misaligned with the decision interval')
    records = evidence['records']
    by_zone = {record['zone']: record for record in records}
    zones = task.action_profile.controlled_entities
    if len(by_zone) != len(records) or set(by_zone) != set(zones):
        raise ValueError('native target zone coverage mismatch')
    action = []
    for zone in zones:
        record = by_zone[zone]
        if evidence['schema']==TYPED_SCHEMA and (record.get('control_type')!=4
                or record.get('heating_active') is not True or record.get('cooling_active') is not True):
            raise ValueError('native control mode cannot be encoded as a dual-setpoint imitation target without an explicit projection')
        pair = fixed_thermostat_action(task, heating_c=record['heating_setpoint_c'],
                                      cooling_c=record['cooling_setpoint_c'])[:2]
        action.extend(pair)
    return action
