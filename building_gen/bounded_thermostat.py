"""Versioned thermostat mappings with cooling ceilings and minimum gaps."""
import math

TRANSFORM_ID='bounded_thermostat_15_30_v1'
WIDE_TRANSFORM_ID='bounded_thermostat_12_32_v1'
PROFILES={TRANSFORM_ID:(15.,23.,30.),WIDE_TRANSFORM_ID:(12.,24.,32.)}

def limits(transform_id=TRANSFORM_ID):
    try:return PROFILES[transform_id]
    except KeyError:raise ValueError('unknown bounded thermostat transform') from None

def validate_ranges(heating,offset,transform_id=TRANSFORM_ID):
    low,high,ceiling=limits(transform_id)
    if tuple(heating)!=(low,high) or tuple(offset)!=(2.,ceiling-low):
        raise ValueError(f'{transform_id} requires heating [{low},{high}] and offset envelope [2,{ceiling-low}]')

def decode(heating_action,cooling_action,transform_id=TRANSFORM_ID):
    if any(isinstance(x,bool) or not math.isfinite(x) or not 0<=x<=1 for x in (heating_action,cooling_action)):
        raise ValueError('bounded thermostat actions must be finite and within [0,1]')
    low,high,ceiling=limits(transform_id)
    heating=low+(high-low)*float(heating_action)
    cooling=heating+2.+float(cooling_action)*(ceiling-heating-2.)
    return heating,cooling

def encode(heating,cooling,transform_id=TRANSFORM_ID):
    if any(isinstance(x,bool) or not math.isfinite(x) for x in (heating,cooling)):
        raise ValueError('thermostat targets must be finite')
    low,high,ceiling=limits(transform_id)
    if not low<=heating<=high or not heating+2<=cooling<=ceiling:
        raise ValueError('targets outside bounded thermostat feasible region')
    return (heating-low)/(high-low),(cooling-heating-2.)/(ceiling-heating-2.)
