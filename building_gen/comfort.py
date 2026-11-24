"""Deterministic comfort bounds, independent of controller setpoints."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class ComfortBand:
    minimum_c: float
    maximum_c: float

    def __post_init__(self):
        for value in (self.minimum_c, self.maximum_c):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('comfort bounds must be finite numbers')
        if self.minimum_c >= self.maximum_c:
            raise ValueError('comfort minimum must be smaller than maximum')

    def violation_c(self, temperature_c: float) -> float:
        if isinstance(temperature_c, bool) or not isinstance(temperature_c, (int, float)) or not math.isfinite(temperature_c):
            raise ValueError('temperature must be finite')
        return max(self.minimum_c-temperature_c, 0.0, temperature_c-self.maximum_c)


@dataclass(frozen=True)
class WeeklyComfortPeriod:
    """Sunday=1; local standard-time minutes; start inclusive, end exclusive.

    Split overnight intervals across their two days. No implicit timezone,
    daylight-saving, holiday or occupancy inference is performed.
    """
    weekdays: tuple[int, ...]
    start_minute: int
    end_minute: int
    band: ComfortBand
    zone: str | None = None

    def __post_init__(self):
        if not isinstance(self.weekdays, tuple) or not self.weekdays or len(set(self.weekdays)) != len(self.weekdays):
            raise ValueError('weekdays must be a nonempty tuple of unique days')
        if any(type(day) is not int or not 1 <= day <= 7 for day in self.weekdays):
            raise ValueError('weekdays must be integers from 1 through 7')
        if type(self.start_minute) is not int or type(self.end_minute) is not int or not 0 <= self.start_minute < self.end_minute <= 1440:
            raise ValueError('comfort interval requires 0 <= start < end <= 1440')
        if not isinstance(self.band, ComfortBand):
            raise TypeError('period band must be a ComfortBand')
        if self.zone is not None and (not isinstance(self.zone, str) or not self.zone.strip() or self.zone != self.zone.strip()):
            raise ValueError('zone must be an exact nonempty identifier')


@dataclass(frozen=True)
class ComfortSchedule:
    """Resolve zone periods before shared periods, then zone defaults, then default."""
    default: ComfortBand
    zone_defaults: tuple[tuple[str, ComfortBand], ...] = ()
    periods: tuple[WeeklyComfortPeriod, ...] = ()
    monthly_defaults: tuple[tuple[int, ComfortBand], ...] = ()

    def widened(self, total_width_c: float):
        """Widen every band symmetrically; preserve timing and zone precedence."""
        if (isinstance(total_width_c, bool) or not isinstance(total_width_c, (int, float))
                or not math.isfinite(total_width_c) or total_width_c <= 0):
            raise ValueError('total comfort-band widening must be positive and finite')
        half = total_width_c / 2
        def widen(band):
            return ComfortBand(band.minimum_c - half, band.maximum_c + half)
        return replace(self, default=widen(self.default),
                       zone_defaults=tuple((zone, widen(band)) for zone, band in self.zone_defaults),
                       monthly_defaults=tuple((month, widen(band)) for month, band in self.monthly_defaults),
                       periods=tuple(replace(period, band=widen(period.band)) for period in self.periods))

    def __post_init__(self):
        if not isinstance(self.default, ComfortBand):
            raise TypeError('default must be a ComfortBand')
        if not isinstance(self.zone_defaults, tuple) or not isinstance(self.periods, tuple):
            raise TypeError('schedule collections must be immutable tuples')
        if not isinstance(self.monthly_defaults, tuple):
            raise TypeError('monthly defaults must be an immutable tuple')
        months=[]
        for month, band in self.monthly_defaults:
            if type(month) is not int or not 1 <= month <= 12 or not isinstance(band, ComfortBand):
                raise ValueError('invalid monthly comfort default')
            months.append(month)
        if len(set(months)) != len(months):
            raise ValueError('duplicate monthly comfort defaults')
        zones=[]
        for pair in self.zone_defaults:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('zone defaults require (zone, band) pairs')
            zone, band=pair
            if not isinstance(zone, str) or not zone.strip() or zone != zone.strip() or not isinstance(band, ComfortBand):
                raise ValueError('invalid zone default')
            zones.append(zone)
        if len(set(zones)) != len(zones):
            raise ValueError('duplicate zone defaults')
        if any(not isinstance(period, WeeklyComfortPeriod) for period in self.periods):
            raise TypeError('periods must contain WeeklyComfortPeriod values')
        for index, left in enumerate(self.periods):
            for right in self.periods[index+1:]:
                if (left.zone == right.zone and set(left.weekdays) & set(right.weekdays)
                    and max(left.start_minute,right.start_minute) < min(left.end_minute,right.end_minute)):
                    raise ValueError('overlapping comfort periods at the same precedence')

    def band_at(self, zone: str, *, weekday: int, minute: float, month: int | None = None) -> ComfortBand:
        if not isinstance(zone, str) or not zone.strip() or zone != zone.strip():
            raise ValueError('zone must be an exact nonempty identifier')
        if type(weekday) is not int or not 1 <= weekday <= 7:
            raise ValueError('weekday must be an integer from 1 through 7')
        if isinstance(minute, bool) or not isinstance(minute, (int,float)) or not math.isfinite(minute) or not 0 <= minute < 1440:
            raise ValueError('minute must be finite and in [0,1440)')
        if self.monthly_defaults and (type(month) is not int or not 1 <= month <= 12):
            raise ValueError('monthly comfort requires a valid calendar month')
        for scope in (zone, None):
            for period in self.periods:
                if period.zone == scope and weekday in period.weekdays and period.start_minute <= minute < period.end_minute:
                    return period.band
        return dict(self.zone_defaults).get(zone,dict(self.monthly_defaults).get(month,self.default))

    def validate_zones(self, controlled_zones: tuple[str, ...]) -> None:
        """Reject misspelled overrides before starting a simulator."""
        declared = {zone for zone, _ in self.zone_defaults}
        declared.update(period.zone for period in self.periods if period.zone is not None)
        unknown = declared - set(controlled_zones)
        if unknown:
            raise ValueError(f'comfort schedule references uncontrolled zones: {sorted(unknown)}')

    def to_mapping(self) -> dict:
        def band(value):
            return {'minimum_c': float(value.minimum_c), 'maximum_c': float(value.maximum_c)}
        result = {'schema_version': 'hot-comfort-schedule/1.0.0', 'default': band(self.default),
                'zone_defaults': {zone: band(value) for zone, value in sorted(self.zone_defaults)},
                'periods': [{'weekdays': sorted(period.weekdays), 'start_minute': period.start_minute,
                             'end_minute': period.end_minute, 'band': band(period.band), 'zone': period.zone}
                            for period in sorted(self.periods, key=lambda p: (p.zone or '', sorted(p.weekdays), p.start_minute))]}

        if self.monthly_defaults:
            result['schema_version']='hot-comfort-schedule/1.1.0'
            result['monthly_defaults']={str(month):band(value) for month,value in sorted(self.monthly_defaults)}
        return result

    @classmethod
    def from_mapping(cls, value: dict) -> 'ComfortSchedule':
        def exact(mapping, keys):
            if not isinstance(mapping, dict) or set(mapping) != set(keys):
                raise ValueError(f'expected exactly these comfort fields: {sorted(keys)}')
        def band(mapping):
            exact(mapping, ('minimum_c', 'maximum_c'))
            return ComfortBand(**mapping)
        monthly = isinstance(value,dict) and value.get('schema_version')=='hot-comfort-schedule/1.1.0'
        exact(value, ('schema_version', 'default', 'zone_defaults', 'periods') + (('monthly_defaults',) if monthly else ()))
        if value['schema_version'] not in ('hot-comfort-schedule/1.0.0','hot-comfort-schedule/1.1.0'):
            raise ValueError('unsupported comfort schedule schema')
        if not isinstance(value['zone_defaults'], dict) or not isinstance(value['periods'], list):
            raise ValueError('zone_defaults must be an object and periods must be a list')
        periods=[]
        for entry in value['periods']:
            exact(entry, ('weekdays', 'start_minute', 'end_minute', 'band', 'zone'))
            if not isinstance(entry['weekdays'], list):
                raise ValueError('weekdays must be a list')
            periods.append(WeeklyComfortPeriod(tuple(entry['weekdays']), entry['start_minute'],
                                                entry['end_minute'], band(entry['band']), entry['zone']))
        monthly_values=value.get('monthly_defaults',{})
        if not isinstance(monthly_values,dict) or any(k not in {str(i) for i in range(1,13)} for k in monthly_values):
            raise ValueError('monthly default keys must be canonical month strings')
        return cls(band(value['default']), tuple((zone, band(bounds)) for zone,bounds in value['zone_defaults'].items()), tuple(periods), tuple((int(m),band(b)) for m,b in monthly_values.items()))

    def interval_segments(self, zone: str, *, weekday: int, start_minute: float,
                          duration_minutes: float, month: int | None = None) -> tuple[tuple[int, float, float, ComfortBand], ...]:
        """Resolve each constant-band part of an interval, including week rollover.

        Durations are positive and at most one day. Coordinates in each returned
        segment are minutes within that segment's local standard-time day.
        """
        self.band_at(zone, weekday=weekday, minute=start_minute, month=month)
        if (isinstance(duration_minutes, bool) or not isinstance(duration_minutes, (int, float))
                or not math.isfinite(duration_minutes) or not 0 < duration_minutes <= 1440):
            raise ValueError('duration must be finite and in (0,1440] minutes')
        if self.monthly_defaults and start_minute + duration_minutes > 1440:
            raise ValueError('split monthly comfort intervals at midnight using the new calendar month')
        segments=[]
        remaining=float(duration_minutes)
        current=float(start_minute)
        day=weekday
        while remaining > 0:
            end=min(1440.0,current+remaining)
            boundaries={current,end}
            for period in self.periods:
                if day in period.weekdays and period.zone in (None,zone):
                    boundaries.update(point for point in (period.start_minute,period.end_minute) if current < point < end)
            ordered=sorted(boundaries)
            for left,right in zip(ordered,ordered[1:]):
                segments.append((day,left,right,self.band_at(zone,weekday=day,minute=left,month=month)))
            remaining=max(0.0,remaining-(end-current))
            current=0.0
            day=day % 7+1
        return tuple(segments)

    def occupied_exposure(self, zone: str, *, temperature_c: float, occupants: float,
                          weekday: int, start_minute: float, duration_minutes: float, month: int | None = None) -> float:
        """Person·°C²·hours for constant sampled temperature/count over an interval.

        This splits scoring bounds, not physical simulator samples. It makes no
        claim to recover sub-interval temperature or occupancy variation.
        """
        if isinstance(occupants,bool) or not isinstance(occupants,(int,float)) or not math.isfinite(occupants) or occupants < 0:
            raise ValueError('occupants must be finite and nonnegative')
        segments=self.interval_segments(zone,weekday=weekday,start_minute=start_minute,duration_minutes=duration_minutes,month=month)
        return math.fsum(occupants*band.violation_c(temperature_c)**2*(right-left)/60
                         for _,left,right,band in segments)
