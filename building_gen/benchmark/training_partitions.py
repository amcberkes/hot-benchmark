"""Explicit calendar partitions for training with held-out evaluation dates."""
from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from collections.abc import Sequence

from building_gen.benchmark.specs import CalendarWindowSpec


def eligible_training_windows(*, episode_length_hours: int,
                              reserved_windows: Sequence[CalendarWindowSpec],
                              calendar_year: int | None = None) -> tuple[CalendarWindowSpec, ...]:
    """Enumerate within-month training windows disjoint from all reserved dates.

    Endpoints are inclusive. An episode is excluded if any day it covers is
    reserved, not merely when its start is reserved. This is an explicit pool
    builder; the historical month-stratified training schedule is unchanged.
    """
    if type(episode_length_hours) is not int or episode_length_hours <= 0 or episode_length_hours % 24:
        raise ValueError('training episode length must be a positive multiple of 24 hours')
    if episode_length_hours > 28 * 24:
        raise ValueError('within-month training episodes cannot exceed 28 days')
    # Use the same calendar validation and representative year as TaskSpec windows.
    first, _ = CalendarWindowSpec('calendar_check', 1, 1, 1, 1).date_bounds_for_calendar_year(calendar_year)
    year = first.year
    reserved = []
    for window in reserved_windows:
        if not isinstance(window, CalendarWindowSpec):
            raise TypeError('reserved dates must be CalendarWindowSpec values')
        reserved.append(window.date_bounds_for_calendar_year(calendar_year))
    duration = episode_length_hours // 24
    windows = []
    for month in range(1, 13):
        for day in range(1, monthrange(year, month)[1] - duration + 2):
            start = date(year, month, day)
            end = start + timedelta(days=duration - 1)
            if any(max(start, left) <= min(end, right) for left, right in reserved):
                continue
            windows.append(CalendarWindowSpec(
                f'train-m{month:02d}d{day:02d}-{duration}d', month, day, end.month, end.day))
    if not windows:
        raise ValueError('reserved dates leave no eligible training episodes')
    return tuple(windows)


class PartitionedCalendarSchedule:
    """Seeded random-access sampling from an explicit eligible window pool.

    Each block visits every eligible month once. Starts within a month are
    permuted without replacement before repetition. Entirely reserved months
    are absent; the pool identity records that change in training distribution.
    """
    strategy_id = 'reserved-date-month-stratified-v1'

    def __init__(self, *, seed: int, episode_length_hours: int,
                 reserved_windows: Sequence[CalendarWindowSpec], calendar_year: int | None = None):
        import hashlib
        import json
        if type(seed) is not int or seed < 0:
            raise ValueError('schedule seed must be a nonnegative integer')
        self.seed = seed
        self.calendar_year = calendar_year
        self.windows = eligible_training_windows(episode_length_hours=episode_length_hours,
                                                reserved_windows=reserved_windows, calendar_year=calendar_year)
        self._months = tuple(sorted({window.start_month for window in self.windows}))
        self._by_month = {month: tuple(window for window in self.windows if window.start_month == month)
                          for month in self._months}
        self._identity = {
            'strategy_id': self.strategy_id, 'seed': seed, 'calendar_year': calendar_year,
            'episode_length_hours': episode_length_hours,
            'eligible_windows': [[w.start_month, w.start_day, w.end_month, w.end_day] for w in self.windows],
        }
        payload = json.dumps(self._identity, sort_keys=True, separators=(',', ':')).encode()
        self.pool_sha256 = hashlib.sha256(payload).hexdigest()

    def identity_payload(self) -> dict:
        from copy import deepcopy
        return {**deepcopy(self._identity), 'pool_sha256': self.pool_sha256}

    def _digest(self, *parts) -> bytes:
        import hashlib
        import json
        return hashlib.sha256(json.dumps([self.pool_sha256, *parts], separators=(',', ':')).encode()).digest()

    def sample(self, episode_index: int) -> CalendarWindowSpec:
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError('episode index must be a nonnegative integer')
        block, position = divmod(episode_index, len(self._months))
        months = sorted(self._months, key=lambda month: (self._digest('month', block, month), month))
        month = months[position]
        windows = self._by_month[month]
        cycle, day_position = divmod(block, len(windows))
        order = sorted(range(len(windows)), key=lambda index: (self._digest('day', month, cycle, index), index))
        return windows[order[day_position]]

    def child_seed(self, episode_index: int) -> int:
        # Reuse index validation without storing or advancing mutable RNG state.
        self.sample(episode_index)
        return int.from_bytes(self._digest('child', episode_index), 'big') % (2**32)


    def training_sample(self, episode_index: int):
        """Adapt the partition schedule to the Gym calendar episode contract."""
        from building_gen.benchmark.training_calendar import CalendarTrainingSample
        window = self.sample(episode_index)
        block = episode_index // len(self._months)
        return CalendarTrainingSample(
            schedule_seed=self.seed, episode_index=episode_index, calendar_block=block,
            month_visit_index=block, day_cycle=block // len(self._by_month[window.start_month]),
            child_seed=self.child_seed(episode_index), window=window,
            strategy_id=self.strategy_id, pool_sha256=self.pool_sha256)
