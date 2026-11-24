"""Isolated, mass-preserving shifts of explicitly partitioned compact occupancy schedules.

Only People references and new cloned schedules change. Design-day blocks are
preserved. Nonzero occupancy may not cross midnight: such a shift needs a
calendar-aware transformation and is rejected rather than wrapped or truncated.
"""

from copy import deepcopy
import math
import re


def shift_compact(schedule, minutes):
    if type(minutes) is not int or not -1439 <= minutes <= 1439:
        raise ValueError("shift must be integer minutes inside one day")
    data = schedule["data"]
    output = []
    i = 0
    blocks = []
    seen = set()
    ordinary = {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "holiday",
        "customday1",
        "customday2",
    }
    designs = {"summerdesignday", "winterdesignday"}
    expand = {
        "weekdays": {"monday", "tuesday", "wednesday", "thursday", "friday"},
        "weekends": {"saturday", "sunday"},
    }
    while i < len(data):
        item = data[i]
        value = item["field"]
        if isinstance(value, str) and value.lower().startswith("through:"):
            output.append(deepcopy(item))
            seen = set()
            i += 1
            continue
        if not isinstance(value, str) or not value.lower().startswith("for:"):
            raise ValueError("expected explicit Through/For blocks; interpolation is unsupported")
        selector = [
            {"holidays": "holiday"}.get(v, v) for v in value.split(":", 1)[1].strip().lower().split()
        ]
        header = deepcopy(item)
        if "allotherdays" in selector:
            if not designs <= seen:
                raise ValueError("AllOtherDays requires both design days explicitly covered first")
            explicit = set().union(*(expand.get(v, {v}) for v in selector if v != "allotherdays"))
            selector = sorted(explicit | ((ordinary | designs) - seen))
            header["field"] = "For: " + " ".join(selector)
        seen.update(set().union(*(expand.get(v, {v}) for v in selector)))
        design = bool(set(selector) & {"summerdesignday", "winterdesignday"})
        allowed = {
            "weekdays",
            "weekends",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "holiday",
            "customday1",
            "customday2",
        }
        if (
            not selector
            or (design and not set(selector) <= {"summerdesignday", "winterdesignday"})
            or (not design and not set(selector) <= allowed)
        ):
            raise ValueError("design days must be explicitly separated from ordinary days")
        i += 1
        intervals = []
        start = 0
        raw = []
        while (
            i < len(data)
            and isinstance(data[i]["field"], str)
            and data[i]["field"].lower().startswith("until:")
        ):
            match = re.fullmatch(r"Until:\s*(\d{1,2}):(\d{2})", data[i]["field"], re.IGNORECASE)
            if match is None or i + 1 >= len(data):
                raise ValueError("invalid Until pair")
            hour, minute = map(int, match.groups())
            end = hour * 60 + minute
            fraction = data[i + 1]["field"]
            if (
                minute >= 60
                or not start < end <= 1440
                or isinstance(fraction, bool)
                or not isinstance(fraction, (float, int))
                or not math.isfinite(fraction)
                or fraction < 0
            ):
                raise ValueError("invalid interval or occupancy fraction")
            raw.extend(deepcopy(data[i : i + 2]))
            intervals.append((start, end, fraction))
            start = end
            i += 2
        if start != 1440:
            raise ValueError("each day must end at 24:00")
        output.append(header)
        if design:
            output.extend(raw)
            continue
        shifted = []
        for start, end, fraction in intervals:
            a, b = start + minutes, end + minutes
            if fraction and (a < 0 or b > 1440):
                raise ValueError("nonzero occupancy would cross midnight")
            if max(a, 0) < min(b, 1440):
                shifted.append((max(a, 0), min(b, 1440), fraction))
        edges = sorted({0, 1440} | {v for a, b, _ in shifted for v in (a, b)})
        segments = []
        for a, b in zip(edges, edges[1:]):
            fraction = next((v for left, right, v in shifted if left <= a and b <= right), 0.0)
            if segments and segments[-1][1] == fraction:
                segments[-1] = (b, fraction)
            else:
                segments.append((b, fraction))
        before = sum((b - a) * v for a, b, v in intervals)
        after = 0.0
        start = 0
        for end, fraction in segments:
            output.extend([{"field": f"Until: {end//60:02d}:{end%60:02d}"}, {"field": fraction}])
            after += (end - start) * fraction
            start = end
        if not math.isclose(before, after, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("daily occupancy mass changed")
        blocks.append(
            dict(selector=header["field"], fraction_minutes_before=before, fraction_minutes_after=after)
        )
    result = deepcopy(schedule)
    result["data"] = output
    return result, blocks


def occupancy_timing_variant(model, minutes):
    result = deepcopy(model)
    proof = []
    names = sorted({p["number_of_people_schedule_name"] for p in result["People"].values()})
    for old in names:
        if old not in result.get("Schedule:Compact", {}):
            raise ValueError("only compact occupancy schedules are supported")
        new = f"HOT_OCC_SHIFT_{minutes}_{old}"
        if any(
            new.casefold() == k.casefold()
            for kind, objects in result.items()
            if kind.startswith("Schedule:")
            for k in objects
        ):
            raise ValueError("shifted schedule name already exists")
        schedule, blocks = shift_compact(result["Schedule:Compact"][old], minutes)
        result["Schedule:Compact"][new] = schedule
        for people in result["People"].values():
            if people["number_of_people_schedule_name"] == old:
                people["number_of_people_schedule_name"] = new
        proof.append(
            dict(
                source_schedule=old,
                shifted_schedule=new,
                shift_minutes=minutes,
                ordinary_day_blocks=blocks,
                design_days="unchanged",
            )
        )
    return result, proof
