"""Check scheduled comfort bounds and applied thermostat commands."""

import json
import math


def actuator_readbacks(rows, zones):
    count = 0
    largest = 0.0
    for row in rows:
        if (
            row["actuator_readback_status"] != "available"
            or row["actuator_readback_phase"] != "end_zone_timestep"
        ):
            raise ValueError("Missing applied actuator readback")
        requested = json.loads(row["submitted_action_physical_json"])
        actual = json.loads(row["actuator_readback_physical_json"])
        if [v["zone"] for v in requested] != list(zones) or len(actual) != len(requested):
            raise ValueError("Wrong controlled-zone coverage")
        for a, b in zip(actual, requested):
            if a["zone"] != b["zone"]:
                raise ValueError("Readback zone mismatch")
            for key in ("heating_setpoint_c", "cooling_setpoint_c"):
                if not math.isfinite(a[key]) or not math.isfinite(b[key]):
                    raise ValueError("Nonfinite command/readback")
                error = abs(a[key] - b[key])
                largest = max(largest, error)
                if error > 1e-6:
                    raise ValueError("Applied setpoint differs from command")
        count += 1
    return count, largest


def verify_segment_bands(segments, schedule, month):
    for segment in segments:
        band = schedule.band_at(
            segment["zone_id"],
            weekday=segment["weekday"],
            minute=(segment["start_minute"] + segment["end_minute"]) / 2,
            month=month,
        )
        if (segment["minimum_c"], segment["maximum_c"]) != (band.minimum_c, band.maximum_c):
            raise ValueError("recorded comfort band differs from declared scoring schedule")
