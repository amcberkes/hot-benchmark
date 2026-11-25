"""Weather, schedule and similarity diagnostics for the stored experiment."""

import json
import numpy as np

from .matrix import read_csv, write_csv


def diagnostics(data, out):

    def load(name):
        return json.loads((data / name).read_text())

    weekly = load("schedules.json")
    weeks = np.array(list(weekly.values()))
    days = weeks.reshape(-1, 7, 24).mean(axis=1)
    hours = (weeks > 0.1).sum(axis=1)
    peaks = weeks.max(axis=1)
    weather = read_csv(data / "weather.csv")
    coeff = load("coefficients.json")
    # On the diagnostic Cartesian grid each component can be integrated on its own axis.
    diagnostic = load("sensitivity.json")
    scales = diagnostic["scales"]
    geom = np.array(
        [[float(row[k]) for k in scales["geometry_scales"]] for row in read_csv(data / "geometry.csv")]
    )
    norm = np.linalg.norm(days, axis=1)
    cosine = (days @ days.T) / np.outer(norm, norm)

    def distance(v):
        return np.abs(v[:, None] - v[None, :])

    resistance = np.array([0.5, 1, 2])
    climate = distance(np.array(diagnostic["weather"]))
    matrices = dict(
        geometry=np.exp(-(distance(geom) / list(scales["geometry_scales"].values())).sum(axis=2)),
        thermal=np.exp(-distance(resistance) - distance(1 / resistance)),
        climate=np.exp(
            -((climate[:, :, :2] / 1000) ** 2).sum(axis=2) - (climate[:, :, 2:] / 10).sum(axis=2)
        ),
        function=np.exp(-distance(hours) / 168 - distance(peaks)) * cosine,
    )
    variances = {name: v.var() for name, v in matrices.items()}
    delta = np.abs(np.subtract(*diagnostic["seattle"]))
    delta[:2] **= 2
    write_csv(
        out / "schedule-cosines.csv",
        ["schedule", *weekly],
        [(name, *row) for name, row in zip(weekly, cosine)],
    )
    return dict(
        variance_share_percent={k: 100 * v / sum(variances.values()) for k, v in variances.items()},
        seattle_climate_scores=[
            np.exp(-np.dot(delta, w)) for w in [coeff["climate_alpha"], [1e-6, 1e-6, 0.1, 0.1]]
        ],
        weather_ranges={
            k: [min(float(r[k]) for r in weather), max(float(r[k]) for r in weather)]
            for k in ("hdd", "cdd", "tmax", "tmin")
        },
    )
