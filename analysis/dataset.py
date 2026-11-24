"""Dataset descriptors, controller settings and the coefficient diagnostic."""

import json
import re
import shutil
import numpy as np

from .matrix import SOURCES, TARGETS, read_csv, write_csv


def reproduce(root, out):
    data = root / "data"

    def load(name):
        return json.loads((data / name).read_text())

    for i, name in enumerate(("resources", "axes", "geometry"), 1):
        shutil.copyfile(data / f"{name}.csv", out / f"table-3.{i}.csv")
    weekly = load("schedules.json")
    weeks = np.array(list(weekly.values()))
    days = weeks.reshape(-1, 7, 24).mean(axis=1)
    hours = (weeks > 0.1).sum(axis=1)
    peaks = weeks.max(axis=1)
    write_csv(
        out / "table-3.4.csv",
        ["schedule", "weekly_hours", "peak", "variance"],
        zip(weekly, hours, peaks, days.var(axis=1)),
    )
    locations = load("locations.json")
    write_csv(
        out / "table-A.1.csv",
        ["zone", "base", "examples"],
        [
            (z, base[0], ", ".join(re.sub(r" \(.*\)$", "", v[0]) for v in more))
            for z, (base, more) in locations.items()
        ],
    )
    weather = read_csv(data / "weather.csv")
    shown = sorted((r for r in weather if r["table_location"]), key=lambda r: -float(r["hdd"]))
    write_csv(
        out / "table-A.2.csv",
        ["location", "heating_degree_days", "cooling_degree_days", "max_c", "min_c"],
        [
            (r["table_location"], round(float(r["hdd"])), round(float(r["cdd"])), r["tmax"], r["tmin"])
            for r in shown
        ],
    )
    config = root / "simulation/config"
    ppo = json.loads((config / "ppo.json").read_text())
    suite = json.loads((config / "test.json").read_text())
    task = suite["tasks"][0]
    frequency = task["episode"]["timesteps_per_hour"]
    steps = 483840
    setup = [
        ("targets", len(TARGETS)),
        ("sources", len(SOURCES)),
        ("source_tasks", " ".join(SOURCES)),
        ("target_tasks", " ".join(TARGETS)),
        ("pairs", len(SOURCES) * len(TARGETS)),
        ("controller", ppo["policy_architecture"]),
        ("training_steps", steps),
        ("control_minutes", 60 // frequency),
        ("reference", "target historical controller"),
        ("outcomes", "HVAC kWh/m² and occupied discomfort in °C²"),
        ("success", "lower energy and no greater discomfort"),
        ("training_seeds", 10),
        ("validation", "days 1–7 each month"),
        ("test", "days 15–21 each month"),
    ]
    write_csv(out / "table-3.5.csv", ["setting", "value"], setup)
    write_csv(
        out / "table-3.6.csv",
        ["channel", "count", "quantity"],
        [
            ("zone", "5n", "temperature, people, occupied flag and two comfort limits"),
            ("outdoor", "2", "dry-bulb temperature and relative humidity"),
            ("HVAC", "2", "electricity and gas power per controlled area"),
            ("calendar", "7", "hour, weekday, day of year and two pairs of cyclic features"),
            ("heating", "n", "h = 12 + 12a °C"),
            ("cooling", "n", "c = h + 2 + (30 - h)b °C"),
        ],
    )
    write_csv(
        out / "table-A.3.csv",
        ["inputs", "count", "encoding"],
        [
            ("temperature and band", "3n", "clip (T-10)/30 to [0,1]"),
            ("occupancy", "2n", "N/(N+10) and N>0"),
            ("outdoors", "2", "(T+30)/80 and RH/100, bounded"),
            ("HVAC", "2", "p/(p+0.05), p in kW/m²"),
            ("calendar", "7", "normalized hour, weekday, year day and sine/cosine of hour and weekday"),
        ],
    )
    write_csv(
        out / "table-A.4.csv",
        ["setting", "value"],
        [
            ("control_minutes", 60 // frequency),
            *(
                (k, ppo[k])
                for k in (
                    "learning_rate",
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "gamma",
                    "gae_lambda",
                    "ent_coef",
                )
            ),
            ("log_std_init", ppo["policy_kwargs"]["log_std_init"]),
            ("training_reward_scale", ppo["training_reward_scale"]),
            ("training_steps", steps),
            ("sampled_weeks", steps // (168 * frequency)),
            ("rollout_batches", steps // ppo["n_steps"]),
        ],
    )
    coeff = load("coefficients.json")
    features = {
        "geometry_eta": ["floor area", "zones", "window-to-wall percentage", "height"],
        "thermal_beta": ["resistance multiplier", "normalized wall transmittance"],
        "climate_alpha": ["HDD squared", "CDD squared", "maximum temperature", "minimum temperature"],
        "function_delta": ["weekly operating hours", "relative peak"],
        "weights": ["geometry", "thermal", "climate", "function"],
    }
    rows = [
        (axis, name, w, (w**-0.5 if "squared" in name else 1 / w) if axis != "weights" else "")
        for axis, fs in features.items()
        for name, w in zip(fs, coeff[axis], strict=True)
    ]
    write_csv(out / "table-A.14.csv", ["component", "feature", "coefficient", "reference_difference"], rows)
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
