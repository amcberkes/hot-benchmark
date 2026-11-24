"""Paired transfer statistics for the two monthly calendars."""

import csv
import json
from itertools import product

import numpy as np

SOURCES = ("OF-00", "OF-01", "OF-10", "OS-00", "RF-00", "RS-00")
TARGETS = tuple(
    f"{city}-{occ}{env}" for city, occ, env in product(("OF", "OS", "RF", "RS"), (0, 1), (0, 1))
)
GROUPS = ("Building type and climate", "Climate only", "Building type only", "Neither")
RULES = (
    "HOT similarity score",
    "Climate matching",
    "Random choice",
    "Best single source, validation weeks",
    "Keep the historical controller",
    "Best choice in hindsight",
)
CONTRASTS = ("OF-01", "OF-10", "OS-00", "RF-00", "RS-00", "RS-11")
DRAWS = np.random.default_rng(20261113).integers(0, 10, (20000, 10))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def interval(values, alpha=0.05):
    draws = values[DRAWS].mean(axis=1)
    return [float(values.mean()), *np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist()]


def load_endpoints(data):
    """Index the full calendar × source × target × seed grid explicitly."""
    policies = np.full((2, len(SOURCES), len(TARGETS), 10, 2), np.nan)
    references = np.full((2, len(TARGETS), 2), np.nan)
    for file in ("historical", "outcomes"):
        for row in read_csv(data / f"{file}.csv"):
            cal = ("validation", "test").index(row["calendar"])
            target = TARGETS.index(row["target"])
            if file == "outcomes":
                seed = int(row["seed"])
                if seed not in range(10):
                    raise ValueError(f"Unexpected seed {seed}")
                slot = policies[cal, SOURCES.index(row["source"]), target, seed]
            else:
                slot = references[cal, target]
            if np.isfinite(slot).any():
                raise ValueError(f"Duplicate {file} record: {row}")
            energy, exposure, people = (
                float(row[k]) for k in ("energy_kwh_m2", "squared_exposure_person_c2_h", "person_hours")
            )
            if not all(np.isfinite([energy, exposure, people])) or min(energy, exposure) < 0 or people <= 0:
                raise ValueError(f"Invalid physical totals: {row}")
            slot[:] = energy, exposure / people
    if not np.isfinite(policies).all() or not np.isfinite(references).all() or (references <= 0).any():
        raise ValueError("Incomplete grid or undefined historical denominator")
    return policies, references


def similarity(left, right, coeff):
    components = []
    for field, key in [
        ("geometry", "geometry_eta"),
        ("thermal", "thermal_beta"),
        ("climate", "climate_alpha"),
    ]:
        difference = np.abs(np.subtract(left[field], right[field]))
        if field == "climate":
            difference[:2] **= 2
        components.append(np.exp(-np.dot(difference, coeff[key])))
    h, p, x = left["function"]
    j, q, y = right["function"]
    cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    components.append(np.exp(-np.dot([abs(h - j), abs(p - q)], coeff["function_delta"])) * cosine)
    return np.dot(coeff["weights"], components), components[2]


def selection(data, out, policies, references, archived):
    contexts = json.loads((data / "contexts.json").read_text())
    coeff = json.loads((data / "coefficients.json").read_text())
    fit = []
    for scores in policies[0]:
        feasible = scores[:, :, 1] <= references[0, :, None, 1]
        relative = scores[:, :, 0] / references[0, :, None, 0]
        fit.append((-feasible.sum(), relative[feasible].mean() if feasible.any() else np.inf))
    fixed = [s for s, score in enumerate(fit) if score == min(fit)]
    choices, records = {}, []
    for t, target in enumerate(TARGETS):
        scores = np.array([similarity(contexts[source], contexts[target], coeff) for source in SOURCES])
        hot, climate = [np.flatnonzero(col == col.max()).tolist() for col in scores.T]
        choices[target] = tuple(
            " ".join(SOURCES[s] for s in sorted(rule)) for rule in (hot, climate, fixed)
        )
        for seed in range(10):
            base = references[1, t]
            outcomes = np.vstack([policies[1, :, t, seed], base])
            energy, comfort = outcomes.T
            feasible = comfort <= base[1]
            best = min(energy[feasible])
            oracle = np.flatnonzero(feasible & (energy == best))
            for rule, selected in zip(
                RULES, (hot, climate, range(len(SOURCES)), fixed, [len(SOURCES)], oracle)
            ):
                kept = feasible[selected]
                excess = 100 * (energy[selected] - best) / base[0]
                records.append(
                    (
                        target,
                        seed,
                        rule,
                        kept.mean(),
                        (kept & (energy[selected] < base[0])).mean(),
                        (kept * excess).mean(),
                    )
                )
    expected = {
        r["target"]: (r["hot"], r["climate"], r["fixed"]) for r in read_csv(data / "frozen-selection.csv")
    }
    if any(choices[t][:2] != expected[t][:2] for t in TARGETS) or archived and choices != expected:
        raise ValueError("Derived choices disagree with the frozen validation and descriptor rules")
    write_csv(
        out / "selector-choices.csv",
        ["target", "HOT", "climate", "fixed"],
        [(t, *v) for t, v in choices.items()],
    )
    head = ["rule", "comfort_percent", "joint_percent", "excess_percent_of_historical"]
    rows, main, vectors = [], [], []
    for name, targets in [
        ("non-source", set(TARGETS) - set(SOURCES)),
        ("source", set(SOURCES)),
        ("all", set(TARGETS)),
    ]:
        for rule in RULES:
            values = np.array(
                [[seed, f, j, e] for t, seed, r, f, j, e in records if t in targets and r == rule]
            )
            _, f, joint, excess = values.T
            row = [rule, 100 * f.mean(), 100 * joint.mean(), excess.sum() / f.sum() if f.any() else ""]
            rows.append([name, *row])
            if name == "non-source":
                vectors.append(np.array([100 * joint[values[:, 0] == k].mean() for k in range(10)]))
                if rule != RULES[4]:
                    main.append([rule, row[2], row[1], row[3]])
    write_csv(out / "table-A.15.csv", ["targets", *head], rows)
    write_csv(out / "table-3.8.csv", [head[0], head[2], head[1], head[3]], main)
    return {rule: interval(vectors[0] - v) for rule, v in zip(RULES[1:3], vectors[1:3])}


def reproduce(data, out, measurements=None):
    archived = measurements is None
    policies, refs = load_endpoints(data if archived else measurements)
    diff = policies - refs[:, None, :, None, :]
    pct = 100 * diff / refs[:, None, :, None, :]
    means = pct.mean(axis=3)
    comfort = diff[..., 1] <= 0
    joint = (diff[..., 0] < 0) & comfort
    wins, kept = joint.sum(axis=3), comfort.sum(axis=3)
    group = np.array([[2 * (s[1] != t[1]) + (s[0] != t[0]) for t in TARGETS] for s in SOURCES])
    for cal, table in enumerate(("A.12", "3.7")):
        rows = []
        for g, name in enumerate(GROUPS):
            mask = group == g
            rows.append(
                [
                    name,
                    mask.sum(),
                    wins[cal][mask].mean(),
                    kept[cal][mask].mean(),
                    *means[cal][mask].mean(axis=0),
                ]
            )
        write_csv(
            out / f"table-{table}.csv",
            ["group", "pairs", "joint_of_10", "comfort_of_10", "energy_pct", "discomfort_pct"],
            rows,
        )
    for table, s in enumerate(("OF-00", "OF-10", "OF-01", "OS-00", "RF-00", "RS-00"), 5):
        i = SOURCES.index(s)
        rows = [
            [
                t,
                *interval(pct[1, i, j, :, 0]),
                *interval(pct[1, i, j, :, 1]),
                diff[1, i, j, :, 1].mean(),
                wins[1, i, j],
                kept[1, i, j],
            ]
            for j, t in enumerate(TARGETS)
        ]
        write_csv(
            out / f"table-A.{table}.csv",
            [
                "target",
                "energy_pct",
                "lower_E",
                "upper_E",
                "discomfort_pct",
                "lower_C",
                "upper_C",
                "discomfort_difference_c2",
                "joint_of_10",
                "comfort_of_10",
            ],
            rows,
        )
    write_csv(out / "table-A.11.csv", ["target", *SOURCES], zip(TARGETS, *wins[0]))

    def signs(a):
        return np.where(np.abs(a) <= 1e-10, 0, np.sign(a))

    avg = diff.mean(axis=3)
    flips = signs(avg[0]) != signs(avg[1])
    verdict = (avg[..., 0] < 0) & (avg[..., 1] <= 0)
    changed = {}
    for i, s in enumerate(SOURCES):
        for j, t in enumerate(TARGETS):
            flags = (*flips[i, j], verdict[0, i, j] != verdict[1, i, j])
            reasons = [
                key for key, flag in zip(("energy sign", "discomfort sign", "joint verdict"), flags) if flag
            ]
            if reasons:
                changed[s, t] = ", ".join(reasons)
    roster = {(r["source"], r["target"]) for r in read_csv(data / "calendar-disagreements.csv")}
    if archived and changed.keys() != roster:
        raise ValueError("Calendar comparisons disagree with the archived roster")
    write_csv(
        out / "table-A.13.csv",
        ["source", "target", "what_changes"],
        [(s, t, v) for (s, t), v in changed.items()],
    )
    i = SOURCES.index("RF-00")
    rows = []
    for t in ("RF-00", "RS-00", "OS-00", "OF-00"):
        j = TARGETS.index(t)
        rows.append(
            [t, *means[:, i, j, 0], *means[:, i, j, 1], *wins[:, i, j], ("RF-00", t) not in changed]
        )
    write_csv(
        out / "table-3.9.csv",
        [
            "target",
            "energy_val_pct",
            "energy_test_pct",
            "comfort_val_pct",
            "comfort_test_pct",
            "joint_val",
            "joint_test",
            "agree",
        ],
        rows,
    )
    contrasts = [
        [
            t,
            *interval(diff[1, 0, TARGETS.index(t), :, 0], 0.05 / 12),
            *interval(diff[1, 0, TARGETS.index(t), :, 1], 0.05 / 12),
        ]
        for t in CONTRASTS
    ]
    write_csv(
        out / "figure-A.2.csv",
        ["target", "energy", "lower_E", "upper_E", "discomfort", "lower_C", "upper_C"],
        contrasts,
    )
    write_csv(
        out / "transfer-matrix.csv",
        ["source", "target", "joint_of_10", "comfort_of_10", "energy_pct", "discomfort_pct"],
        [
            (s, t, wins[1, i, j], kept[1, i, j], *means[1, i, j])
            for i, s in enumerate(SOURCES)
            for j, t in enumerate(TARGETS)
        ],
    )
    axes = np.array([[sum(a != b for a, b in zip(s, t)) for t in TARGETS] for s in SOURCES])
    within = group == 0
    within_variants = {
        label: float(
            wins[1][
                within
                & np.array(
                    [
                        [tuple(a != b for a, b in zip(s[-2:], t[-2:])) == pattern for t in TARGETS]
                        for s in SOURCES
                    ]
                )
            ].mean()
        )
        for pattern, label in [
            ((False, False), "own task"),
            ((True, False), "occupancy only"),
            ((False, True), "envelope only"),
            ((True, True), "both"),
        ]
    }
    return dict(
        test_joint=int(wins[1].sum()),
        mean_joint_within_type_and_climate=within_variants,
        validation_joint=int(wins[0].sum()),
        discordant_pairs=len(changed),
        discordant_targets=len({t for s, t in changed}),
        discordant_across_climates=sum(s[1] != t[1] for s, t in changed),
        mean_joint_by_axes_changed={int(k): float(wins[1][axes == k].mean()) for k in np.unique(axes)},
        hot_advantage_pp=selection(data, out, policies, refs, archived),
    )
