"""Build the frozen simulation inputs from the public HOT templates and weather files."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from .occupancy import occupancy_timing_variant

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config"


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def download(manifest, remote, expected):
    """Retrieve a stored trajectory archive using the reader's Hugging Face credentials."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(manifest["repository"], remote, revision=manifest["revision"], repo_type="dataset")
    )
    if digest(path) != expected:
        raise ValueError(f"Archive SHA-256 mismatch for {remote}")
    return path


def public_file(spec, name, sha):
    url = f"https://huggingface.co/datasets/{spec['repository']}/resolve/{spec['revision']}/{name}"
    raw = subprocess.check_output(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--max-time",
            "120",
            url,
        ]
    )
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Public input SHA-256 mismatch for {name}")
    return raw


def building(template, kind, reporting):
    model = json.loads(template)
    next(iter(model["Building"].values()))["maximum_number_of_warmup_days"] = 200
    next(iter(model["Timestep"].values()))["number_of_timesteps_per_hour"] = 12
    meters = json.loads((CONFIG / f"{kind}-meters.json").read_text())
    model["Meter:Custom"] = meters["Meter:Custom"]
    model["Output:Meter"] = {**reporting["Output:Meter"], **meters["Output:Meter"]}
    model["Output:Variable"] = reporting["Output:Variable"]
    if kind == "office":
        model["Output:VariableDictionary"] = {"HOT diagnostic dictionary": {"key_field": "IDF"}}
    else:
        model["ZoneAirHeatBalanceAlgorithm"] = {
            "HOT diagnostic solver": {"algorithm": "AnalyticalSolution"}
        }
    return model


def higher_resistance(model):
    for section in ("Material", "Material:NoMass"):
        for material in model.get(section, {}).values():
            if "thermal_resistance" in material:
                material["thermal_resistance"] = float(material["thermal_resistance"]) * 2.0
            elif "conductivity" in material:
                if 0 < material.get("thickness", 0) < 0.001:
                    material["thickness"] *= 2.0
                else:
                    material["conductivity"] = max(float(material["conductivity"]) / 2.0, 0.001)


def assets(directory, check=False):
    spec = json.loads((HERE / "assets.json").read_text())
    reporting = json.loads((CONFIG / "reporting.json").read_text())
    models = {}
    for name, sha in spec["files"].items():
        dest = directory / name
        if dest.is_file() or check:
            if digest(dest) != sha:
                raise ValueError(f"Input differs from the registered version: {dest}")
            continue
        if name.endswith(".epw"):
            raw = public_file(spec, "weather/base/" + name, sha)
        elif name.endswith(".json"):
            raw = (CONFIG / "reward" / name).read_bytes()
        else:
            kind = "office" if "office" in name else "retail"
            if kind not in models:
                ref = spec["templates"][kind]
                models[kind] = building(public_file(spec, ref["path"], ref["sha256"]), kind, reporting)
            model = deepcopy(models[kind])
            if "_o1_" in name:
                model, _ = occupancy_timing_variant(model, -120)
            if "_t1." in name:
                higher_resistance(model)
            raw = (json.dumps(model, indent=2) + "\n").encode()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"Reconstructed input SHA-256 mismatch: {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as stream:
            tmp = Path(stream.name)
        try:
            tmp.write_bytes(raw)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
    return len(spec["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify local files offline")
    args = parser.parse_args()
    print(f"{assets(args.directory, args.check)} input files verified")
