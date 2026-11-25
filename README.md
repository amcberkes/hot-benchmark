# HOT transfer benchmark

Run HVAC controller transfer experiments with the [HOT dataset](https://huggingface.co/datasets/BuildingBench/HOT). Train a policy in one building, deploy it in another, and compare it with the controller already operating there. The supplied experiments vary building type, climate, occupancy schedule and thermal resistance.

## Run a transfer experiment

The example below trains one policy in the default Fairbanks office and evaluates it in Seattle retail with an earlier occupancy schedule and higher thermal resistance. It also runs the target’s historical controller on the same weeks.

Install EnergyPlus 24.1.0 (build `9d7789a3ac`), Git and Python 3.12. From the repository root:

```sh
python3.12 -m venv ../hot-sim-env
. ../hot-sim-env/bin/activate
python -m pip install -r simulation/requirements.txt
export ENERGYPLUS_PATH=/path/to/EnergyPlus-24-1-0
export PYTHONPATH="$ENERGYPLUS_PATH${PYTHONPATH:+:$PYTHONPATH}"
python -m simulation.fetch ../hot-inputs
python -m simulation.run all --source OF-00 --target RS-11 --seed 0 --calendar test \
  --data ../hot-inputs --work ../hot-example
```

The input builder downloads public building templates and weather files with `curl` and verifies their checksums. Add `--plan` to the last command to inspect the training and evaluation steps. The simulation guide explains the task codes, how to reuse a trained policy on another target, change the evaluation weeks, or run the full matrix. [Simulation guide](simulation/README.md)

## Reproduce the HOT follow-up study

The follow-up study evaluated six source tasks on sixteen targets with ten training seeds per source. The repository includes every seed’s validation and test outcomes, each target’s historical reference, and the rules for selecting a source. A transfer succeeds when it uses less HVAC energy per square metre and causes no greater occupied discomfort than the historical controller.

To regenerate the results offline, install the regular and bold TeX Gyre Pagella fonts and create a separate Python environment. [Plot fonts](https://ctan.org/pkg/tex-gyre)

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --out results
```

A fresh output directory contains 13 CSV tables at the published precision and three PDF figures, all computed from the follow-up study’s stored outcomes.

Use `--tables-only` for CSV output or `--details` to retain the numerical summary and intermediate measurements. To check the table contents and vector drawings against a local copy of the thesis, install `PyMuPDF` and run `python -m analysis.verify /path/to/thesis.pdf results`. The simulation guide also covers re-auditing original five-minute trajectories and archive access.

## Dataset and citation

A. Berkes, Y. Bengio, D. Rolnick and D. Vakalis, *A HOT Dataset: 150,000 Buildings for HVAC Operations Transfer Research*, BuildSys 2025. [Paper](https://doi.org/10.1145/3736425.3770110) · [Dataset revision used here](https://huggingface.co/datasets/BuildingBench/HOT/tree/06e84189e6c1e196f689dee26acd49533aa5f3af) · [Dataset generator](https://github.com/RolnickLab/Building-Generator/tree/4afcdc5fed026f2bc5d5310f3c31be42fd23aafa)
