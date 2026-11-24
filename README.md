# HOT transfer experiments

Code and measurements for Chapter 3 and Appendix A of the thesis. Six source tasks, ten training seeds per source and sixteen target tasks give 960 policy evaluations on each of two calendars. Each evaluation compares a policy with the target building’s historical controller.

## Reproduce the tables and figures

From the repository root, with Python 3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --out results
```

The command writes 24 numbered CSV tables, nine PDF figures and a numerical summary. Tables run offline with `--tables-only`. Drawing all figures requires `pdflatex` with TikZ, `standalone` and either `newpx` or `pxfonts`. The two maps use four checksum-verified Natural Earth layers, downloaded with `curl` on first use. To reuse them offline, pass `--maps /path/to/natural-earth`.

| Exhibits | Source |
| --- | --- |
| Tables 3.1–3.4, A.1–A.2 and Figures 3.1–3.3, A.1 | Dataset inventory, geometry, schedules, weather descriptors and publication inputs |
| Figures 3.4–3.5 | Redraws of the HOT paper’s weather-transfer scores and variability comparison |
| Tables 3.5–3.6 and A.3–A.4 | Task design, observations, actions and PPO settings |
| Tables 3.7, 3.9, A.5–A.13 and Figures 3.6–3.7, A.2 | Seed outcomes and target-specific historical controls |
| Tables 3.8, A.14–A.15 | Task descriptors, similarity coefficients, validation-based source selection and test outcomes |

For the published matrix, 406 of 960 test deployments use less HVAC energy and have no greater occupied discomfort. On the ten targets outside the source set, HOT source selection meets both criteria in 82% of deployments and climate matching in 61%. The paired difference is 21 percentage points (95% interval 12.75–30.50).

If you have the thesis PDF, install `PyMuPDF` in the analysis environment and run `python -m analysis.verify /path/to/thesis.pdf results`. This checks the numeric labels and their positions in all nine figures and all 864 printed entries in Tables A.5–A.10.

## Read the measurements

`data/outcomes.csv` contains one row per calendar, source, target and training seed. `data/historical.csv` contains one reference per calendar and target. Both store HVAC energy in kWh/m², summed squared temperature-band violations in person·°C²·h, and person-hours. Divide the second total by the third to obtain occupied discomfort in °C². Validation covers days 1–7 of each month and test covers days 15–21.

Task codes give building (O = office, R = retail), climate (F = Fairbanks, S = Seattle), occupancy and envelope. For example, `OS-10` is the Seattle office with occupancy shifted two hours earlier and the default envelope. The suffix `01` selects higher thermal resistance and `11` applies both changes.

`contexts.json` and `coefficients.json` supply the geometry, thermal, climate and function terms of the HOT score. The analysis recomputes source choices and checks them against `frozen-selection.csv`. Tied sources share weight equally. Confidence intervals resample the ten training seeds 20,000 times with a fixed random seed. Figure A.2 uses a Bonferroni adjustment across two outcomes for six predeclared transfers. The analysis also checks the 22 source–target pairs in `calendar-disagreements.csv`. CSV outputs retain the precision used for calculation.

## Re-audit or rerun

The simulation guide gives commands to remeasure an archived trajectory, reconstruct all 17 physical inputs from public templates and versioned settings, train a policy, and run the full matrix or a smaller experiment. Re-auditing original trajectories requires access to the archives. Fresh simulation uses public inputs. [Simulation guide](simulation/README.md)

To recompute the four weather descriptors for all 192 weather files, run `python -m analysis.weather ../hot-weather`. The first run downloads the pinned EPWs. [HOT dataset](https://huggingface.co/datasets/BuildingBench/HOT/tree/06e84189e6c1e196f689dee26acd49533aa5f3af)

The dataset and its generation pipeline accompany A. Berkes, Y. Bengio, D. Rolnick and D. Vakalis, *A HOT Dataset: 150,000 Buildings for HVAC Operations Transfer Research*, BuildSys 2025. [Paper](https://doi.org/10.1145/3736425.3770110) · [Dataset generator](https://github.com/RolnickLab/Building-Generator/tree/4afcdc5fed026f2bc5d5310f3c31be42fd23aafa)
