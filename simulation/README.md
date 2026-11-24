# EnergyPlus experiments

Run these commands from the repository root. Use a separate Python environment for the simulator and the figure renderer.

## Audit an original evaluation

A local archive can be checked with Python 3.12 alone:

```sh
python -m simulation.audit test OF-00 OF-00 0 --archive /path/to/run.tar.gz
```

Arguments are calendar, source, target and seed. Use `historical` as the source and seed `0` for a target’s reference controller. The audit checks the compressed archive and trajectory hashes, remeasures energy and occupied exposure for all 24,192 five-minute intervals, checks clocks and simulator diagnostics, and reconciles the totals with the stored outcome. For trained policies it also compares each commanded setpoint with its actuator readback.

`archives.csv` lists all 1,920 policy and 32 historical evaluations. All policy archives, all 16 test references and seven validation references have pinned locations in the access-controlled Hugging Face archive. With access and the `huggingface-hub` package installed, omit `--archive` to download a case. For the eight office validation references and the `RS-11` validation reference, the manifest contains checksums only. Supply the matching archive to re-audit one of those cases, or rerun its reference as below.

## Install and build the inputs

Install the EnergyPlus 24.1.0 release for your platform (build `9d7789a3ac`). Set `ENERGYPLUS_PATH` to the directory containing the executable and `pyenergyplus`. The commands below create sibling directories for the simulator environment, inputs and outputs. [EnergyPlus 24.1.0](https://github.com/NREL/EnergyPlus/releases/tag/v24.1.0)

```sh
python3.12 -m venv ../hot-sim-env
. ../hot-sim-env/bin/activate
python -m pip install -r simulation/requirements.txt
export ENERGYPLUS_PATH=/path/to/EnergyPlus-24-1-0
export PYTHONPATH="$ENERGYPLUS_PATH${PYTHONPATH:+:$PYTHONPATH}"
python -m simulation.fetch ../hot-inputs
python -m simulation.fetch ../hot-inputs --check
```

The input builder uses `curl` to retrieve two public Seattle building templates and the Fairbanks and Seattle weather files. It applies the historical meter definitions, reporting and solver settings, five-minute timestep, and occupancy and thermal variants. The six source-specific reward calibrations are versioned under `config/reward/`. Each resulting file must match the original SHA-256 in `assets.json` before a run can start.

## Run the thesis protocol

For a single source, target, seed and calendar:

```sh
python -m simulation.run all --source OF-00 --target RS-11 --seed 0 --calendar test \
  --data ../hot-inputs --work ../hot-example
```

Append `--plan` to print commands without executing them. The four stages run in order:

| Stage | Work performed |
| --- | --- |
| `teacher` | Run four source-local demonstration weeks, audit the trajectories and export the examples. |
| `train` | Fit a seed-specific shared-zone actor, then run CPU PPO for 483,840 transitions. Verify the training calendar, simulator diagnostics and checkpoint hashes. |
| `native` | Run the target’s historical controller over all twelve weeks of the selected calendar. |
| `policies` | Evaluate the final checkpoint on the same target and weeks. Record physical outcomes and actuator readbacks. |

Each stage writes its own directory and refuses to overwrite an existing case. Stages can be invoked separately. For example, after the run above, use `native --target OS-10 --calendar test` and then `policies --source OF-00 --target OS-10 --seed 0 --calendar test` with the same `--data` and `--work` to evaluate the trained policy on another target. Run from a clean Git checkout so that the receipts identify the source revision.

For the full matrix use a fresh work directory. This trains 60 policies and runs 1,952 calendar evaluations. Reserve storage for checkpoints, EnergyPlus working files and five-minute trajectories. Results from another platform or numerical library build can differ from the archived outcomes.

```sh
python -m simulation.run all --data ../hot-inputs --work ../hot-full
python -m simulation.collect ../hot-full ../hot-measurements
# Use the analysis environment from the repository README for plotting.
. .venv/bin/activate
python reproduce.py --measurements ../hot-measurements --out ../hot-rerun-results
```

## Change the experiment

The JSON suites specify building and weather hashes, controlled zones, observations, actions, rewards and weekly windows. To try another evaluation week, copy `simulation/config/test.json`, edit the `evaluation_protocol.windows` of `native_office_c0_o0_t0`, then run the historical controller:

```sh
python -m building_gen.cli benchmark run-baselines --suite ../my-suite.json \
  --experiment native_office_c0_o0_t0 --data-root ../hot-inputs \
  --work-dir ../my-week/work --output-dir ../my-week/reference
```

For a trained policy, run `python -m building_gen.cli evaluate-baseline --help`. Use the same suite and protocol, supply the source and target task names, and point `--run-dir` at a checkpoint directory. The low-level commands write result JSON, trajectory CSV and simulator diagnostics for custom designs. The numbered thesis tables expect the original six-source, sixteen-target, two-calendar design.
