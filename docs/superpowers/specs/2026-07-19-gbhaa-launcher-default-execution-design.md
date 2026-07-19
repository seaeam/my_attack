# GBHAA Launcher Default Execution Design

## Goal

Make every script under `gbhaa_experiments/launchers/` use the Python interpreter from the currently active environment and execute the formal experiment by default.

## Scope

- Update all seven launcher scripts.
- Update `gbhaa_experiments/README.md` and `gbhaa_experiments/EXPERIMENT_SUPPLEMENT_PLAN_ZH.md` so their commands match the new behavior.
- Do not change experiment matrices, model parameters, datasets, seeds, budgets, result locations, or low-level Python runner interfaces.
- Do not start formal experiments as part of this change.
- Per user instruction, do not add or run unit tests.

## Launcher Behavior

Each launcher resolves Python as follows:

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
```

This uses the interpreter from an active Conda, virtualenv, or PATH-based environment while retaining the existing `PYTHON_BIN=/custom/path` override.

Each launcher defaults to formal execution:

```bash
EXECUTE="${EXECUTE:-1}"
```

Therefore `bash gbhaa_experiments/launchers/<name>.sh` runs the experiment. Setting `EXECUTE=0` remains an explicit, optional command-preview mode. The low-level `run_matrix.py` and `run_stealthiness.py` `--execute` interfaces remain unchanged; launchers supply the flag by default.

## Documentation

- Remove statements saying launchers are dry-run by default.
- Show direct launcher invocation as the normal formal-run command.
- Document `EXECUTE=0` only as an optional preview mechanism.
- Replace hard-coded `/opt/miniconda3/bin/python` examples with `python` or the current-environment `PYTHON_BIN` pattern.
- Remove the machine-specific `/Users/bytedance/Downloads/Github/my_attack` repository path and state that commands run from the repository root.
- Remove redundant `EXECUTE=1` from formal-run examples.

## Validation

Without starting experiments:

- Run `bash -n` over all launcher scripts.
- Search for stale `/opt/miniconda3/bin/python` and `/Users/bytedance/Downloads/Github/my_attack` paths.
- Search for `EXECUTE="${EXECUTE:-0}"` and outdated default-dry-run wording.
- Inspect the final diff for unintended experiment configuration changes.
