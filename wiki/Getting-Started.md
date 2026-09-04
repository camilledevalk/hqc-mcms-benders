# Getting Started
This is a quick starter guide to running the code on this branch, which is trimmed down to reproducing the experiments in the IEEE paper. (The full development setup — including the frontend demo, Docker image, and CI — lives on the `research`/`dev` branches instead.)

## Setup

### 1. Clone the repository:
```sh
git clone git@bitbucket.org:capgemini-quantum-lab/logistiqs.git
```

### 2. Install Python dependencies
Our solver is written in Python. Dependencies are pinned exactly via [uv](https://docs.astral.sh/uv/) to keep the environment reproducible. Install uv, then run:
```sh
uv sync
```

### 3. (Optional) Set up PyTest in VS Code
Our test suite for the solver and our other Python code is written using [PyTest](https://pytest.org/). To set up running PyTest from [VS Code](https://code.visualstudio.com/), add the following to your settings.json (Press `Ctrl+Shift+P` → type `Preferences: Open Settings (JSON)` → hit Enter).

```json
{
  "python.testing.pytestArgs": ["tests"],
  "python.testing.unittestEnabled": false,
  "python.testing.pytestEnabled": true
}
```

## Reproducing the paper's experiments
Work through the notebooks in `notebooks/` in numeric order — see the top-level [README](../README.md) for the summary of what each step does.

## Calling the Bender's decomposition solver directly from CLI
To call the Bender's decomposition solver (`BenderMILPSolver`) directly from the
command line, you can use the command-line interface (CLI) provided in the `run` module
of the `milp_engine` package. An example command to run the solver is as follows:

```sh
python -m milp_engine.run --problem-name binary_choice_milp --criterion coverage --strategy take-all --no-parallel
```

A comprehensive description of all the CLI options can be found in the [BenderMILPSolver CLI wiki](wiki/MILP-Engine/BenderMILPSolver-CLI.md).
