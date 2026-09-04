# BenderMILPSolver CLI
We built a Command-Line Interface (CLI) around the [BenderMILPSolver](Solvers.md). This allows the solver to be ran on arbitrary problems using the command-line, which can be helpful when benchmarking or running the solver in a single instance. The CLI is implemented in the `run` module of the `milp_engine` package

## Usage
An example command to run the solver is as follows:

```sh
python -m milp_engine.run --problem-name binary_choice_milp --criterion coverage --strategy take-all --no-parallel
```

This runs the Bender's Decomposition solver on the `binary_choice_milp` problem with the
coverage criterion and the take-all strategy, without parallel processing.
You can adjust the parameters as needed. Here is an overview of the available
options:

- `--problem-name`: The name of one of the pre-defined problems. Available options include:
  - `binary_choice_milp`: a toy problem about a binary choice between two sets of constraints.
  - `savings_milp`: a toy problem about optimally investing in savings accounts.
- `--mps-file`: The path to an MPS file containing a custom problem definition.
- `--y-prefix`: The prefix to determine which variables are the complicating (y) variables.
- `--n-subproblems`: The number of subproblems that are solved per iteration.
- `--max-iterations`: The maximum number of iterations to run the solver.
- `--convergence-bound`: The convergence threshold for the solver.
- `--no-parallel`: If set, the solver will not use parallel processing.
- `--use-profiler`: If set, the solver will use a profiler to measure performance.
- `--use-quantum`: If set, the solver will perform heuristics using a Quantum Computer back-end.
- `--criterion`: The criterion to use for the solver (one of: `coverage`, `exclusion`).
- `--strategy`: The strategy to use for the solver (one of: `max-coverage`, `min-set-cover`, `take-all`).
- `--max-cuts`: The maximum number of cuts to select when using the maximum coverage strategy.
- `--backend`: This takes either the value `"fermioniq"` or `"qiskit"`. When `--use-quantum` is enabled, this setting defines the quantum backend used: either Fermioniq (a quantum emulator) or qiskit (actual quantum circuit).
- `--backend-options`: a JSON of extra backend options.
- `--solver-options`: a JSON of extra solver options.
