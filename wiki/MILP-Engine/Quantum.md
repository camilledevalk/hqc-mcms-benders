# Quantum Computing in MILP Engine

## Overview

This page describes how quantum computing is integrated into the MILP Engine framework, specifically for solving cut selection problems within Benders Decomposition. First, the cut selection formulation is explained and how that can be solved on a quantum device using the Quantum Approximate Optimization Algorithm (QAOA). To perform the QAOA, there are multiple backends (Qiskit, and Fermioniq), which are also explained. Then, we describe the execution flow and how to configure the solvers.

- [Quantum Computing in MILP Engine](#quantum-computing-in-milp-engine)
  - [Overview](#overview)
  - [Problem Formulation](#problem-formulation)
    - [From Cut Selection to QUBO](#from-cut-selection-to-qubo)
    - [QAOA Circuit Generation](#qaoa-circuit-generation)
  - [Quantum Backends](#quantum-backends)
    - [Fermioniq Backend](#fermioniq-backend)
    - [Qiskit Backend](#qiskit-backend)
      - [Sampler\_sim vs Sampler\_hardware (Qiskit)](#sampler_sim-vs-sampler_hardware-qiskit)
  - [Algorithm Implementation](#algorithm-implementation)
    - [QAOA Execution Flow](#qaoa-execution-flow)
    - [Solution Extraction](#solution-extraction)
  - [Configuration and Usage](#configuration-and-usage)
    - [Parameter Selection](#parameter-selection)

## Problem Formulation

### From Cut Selection to QUBO

In this codebase the transformation from an LP cut-selection problem to a QUBO is handled by utility functions in `milp_engine.utils.quantum_utils`.

- `lp_to_quadratic_program(optimization_problem: pulp.LpProblem) -> qiskit_optimization.QuadraticProgram`:
  - Writes the PuLP problem to a temporary LP file, reads it into Qiskit's `QuadraticProgram` and returns it.
  - The LP is written to a unique temporary file (via `tempfile.NamedTemporaryFile`) and removed after conversion. This is safe for parallel workers.

- `lp_to_qubo(optimization_problem: pulp.LpProblem) -> qiskit_optimization.QuadraticProgram`:
  - Validates that all decision variables are binary (integer with bounds 0/1) and then converts the QuadraticProgram to QUBO form using `QuadraticProgramToQubo`.
  - Raises `ValueError` if non-binary variables are present. The QUBO returned is used to build the QAOA cost operator.

### QAOA Circuit Generation

Implementation details in `milp_engine.utils.quantum_utils.create_qaoa_circuit`:

- Input: a QUBO `QuadraticProgram` and `reps` (the QAOA depth / number of layers).
- The function calls `qubo.to_ising()` to obtain the cost operator as a `SparsePauliOp` and constructs a `QAOAAnsatz` with `cost_operator` and `reps`.
- `basis_gates` may be supplied as a list, the string `'qiskit'` (which expands to a gate-set from `qcshared.utils.gate_maps.qiskit_gate_map`) or `None` (no transpilation gate restriction). If invalid, a `ValueError` is raised.
- The constructed ansatz is transpiled with the provided `optimization_level` (default 3) and returned together with the cost operator. The returned `QuantumCircuit` is parameterized (angles are QAOA parameters) and ready for simulation or execution on hardware.

For readers unfamiliar with quantum algorithms, a short practical note on QAOA helps explain the surrounding code and why some backends may run a VQE-style optimizer over circuit parameters:

- QAOA (Quantum Approximate Optimization Algorithm) encodes a combinatorial optimization problem into a cost unitary $\hat{U}_C$ (constructed from the QUBO) and applies a mixer unitary $\hat{H}_M$ to explore feasible states. In our code base, we use the default mixer and cost unitaries.

- QAOA is a specialization of VQE (Variational Quantum Eigensolver): VQE minimizes an observable's expectation value over a parameterized state and QAOA uses the alternating cost/mixer ansatz with the QUBO-derived Hamiltonian as the observable. Some providers (like Fermioniq) can run the classical optimization loop remotely and return optimized parameters and samples; the code exposes `fermioniq_options['optimizer']` to toggle this behaviour. Using provider-side optimization reduces round trips and leverages provider tooling, at the cost of reduced transparency and dependence on remote optimizer settings.

A brief note on *observable*: an (observable in this context) comes from quantum mechanics and is a Hermitian operator (matrix) $H$ that represents a measurable quantity. For QAOA this is the QUBO-derived Hamiltonian. The algorithm prepares a parameterized state $|\psi(\theta)\rangle$ and estimates the expectation value
$$
\langle H \rangle = \langle \psi(\theta) | H | \psi(\theta) \rangle,
$$
by measuring the state repeatedly. Measurements return eigenvalues of $H$ with probabilities given by the Born rule, and the expectation is the probability-weighted average of those outcomes.

## Quantum Backends

### Fermioniq Backend

The project contains a `FermioniqBackend` implementation in `milp_engine.solvers.quantum_solver.FermioniqBackend`. Fermioniq has a quantum-inpsired quantum emulator called `Ava`. More information on that can be found [online here](https://www.fermioniq.com/ava).

Key points:

- Initialization requires a `tokens.json` in the current working directory (used to authenticate the Fermioniq client). If this file is missing a `FileNotFoundError` is raised.
- Publicly supported remote configs: `cpu-2`, `cpu-4`, `cpu-8`, `gpu-h100` (the backend string is provided via `fermioniq_options["remote_config"]`).
- Important configuration options (passed through `fermioniq_options`):
  - `n_shots` (int, default 1000): number of sampling shots emitted by the backend when sampling mode is enabled.
  - `optimization_level` (int, default 3): used when transpiling the QAOA ansatz before submission.
  - `optimizer` (dict): nested options controlling whether Fermioniq itself runs a classical optimizer (VQE-like) over circuit parameters. Example fields:
    - `enabled` (bool): if True, Fermioniq will run an optimizer (COBYLA or SPSA currently supported) to minimize the expectation of the supplied observable.
    - `evaluation_mode` (str, default `"contract"`): evaluation mechanism for expectation values (currently `contract`).
    - `optimizer_name` (str, default `"cobyla"`): which classical optimizer to use.
    - `initial_param_values`, `initial_param_noise` and `optimizer_settings` may also be provided.

Execution flow:

- The backend builds the QAOA circuit via `create_qaoa_circuit(qubo, reps, optimization_level=...)` and constructs a Fermioniq `EmulatorJob` with a `standard_config` modified by the provided fermioniq options.
- The observable is set to the QAOA cost operator and sampling can be enabled (the config writes sampling to `config["output"]["sampling"]`). The job is scheduled with the Fermioniq client and waited upon.
- The returned object includes `job_outputs` and sampling data. The backend also computes an "achieved_fidelity" value from job metadata and compares it against a `fidelity_threshold` passed to `run_qaoa`; if fidelity is below the threshold, a `UserWarning` is issued.

Notes and limitations:

- Fermioniq runs remotely and requires credentials. A tokens.json file containing the API token must be present in the working directory — Fermioniq issues these tokens and provides instructions when you purchase access to their compute services. The backend also requires network access to submit jobs.
- The backend expects a QUBO (binary-only) and will set `self.decision_variables` from `qubo.variables` for downstream interpretation.

### Qiskit Backend

The `QiskitBackend` implementation in `milp_engine.solvers.quantum_solver` supports both local simulation (via `qiskit.primitives.Sampler`) and hardware execution through IBM Quantum Runtime.

Important options (passed via `qiskit_options` on backend init):

- `n_shots` (int, default 1000)
- `use_hardware` (bool, default False). If True, the backend expects a `token_path` pointing to a file containing an IBM token string. The token is saved using `QiskitRuntimeService.save_account` and `QiskitRuntimeService()` is used to obtain a backend.
- `ibm_hardware_name` (str, default `"ibm_strasbourg"`) to select the IBM device.

Execution flow (simplified):

- The code constructs a `QuadraticProgram` (via `lp_to_quadratic_program`) and a QUBO (via `lp_to_qubo`).
For hardware mode the code configures a `Sampler_hardware` with a pass manager tuned for the selected backend. For local simulation it initializes `Sampler_sim` with given shot options.

The QAOA minimum eigen-solver is constructed (Qiskit's QAOA wrappers) with the classical optimizer `COBYLA` instantiated using the `optimizer_settings` from the parent class. The `reps` parameter controls circuit depth.

After solving, `results.samples` is used to build a probability distribution; then the code samples `n_shots` draws using numpy to produce a frequency dictionary. The backend sets `self.decision_variables` from the quadratic program's variables for later mapping.

#### Sampler_sim vs Sampler_hardware (Qiskit)

The `QiskitBackend` supports two sampler modes. The distinction is important for reproducibility, noise behavior, and how circuits are transpiled.

Sampler_sim (local simulation):

- Uses Qiskit's local sampling primitives (typically Aer backends or the `Sampler` primitive configured for simulation). It runs entirely on the client machine and simulates ideal (or optionally noisy) device behaviour depending on supplied noise models.
- Deterministic behaviour: for a given random seed and configuration (and ignoring explicit sampling randomness when `n_shots` > 1), simulation results are reproducible. The codebase can set seeds for the classical optimizer and numpy sampling to improve reproducibility.
- No hardware coupling: no pass manager tailored to a real device is required; the transpiler may still run but is not constrained by device-specific basis gates or coupling maps unless the code supplies them.
- Fast turnaround and lower latency — good for development, unit tests and small instances.

Sampler_hardware (IBM runtime / real hardware):

- Configures a sampler that targets an actual IBM device via the Qiskit Runtime (or a remote service). The backend will set a pass manager and transpilation strategy matching the target device's native gate set, topology, and calibration constraints.
- Noise and real-device effects: results are subject to decoherence, readout errors, gate infidelities and queue delays. The job metadata includes device calibration info which the code may use to compute `achieved_fidelity`.
- Transpilation differences: the hardware sampler will run the full pass manager and mapping to satisfy device coupling maps and basis gates. This can change circuit depth and structure relative to simulation and influence final performance.
- Non-deterministic timing: jobs may be queued and take longer to return. Results are not reproducible in the same way as a simulator because hardware noise varies over time.

When to use which:

- Use `Sampler_sim` for development, debugging, unit tests, and to verify mapping/interpretation logic.
- Use `Sampler_hardware` when you want to evaluate performance on real devices or to test noise-resilience. Expect longer runtimes and to manage API tokens and rate limits.

Notes:

- Running on IBM hardware requires an API token and may incur job queueing time; results include noise and real-device fidelities.
- Local simulation with Aer is deterministic given the same random seed and configuration, but still subject to statistical sampling when `n_shots` > 1.

## Algorithm Implementation

### QAOA Execution Flow

High-level flow implemented by `milp_engine.solvers.quantum_solver.QuantumSolver`:

- Prepare the binary LP `optimization_problem` representing the cut selection.

- Instantiate the selected backend (`FermioniqBackend` or `QiskitBackend`) with desired options and a `solve_method` (currently only `"qaoa"`).

- Call `QuantumSolver.solve_with_binary_indicator_matrix(...)`, which dispatches to `backend.run_qaoa(...)`.

  - For Fermioniq the call signature is `run_qaoa(optimization_problem, reps, fidelity_threshold)`.
  - For Qiskit the call is `run_qaoa(optimization_problem, reps)` and it returns a tuple `(results, frequency_dict)`.

- The returned raw samples are passed to `interpret_samples(samples, problem)` to compute feasibility flags and LP objective values for each sampled bitstring.

Important implementation notes:

- Bit-ordering matters (most significant bit vs least-significant bit). The code stores `decision_variables` from the QUBO conversion — use that to map indices.
- Feasibility is checked by evaluating LP constraints under the candidate assignment using a small numerical tolerance.
- If no feasible sample is found the caller currently raises `ValueError`, but a fallback (least-infeasible solution) may be more robust in practice.

**Important note on samples:** Raw samples just contain information about how often which bitstring was sampled. These bitstrings also contain all auxilary variables that were introduced with creating the QUBO. With `interpret_samples`, the samples are combined based on only the actual decision variables present in the linear program (`problem`). This function also calculates the costs of the variable assignment based on the bitstring. (See also [solution extraction](#solution-extraction)).

`QuantumSolver._get_solution_from_output(...)` attempts to find the lowest-cost feasible bitstring. It:

- Filters decision variables to those named with the `x_` prefix (the expected variable naming convention used by the cut-selection problems).

- Determines the solution class (set cover vs maximum coverage) for each sampled bitstring using helper functions.

- Maps the chosen bitstring back to the original PuLP variables and sets their `.varValue` accordingly.

The optimization problem is returned with `status` set to `LpStatusOptimal` and decision variable values populated.

Classical optimization options:

- The classical optimizer settings live in the backend parent class as `self.optimizer_settings` (defaults include `rhobeg: np.pi/2`, `max_iter: 250`, `tol: 1e-4`, `catol: 2e-4`).
- Fermioniq also accepts optimizer-specific options via `fermioniq_options["optimizer"]` which can enable Fermioniq-side optimization of QAOA parameters (VQE-style).

### Solution Extraction

The codebase provides robust utils for interpreting samples and mapping bitstrings back to LP variables:


- `interpret_samples(samples: Dict[str, int], problem, qubo=None)` returns two dicts:
  - `samples_interpreted`: per-bitstring info including `cost_lp`, `frequency`, `feasible` and optionally `cost_qubo`.
  - `solutions_interpreted`: aggregated information keyed by solution bitstring (with consolidated frequencies).

- `QuantumSolver._get_solution_from_output` loops over samples (sorted by frequency) and uses `get_solution_class_from_output(...)` together with helper functions in `milp_engine.utils.milp_engine_utils` (notably `determine_set_cover_class` and `determine_max_coverage_class`) to decide whether a bit in the protected solution corresponds to the solution class (0 or 1). The mapping expects decision variables to be named with the `x_` prefix to be considered as selection bits.

- If none of the sampled bitstrings is feasible, the function raises a `ValueError("No feasible solution found in the provided QAOA samples.")`.

## Configuration and Usage

### Parameter Selection

Parameters you can tune in practice:


- `reps` (int): the number of QAOA layers; passed to `create_qaoa_circuit` and the QAOA solver. Higher values increase expressivity but also circuit depth and runtime.
- `n_shots` (int): number of measurement shots to collect (Fermioniq and Qiskit options). Larger values reduce sampling variance but increase runtime.
- `optimization_level` (int, default 3): transpiler optimization level used when building the QAOA circuit.
- `optimizer_settings` (dict): classical optimizer configuration used by COBYLA/SPSA. Defaults are provided by `QuantumOptimisationBackend` but can be overridden.
- `fidelity_threshold` (float, default 0.99): used by `FermioniqBackend.run_qaoa` to trigger a warning when job fidelity is lower than desired.

Example: quick usage snippet

```python
from milp_engine.solvers.quantum_solver import QuantumSolver, QiskitBackend
from milp_engine.utils.quantum_utils import lp_to_qubo
import pulp

# Prepare a (binary) PuLP problem named 'cut_selection' with x_ prefixed decision vars
prob = pulp.LpProblem('cut_selection', pulp.LpMaximize)
# ... build your PuLP problem here using binary vars named x_0, x_1, ...

# Create a Qiskit backend (local simulation)

qiskit_backend = QiskitBackend(qiskit_options={"n_shots": 2000, "use_hardware": False})

# Wrap in a QuantumSolver
qsolver = QuantumSolver(backend=qiskit_backend, solver_options={"reps": 2}, solve_method="qaoa")

# Solve (binary_indicator_matrix required by the API; construct from your problem)
# For demonstration, we pass a dummy identity matrix if you have N variables
import numpy as np
N = len([v for v in prob.variables() if v.name.startswith('x_')])
binary_indicator_matrix = np.eye(N, dtype=int)


solved_prob = qsolver.solve_with_binary_indicator_matrix(prob, binary_indicator_matrix, max_number_covered=None)
print([v.varValue for v in solved_prob.variables()])
```

Notes about the snippet:

- The function `solve_with_binary_indicator_matrix(...)` expects a NumPy `binary_indicator_matrix` that encodes coverage relationships used to determine solution class (set cover vs maximum coverage). The code relies on helper functions in `milp_engine.utils.milp_engine_utils` to interpret this matrix.
- Decision variables intended to represent selection bits must be named with the `x_` prefix (this is how the solver filters them out from other LP variables).
