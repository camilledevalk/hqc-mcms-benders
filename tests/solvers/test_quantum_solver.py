import io
import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pulp
import pytest
from _pytest.monkeypatch import MonkeyPatch
from pytest_mock import MockerFixture
from qiskit_optimization import QuadraticProgram

from milp_engine.solvers.quantum_solver import (
    FermioniqBackend,
    JuliaMPSBackend,
    QiskitBackend,
    QuantumSolver,
    get_relevant_bits_from_output,
    get_solution_class_from_output,
    map_solution_class_to_decision_variables,
)
from milp_engine.strategies.minimum_set_cover_strategy import MinimumSetCoverStrategy


@pytest.fixture
def mock_token_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    # Write the tokens.json file inside it
    token_path = tmp_path / "tokens.json"
    # token_path = token_dir / "tokens.json"

    token_path.write_text(json.dumps({"id": "test_id", "secret": "test_secret"}))

    # Patch os.getcwd to return the temp path
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    return token_path


def test_quantum_solver_init(monkeypatch: MonkeyPatch, mock_token_file: Path) -> None:
    class DummyClient:
        id: str
        secret: str

        def __init__(self, access_token_id: str, access_token_secret: str):
            self.id = access_token_id
            self.secret = access_token_secret

    monkeypatch.setattr("fermioniq.Client", DummyClient)

    solver = QuantumSolver(backend=FermioniqBackend())

    assert isinstance(solver.backend, FermioniqBackend)
    assert solver.backend.client.id == "test_id"  # type: ignore[attr-defined]
    assert solver.backend.client.secret == "test_secret"  # type: ignore[attr-defined]
    assert solver.backend.fermioniq_optimizer_config["enabled"] is True


@pytest.mark.parametrize(
    "fermioniq_options",
    [
        {"dmrg": {"D": 1234}},
        {"group_size": 1},
        {"output": {"expectation_values": {"enabled": True}}},
        {"remote_config": "cpu-8"},
    ],
)
def test_quantum_solver_options(
    fermioniq_options: Dict, monkeypatch: MonkeyPatch, mock_token_file: Path
) -> None:

    class DummyJobResult:
        def __init__(self, config: dict) -> None:
            self.config = config
            self.job_outputs = [
                {
                    "metadata": {
                        "fidelity_product": 1.0,
                        "extrapolated_2qubit_gate_fidelity": 1.0,
                    }
                }
            ]
            # Store your fake samples internally
            self._samples = {"10": 1}

        def samples(self, i: int, j: int) -> Dict[str, int]:
            """Mimic the real API's `samples` method by returning stored sample data.

            :param i: Index of the job output to access.
            :param j: Index of the sample within the job output.

            :return:A dictionary representing sample data.
            """
            return self._samples

    class DummyEmulatorJob:
        job_id: str = "dummy_job_id"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.config = kwargs.get("config", {})

    class DummyClient:
        id: str
        secret: str

        def __init__(self, access_token_id: str, access_token_secret: str):
            self.id = access_token_id
            self.secret = access_token_secret

        def schedule_async(self, job: Any) -> None:
            self.job = job

        def get_status(self, job: Any) -> str:
            return "finished"

        def get_results(self, job_id: str) -> DummyJobResult:
            return DummyJobResult(config=self.job.config)

    monkeypatch.setattr("fermioniq.Client", DummyClient)
    monkeypatch.setattr("fermioniq.EmulatorJob", DummyEmulatorJob)

    solver = QuantumSolver(
        backend=FermioniqBackend(fermioniq_options=fermioniq_options)
    )

    # Build a simple optimization problem
    optimization_problem = pulp.LpProblem("Maximize_x0_minus_x1", pulp.LpMaximize)
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    optimization_problem += x_0 - x_1

    output, _ = solver.backend.run_qaoa(
        optimization_problem, reps=1, fidelity_threshold=0.99
    )

    def deep_compare(
        actual: Dict, expected: Dict, explicit_not: str = "remote_config"
    ) -> None:
        """Recursively compare two dictionaries for deep equality.

        :param actual: The dictionary produced by the code under test
            (e.g., output.config[0]).
        :param expected: The dictionary containing the expected values
            (e.g., fermioniq_options).
        :param explicit_not: The key in expected that must NOT be present in actual

        """
        for k, v in expected.items():
            if k == explicit_not:
                # Check that the key is not present in actual
                assert (
                    k not in actual
                ), f"Key '{k}' from expected found in actual output.config[0]"
                continue
            assert (
                k in actual
            ), f"Key '{k}' from expected not found in actual output.config[0]"
            if isinstance(v, dict):
                assert isinstance(
                    actual[k], dict
                ), f"Key '{k}' in actual output.config[0] is not a dict"
                deep_compare(actual[k], v)
            else:
                assert actual[k] == v, (
                    f"Value for key '{k}' does not match:"
                    f" actual={actual[k]} expected={v}"
                )

    deep_compare(output.config[0], fermioniq_options)


def test_get_relevant_bits_from_output() -> None:

    # Create a binary maximization problem
    optimization_problem = pulp.LpProblem("Maximize_x0_minus_x1", pulp.LpMaximize)

    # Define binary decision variables
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    x_2 = pulp.LpVariable("x_2", cat="Binary")

    # Objective function: Maximize x_0 - x_1
    optimization_problem += x_0 + x_1 - x_2, "Objective"

    # Define the decision variables
    decision_vars = [x_0, x_1, x_2]

    binary_indicator_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    max_number_covered = None  # Therefore, strategy is Minimum Set Cover

    # Create a QAOA output sample
    output_samples = {"110": 5, "101": 3}

    # Get the relevant bits from the original bitstring
    relevant_bits, x_decision_variables = get_relevant_bits_from_output(
        output_samples, decision_vars
    )

    # Test the get_relevant_bits_from_output function
    assert relevant_bits == {"110": ["1", "1", "0"], "101": ["1", "0", "1"]}

    # Initialize empty dictionary to hold the mapped solution class
    mapped_bitstring = {}
    solution_class = {}
    for bitstring, filtered_bits in relevant_bits.items():
        # Get the solution class
        sol_class = get_solution_class_from_output(
            max_number_covered, binary_indicator_matrix, filtered_bits
        )
        solution_class[bitstring] = sol_class
        # Map the solution class to the decision variables
        solution_bitstring = map_solution_class_to_decision_variables(
            x_decision_variables, filtered_bits, sol_class
        )

        # Add mapping to the dictionary
        mapped_bitstring[bitstring] = solution_bitstring

    # Test the get_solution_class_from_output function
    assert solution_class == {
        "110": "1",
        "101": "0",
    }

    # Test the map_solution_class_to_decision_variables function
    assert mapped_bitstring == {
        "110": {"x_0": 1.0, "x_1": 1.0, "x_2": 0.0},
        "101": {"x_0": 0.0, "x_1": 1.0, "x_2": 0.0},
    }


def test_get_solution_from_output(
    monkeypatch: MonkeyPatch, mock_token_file: Path, mocker: MockerFixture
) -> None:
    # Create a binary maximization problem
    optimization_problem = pulp.LpProblem("Maximize_x0_minus_x1", pulp.LpMaximize)

    # Define binary decision variables
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    x_2 = pulp.LpVariable("x_2", cat="Binary")

    # Objective function: Maximize x_0 - x_1
    optimization_problem += x_0 + x_1 - x_2, "Objective"

    decision_vars = [x_0, x_1, x_2]

    binary_indicator_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    max_number_covered = None  # Therefore, strategy is Minimum Set Cover

    solvers = [
        QuantumSolver(backend=FermioniqBackend()),
        QuantumSolver(backend=QiskitBackend()),
        QuantumSolver(backend=JuliaMPSBackend()),
    ]

    for solver in solvers:
        output_samples = {"110": 3, "101": 5}
        solved_optimization_problem = solver._get_solution_from_output(
            output_samples,
            optimization_problem,
            binary_indicator_matrix,
            max_number_covered,
            decision_vars,
        )

        assert isinstance(solved_optimization_problem, pulp.LpProblem)

        # Obtain the optimal solution for each decision variable
        optimal_solution = {}
        for var in solved_optimization_problem.variables():
            optimal_solution[cast(str, var.name)] = var.varValue

        # Test the optimal solution
        assert optimal_solution == {"x_0": 1.0, "x_1": 1.0, "x_2": 0.0}

        # Check that varValue is set and binary for all decision variables in
        # optimization problem
        for var in solved_optimization_problem.variables():
            assert var.varValue in [
                0,
                1,
            ], f"Variable {var.name} value {var.varValue} is not binary"

        # Check problem status is 'Optimal'
        assert solved_optimization_problem.status == pulp.constants.LpStatusOptimal


def test_get_solution_from_output_rare_solution(
    monkeypatch: MonkeyPatch, mock_token_file: Path, mocker: MockerFixture
) -> None:
    strategy = MinimumSetCoverStrategy()

    binary_indicator_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    optimization_problem, decision_vars = strategy.create_optimization_problem(
        binary_indicator_matrix=binary_indicator_matrix
    )

    max_number_covered = None  # Therefore, strategy is Minimum Set Cover

    solvers = [
        QuantumSolver(backend=FermioniqBackend()),
        QuantumSolver(backend=QiskitBackend()),
        QuantumSolver(backend=JuliaMPSBackend()),
    ]

    for solver in solvers:
        output_samples = {"111000": 5, "101010": 3, "111111": 1}
        solved_optimization_problem = solver._get_solution_from_output(
            output_samples,
            optimization_problem,
            binary_indicator_matrix,
            max_number_covered,
            decision_vars,
        )
        assert isinstance(solved_optimization_problem, pulp.LpProblem)

        # Check that varValue is set and binary for all decision variables in
        # optimization problem
        for var in solved_optimization_problem.variables():
            assert var.varValue == 1

        # Check problem status is 'Optimal'
        assert solved_optimization_problem.status == pulp.constants.LpStatusOptimal


def test_get_solution_from_output_no_solution(
    monkeypatch: MonkeyPatch, mock_token_file: Path, mocker: MockerFixture
) -> None:

    strategy = MinimumSetCoverStrategy()

    binary_indicator_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    optimization_problem, decision_vars = strategy.create_optimization_problem(
        binary_indicator_matrix=binary_indicator_matrix
    )

    max_number_covered = None  # Therefore, strategy is Minimum Set Cover

    solvers = [
        QuantumSolver(backend=FermioniqBackend()),
        QuantumSolver(backend=QiskitBackend()),
        QuantumSolver(backend=JuliaMPSBackend()),
    ]

    samples = {"111000": 5, "101010": 3}
    for solver in solvers:
        with pytest.raises(
            ValueError, match="No feasible solution found in the provided QAOA samples."
        ):
            solver._get_solution_from_output(
                samples,
                optimization_problem,
                binary_indicator_matrix,
                max_number_covered,
                decision_vars,
            )


@pytest.mark.parametrize(
    "fidelity,expect_warning",
    [
        (1.0, False),
        (0.98, True),
    ],
)
def test_run_qaoa_fermioniq(
    monkeypatch: MonkeyPatch, fidelity: float, expect_warning: bool
) -> None:
    # Dummy classes and functions
    class DummyJobResult:
        def __init__(self) -> None:
            self.job_outputs = [
                {
                    "metadata": {
                        "fidelity_product": fidelity,
                        "extrapolated_2qubit_gate_fidelity": fidelity,
                    }
                }
            ]
            # Store your fake samples internally
            self._samples = {"10": 1}

        def samples(self, i: int, j: int) -> Dict[str, int]:
            """Mimic the real API's `samples` method by returning stored sample data.

            :param i: Index of the job output to access.
            :param j: Index of the sample within the job output.

            :return:A dictionary representing sample data.
            """
            return self._samples

    class DummyClient:
        id: str
        secret: str

        def schedule_async(self, job: Any) -> None:
            self.job = job

        def get_status(self, job: Any) -> str:
            return "finished"

        def get_results(self, job_id: str) -> DummyJobResult:
            return DummyJobResult()

    class DummyEmulatorJob:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.job_id = "dummy_job_id"

    monkeypatch.setattr("fermioniq.Client", lambda *a, **kw: DummyClient())
    monkeypatch.setattr("fermioniq.EmulatorJob", DummyEmulatorJob)
    monkeypatch.setattr(
        "fermioniq.config.defaults.standard_config",
        lambda c: {"output": {}, "dmrg": {}},
    )
    # Patch open only for the token file
    original_open = open

    def selective_open(file: str, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "tokens.json" in str(file):
            return io.StringIO(json.dumps({"id": "id", "secret": "secret"}))
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", selective_open)

    solver = QuantumSolver(backend=FermioniqBackend())

    # Build a simple optimization problem
    optimization_problem = pulp.LpProblem("Maximize_x0_minus_x1", pulp.LpMaximize)
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    optimization_problem += x_0 - x_1

    if expect_warning:
        with pytest.warns(UserWarning):
            output, samples = solver.backend.run_qaoa(
                optimization_problem, reps=1, fidelity_threshold=0.99
            )
    else:
        output, samples = solver.backend.run_qaoa(
            optimization_problem, reps=1, fidelity_threshold=0.99
        )

    # Since _get_solution_from_output returns a dict here
    assert samples == {"10": 1}
    assert isinstance(output, DummyJobResult)


def test_run_qaoa_qiskit(monkeypatch: MonkeyPatch) -> None:
    # Dummy classes and functions
    class DummySample:
        """Mimics Qiskit's OptimizationSample."""

        def __init__(self, x: list[int], probability: float) -> None:
            self.x = x
            self.probability = probability

    class DummyMinimumEigenOptimizerResult:
        """Mimics Qiskit's MinimumEigenOptimizerResult for testing."""

        def __init__(self) -> None:
            # Create samples with bitstring solutions and their probabilities
            self.samples = [
                DummySample([1, 0], 1.0),
            ]

    class DummyMinimumEigenOptimizer:
        def solve(self, qp: QuadraticProgram) -> DummyMinimumEigenOptimizerResult:
            return DummyMinimumEigenOptimizerResult()

    monkeypatch.setattr(
        "qiskit_optimization.algorithms.MinimumEigenOptimizer",
        lambda *a, **kw: DummyMinimumEigenOptimizer(),
    )

    solver = QuantumSolver(backend=QiskitBackend())

    # Build a simple optimization problem
    optimization_problem = pulp.LpProblem("Maximize_x0_minus_x1", pulp.LpMaximize)
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    optimization_problem += x_0 - x_1

    output, samples = solver.backend.run_qaoa(optimization_problem, reps=1)

    # Since _get_solution_from_output returns a dict here
    assert samples == {"10": 1000}  # Since number of shots is defaulted 1000
    assert isinstance(output, DummyMinimumEigenOptimizerResult)


def test_solve_with_binary_indicator_matrix(monkeypatch: MonkeyPatch) -> None:
    # Create a simple optimization problem: minimize x0 + x1
    prob = pulp.LpProblem("TestProblem", pulp.LpMinimize)
    x0 = pulp.LpVariable("x_0", 0, 1, cat="Binary")
    x1 = pulp.LpVariable("x_1", 0, 1, cat="Binary")
    prob += x0 + x1

    # Simple 2x2 binary indicator matrix
    binary_indicator_matrix = np.array([[0, 1], [1, 0]])

    # Patch open only for the token file needed for fermioniq backend
    original_open = open

    def selective_open(file: str, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "tokens.json" in str(file):
            return io.StringIO(json.dumps({"id": "id", "secret": "secret"}))
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", selective_open)

    solvers = [
        QuantumSolver(backend=FermioniqBackend()),
        QuantumSolver(backend=QiskitBackend()),
        QuantumSolver(backend=JuliaMPSBackend()),
    ]

    # Initialize the solver with qiskit backend
    for solver in solvers:
        # Mock backend and its decision variables
        monkeypatch.setattr(
            solver.backend,
            "run_qaoa",
            lambda optimization_problem, reps, fidelity_threshold: (
                {"dummy_output": 1},
                {"01": 3, "10": 1},  # bitstring with highest freq
            ),
        )
        solver.backend.decision_variables = [x0, x1]
        # Solve
        solved_prob = solver.solve_with_binary_indicator_matrix(
            prob, binary_indicator_matrix
        )

        # Check if problem is solved and variables are set according to our mock
        assert solved_prob.status == pulp.constants.LpStatusOptimal
        # Since sol_class=0 and bitstring=01 -> x0=1 if bit=0 else 0
        # Here first bit=0 -> x0=1, second bit=1 -> x1=0
        assert x0.varValue in (0, 1)
        assert x1.varValue in (0, 1)


def test_julia_mps_backend_init() -> None:
    """Test JuliaMPSBackend constructor with default and custom options."""
    # Default options
    backend = JuliaMPSBackend()
    assert backend.julia_executable == "julia"
    assert backend.julia_project_path == "julia/MPS_JuliQAOA"
    assert backend.n_shots == 1000
    assert backend.cutoff == 1e-6
    assert backend.maxdim == 64
    assert backend.optimizer == "cobyla"
    assert backend.maxiter == 1000

    # Custom options
    backend = JuliaMPSBackend(
        julia_options={
            "julia_executable": "/opt/julia/bin/julia",
            "julia_project_path": "/home/user/MPS_JuliQAOA",
            "n_shots": 500,
            "cutoff": 1e-8,
            "maxdim": 128,
            "optimizer": "particle_swarm",
            "maxiter": 2000,
        }
    )
    assert backend.julia_executable == "/opt/julia/bin/julia"
    assert backend.julia_project_path == "/home/user/MPS_JuliQAOA"
    assert backend.n_shots == 500
    assert backend.cutoff == 1e-8
    assert backend.maxdim == 128
    assert backend.optimizer == "particle_swarm"
    assert backend.maxiter == 2000


def test_run_qaoa_julia_mps(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test JuliaMPSBackend.run_qaoa with a mocked subprocess."""
    julia_output = {
        "expectation_value": -1.5,
        "angles": [0.5, 1.2],
        "p": 1,
        "nqubits": 2,
        "converged": True,
        "iterations": 42,
        "status": "success",
        "samples": {"10": 700, "01": 250, "00": 30, "11": 20},
    }

    # Build a QUBO directly to avoid docplex/cplex LP file parsing
    dummy_qubo = QuadraticProgram("test_qubo")
    dummy_qubo.binary_var("x_0")
    dummy_qubo.binary_var("x_1")
    dummy_qubo.maximize(linear=[1, -1])

    monkeypatch.setattr(
        "milp_engine.solvers.quantum_solver.lp_to_qubo", lambda prob: dummy_qubo
    )

    def mock_subprocess_run(
        cmd: list, capture_output: bool, text: bool, check: bool
    ) -> Any:
        """Write the julia output JSON to the output path and return success."""
        input_path = cmd[-2]
        output_path = cmd[-1]

        # Verify input JSON was written correctly
        with open(input_path, "r") as f:
            input_data = json.load(f)
        assert "z_interactions" in input_data
        assert input_data["nqubits"] == 2
        assert input_data["p"] == 1
        assert input_data["optimize"] is True
        assert input_data["n_shots"] == 50

        # Write the fake Julia output
        with open(output_path, "w") as f:
            json.dump(julia_output, f)

        class FakeResult:
            returncode = 0
            stdout = ""
            stderr = ""

        return FakeResult()

    monkeypatch.setattr(
        "milp_engine.solvers.quantum_solver.subprocess.run", mock_subprocess_run
    )

    backend = JuliaMPSBackend(julia_options={"n_shots": 50, "maxiter": 100})

    # Build a simple optimization problem
    optimization_problem = pulp.LpProblem("Maximize_x0_minus_x1", pulp.LpMaximize)
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    optimization_problem += x_0 - x_1

    metadata, samples = backend.run_qaoa(optimization_problem, reps=1)

    # Verify samples
    assert samples == {"10": 700, "01": 250, "00": 30, "11": 20}
    assert sum(samples.values()) == 1000

    # Verify metadata
    assert metadata["expectation_value"] == -1.5
    assert metadata["angles"] == [0.5, 1.2]
    assert metadata["p"] == 1
    assert metadata["nqubits"] == 2
    assert metadata["converged"] is True
    assert metadata["iterations"] == 42
    assert metadata["status"] == "success"
    assert "offset" in metadata
    assert "qubo_objective" in metadata

    # Verify decision variables were set
    assert len(backend.decision_variables) == 2


def test_run_qaoa_julia_mps_subprocess_failure(monkeypatch: MonkeyPatch) -> None:
    """Test JuliaMPSBackend.run_qaoa raises RuntimeError on subprocess failure."""
    # Build a QUBO directly to avoid docplex/cplex LP file parsing
    dummy_qubo = QuadraticProgram("test_qubo")
    dummy_qubo.binary_var("x_0")
    dummy_qubo.binary_var("x_1")
    dummy_qubo.minimize(linear=[1, 1])

    monkeypatch.setattr(
        "milp_engine.solvers.quantum_solver.lp_to_qubo", lambda prob: dummy_qubo
    )

    def mock_subprocess_run(
        cmd: list, capture_output: bool, text: bool, check: bool
    ) -> Any:
        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = "ERROR: LoadError: some julia error"

        return FakeResult()

    monkeypatch.setattr(
        "milp_engine.solvers.quantum_solver.subprocess.run", mock_subprocess_run
    )

    backend = JuliaMPSBackend()

    optimization_problem = pulp.LpProblem("test", pulp.LpMinimize)
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    optimization_problem += x_0 + x_1

    with pytest.raises(RuntimeError, match="Julia MPS-QAOA failed"):
        backend.run_qaoa(optimization_problem, reps=1)


@pytest.fixture
def mock_ibm_token_file(tmp_path: Path) -> Path:
    token_path = tmp_path / "token.txt"
    token_path.write_text("DUMMY_TOKEN")
    return token_path


def test_run_qaoa_qiskit_real_hardware(
    monkeypatch: MonkeyPatch, mock_ibm_token_file: Path
) -> None:
    # Dummy classes and functions
    class DummySample:
        """Mimics Qiskit's OptimizationSample."""

        def __init__(self, x: list[int], probability: float) -> None:
            self.x = x
            self.probability = probability

    class DummyMinimumEigenOptimizerResult:
        """Mimics Qiskit's MinimumEigenOptimizerResult for testing."""

        def __init__(self) -> None:
            # Create samples with bitstring solutions and their probabilities
            self.samples = [
                DummySample([1, 0], 1.0),
            ]

    class DummyMinimumEigenOptimizer:
        def solve(self, qp: QuadraticProgram) -> DummyMinimumEigenOptimizerResult:
            return DummyMinimumEigenOptimizerResult()

    monkeypatch.setattr(
        "qiskit_optimization.algorithms.MinimumEigenOptimizer",
        lambda *a, **kw: DummyMinimumEigenOptimizer(),
    )

    class DummyBackend:
        """Mimics an IBM hardware backend."""

        def __init__(self, backend: str = "") -> None:
            self.backend = backend

    class DummyService:

        def __init__(self) -> None:
            pass

        @classmethod
        def save_account(
            self, token: str, overwrite: bool, channel: str, instance: str
        ) -> None:
            pass

        def backend(self, name: str) -> DummyBackend:
            return DummyBackend()

    class DummySession:
        def __init__(self, backend: DummyBackend) -> None:
            self.backend = backend

        def __enter__(self) -> "DummySession":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    class DummySampler:
        """Fake sampler that mimics Sampler_hardware."""

        def __init__(
            self,
            options: Dict[str, Any],
            backend: DummyBackend | None = None,
            mode: DummySession | None = None,
        ) -> None:
            self.backend = backend
            self.default_shots = options.get("default_shots", 1024)

    monkeypatch.setattr(
        "milp_engine.solvers.quantum_solver.QiskitRuntimeService", DummyService
    )

    monkeypatch.setattr("milp_engine.solvers.quantum_solver.Session", DummySession)

    monkeypatch.setattr(
        "milp_engine.solvers.quantum_solver.Sampler_hardware", DummySampler
    )
    monkeypatch.setattr(
        "milp_engine.solvers.quantum_solver.generate_preset_pass_manager",
        lambda *a, **kw: None,
    )

    solver = QuantumSolver(
        backend=QiskitBackend(
            qiskit_options={
                "use_hardware": True,
                "token_path": str(mock_ibm_token_file),
            }
        )
    )

    # Build a simple optimization problem
    optimization_problem = pulp.LpProblem("Maximize_x0_minus_x1", pulp.LpMaximize)
    x_0 = pulp.LpVariable("x_0", cat="Binary")
    x_1 = pulp.LpVariable("x_1", cat="Binary")
    optimization_problem += x_0 - x_1

    output, samples = solver.backend.run_qaoa(optimization_problem, reps=1)

    # Since _get_solution_from_output returns a dict here
    assert samples == {"10": 1000}  # Since number of shots is defaulted 1000
    assert isinstance(output, DummyMinimumEigenOptimizerResult)
