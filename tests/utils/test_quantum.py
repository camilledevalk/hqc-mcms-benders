# tests/utils/test_quantum.py
import os
from typing import Callable, Dict

import numpy as np
import pulp
import pytest
import qiskit
from _pytest.fixtures import FixtureRequest  # for typing `request`
from qcshared.utils.gate_maps import qiskit_gate_map
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram

from milp_engine.utils.quantum import (
    create_qaoa_circuit,
    interpret_samples,
    lp_to_quadratic_program,
    lp_to_qubo,
    qubo_to_z_interactions,
)

PARAMS = [
    # Set Cover problem instances
    ("set_cover_problem", {"num_elements": 5, "seed": 42}),
    ("set_cover_problem", {"num_elements": 10, "seed": 123}),
    ("set_cover_problem", {"num_elements": 7, "seed": 99}),
    # Maximum Coverage problem instances
    (
        "maximum_coverage_problem",
        {"num_cuts": 6, "num_vars": 4, "max_cuts": 3, "seed": 21},
    ),
    (
        "maximum_coverage_problem",
        {"num_cuts": 10, "num_vars": 5, "max_cuts": 4, "seed": 17},
    ),
    (
        "maximum_coverage_problem",
        {"num_cuts": 8, "num_vars": 6, "max_cuts": 2, "seed": 77},
    ),
]


@pytest.mark.parametrize("problem_factory_fixture, kwargs", PARAMS)
def test_lp_to_qubo_conversion(
    request: FixtureRequest, problem_factory_fixture: str, kwargs: Dict[str, int]
) -> None:
    """Test the conversion of a PuLP problem to a Qiskit QuadraticProgram.

    :param request: pytest request fixture to access other fixtures dynamically.
    :param problem_factory_fixture: A PuLP LpProblem instance to convert.
    :param kwargs: A dictionary of keyword arguments specifying the optimization
        problem.
    """
    optimization_problem_instance = request.getfixturevalue(problem_factory_fixture)
    lp_problem = optimization_problem_instance(**kwargs)

    qubo = lp_to_qubo(lp_problem)

    # Check the type of the result
    assert isinstance(qubo, QuadraticProgram)

    # Check the name of the problem
    assert qubo.name == lp_problem.name

    # QUBO should have no constraints
    assert len(qubo.linear_constraints) == 0


@pytest.mark.parametrize("problem_factory_fixture, kwargs", PARAMS)
def test_lp_to_quadratic_program_conversion(
    request: FixtureRequest, problem_factory_fixture: str, kwargs: Dict[str, int]
) -> None:
    """Test the conversion of a PuLP problem to a Qiskit QuadraticProgram.

    :param request: pytest request fixture to access other fixtures dynamically.
    :param problem_factory_fixture: A PuLP LpProblem instance to convert.
    :param kwargs: A dictionary of keyword arguments specifying the optimization
        problem.
    """
    optimization_problem_instance = request.getfixturevalue(problem_factory_fixture)
    lp_problem = optimization_problem_instance(**kwargs)
    quadratic_program = lp_to_quadratic_program(lp_problem)

    # Check the type of the result
    assert isinstance(quadratic_program, QuadraticProgram)

    # Check the name of the problem
    assert quadratic_program.name == lp_problem.name

    # Check the number of variables
    assert len(quadratic_program.variables) == len(lp_problem.variables())

    # Check the number of constraints
    assert len(quadratic_program.linear_constraints) == len(lp_problem.constraints)

    # Extract all constraint names from the QuadraticProgram
    qiskit_constraint_names = {
        constraint.name for constraint in quadratic_program.linear_constraints
    }

    # Check that each name from PuLP is present in Qiskit
    for name in lp_problem.constraints:
        assert name in qiskit_constraint_names

    # Check the linear objective coefficients
    objective_vars = lp_problem.objective.keys()
    for v in objective_vars:
        assert quadratic_program.objective.linear[v.name] == pytest.approx(
            lp_problem.objective[v]
        )

    # Check the quadratic objective coefficients
    for vi in lp_problem.variables():
        for vj in lp_problem.variables():
            assert quadratic_program.objective.quadratic[
                vi.name, vj.name
            ] == pytest.approx(0.0)

    # Qiskit sense to PulP sense map
    sense_map = {
        0: -1,  # LE
        1: 1,  # GE
        2: 0,  # EQ
    }
    # Check the constraints
    for i, constraint in enumerate(quadratic_program.linear_constraints):
        pulp_constraint = list(lp_problem.constraints.values())[i]
        expected_pulp_sense = sense_map[constraint.sense.value]
        assert expected_pulp_sense == pulp_constraint.sense
        assert constraint.rhs == pytest.approx(
            -1 * pulp_constraint.constant
        )  # Since pulp want the rhs to be zero

        constraint_dict = constraint.linear.to_dict()

        for var in quadratic_program.variables:
            assert constraint_dict.get(var.name, 0) == pytest.approx(
                pulp_constraint.get(var.name, 0)
            )


def test_lp_to_quadratic_program_temp_file_cleanup(
    set_cover_problem: Callable[[int, int], pulp.LpProblem],
    tmp_path: os.PathLike,
) -> None:
    """Test that temporary files are cleaned up after conversion.

    :param set_cover_problem: A PuLP LpProblem Minimum Set Cover
        instance to convert.
    :param tmp_path: pytest temporary directory fixture.
    """
    lp_problem = set_cover_problem(5, 20)
    lp_to_quadratic_program(lp_problem)

    # Assert no leftover .lp files in the OS temp directory
    import tempfile

    leftover = [f for f in os.listdir(tempfile.gettempdir()) if f.endswith("-tmp.lp")]
    assert not leftover, f"Leftover temp files found: {leftover}"


@pytest.mark.parametrize(
    "reps,basis_gates",
    [
        (2, None),
        (5, None),
        (10, None),
        (2, "qiskit"),
        (2, ["cx", "rz", "sx", "x", "y", "z"]),
    ],
)
@pytest.mark.parametrize("problem_factory_fixture, kwargs", PARAMS)
def test_create_qaoa_circuit_parametrized(
    request: FixtureRequest,
    problem_factory_fixture: str,
    kwargs: Dict[str, int],
    reps: int,
    basis_gates: str | list | None,
) -> None:
    """Parametrized test for QAOA circuit creation with various reps and basis_gates.

    :param request: pytest request fixture to access other fixtures dynamically.
    :param problem_factory_fixture: The name of the fixture for the problem instance.
    :param kwargs: Dictionary of keyword arguments for the problem instance.
    :param reps: Number of QAOA repetitions.
    :param basis_gates: Basis gates to use for the circuit, or None.
    """
    optimization_problem_instance = request.getfixturevalue(problem_factory_fixture)
    lp_problem = optimization_problem_instance(**kwargs)
    qubo = lp_to_qubo(lp_problem)

    if basis_gates is None:
        circuit, operator = create_qaoa_circuit(qubo, reps)
    else:
        circuit, operator = create_qaoa_circuit(
            qubo, reps=reps, basis_gates=basis_gates
        )

    assert isinstance(circuit, qiskit.QuantumCircuit)
    assert isinstance(operator, SparsePauliOp)

    op_expected, _ = qubo.to_ising()
    assert operator == op_expected

    if basis_gates == "qiskit":
        basis_gates_list = list(qiskit_gate_map.keys())
        for gate in circuit.count_ops().keys():
            assert (
                gate in basis_gates_list
            ), f"Gate {gate} not in basis gates {basis_gates_list}"
    elif isinstance(basis_gates, list):
        for gate in circuit.count_ops().keys():
            assert gate in basis_gates, f"Gate {gate} not in basis gates {basis_gates}"


def test_interpret_samples() -> None:
    # Setup dummy samples
    samples = {
        "01": 3,
        "10": 1,
        "11": 2,
    }

    # Create a simple pulp problem with two binary variables named "x_0", "x_1"
    prob = pulp.LpProblem("test", pulp.LpMinimize)
    x0 = pulp.LpVariable("x_0", 0, 1, cat="Binary")
    x1 = pulp.LpVariable("x_1", 0, 1, cat="Binary")
    prob += x0 + x1  # objective (not used in mock)

    samples_interpreted, solutions_interpreted = interpret_samples(samples, prob)

    # Check keys and structure in samples_interpreted
    assert set(samples_interpreted.keys()) == {"01", "10", "11"}
    for bitstring, info in samples_interpreted.items():
        assert "cost_lp" in info
        assert "frequency" in info
        assert "feasible" in info
        assert info["frequency"] == samples[bitstring]

    # Check feasibility logic matches mock
    assert samples_interpreted["10"]["feasible"] is True
    assert samples_interpreted["01"]["feasible"] is True
    assert samples_interpreted["11"]["feasible"] is True

    # Check costs are correct or NaN if infeasible
    assert samples_interpreted["10"]["cost_lp"] == 1.0
    assert samples_interpreted["01"]["cost_lp"] == 1.0
    assert samples_interpreted["11"]["cost_lp"] == 2.0

    # Check solutions_interpreted has combined frequencies
    # "01" -> solution string "01", frequency 3
    # "10" -> solution string "10", frequency 1
    # "11" -> solution string "11", frequency 2
    assert solutions_interpreted["01"]["frequency"] == 3
    assert solutions_interpreted["10"]["frequency"] == 1
    assert solutions_interpreted["11"]["frequency"] == 2

    # Check cost and feasibility in solutions_interpreted match samples_interpreted
    # for each solution
    for sol_str, sol_info in solutions_interpreted.items():
        # Find a sample bitstring with same solution string
        matching_samples = [s for s in samples if s == sol_str]
        if matching_samples:
            sample_info = samples_interpreted[matching_samples[0]]
            assert sol_info["cost_lp"] == sample_info["cost_lp"]
            assert sol_info["feasible"] == sample_info["feasible"]


def test_interpret_samples_qubo() -> None:
    # Setup dummy samples
    samples = {
        # optimal:
        "01000": 3,
        # feasible:
        "10101": 1,
        # infeasible:
        "00101": 2,
        # same 'solution', different qubo sample.
        # This is to check the addition in solution_interpreted
        "00100": 4,
    }

    # Create a simple pulp problem with two binary variables named "x_0", "x_1"
    prob = pulp.LpProblem("test", pulp.LpMinimize)
    x0 = pulp.LpVariable("x_0", 0, 1, cat="Binary")
    x1 = pulp.LpVariable("x_1", 0, 1, cat="Binary")
    x2 = pulp.LpVariable("x_2", 0, 1, cat="Binary")
    prob += x0 + x1 + x2  # objective (not used in mock)
    prob += (
        x0 + x1 >= x2,
        "c1",
    )  # constraint: if x2=1, then at least one of x0 or x1 must be 1

    qubo = lp_to_qubo(prob)
    samples_interpreted, solutions_interpreted = interpret_samples(
        samples, prob, qubo=qubo
    )

    # Check feasibility logic matches mock
    assert samples_interpreted["01000"]["feasible"] is True
    assert samples_interpreted["10101"]["feasible"] is True
    assert samples_interpreted["00101"]["feasible"] is False
    assert samples_interpreted["00100"]["feasible"] is False

    # Check costs are correct or NaN if infeasible
    assert samples_interpreted["01000"]["cost_lp"] == 1.0
    assert samples_interpreted["10101"]["cost_lp"] == 2.0
    assert samples_interpreted["00101"]["cost_lp"] is float(np.nan)

    # Check the cost_qubo entries exist and are correct
    assert "cost_qubo" in samples_interpreted["01000"]
    assert samples_interpreted["01000"]["cost_qubo"] == 5.0
    assert samples_interpreted["10101"]["cost_qubo"] == 6.0
    assert samples_interpreted["00101"]["cost_qubo"] == 17.0
    assert samples_interpreted["00100"]["cost_qubo"] == 5.0

    # Check solutions_interpreted has combined frequencies
    # "01000" -> solution string "010", frequency 3
    # "10101" -> solution string "101", frequency 1
    # "00101" -> solution string "001", frequency 2
    # "00100" -> solution string "001", frequency 4 (added to previous)
    assert solutions_interpreted["010"]["frequency"] == 3
    assert solutions_interpreted["101"]["frequency"] == 1
    assert solutions_interpreted["001"]["frequency"] == 6


def test_qubo_to_z_interactions_matches_to_ising() -> None:
    """Verify qubo_to_z_interactions matches Qiskit ``to_ising``.

    This checks that both approaches produce the same Hamiltonian.
    """
    qubo = QuadraticProgram("test")
    qubo.binary_var("x0")
    qubo.binary_var("x1")
    qubo.binary_var("x2")
    qubo.minimize(linear=[-1, -1, -1], quadratic={(0, 1): 2, (1, 2): 2})

    z_interactions, offset = qubo_to_z_interactions(qubo)
    op, qiskit_offset = qubo.to_ising()

    # Offsets must match
    assert offset == pytest.approx(qiskit_offset)

    # Reconstruct Hamiltonian from z_interactions and compare to qiskit operator
    # Build a dict of {frozenset(qubits): weight} from our output (1-indexed)
    our_terms: Dict[tuple, float] = {}
    for term in z_interactions:
        key = tuple(sorted(term["qubits"]))
        our_terms[key] = our_terms.get(key, 0.0) + term["weight"]

    # Build the same from the Qiskit SparsePauliOp (0-indexed)
    qiskit_terms: Dict[tuple, float] = {}
    for pauli_op in op:
        pauli = pauli_op.paulis[0]
        coeff = pauli_op.coeffs[0].real
        z_indices = tuple(sorted(i + 1 for i in range(len(pauli.z)) if pauli.z[i]))
        if z_indices:
            qiskit_terms[z_indices] = qiskit_terms.get(z_indices, 0.0) + coeff

    # Same number of terms
    assert set(our_terms.keys()) == set(qiskit_terms.keys())
    for key in our_terms:
        assert our_terms[key] == pytest.approx(qiskit_terms[key])


def test_qubo_to_z_interactions_single_variable() -> None:
    """Test with a single-variable QUBO: min -x0."""
    qubo = QuadraticProgram("single")
    qubo.binary_var("x0")
    qubo.minimize(linear=[-1])

    z_interactions, offset = qubo_to_z_interactions(qubo)

    # x0 = (1 - Z1)/2 => -x0 = -(1 - Z1)/2 = -1/2 + Z1/2
    # So offset = -1/2, Z1 coeff = 1/2
    assert offset == pytest.approx(-0.5)
    assert len(z_interactions) == 1
    assert z_interactions[0]["qubits"] == [1]
    assert z_interactions[0]["weight"] == pytest.approx(0.5)


def test_qubo_to_z_interactions_uses_1_indexing() -> None:
    """Verify all qubit indices are 1-based (Julia convention)."""
    qubo = QuadraticProgram("indexing")
    for i in range(4):
        qubo.binary_var(f"x{i}")
    qubo.minimize(quadratic={(0, 3): 1.0})

    z_interactions, _ = qubo_to_z_interactions(qubo)

    all_qubits = set()
    for term in z_interactions:
        all_qubits.update(term["qubits"])
    # All qubits should be >= 1 (1-indexed)
    assert min(all_qubits) >= 1
    # For a (0,3) quadratic term, we expect qubits 1 and 4
    zz_terms = [t for t in z_interactions if len(t["qubits"]) == 2]
    assert any(sorted(t["qubits"]) == [1, 4] for t in zz_terms)


def test_qubo_to_z_interactions_rejects_constraints() -> None:
    """Test that a QuadraticProgram with constraints raises ValueError."""
    qp = QuadraticProgram("constrained")
    qp.binary_var("x0")
    qp.binary_var("x1")
    qp.minimize(linear=[1, 1])
    qp.linear_constraint(linear={"x0": 1, "x1": 1}, sense="<=", rhs=1)

    with pytest.raises(ValueError, match="no constraints"):
        qubo_to_z_interactions(qp)


@pytest.mark.parametrize("problem_factory_fixture, kwargs", PARAMS)
def test_qubo_to_z_interactions_parametrized(
    request: FixtureRequest, problem_factory_fixture: str, kwargs: Dict[str, int]
) -> None:
    """Test qubo_to_z_interactions matches to_ising for all problem instances."""
    optimization_problem_instance = request.getfixturevalue(problem_factory_fixture)
    lp_problem = optimization_problem_instance(**kwargs)
    qubo = lp_to_qubo(lp_problem)

    z_interactions, offset = qubo_to_z_interactions(qubo)
    _, qiskit_offset = qubo.to_ising()

    # Offsets must match
    assert offset == pytest.approx(qiskit_offset)

    # All interactions should be well-formed
    for term in z_interactions:
        assert "qubits" in term
        assert "weight" in term
        assert all(q >= 1 for q in term["qubits"])
        assert all(q <= qubo.get_num_vars() for q in term["qubits"])
        assert abs(term["weight"]) > 1e-15
