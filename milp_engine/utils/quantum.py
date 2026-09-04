# milp_engine/utils/quantum.py
"""This module contains utility function for the milp_engine."""
import os
import tempfile
from typing import Dict, Tuple

import numpy as np
import pulp
from qcshared.utils.gate_maps import qiskit_gate_map
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from milp_engine.utils.lp import get_objective_val, is_feasible


def lp_to_quadratic_program(
    optimization_problem: pulp.LpProblem,
) -> QuadraticProgram:
    """Convert a PuLP problem to a Qiskit QuadraticProgram.

    Uses docplex to convert the problem to a Qiskit-compatible format.

    :param optimization_problem: The PuLP problem to convert.
    :return: The converted Qiskit QuadraticProgram.
    """
    # Write to a unique temp file (safe for parallel workers)
    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as tmp:
        path = tmp.name

    try:
        optimization_problem.writeLP(path)

        # Use Qiskit to read the LP file
        quadratic_program = QuadraticProgram()
        quadratic_program.read_from_lp_file(path)

        # Monkey patch the name, because the name is not parsed correctly
        quadratic_program.name = optimization_problem.name
    finally:
        os.remove(path)

    return quadratic_program


def lp_to_qubo(
    optimization_problem: pulp.LpProblem,
) -> QuadraticProgram:
    """Convert a PuLP problem to a Qiskit QuadraticProgram in QUBO formay.

    Uses docplex to convert the problem to a Qiskit-compatible format.

    :param optimization_problem: The PuLP problem to convert.
    :return: The converted Qiskit QuadraticProgram that is a QUBO.
    :raises ValueError: If the LP problem contains non-binary variables.
    """
    # Check if all decision variables are binary
    for var in optimization_problem.variables():
        if var.cat != pulp.LpInteger or var.lowBound != 0 or var.upBound != 1:
            raise ValueError(
                f"Non-binary variable detected: {var.name} (type: {var.cat})"
            )

    quadratic_program = lp_to_quadratic_program(optimization_problem)

    # Convert to QUBO
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)

    return qubo


def create_qaoa_circuit(
    qubo: QuadraticProgram,
    reps: int,
    basis_gates: str | list | None = None,
    optimization_level: int = 3,
) -> Tuple[QuantumCircuit, SparsePauliOp]:
    """Generate a parameterized QAOA circuit from a given QUBO problem.

    :param qubo: A QUBO-formulated optimization problem.
    :param reps: Number of QAOA repetitions (circuit depth).
    :param basis_gates: Optional basis gates for the circuit.
        If a list, it should contain the names of the gates to use.
        If "qiskit", it uses the default Qiskit gate set.
        If None, there is no transpilation gates are used.
    :param optimization_level: Optimization level for transpilation.
        Defaults to 3, which is the highest level of optimization.
    :return: A parameterized quantum circuit implementing the QAOA ansatz.
    :raises ValueError: If basis_gates is not a list, "qiskit", or None.
    """
    operator, _ = qubo.to_ising()

    # Parse basis_gates
    match basis_gates:
        case list():
            flatten = True
        case "qiskit":
            flatten = True
            basis_gates = list(qiskit_gate_map.keys())
        case None:
            basis_gates = None
            flatten = False
        case _:
            raise ValueError(
                "basis_gates must be a list of gate names, 'qiskit', or None."
            )

    # Create the QAOA circuit
    circuit = QAOAAnsatz(cost_operator=operator, reps=reps, flatten=flatten)

    # Transpile the circuit with the specified basis gates and optimization level
    circuit_transpiled = transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )

    return circuit_transpiled, operator


def interpret_samples(
    samples: Dict[str, int],
    problem: pulp.LpProblem | QuadraticProgram,
    qubo: QuadraticProgram | None = None,
) -> tuple[Dict[str, Dict[str, float | int]], Dict[str, Dict[str, float | int]]]:
    """Interpret samples from a quantum optimization run.

    Each sample (bitstring) is mapped to:
    - Whether it is feasible for the given problem
    - Its objective/cost value
      - cost_lp: Cost according to the original LP problem
      - cost_qubo: Cost according to the QUBO formulation (if qubo is provided)
      * The difference is that the QUBO cost includes penalties for constraint
        violations.
    - How frequently it appeared in the sampling results

    :param samples: Dictionary of samples with bitstrings as keys and frequencies
        as values.
    :param problem: The problem instance to interpret the samples against.
    :param qubo: Optional QUBO instance. If not provided, assumes variable order
        matches the problem. If provided, also calculate the qubo costs.
    :return: Tuple of interpreted samples and solutions.
    """
    problem_variables = (
        problem.variables()
        if isinstance(problem, pulp.LpProblem)
        else problem.variables
    )
    problem_vars_str = set([str(v.name) for v in problem_variables])
    samples_interpreted: Dict[str, Dict[str, float | int]] = {}
    solutions_interpreted: Dict[str, Dict[str, float | int]] = {}

    # If qubo is not provided, assume variable order matches cut_selection_problem
    if qubo is None:
        qubo_vars = list(problem_variables)
    else:
        qubo_vars = qubo.variables

    for bitstring, frequency in samples.items():
        solution = {}
        solution_list = ["0"] * len(problem_variables)
        for i in range(len(qubo_vars)):
            var_name = (
                str(qubo_vars[i].name)
                if hasattr(qubo_vars[i], "name")
                else str(qubo_vars[i])
            )
            if var_name in problem_vars_str:
                val = 1 if bitstring[i] == "1" else 0
                solution[var_name] = float(val)
                solution_list[i] = "1" if bitstring[i] == "1" else "0"
        solution_string = "".join(solution_list)

        solution_feasible = is_feasible(problem, solution)
        objective_value_lp = (
            get_objective_val(problem, solution) if solution_feasible else np.nan
        )

        samples_interpreted[bitstring] = {
            "cost_lp": objective_value_lp,
            "frequency": int(frequency),
            "feasible": solution_feasible,
        }

        if qubo is not None:
            # Also calculate the QUBO cost
            qubo_solution = {
                var: float(bitstring[i]) for var, i in qubo.variables_index.items()
            }
            objective_value_qubo = get_objective_val(qubo, qubo_solution)
            samples_interpreted[bitstring]["cost_qubo"] = objective_value_qubo

        if solution_string not in solutions_interpreted:
            solutions_interpreted[solution_string] = {
                "cost_lp": objective_value_lp,
                "frequency": int(frequency),
                "feasible": solution_feasible,
            }
        else:
            if solutions_interpreted[solution_string]["frequency"] is not None:
                solutions_interpreted[solution_string]["frequency"] += int(frequency)
            else:
                solutions_interpreted[solution_string]["frequency"] = int(frequency)

    return samples_interpreted, solutions_interpreted


def qubo_to_z_interactions(
    qubo: QuadraticProgram,
) -> Tuple[list, float]:
    """Convert a QUBO to Z-interaction terms for JuliQAOA.

    Replicates the logic of ``qiskit_optimization.translators.ising.to_ising``
    but outputs lightweight dicts instead of ``SparsePauliOp``.  Uses the
    binary-to-spin mapping  x_i = (1 - Z_i) / 2.

    Qubit indices are **1-based** (Julia convention).

    :param qubo: A QUBO-formulated ``QuadraticProgram`` (no constraints,
        all binary variables).
    :return: A tuple ``(z_interactions, offset)`` where *z_interactions* is
        a list of ``{"qubits": [int, ...], "weight": float}`` dicts and
        *offset* is the constant energy shift.
    :raises ValueError: If the QUBO contains non-binary variables or
        constraints.
    """
    if qubo.get_num_vars() > qubo.get_num_binary_vars():
        raise ValueError("All variables must be binary.")
    if qubo.linear_constraints or qubo.quadratic_constraints:
        raise ValueError("QUBO must have no constraints.")

    offset = float(qubo.objective.constant * qubo.objective.sense.value)

    # Accumulate weights keyed by sorted qubit tuple (1-indexed)
    weights: Dict[tuple, float] = {}

    def _add(qubits: tuple, w: float) -> None:
        weights[qubits] = weights.get(qubits, 0.0) + w

    sense = qubo.objective.sense.value  # +1 minimise, -1 maximise

    # Linear terms:  coef * sense / 2 * (I - Z_i)
    for idx, coef in qubo.objective.linear.to_dict().items():
        weight = coef * sense / 2.0
        _add((int(idx) + 1,), -weight)  # -weight * Z_i
        offset += weight

    # Quadratic terms
    for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
        weight = coeff * sense / 4.0
        i1, j1 = int(i) + 1, int(j) + 1

        if i == j:
            offset += weight
        else:
            _add(tuple(sorted((i1, j1))), weight)  # weight * Z_i Z_j

        _add((i1,), -weight)  # -weight * Z_i
        _add((j1,), -weight)  # -weight * Z_j
        offset += weight

    # Build list, dropping near-zero terms
    z_interactions = [
        {"qubits": list(qubits), "weight": w}
        for qubits, w in weights.items()
        if abs(w) > 1e-15
    ]

    return z_interactions, offset
