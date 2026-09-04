# tests/utils/test_base_solver.py
from typing import Dict, List, Tuple

import numpy as np
import pulp
import pytest

from milp_engine.solvers.base_solver import Solver, _build_exclusion_constraints
from milp_engine.solvers.highs_solver import HiGHSSolver
from milp_engine.solvers.pulp_solver import PuLPSolver
from milp_engine.solvers.scip_solver import ScipSolver
from milp_engine.utils.lp import is_feasible


@pytest.mark.parametrize(
    "milp_tag, solver",
    [
        (tag, solver)
        for tag in [
            "binary_choice_milp",
            "savings_milp",
        ]
        for solver in [ScipSolver(), PuLPSolver(), HiGHSSolver()]
    ],
)
def test_solve(
    milp_tag: str,
    solver: Solver,
    milps: Dict[str, Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]],
) -> None:
    milp, _, solution = milps[milp_tag]

    solver = ScipSolver()
    solved_milp = solver.solve(milp)

    assert solved_milp.status == pulp.LpStatusOptimal
    for var_name, var_value in solution.items():
        assert solved_milp.variablesDict()[var_name].varValue == var_value


def test_enumerate_feasible_solutions_iterative_integer() -> None:
    # Define a maximization problem with two integer variables
    problem = pulp.LpProblem("mock_problem", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0, upBound=5, cat="Integer")
    x2 = pulp.LpVariable("x2", lowBound=1, upBound=4, cat="Integer")

    # Objective: maximize x1 + x2
    problem += x1 + x2, "Maximize"
    # Constraint: sum of x1 and x2 should be at most 8
    problem += x1 + x2 <= 8, "Constraint1"

    solver = PuLPSolver()
    multiple_solutions = solver.enumerate_feasible_solutions(problem, 2)

    # Ensure each solution is feasible
    for solution in multiple_solutions:
        assert is_feasible(problem, solution)

    # Check that all solutions are unique
    unique_solutions = [frozenset(sol.items()) for sol in multiple_solutions]
    assert len(unique_solutions) == len(set(unique_solutions))

    # Check that the solutions contain the same variables as the original problem
    for solution in multiple_solutions:
        assert set(solution.keys()) == {v.name for v in problem.variables()}


def test_enumerate_feasible_solutions_iterative_binary() -> None:
    # Define a maximization problem with binary (0/1) variables
    problem = pulp.LpProblem("mock_problem", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0, upBound=1, cat="Integer")
    x2 = pulp.LpVariable("x2", lowBound=0, upBound=1, cat="Integer")

    problem += x1 + x2, "Maximize"
    problem += x1 + x2 <= 8, "Constraint1"

    solver = PuLPSolver()
    multiple_solutions = solver.enumerate_feasible_solutions(problem, 3)

    assert len(multiple_solutions) == 3

    # Ensure each solution is feasible
    for solution in multiple_solutions:
        assert is_feasible(problem, solution)

    # Ensure all returned solutions are distinct
    unique_solutions = [frozenset(sol.items()) for sol in multiple_solutions]
    assert len(unique_solutions) == len(set(unique_solutions))

    # Check that the solutions contain the same variables as the original problem
    for solution in multiple_solutions:
        assert set(solution.keys()) == {v.name for v in problem.variables()}


def test_enumerate_feasible_solutions_iterative_combination() -> None:
    # Define a maximization problem with a binary and general integer variable
    problem = pulp.LpProblem("mock_problem", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0, upBound=1, cat="Integer")
    x2 = pulp.LpVariable("x2", lowBound=0, upBound=10, cat="Integer")

    problem += x1 + x2, "Maximize"
    problem += x1 + x2 <= 8, "Constraint1"

    solver = PuLPSolver()
    multiple_solutions = solver.enumerate_feasible_solutions(problem, 5)

    assert len(multiple_solutions) == 5

    # Ensure each solution is feasible
    for solution in multiple_solutions:
        assert is_feasible(problem, solution)

    # Ensure all solutions are different
    unique_solutions = [frozenset(sol.items()) for sol in multiple_solutions]
    assert len(unique_solutions) == len(set(unique_solutions))

    # Check that the solutions contain the same variables as the original problem
    for solution in multiple_solutions:
        assert set(solution.keys()) == {v.name for v in problem.variables()}


def test_enumerate_feasible_solutions_iterative_unbounded(
    unbounded_max_problem: Tuple[pulp.LpProblem, np.ndarray],
) -> None:
    problem, ray = unbounded_max_problem
    solver = PuLPSolver()
    solutions = solver.enumerate_feasible_solutions(problem, 2)

    # PuLPSolver should not be able to find feasible solutions (no extreme ray support)
    assert len(solutions) == 0

    # Check that the solutions contain the same variables as the original problem
    for solution in solutions:
        assert set(solution.keys()) == {v.name for v in problem.variables()}


def test_enumerate_feasible_solutions_iterative_unbounded_rich(
    unbounded_max_problem: Tuple[pulp.LpProblem, np.ndarray],
) -> None:
    problem, ray = unbounded_max_problem
    for var in problem.variables():
        # enumerate_feasible_solutions only works with integer variables
        var.cat = pulp.LpInteger

    solver = HiGHSSolver()
    solutions = solver.enumerate_feasible_solutions(problem, 5)

    assert len(solutions) == 5

    # Ensure each solution is feasible
    for solution in solutions:
        assert is_feasible(problem, solution)

    # Ensure all solutions are different
    unique_solutions = [frozenset(sol.items()) for sol in solutions]
    assert len(unique_solutions) == len(set(unique_solutions))

    # Check that the solutions contain the same variables as the original problem
    for solution in solutions:
        assert set(solution.keys()) == {v.name for v in problem.variables()}


def test_build_exclusion_constraints_typing() -> None:
    # Create a dummy LP problem
    prob = pulp.LpProblem("TestProblem", pulp.LpMinimize)
    x = pulp.LpVariable("x", cat=pulp.LpInteger, lowBound=0, upBound=1)
    y = pulp.LpVariable("y", cat=pulp.LpInteger, lowBound=0, upBound=10)

    # Assign dummy solution values to simulate prior solving
    x.varValue = 1
    y.varValue = 5

    # Call the function under test
    (
        modified_prob,
        exclusion_binary,
        exclusion_combs,
        num_int_vars,
    ) = _build_exclusion_constraints(
        problem=prob, original_vars=[x, y], solution_index=0
    )

    # Type assertions
    assert isinstance(modified_prob, pulp.LpProblem)
    assert isinstance(exclusion_binary, list)
    assert isinstance(exclusion_combs, list)
    assert all(isinstance(expr, pulp.LpAffineExpression) for expr in exclusion_binary)
    assert all(isinstance(var, pulp.LpVariable) for var in exclusion_combs)
    assert isinstance(num_int_vars, int)
