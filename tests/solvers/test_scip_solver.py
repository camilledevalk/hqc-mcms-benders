# tests/solvers/test_scip_solver.py
from typing import Tuple

import numpy as np
import pulp
import pytest

from milp_engine.instances.milps import basic_milp
from milp_engine.solvers.scip_solver import ScipSolver
from milp_engine.utils.lp import is_feasible


@pytest.mark.parametrize(
    "cat_x1, cat_x2, expected_solutions",
    [
        ("Integer", "Integer", 2),
        ("Binary", "Binary", 3),
        ("Integer", "Binary", 5),
    ],
)
def test_enumerate_feasible_solutions_scip_parametrized(
    cat_x1: str, cat_x2: str, expected_solutions: int
) -> None:
    milp = basic_milp(cat_x1, cat_x2)

    solver = ScipSolver()

    multiple_solutions = solver.enumerate_feasible_solutions(milp, expected_solutions)

    assert len(multiple_solutions) == expected_solutions

    # Ensure each solution is feasible
    for solution in multiple_solutions:
        assert is_feasible(milp, solution)

    # Ensure all solutions are different
    unique_solutions = [frozenset(sol.items()) for sol in multiple_solutions]
    assert len(unique_solutions) == len(set(unique_solutions))


def test_enumerate_feasible_solutions_scip_unbounded(
    unbounded_max_problem: Tuple[pulp.LpProblem, np.ndarray],
) -> None:
    problem, _ = unbounded_max_problem

    solver = ScipSolver()

    multiple_solutions = solver.enumerate_feasible_solutions(problem, 2)

    # PuLPSolver should not be able to find feasible solutions (no extreme ray support)
    assert len(multiple_solutions) == 2
