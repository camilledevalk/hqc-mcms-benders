import itertools
from typing import Dict, List, Tuple

import pulp
import pytest

from milp_engine.solvers.pulp_solver import PuLPSolver
from milp_engine.utils.lp import get_objective_val, is_feasible


@pytest.mark.parametrize(
    "time_limit, gap_tolerance", list(itertools.product([0.10, None], [0.05, None]))
)
def test_solve_with_time_limit_and_gap_tolerance(
    time_limit: float | None, gap_tolerance: float | None
) -> None:
    # Mock the optimization problem details
    problem = pulp.LpProblem("Mock problem", pulp.LpMinimize)

    # Define binary decision variables for each element
    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(1)]

    # Objective function: Minimize the sum of selected elements
    problem.setObjective(pulp.lpSum(x))

    # Define solver with parameterized time_limit and gap_tolerance
    solver = PuLPSolver(time_limit=time_limit, gap_tolerance=gap_tolerance)

    # Test if the solver was called with the correct parameters
    result = solver.solve(problem)

    # Ensure the optimization problem is returned as expected
    assert result == problem


@pytest.mark.parametrize(
    "x, expected_results, solutions",
    [
        (
            [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(3)],
            [(True, 2), (False, 2), (False, 3), (False, 2)],
            [
                {"x_0": 1, "x_1": 1, "x_2": 0},
                {"x_0": 1, "x_1": 0, "x_2": 1},
                {"x_0": 1, "x_1": 1, "x_2": 1},
                {"x_0": 1.1, "x_1": 0.9, "x_2": 0},
            ],
        ),
        (
            [
                pulp.LpVariable(f"x_{i}", lowBound=-2, upBound=2, cat=pulp.LpContinuous)
                for i in range(3)
            ],
            [(True, 2), (False, 2), (False, 3), (True, 2.0)],
            [
                {"x_0": 1, "x_1": 1, "x_2": 0},
                {"x_0": 1, "x_1": 0, "x_2": 1},
                {"x_0": 1, "x_1": 1, "x_2": 1},
                {"x_0": 1.1, "x_1": 1.4, "x_2": -0.5},
            ],
        ),
    ],
)
def test_evaluate_solution(
    x: List[pulp.LpVariable],
    expected_results: List[Tuple[bool, float | int]],
    solutions: List[Dict[str, float | int]],
) -> None:
    # Mock the optimization problem details
    problem = pulp.LpProblem("Mock problem", pulp.LpMinimize)

    # Objective function: Minimize the sum of selected elements
    problem.setObjective(pulp.lpSum(x))

    # Define problem constraints
    problem.addConstraint(x[0] + x[1] + x[2] <= 2)
    problem.addConstraint(x[0] + x[1] >= 2)

    # Check evaluation for every solution
    for i, solution in enumerate(solutions):
        result = is_feasible(problem, solution), get_objective_val(problem, solution)
        # Check that the result is feasible and objective value is correct
        assert result == expected_results[i]


def test_evaluate_solution_missing_variable() -> None:
    # Mock the optimization problem details
    problem = pulp.LpProblem("Mock problem", pulp.LpMinimize)

    # Define binary decision variables for each element
    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(1)]

    # Objective function: Minimize the sum of selected elements
    problem.setObjective(pulp.lpSum(x))

    # Define empty solution
    solution: Dict[str, float] = {}

    feasible = is_feasible(problem, solution)

    # Ensure that the solution is not feasible because of missing variable
    assert not feasible
