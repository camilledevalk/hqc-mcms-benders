from typing import Tuple, cast

import numpy as np
import pulp

from milp_engine.solvers.highs_solver import HiGHSSolver


def test_HiGHS_extreme_ray(
    unbounded_max_problem: Tuple[pulp.LpProblem, np.ndarray],
) -> None:
    problem, expected_ray = unbounded_max_problem
    solver = HiGHSSolver()
    solved_problem, found_ray = solver.solve_and_find_extreme_ray(problem)

    assert solved_problem.status == pulp.LpStatusUnbounded
    assert found_ray is not None
    assert np.allclose(cast(np.ndarray, found_ray), expected_ray)
