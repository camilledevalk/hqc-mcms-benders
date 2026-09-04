# milp_engine/solvers/pulp_solver.py
"""This module contains the HiGHSSolver.

This solver is a wrapper around the HiGHS solver, which is a high-performance
linear programming (LP) and mixed-integer programming (MIP) solver.

It implements the ExtremeRaySolver interface, which extends the Solver interface
with additional functionality for finding extreme rays in unbounded
problems.
"""
from typing import Tuple

import numpy as np
import pulp

from milp_engine.solvers.base_solver import ExtremeRaySolver, Solver


class HiGHSSolver(ExtremeRaySolver, Solver):
    """A solver class relying on PuLP.solve."""

    def __init__(
        self, time_limit: float | None = None, gap_tolerance: float | None = None
    ):
        """Initialise HiGHSSolver with time limit and gap tolerance.

        :param time_limit: maximum time in seconds to run the solver.
        :param gap_tolerance: maximum optimality gap for solver
        """
        self.solver = HiGHSCustom(msg=False)
        assert self.solver.available(), (
            "HiGHS solver is not available. "
            "Use `pip install highspy` to install the HiGHS solver."
        )
        self.solver.timeLimit = time_limit
        self.solver.gapRel = gap_tolerance

    def solve(self, problem: pulp.LpProblem) -> pulp.LpProblem:
        """Solves the given optimization problem using PuLP and HiGHS' python API.

        :param problem: the optimization problem to be solved.
        :return: the solved optimization problem.
        """
        problem.solve(self.solver)
        return problem

    def solve_and_find_extreme_ray(
        self,
        problem: pulp.LpProblem,
    ) -> Tuple[pulp.LpProblem, np.ndarray | None]:
        """Solve the given LP problem and return the solved problem and primal ray.

        :param problem: the LP problem to be solved.
        :return: a tuple containing the solved problem and the primal ray if
            unbounded, otherwise None.
        """
        _, ray = self.solver.solve_and_get_ray(problem)  # status in the problem object
        return problem, ray


class HiGHSCustom(pulp.HiGHS):
    """Custom HiGHS solver class to handle specific solver configurations."""

    def solve_and_get_ray(self, lp: pulp.LpProblem) -> Tuple[int, np.ndarray | None]:
        """Solve the given LP problem and return the status and primal ray.

        :param lp: the LP problem to be solved.
        :return: a tuple containing the solver status and the primal ray if
            unbounded, otherwise None.
        """
        status = super().actualSolve(lp)
        h = lp.solverModel
        ray = h.getPrimalRay()[2] if h.getPrimalRayExist() else None

        # Normalize ray if not None and status is unbounded
        if ray is not None and status == pulp.LpStatusUnbounded:
            ray = ray / np.linalg.norm(ray)

        return status, ray
