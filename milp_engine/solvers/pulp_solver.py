# milp_engine/solvers/pulp_solver.py
"""This module contains the PuLPSolver implementation.

The 'PuLPSolver' class implements this interface using the classical PuLP
library. It solves MILPs via the CBC solver.
"""
import warnings

import numpy as np
import pulp

from milp_engine.solvers.base_solver import CutSelectionSolver, Solver


class PuLPSolver(Solver, CutSelectionSolver):
    """A solver class relying on PuLP.solve."""

    def __init__(
        self, time_limit: float | None = None, gap_tolerance: float | None = None
    ):
        """Initialise object with time limit and gap tolerance.

        :param time_limit: maximum time in seconds to run the solver.
        :param gap_tolerance: maximum optimality gap for solver
        """
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance

    def solve(self, problem: pulp.LpProblem) -> pulp.LpProblem:
        """Solves the given linear problem using PuLP's (CBC) solver.

        :param problem: the linear problem to be solved.
        :return: the solved linear problem.
        """
        # Default to the CBC solver
        solver = pulp.PULP_CBC_CMD(msg=False)

        # Support for additional solvers will be added in the future, as their
        # input parameters vary.

        # Set common solver parameters if available
        if hasattr(solver, "timeLimit") and self.time_limit is not None:
            solver.timeLimit = self.time_limit
            print("Set solver time limit")
        else:
            warnings.warn("No time limit set for the solver.")
        if hasattr(solver, "gapRel"):
            solver.gapRel = self.gap_tolerance
            print("Set solver optimality gap")
        else:
            warnings.warn("No optimality gap set for the solver.")

        problem.solve(solver)
        return problem

    def solve_with_binary_indicator_matrix(
        self,
        problem: pulp.LpProblem,
        binary_indicator_matrix: np.ndarray,
        max_number_covered: int | None = None,
    ) -> pulp.LpProblem:
        """Solves the given optimization problem using PuLP's solver.

        This is simply a wrapper for the 'solve' method.

        :param problem: the linear problem to be solved.
        :param binary_indicator_matrix: NxN numpy array representing the
            binary indicator matrix.
        :param max_number_covered: Maximum number of elements that may be
            selected or covered during the solution process. Defaults to 0.
        :return: the solved linear problem.

        Note: this function is simply a wrapper for the PuLPSolver.solve method.
        """
        return self.solve(problem)
