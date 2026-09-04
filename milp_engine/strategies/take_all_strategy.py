# milp_engine/strategies/take_all_strategy.py
"""This module contains the basic 'TakeAllStrategy'.

This strategy simply selects all cuts, always.
"""
from typing import Tuple

import numpy as np
import pulp

from milp_engine.data.cut import Cut
from milp_engine.solvers.base_solver import CutSelectionSolver, Solver
from milp_engine.strategies.base_strategy import Strategy


class TakeAllStrategy(Strategy):
    """Class defining naive cut selection strategy."""

    def create_optimization_problem(
        self, binary_indicator_matrix: np.ndarray | None
    ) -> Tuple[pulp.LpProblem, list]:
        """Non-implemented method for creating an optimization problem.

        :param binary_indicator_matrix: A numpy array representing
            the binary indicator matrix.
        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("No need to create an optimization problem.")

    def select_cuts(
        self,
        cuts: list[Cut],
        binary_indicator_matrix: np.ndarray | None,
        solver: Solver | CutSelectionSolver,
        max_number_covered: int | None = None,
    ) -> list[Cut]:
        """Simply returns all cuts immediately.

        Duplicate cuts are removed.

        :param cuts: List of cuts used in Benders decomposition.
        :param binary_indicator_matrix: A numpy array representing
            the binary indicator matrix.
        :param solver: A solver to use for solving the optimization problem.
        :param max_number_covered: Maximum number of elements that may be
            selected or covered during the solution process.
        :return: A set of cuts to add to the Master Problem.
        """
        return cuts
