# milp_engine/strategies/base_strategy.py
"""This module contains the abstract class 'Strategy'.

Abstract class 'Strategy' represents the strategy for selecting cuts based on
a binary indicator matrix.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pulp

from milp_engine.data.cut import Cut
from milp_engine.solvers.base_solver import CutSelectionSolver


class Strategy(ABC):
    """Abstract class defining strategy."""

    @abstractmethod
    def create_optimization_problem(
        self, binary_indicator_matrix: np.ndarray
    ) -> Tuple[pulp.LpProblem, list]:
        """Abstract method for creating an optimization problem.

        :param binary_indicator_matrix: A numpy array representing
            the binary indicator matrix.
        :return: An optimization problem and list of decision variables.
        """
        pass

    def select_cuts(
        self,
        cuts: list[Cut],
        binary_indicator_matrix: np.ndarray | None,
        solver: CutSelectionSolver,
    ) -> list[Cut]:
        """Abstract method for selecting the cuts to add to the Master Problem.

        :param cuts: List of cuts used in Benders decomposition.
        :param binary_indicator_matrix: A numpy array representing
            the binary indicator matrix.
        :param solver: A solver to use for solving the optimization problem.
        :return: A list of cuts to add to the Master Problem.
        """
        # Extract the optimization_problem and (decision) variables of
        # the optimization problem (called 'x').
        if binary_indicator_matrix is not None:
            optimization_problem, x = self.create_optimization_problem(
                binary_indicator_matrix
            )

            maximum_number_of_cuts = getattr(self, "maximum_number_of_cuts", None)

            solver.solve_with_binary_indicator_matrix(
                optimization_problem,
                binary_indicator_matrix,
                maximum_number_of_cuts,
            )
            # Retrieve the indices of binary decision variables that are set to one.
            assignment = [i for i in range(len(x)) if pulp.value(x[i]) == 1]
            # Determine the selected cuts based on the binary decision variable
            #     assignment.
            selected_cuts = [cuts[i] for i in assignment]
        else:
            selected_cuts = cuts

        return selected_cuts
