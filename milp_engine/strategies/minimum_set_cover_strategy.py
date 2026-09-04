# milp_engine/strategies/minimum_set_cover_strategy.py
"""This module contains the class 'MinimumSetCoverStrategy'.

The MinimumSetCover strategy aims to select a subset of cuts that cover all
relevant constraints based on the binary indicator matrix. The goal is to minimize
the number of cuts while ensuring full coverage.
"""
from typing import Tuple

import numpy as np
import pulp

from milp_engine.strategies.base_strategy import Strategy


class MinimumSetCoverStrategy(Strategy):
    """Strategy implementation using the Minimum Set Cover approach."""

    def create_optimization_problem(
        self, binary_indicator_matrix: np.ndarray
    ) -> Tuple[pulp.LpProblem, list]:
        """Create a Minimum Set Cover optimization problem.

        :param binary_indicator_matrix: A numpy array representing
            the binary indicator matrix.
        :return: An optimization problem, list of decision variables.
        """
        # Get number of num_elements
        num_elements, num_cuts = binary_indicator_matrix.shape

        # Define LP problem
        problem = pulp.LpProblem("Minimum_Set_Cover", pulp.LpMinimize)

        # Define binary decision variables for each element in the binary indicator
        # matrix
        x = [
            pulp.LpVariable(f"x_{i + 1}", cat=pulp.LpBinary)
            for i in range(num_elements)
        ]

        # Objective function: Minimize the sum of selected elements
        problem.setObjective(pulp.lpSum(x))

        # Constraints: For each edge (i, j), at least one of i or j must be in the cover
        for j in range(num_cuts):
            if binary_indicator_matrix[:, j].sum() == 0:
                continue
            problem += (
                pulp.lpSum(
                    x[i]
                    for i in range(num_elements)
                    if binary_indicator_matrix[i, j] == 1
                )
                >= 1,
                f"M_{j + 1}",
            )
        return problem, x
