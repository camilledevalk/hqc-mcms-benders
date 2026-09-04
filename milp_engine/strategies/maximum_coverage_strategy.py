# milp_engine/strategies/maximum_coverage_strategy.py
"""This module contains the abstract class 'MaximumCoverageStrategy'.

The MaximumCoverage strategy aims to select a subset of cuts that cover
a maximum number of infeasible solutions or a maximum
number of MP decision variables are covered.
"""
import warnings
from typing import Tuple

import numpy as np
import pulp

from milp_engine.strategies.base_strategy import Strategy


class MaximumCoverageStrategy(Strategy):
    """Strategy implementation using the Maximum Coverage approach."""

    def __init__(self, max_cuts: int):
        """Initialize the optimization problem with a maximum number of cuts.

        :param max_cuts: The maximum number of cuts to be selected.
        """
        self.maximum_number_of_cuts = max_cuts

    def create_optimization_problem(
        self,
        binary_indicator_matrix: np.ndarray,
    ) -> Tuple[pulp.LpProblem, list]:
        """Create a Maximum Coverage optimization problem.

        :param binary_indicator_matrix: A numpy array representing
            the binary indicator matrix.
        :return: An optimization problem, list of decision variables.
        """
        # Check for non-zero diagonal elements
        if np.any(np.diag(binary_indicator_matrix) != 0):
            warnings.warn(
                "The binary_indicator_matrix has non-zero values on the diagonal. "
                "These may represent self-connections, which are usually unintended "
                "in coverage problems.",
                UserWarning,
            )
        # Ensure that the maximum number of cuts does not exceed the
        # number of available cuts.
        if self.maximum_number_of_cuts > binary_indicator_matrix.shape[0]:
            ValueError(
                "The maximum number of cuts exceeds the number of available cuts."
            )

        # Define LP problem
        problem = pulp.LpProblem("Minimum_Set_Cover", pulp.LpMaximize)

        # Define binary decision variables for each element in the binary indicator
        # matrix
        x = [
            pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary)
            for i in range(binary_indicator_matrix.shape[0])
        ]
        phi = [
            pulp.LpVariable(f"phi_{j}", cat=pulp.LpBinary)
            for j in range(binary_indicator_matrix.shape[1])
        ]

        # Objective function: Minimize the sum of selected elements
        problem.setObjective(pulp.lpSum(phi))

        # Constraint: limit the number of cuts that can be added
        problem.addConstraint(
            pulp.lpSum(x) <= self.maximum_number_of_cuts, "Max number of cuts"
        )

        # Constraints: Determine coverage of decision variable j
        for j in range(binary_indicator_matrix.shape[1]):
            problem.addConstraint(
                pulp.lpSum(
                    x[i] * binary_indicator_matrix[i, j]
                    for i in range(binary_indicator_matrix.shape[0])
                )
                >= phi[j],
                f"phi_{j}",
            )

        return problem, x
