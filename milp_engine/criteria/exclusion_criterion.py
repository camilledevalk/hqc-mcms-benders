# milp_engine/criteria/exclusion_criterion.py
"""This module contains the class 'ExclusionCriterion'.

The Exclusion criterion creates a binary indicator matrix based on a subset
of general feasibility and optimality cuts that collectively cover all or
most of the decision variables of the master problem.
"""
from typing import List, Tuple, cast

import numpy as np

from milp_engine.criteria.base_criterion import Criterion
from milp_engine.data.bender_state import BenderState
from milp_engine.data.cut import Cut, CutType


class ExclusionCriterion(Criterion):
    """Criterion that creates a binary indicator matrix."""

    @staticmethod
    def construct_binary_indicator_matrix(
        state: BenderState,
    ) -> Tuple[np.ndarray | None, List[Cut], List[Cut]]:
        """Construct a binary indicator matrix based on the given BenderState.

        :param state: An object containing some state variables of the
            Bender solver.
        :return: A binary indicator matrix as a NumPy array,
            the list of cuts used to construct the matrix, and the list of cuts that
            this criterion does not consider.
        :raises ValueError: If cuts and y_stars are different lengths or if either of
            them is empty.
        """
        cuts = cast(List[Cut], state.candidate_cuts)  # Should be defined in BenderState

        assert len(cuts) > 0, "Cuts list must not be empty."

        feasibility_cuts = [cut for cut in cuts if cut.type == CutType.FEASIBILITY]
        optimality_cuts = [cut for cut in cuts if cut.type == CutType.OPTIMALITY]

        if not feasibility_cuts:
            return None, [], optimality_cuts

        y_hats_np = [
            np.array(
                [v for name, v in sorted(y_hat.items()) if name != "z"] + [y_hat["z"]]
            )  # sorted by variable names and with z in the back
            for y_hat, cut in zip(state.y_hats, cuts)
            if cut.type == CutType.FEASIBILITY
        ]

        if len(feasibility_cuts) != len(y_hats_np):
            raise ValueError("Cuts list and y_stars array must be same length.")

        for y_hat_np in y_hats_np:
            if len(y_hat_np) != len(y_hats_np[0]):
                raise ValueError(
                    "All variable assignments must have the same number of elements."
                )

        # Shape: (num_cuts, num_variables)
        cut_matrix = np.stack([cut.coeff_vector for cut in feasibility_cuts])

        # Shape: (num_cuts,)
        constants = np.array([cut.constant for cut in feasibility_cuts])

        # Shape: (num_y_hats, num_variables)
        y_hat_matrix = np.stack(y_hats_np)

        # Shape: (num_cuts, num_y_hats)
        result_matrix = cut_matrix @ y_hat_matrix.T + constants.reshape(-1, 1)

        # A non-positive product indicates that the cut is satisfied by the y_hat.
        #     Cuts always have form "<= 0"
        # Hence, a positive product indicates that the y_hat is infeasible with respect
        #     to the cut.
        binary_indicator_matrix = result_matrix > 0

        return binary_indicator_matrix, feasibility_cuts, optimality_cuts
