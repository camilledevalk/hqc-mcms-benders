# milp_engine/criteria/coverage_criterion.py
"""This module contains the abstract class 'CoverageCriterion'.

The Coverage criterion creates a binary indicator matrix bases on a subset
of general feasibility and optimality cuts that all or most of the decision
variables of the master problem are collectively covered.
"""
from typing import List, Tuple, cast

import numpy as np

from milp_engine.criteria.base_criterion import Criterion
from milp_engine.data.bender_state import BenderState
from milp_engine.data.cut import Cut


class CoverageCriterion(Criterion):
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
        """
        cuts = cast(List[Cut], state.candidate_cuts)  # Should be defined in BenderState

        assert len(cuts) > 0, "Cuts list must not be empty."

        # Shape: (num_cuts, num_variables)
        cut_matrix = np.stack([cut.coeff_vector for cut in cuts])

        # A non-zero coefficient for a variable in a cut indicates that the cut
        # covers that variable.
        binary_indicator_matrix = np.abs(cut_matrix) > 0

        return binary_indicator_matrix, cuts, []
