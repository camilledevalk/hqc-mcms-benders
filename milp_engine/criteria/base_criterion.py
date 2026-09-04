# milp_engine/criteria/base_criterion.py
"""This module contains the abstract class 'Criterion'.

Abstract class 'Criterion' represents a criterion in a
Multiple Cut Multiple Solutions (MCMS) Benders decomposition (BD).
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from milp_engine.data.bender_state import BenderState
from milp_engine.data.cut import Cut


class Criterion(ABC):
    """Abstract class representing a criterion."""

    @staticmethod
    @abstractmethod
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
        pass
