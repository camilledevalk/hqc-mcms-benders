# milp_engine/data/cut.py
"""This module contains the classes 'Cut' and 'CutType'."""
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class CutType(Enum):
    """Enumeration of cut types."""

    FEASIBILITY = "feasibility"
    OPTIMALITY = "optimality"


@dataclass
class Cut:
    """A class to hold a cut.

    It is assumed to always be a '<=' relation with a 0 on the right-hand side.

    :param coeff_vector: The coefficients of the cut in the form of a numpy array.
        The coefficients for y should be indexed following the alphabetical order of
        the variable names. The final element in the vector is the coefficient for the
        z variable.
    :param constant: The constant term of the cut.
    :param type: The type of cut: feasibility cut or optimality cut.
    """

    coeff_vector: np.ndarray
    constant: float
    type: CutType

    def __eq__(self, other: Any) -> bool:
        """Check equality of two Cut instances by doing comparisons on fields.

        :param other: The other Cut instance to compare with.
        :return: True if both instances are equal, False otherwise.
        """
        if not isinstance(other, Cut):
            return False
        return (
            np.array_equal(self.coeff_vector, other.coeff_vector)
            and self.constant == other.constant
            and self.type == other.type
        )
