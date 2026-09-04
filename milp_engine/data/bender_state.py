# milp_engine/data/bender_state.py
"""This module contains the class 'BenderState'."""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pulp

from milp_engine.data.cut import Cut


@dataclass
class BenderState:
    """A class holding some parts of the state of Bender's Decomposition.

    This is used to pass critical information to subprocesses of the solving method
    and to do analysis or debugging of the Bender's Decomposition algorithm.
    :param mp: The master problem as a pulp.LpProblem instance.
    :param y_hats: A list of variable assignment dictionaries for y variables.
    :param cuts: A list of cuts generated from the dual subproblems.
    """

    iteration: int
    mp: pulp.LpProblem  # The master problem at the start of the iteration.
    y_hats: List[Dict[str, float]]
    lower_bound: float  # The LB when the iteration terminated.
    upper_bound: float  # The UB when the iteration terminated.
    candidate_cuts: List[Cut] | None = None
    binary_indicator_matrix: np.ndarray | None = None
    mp_update_cuts: List[Cut] | None = None
    mp_updated: pulp.LpProblem | None = None
    qaoa_fallback_all_cuts: bool = False
    runtimes: Dict[str, float | List[float] | None] = field(
        default_factory=lambda: {
            "mp_solving_time": None,
            "sp_solving_time": None,
            "cut_selection_time": None,
            "iteration_time": None,
            "individual_sp_solving_time": None,
            "individual_sp_solving_time_with_init": None,
        }
    )
