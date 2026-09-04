# tests/criteria/test_criterion.py
import numpy as np
import pulp

from milp_engine.criteria.coverage_criterion import CoverageCriterion
from milp_engine.criteria.exclusion_criterion import ExclusionCriterion
from milp_engine.data.bender_state import BenderState
from milp_engine.data.cut import Cut, CutType


def test_construct_binary_indicator_matrix_exclusion_criterion() -> None:
    y1, y2, z = "y1", "y2", "z"

    cuts = [
        Cut(np.array([-1, -1, 0]), 1000, CutType.FEASIBILITY),
        Cut(np.array([-1, 0, 0]), 1400, CutType.FEASIBILITY),
        Cut(np.array([1, 1, 0]), -2000, CutType.FEASIBILITY),
        Cut(np.array([0, 0, 1]), -500, CutType.FEASIBILITY),
        Cut(np.array([1, 1, 1]), -2500, CutType.FEASIBILITY),
        Cut(np.array([2, 3, 1]), -4000, CutType.OPTIMALITY),
        Cut(np.array([1, 0, -1]), 100, CutType.OPTIMALITY),
    ]

    y_stars = [
        {y1: 500.0, y2: 400.0, z: 100.0},
        {y1: 1300.0, y2: 0.0, z: 600.0},
        {y1: 900.0, y2: 1200.0, z: 400.0},
        {y1: 1000.0, y2: 1000.0, z: 600.0},
        {y1: 900.0, y2: 1600.0, z: 400.0},
        {y1: 1000.0, y2: 1000.0, z: 600.0},
        {y1: 900.0, y2: 1600.0, z: 400.0},
    ]

    expected_matrix = np.array(
        [
            [1, 0, 0, 0, 0],  # 1000 - y1 - y2 check
            [1, 1, 1, 1, 1],  # 1400 - y1 check
            [0, 0, 1, 0, 1],  # y1 + y2 - 2000 check
            [0, 1, 0, 1, 0],  # y3 - 500 check
            [0, 0, 0, 1, 1],  # y1 + y2 + z - 2500 check
        ]
    )
    criterion = ExclusionCriterion()

    state = BenderState(
        iteration=0,
        mp=pulp.LpProblem(),
        y_hats=y_stars,
        candidate_cuts=cuts,
        lower_bound=-1000000.0,
        upper_bound=np.inf,
    )
    matrix, selection_cuts, passed_cuts = criterion.construct_binary_indicator_matrix(
        state
    )

    expected_selection_cuts = cuts[:5]
    assert (
        selection_cuts == expected_selection_cuts
    ), "Selection cuts should match expected cuts"

    expected_passed_cuts = cuts[5:]
    assert passed_cuts == expected_passed_cuts, "Passed cuts should match expected cuts"

    assert isinstance(matrix, np.ndarray), "Output should be a NumPy array"
    assert matrix.shape == (
        len(expected_selection_cuts),
        len(expected_selection_cuts),
    ), "Matrix should be square with shape (len(selection_cuts), len(selection_cuts))"
    assert (
        matrix.shape == expected_matrix.shape
    ), "Matrix shape should match expected result"
    np.testing.assert_array_equal(matrix, expected_matrix)


def test_construct_binary_indicator_matrix_coverage_criterion() -> None:
    y1, y2, y3 = [pulp.LpVariable(v) for v in ("y1", "y2", "y3")]

    cuts = [
        Cut(np.array([-1, -1, 0]), 1000, CutType.FEASIBILITY),
        Cut(np.array([-1, 0, 0]), 1400, CutType.OPTIMALITY),
        Cut(np.array([1, 1, 0]), -2000, CutType.FEASIBILITY),
        Cut(np.array([0, 0, 1]), -500, CutType.OPTIMALITY),
        Cut(np.array([1, 1, 1]), -2500, CutType.FEASIBILITY),
    ]

    expected_matrix = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )

    decision_variables = [y1, y2, y3]
    mp = pulp.LpProblem("Test Problem", pulp.LpMinimize)
    mp += pulp.lpSum(decision_variables)
    criterion = CoverageCriterion()

    state = BenderState(
        iteration=0,
        mp=pulp.LpProblem(),
        y_hats=[],
        candidate_cuts=cuts,
        lower_bound=-1000000.0,
        upper_bound=np.inf,
    )

    matrix, selection_cuts, passed_cuts = criterion.construct_binary_indicator_matrix(
        state
    )

    expected_selection_cuts = cuts
    assert (
        selection_cuts == expected_selection_cuts
    ), "Selection cuts should match all cuts"
    assert passed_cuts == [], "Passed cuts should be empty"

    assert isinstance(matrix, np.ndarray), "Output should be a NumPy array"
    assert matrix.shape == (
        len(cuts),
        len(decision_variables),
    ), "Matrix should be square with shape (len(cuts), len(cuts))"
    assert (
        matrix.shape == expected_matrix.shape
    ), "Matrix shape should match expected result"
    np.testing.assert_array_equal(matrix, expected_matrix)
