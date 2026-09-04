# tests/utils/test_set_cover.py
import numpy as np
import pytest

from milp_engine.utils.set_cover import (
    determine_max_coverage_class,
    determine_set_cover_class,
    is_set_cover,
    num_cuts_covered,
)


@pytest.mark.parametrize(
    "binary_indicator_matrix, cuts_set, expected",
    [
        (np.array([[0, 1], [1, 0]]), {0}, True),
        (np.array([[0, 1], [1, 0]]), {1}, True),
        (np.array([[0, 1], [1, 0]]), set(), False),
        (np.zeros((3, 3), dtype=int), set(), False),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), {1}, True),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), {0, 2}, True),
    ],
)
def test_is_set_cover(
    binary_indicator_matrix: np.ndarray, cuts_set: set[int], expected: bool
) -> None:
    assert is_set_cover(binary_indicator_matrix, cuts_set) == expected


@pytest.mark.parametrize(
    "binary_indicator_matrix, bitstring, expected",
    [
        # 3x3 matrix
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), ["1", "0", "1"], "0"),
        # 4x4 matrix
        (
            np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]),
            ["1", "1", "0", "1"],
            "1",
        ),
        # 5x5 matrix
        (
            np.array(
                [
                    [0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0],
                ]
            ),
            ["0", "1", "0", "1", "0"],
            "1",
        ),
        # 6x6 matrix
        (
            np.array(
                [
                    [0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                ]
            ),
            ["1", "0", "1", "0", "1", "0"],
            "1",
        ),
    ],
)
def test_determine_set_cover_class_valid(
    binary_indicator_matrix: np.ndarray, bitstring: list[str], expected: str
) -> None:
    assert determine_set_cover_class(binary_indicator_matrix, bitstring) == expected


@pytest.mark.parametrize(
    "binary_indicator_matrix, cuts_set, expected_covered_cuts",
    [
        # 3x3
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), {1}, 2),
        # 4x4
        (np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]), {1, 2}, 3),
        # 5x5
        (
            np.array(
                [
                    [0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0],
                ]
            ),
            {0, 2, 4},
            5,
        ),
        # 6x6
        (
            np.array(
                [
                    [0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                ]
            ),
            {0, 2, 4},
            6,
        ),
    ],
)
def test_num_cuts_covered(
    binary_indicator_matrix: np.ndarray,
    cuts_set: set[int],
    expected_covered_cuts: int,
) -> None:
    assert num_cuts_covered(binary_indicator_matrix, cuts_set) == expected_covered_cuts


@pytest.mark.parametrize(
    "binary_indicator_matrix, bitstring, max_selected_cuts, expected_class",
    [
        # 3x3
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), ["0", "1", "0"], 2, "1"),
        # 4x4
        (
            np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]),
            ["1", "0", "0", "0"],
            1,
            "1",
        ),
        # 5x5
        (
            np.array(
                [
                    [0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0],
                ]
            ),
            ["1", "0", "1", "0", "1"],
            3,
            "1",
        ),
        # 6x6
        (
            np.array(
                [
                    [0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                ]
            ),
            ["0", "0", "0", "1", "0", "1"],
            3,
            "1",
        ),
    ],
)
def test_determine_max_coverage_class(
    binary_indicator_matrix: np.ndarray,
    bitstring: list[str],
    max_selected_cuts: int,
    expected_class: str,
) -> None:
    assert (
        determine_max_coverage_class(
            binary_indicator_matrix, bitstring, max_selected_cuts
        )
        == expected_class
    )
