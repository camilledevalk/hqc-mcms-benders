# milp_engine/utils/set_cover.py
"""Utility functions for set cover and max coverage problems."""
import warnings

import numpy as np


def is_set_cover(binary_indicator_matrix: np.ndarray, cuts_set: set) -> bool:
    """Check if the given set of cuts is a set cover .

    :param binary_indicator_matrix: NxN numpy binary indicator matrix.
    :param cuts_set: A set of selected cuts.
    :returns: Bool that represents if a cuts_set is a set cover.
    """
    n = len(binary_indicator_matrix)
    reachable: set = set()

    for v in cuts_set:
        # Add all cuts that can be reached from cut v
        reachable.update(np.where(binary_indicator_matrix[v] > 0)[0])
        # Optionally, include the cut itself
        reachable.add(v)

    return bool(len(reachable) == n)


def determine_set_cover_class(
    binary_indicator_matrix: np.ndarray, bitstring: list[str]
) -> str | None:
    """Find class (0 or 1) in the bitstring for the minimum set cover.

    :param binary_indicator_matrix: NxN numpy binary indicator matrix
    :param bitstring: A list of strings, each '0' or '1', representing cut classes
    :return: Set of cut indices that form the minimal set cover

    Raises a warning: If neither class forms a valid set cover
    """
    class_0 = {i for i, bit in enumerate(bitstring) if bit == "0"}
    class_1 = {i for i, bit in enumerate(bitstring) if bit == "1"}

    valid_0 = is_set_cover(binary_indicator_matrix, class_0)
    valid_1 = is_set_cover(binary_indicator_matrix, class_1)

    if valid_0 and not valid_1:
        return "0"
    elif valid_1 and not valid_0:
        return "1"
    elif valid_0 and valid_1:
        return "0" if len(class_0) < len(class_1) else "1"
    else:
        warnings.warn("Neither class is a correct set cover.")
        return None


def num_cuts_covered(binary_indicator_matrix: np.ndarray, cuts_set: set[int]) -> int:
    """Count how many cuts are covered by selected cuts.

    :param binary_indicator_matrix: NxN numpy binary indicator matrix.
    :param cuts_set: A set of selected cuts.
    :returns: The number of cuts covered.
    """
    n = binary_indicator_matrix.shape[0]

    # Create a binary indicator vector x
    x = np.zeros(n, dtype=int)
    ones = np.ones(n, dtype=int)

    x[list(cuts_set)] = 1

    # The following line makes sure that cuts are only counted once,
    # also if both endpoints lie in the set if cuts.
    covered_cuts = 0.5 * (
        x @ binary_indicator_matrix @ ones + (ones - x) @ binary_indicator_matrix @ x
    )

    return covered_cuts


def determine_max_coverage_class(
    binary_indicator_matrix: np.ndarray,
    bitstring: list[str],
    max_selected_cuts: int,
) -> str | None:
    """Find class (0 or 1) for the optimal cut selection for a Max Coverage.

    :param binary_indicator_matrix: NxN numpy binary indicator matrix
    :param bitstring: List of '0' or '1' strings representing class membership
    :param max_selected_cuts: Max number of cuts that can be selected
    :return: Set of cut indices in the selected class

    Raises a warning: If neither class respect the selection limit.
    """
    class_0 = {i for i, bit in enumerate(bitstring) if bit == "0"}
    class_1 = {i for i, bit in enumerate(bitstring) if bit == "1"}

    valid_classes = []
    if len(class_0) <= max_selected_cuts:
        covered_0 = num_cuts_covered(binary_indicator_matrix, class_0)
        valid_classes.append((covered_0, "0"))
    if len(class_1) <= max_selected_cuts:
        covered_1 = num_cuts_covered(binary_indicator_matrix, class_1)
        valid_classes.append((covered_1, "1"))

    if not valid_classes:
        warnings.warn("Neither class respects the cut selection limit.")
        return None

    # Return the class with the highest cut coverage
    best_class = max(valid_classes, key=lambda x: x[1])
    return best_class[1]
