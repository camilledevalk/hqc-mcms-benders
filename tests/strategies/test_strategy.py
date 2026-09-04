# tests/strategies/test_strategy.py
import numpy as np
import pulp
import pytest

from milp_engine.data.cut import Cut, CutType
from milp_engine.solvers.pulp_solver import PuLPSolver
from milp_engine.strategies.base_strategy import Strategy
from milp_engine.strategies.maximum_coverage_strategy import MaximumCoverageStrategy
from milp_engine.strategies.minimum_set_cover_strategy import MinimumSetCoverStrategy


def test_abstract_method() -> None:
    with pytest.raises(TypeError):  # Abstract classes shouldn't be instantiable
        Strategy()  # type: ignore[abstract]


def test_minimum_set_cover_strategy() -> None:
    """Test that MinimumSetCoverStrategy.select_cuts returns the correct format
    and ensures the number of selected cuts is within expected bounds.
    """
    strategy = MinimumSetCoverStrategy()
    y1, y2, y3, y4 = [Cut(np.random.rand(2), i, CutType.FEASIBILITY) for i in range(4)]

    cuts = [y1, y2, y3, y4]  # Example cuts
    binary_indicator_matrix = np.array(
        [[0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 0]]
    )
    solver = PuLPSolver()
    selected_cuts = strategy.select_cuts(cuts, binary_indicator_matrix, solver)
    assert y1 in selected_cuts and y4 in selected_cuts


def test_minimum_set_cover_strategy_empty_universe() -> None:
    """Test the select_cuts method when the binary indicator matrix represents
    an empty universe.
    """
    strategy = MinimumSetCoverStrategy()
    cuts = [Cut(np.random.rand(2), i, CutType.FEASIBILITY) for i in range(4)]

    binary_indicator_matrix = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )

    solver = PuLPSolver()
    result = strategy.select_cuts(cuts, binary_indicator_matrix, solver)

    assert len(result) == 0  # No connections, no cuts should be selected


def test_minimum_set_cover_strategy_full_universe() -> None:
    """Test the select_cuts method when the binary indicator matrix represents
    a full universe.
    """
    strategy = MinimumSetCoverStrategy()
    cuts = [Cut(np.random.rand(2), i, CutType.FEASIBILITY) for i in range(4)]

    binary_indicator_matrix = np.array(
        [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
    )

    solver = PuLPSolver()
    result = strategy.select_cuts(cuts, binary_indicator_matrix, solver)

    # Expecting at more than half of the elements in the cover
    assert len(result) == 2


def test_minimum_set_cover_strategy_single_edge() -> None:
    """Test the select_cuts method when the binary indicator matrix represents
    an universe with a single connection.
    """
    strategy = MinimumSetCoverStrategy()
    cuts = [
        Cut(np.array([0.0]), 0, CutType.FEASIBILITY),
        Cut(np.array([1.0]), 1, CutType.FEASIBILITY),
    ]

    binary_indicator_matrix = np.array([[0, 1], [1, 0]])

    solver = PuLPSolver()

    result = strategy.select_cuts(cuts, binary_indicator_matrix, solver)

    assert cuts == result


def test_minimum_set_cover_strategy_create_optimization_problem() -> None:
    # Define a simple binary indicator matrix
    binary_indicator_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # Call the function
    strategy = MinimumSetCoverStrategy()  # Replace with your class name if necessary
    problem, x = strategy.create_optimization_problem(binary_indicator_matrix)

    # Test if the problem is of the correct type
    assert isinstance(problem, pulp.LpProblem)

    # Test if the number of decision variables is correct
    assert len(x) == binary_indicator_matrix.shape[0]

    # Test if the objective function is set correctly
    assert problem.objective == pulp.lpSum(x)

    # Test if the constraints are added correctly (edge constraints)
    for j in range(binary_indicator_matrix.shape[0]):
        c = problem.constraints[f"M_{j + 1}"]
        for i in range(binary_indicator_matrix.shape[1]):
            if binary_indicator_matrix[i, j] == 1:
                assert x[i] in c, f"x_{i + 1} should be in constraint M_{j + 1}"


def test_maximum_coverage_strategy() -> None:
    """Test that MaximumCoverageStrategy.select_cuts returns the correct format
    and ensures the number of selected cuts is within expected bounds.
    """
    strategy = MaximumCoverageStrategy(max_cuts=1)
    cuts: list = [Cut(np.array([i]), 0, CutType.FEASIBILITY) for i in range(4)]

    binary_indicator_matrix = np.array(
        [[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    )

    solver = PuLPSolver()
    selected_cuts = strategy.select_cuts(cuts, binary_indicator_matrix, solver)

    assert isinstance(selected_cuts, list), "Output should be a list."
    assert len(selected_cuts) <= len(
        cuts
    ), "Number of selected cuts should be less than or equal to total cuts."


def test_maximum_coverage_strategy_empty_universe() -> None:
    """Test the select_cuts method when the binary indicator matrix represents
    an empty universe.
    """
    strategy = MaximumCoverageStrategy(max_cuts=1)
    cuts: list = [Cut(np.array([i]), 0, CutType.FEASIBILITY) for i in range(4)]

    binary_indicator_matrix = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )

    solver = PuLPSolver()

    result = strategy.select_cuts(cuts, binary_indicator_matrix, solver)

    assert len(result) == 0  # No connections, no cuts should be selected


def test_maximum_coverage_strategy_full_universe() -> None:
    """Test the select_cuts method when the binary indicator matrix represents
    a full universe.
    """
    strategy = MaximumCoverageStrategy(max_cuts=4)
    cuts: list = [Cut(np.array([i]), 0, CutType.FEASIBILITY) for i in range(4)]

    binary_indicator_matrix = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    )

    solver = PuLPSolver()

    result = strategy.select_cuts(cuts, binary_indicator_matrix, solver)

    # Expecting at more than half of the elements in the cover
    assert len(result) == 4


def test_maximum_coverage_strategy_single_edge() -> None:
    """Test the select_cuts method when the binary indicator matrix represents
    an universe with a single connection.
    """
    strategy = MaximumCoverageStrategy(max_cuts=1)
    cuts: list = [Cut(np.array([i]), 0, CutType.FEASIBILITY) for i in range(4)]

    binary_indicator_matrix = np.array([[1, 1], [1, 1]])

    solver = PuLPSolver()

    result = strategy.select_cuts(cuts, binary_indicator_matrix, solver)

    assert len(result) == 1


def test_maximum_coverage_strategy_create_optimization_problem() -> None:
    # Define a simple binary indicator matrix
    binary_indicator_matrix = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])

    # Call the function
    strategy = MaximumCoverageStrategy(
        max_cuts=1
    )  # Replace with your class name if necessary
    problem, x = strategy.create_optimization_problem(binary_indicator_matrix)

    # Test if the problem is of the correct type
    assert isinstance(problem, pulp.LpProblem)

    # Test if the number of decision variables is correct
    assert len(x) == binary_indicator_matrix.shape[0]

    # Test if the objective function is set correctly
    assert problem.objective == pulp.lpSum(x)

    # Test if the constraints are added correctly
    constraint_name = "Max_number_of_cuts"
    assert constraint_name in problem.constraints
    constraint = problem.constraints[constraint_name]
    assert constraint.constant == -1

    for j in range(binary_indicator_matrix.shape[1]):
        constraint_name = f"phi_{j}"
        assert constraint_name in problem.constraints

        constraint = problem.constraints[constraint_name]

        assert (
            len([var for var, _ in problem.constraints[constraint_name].items()])
            <= binary_indicator_matrix.shape[0] + 1
        )
