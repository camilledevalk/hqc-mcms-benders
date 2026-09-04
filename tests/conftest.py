# tests/conftest
"""Contains globally accessible fixtures."""
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pulp
import pytest
import ray
from ray.cluster_utils import Cluster

import milp_engine.instances.milps
from milp_engine.strategies.maximum_coverage_strategy import MaximumCoverageStrategy
from milp_engine.strategies.minimum_set_cover_strategy import MinimumSetCoverStrategy


@pytest.fixture(autouse=True)
def change_test_dir(request: Any, monkeypatch: Any) -> None:
    """Change the current working directory to the directory of the test file.

    This ensures that relative paths in the tests work correctly.
    :param request: The pytest request object.
    :param monkeypatch: The pytest monkeypatch fixture to change the current working
        directory.
    """
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def continuous_min_problem() -> pulp.LpProblem:
    """Produce a simple continuous minimization problem.

    :return: The problem as a pulp.LpProblem instance.
    """
    x1, x2, x3 = [pulp.LpVariable(f"x_{i}", 0) for i in range(1, 4)]

    problem = pulp.LpProblem("continuous_min_problem", pulp.LpMinimize)

    problem += 0.1 * x1 + 0.2 * x2 + 0.3 * x3

    problem += 0.4 * x1 + 0.5 * x2 + 0.6 * x3 >= 1.0, "constraint_1"
    problem += 0.7 * x1 + 0.8 * x2 + 0.9 * x3 >= 2.0, "constraint_2"

    return problem


@pytest.fixture
def unbounded_max_problem() -> Tuple[pulp.LpProblem, np.ndarray]:
    """Produce a simple unbounded maximization problem.

    :return: The problem as a pulp.LpProblem instance and its extreme ray as
        a NumPy array.
    """
    x1, x2 = [pulp.LpVariable(f"x_{i}", 0) for i in range(1, 3)]

    problem = pulp.LpProblem("unbounded_max_problem", pulp.LpMaximize)

    problem += x1 + 2 * x2

    problem += x1 - 2 * x2 >= -5.0, "constraint_1"
    problem += 2 * x2 - x1 >= -5.0, "constraint_2"

    extreme_ray = np.array([2, 1])
    extreme_ray = extreme_ray / np.linalg.norm(extreme_ray)

    return problem, extreme_ray


@pytest.fixture
def basic_milp() -> Tuple[pulp.LpProblem, List[pulp.LpVariable]]:
    """Produce a simple MILP problem.

    :return: The problem as a pulp.LpProblem instance and the list of complicating
        variables.
    """
    x1, x2, x3 = [pulp.LpVariable(f"x_{i}", 0) for i in range(1, 4)]
    y1, y2 = [pulp.LpVariable(f"y_{i}", 0, 1, pulp.LpBinary) for i in range(1, 3)]

    problem = pulp.LpProblem("basic_milp", pulp.LpMinimize)

    problem += 0.1 * x1 + 0.2 * x2 + 0.3 * x3 + 0.4 * y1 + 0.5 * y2

    problem += 0.4 * x1 + 0.5 * x2 + 0.6 * x3 - 1000 * y1 >= 1.0, "constraint_1"
    problem += 0.7 * x1 + 0.8 * x2 + 0.9 * x3 - 2000 * y2 >= 2.0, "constraint_2"

    return problem, [y1, y2]


@pytest.fixture
def basic_mp() -> pulp.LpProblem:
    """Produce a simple master problem without any constraints.

    :return: The simple MP as a pulp.LpProblem instance.
    """
    z = pulp.LpVariable("z", -1e6, 1e6, pulp.LpInteger)
    problem = pulp.LpProblem("basic_mp", pulp.LpMinimize)
    problem += z
    return problem


@pytest.fixture
def binary_choice_milp() -> (
    Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]
):
    """Produce a simple MILP with a binary choice to make.

    The problem has two continuous variables (x1, x2) and two binary variables (y1, y2).
    The constraints that act on x1 and x2 depend on the binary variables y1 and y2.
    The problem has a known solution. Namely: x1 = 5, x2 = 10, y1 = 0, y2 = 1.

    :return: The problem as a pulp.LpProblem instance, the list of complicating
    """
    return milp_engine.instances.milps.binary_choice_milp()


@pytest.fixture
def savings_milp() -> Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]:
    """Produce a MILP representing a savings problem.

    The problem has a known solution. Namely: x_10, ..., x_5 = 100;
    x4, ..., x_1 = 0; y = 400.

    :return: The problem as a pulp.LpProblem instance, the list of complicating
        variables, and the optimal solution.
    """
    return milp_engine.instances.milps.savings_milp()


@pytest.fixture
def dfj_tsp_milp() -> Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]:
    """Fixture to create a TSP problem using the DFJ formulation.

    Note: this is not really a MILP as it only uses integer variables.

    :return: A tuple containing the TSP problem as a pulp.LpProblem instance
        and the optimal solution as a dictionary.
    """
    return milp_engine.instances.milps.dfj_tsp_milp()


@pytest.fixture()
def mtz_tsp_milp() -> Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]:
    """Fixture to create a TSP problem using the MTZ formulation.

    :return: A tuple containing the TSP problem as a pulp.LpProblem instance
        and the optimal solution as a dictionary.
    """
    return milp_engine.instances.milps.mtz_tsp_milp()


@pytest.fixture
def milps(
    binary_choice_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]],
    savings_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]],
    dfj_tsp_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]],
    mtz_tsp_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]],
) -> Dict[str, Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]]:
    """Fixture to provide a dictionary of test MILP instances.

    :param binary_choice_milp: A tuple containing the MILP problem,
        complicating variables, and the optimal solution for the binary choice problem.
    :param savings_milp: A tuple containing the MILP problem, complicating variables,
        and the optimal solution for the savings problem.
    :param dfj_tsp_milp: A tuple containing the TSP problem using the DFJ formulation,
        complicating variables, and the optimal solution.
    :param mtz_tsp_milp: A tuple containing the TSP problem using the MTZ formulation,
        complicating variables, and the optimal solution.
    :return: A dictionary with the problem name as key and a tuple of the problem,
        complicating variables, and optimal solution as value.
    """
    return {
        "binary_choice_milp": binary_choice_milp,
        "savings_milp": savings_milp,
        "dfj_tsp_milp": dfj_tsp_milp,
        "mtz_tsp_milp": mtz_tsp_milp,
    }


@pytest.fixture
def set_cover_problem() -> pulp.LpProblem:
    """Create a randomized Minimum Set Cover optimization problem.

    :return: A `pulp.LpProblem` representing the set cover problem.
    """

    def _create_problem(num_elements: int, seed: int) -> pulp.LpProblem:
        """Create a Minimum Set Cover problem for random universe of elements and sets.

        :param num_elements: Number of elements.
        :param seed: Random seed used to generate the symmetric binary indicator matrix.
        :return: A `pulp.LpProblem` representing the set cover problem.
        """
        rng = np.random.default_rng(seed)
        upper = rng.integers(
            0, 2, size=(num_elements, num_elements)
        )  # binary random matrix
        upper = np.triu(upper, 1)  # zero out the diagonal and lower triangle
        binary_indicator_matrix = upper + upper.T  # make symmetric
        strategy = MinimumSetCoverStrategy()
        problem, _ = strategy.create_optimization_problem(binary_indicator_matrix)
        return problem

    return _create_problem


@pytest.fixture
def maximum_coverage_problem() -> pulp.LpProblem:
    """Create a randomized Maximum Coverage optimization problem.

    :return: A `pulp.LpProblem` representing the maximum coverage problem.
    """

    def _create_problem(
        num_cuts: int, num_vars: int, max_cuts: int, seed: int
    ) -> pulp.LpProblem:
        """Create a randomized Maximum Coverage optimization problem.

        Generates a binary indicator matrix to represent coverage relationships
        between cuts and variables, then formulates a maximum coverage problem
        selecting at most `max_cuts` rows.

        :param num_cuts: Number of available cuts (rows in the matrix).
        :param num_vars: Number of variables to be covered (columns in the matrix).
        :param max_cuts: Maximum number of cuts allowed in the solution.
        :param seed: Random seed used to generate the binary indicator matrix.
        :return: A `pulp.LpProblem` representing the maximum coverage problem.
        """
        rng = np.random.default_rng(seed)
        binary_matrix = rng.integers(0, 2, size=(num_cuts, num_vars))
        strategy = MaximumCoverageStrategy(max_cuts)
        problem, _ = strategy.create_optimization_problem(binary_matrix)
        return problem

    return _create_problem


@pytest.fixture
def ray_test_cluster() -> Generator[Cluster, None, None]:
    """Fixture to initialize Ray for tests.

    This fixture initializes a Ray Cluster with 2 cores before running the tests.
    After the tests are done, it shuts down the cluster, as well as Ray itself.

    :yield: A Ray Cluster instance.
    """
    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 2,
        },
    )

    assert not ray.is_initialized()
    ray.init(address=cluster.address)

    yield cluster

    ray.shutdown()
    cluster.shutdown()


@pytest.fixture
def ray_shutdown() -> Generator[None, None, None]:
    """Fixture to shut down Ray after tests.

    This fixture ensures that Ray is shut down after the tests are completed.

    :yield: None.
    """
    yield
    if ray.is_initialized():
        ray.shutdown()
