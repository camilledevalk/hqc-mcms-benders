# tests/examples/test_tsp.py
from milp_engine.instances.tsp import (
    DFJFormulation,
    MTZFormulation,
    calculate_distances,
)


def test_dfj_formulation() -> None:
    """Weakly test the Dantzig–Fulkerson–Johnson (DFJ) formulation for TSP."""
    cities = {
        "A": (0.0, 0.0),
        "B": (3.0, 0.0),
        "C": (3.0, 4.0),
        "D": (0.0, 4.0),
    }
    distances = calculate_distances(cities)

    # Add some self-loops to throw off the DFJ formulation
    distances[("A", "A")] = 0.0
    distances[("C", "C")] = 0.0

    dfj_formulation = DFJFormulation()
    problem = dfj_formulation.formulate(distances)

    # Simply check sums of variables, objective, and constraints
    assert len(problem.variables()) == 12  # 6 bidirectional edges
    assert len(problem.objective) == 12  # 6 bidirectional edges

    # Check that all edges are present, but not self-loops
    variable_names = set(problem.variablesDict().keys())
    for city_i in cities.keys():
        for city_j in cities.keys():
            if city_i != city_j:
                assert f"y_('{city_i}',_'{city_j}')" in variable_names
            if city_i == city_j:
                assert f"y_('{city_i}',_'{city_j}')" not in variable_names

    # 4 incoming for each city + 4 outgoing for each city
    #   + 10 subtour elimination constraints
    assert len(problem.constraints) == 4 + 4 + 10


def test_mtz_formulation() -> None:
    """Weakly test the Miller-Tucker-Zemlin (MTZ) formulation for TSP."""
    cities = {
        "A": (0.0, 0.0),
        "B": (3.0, 0.0),
        "C": (3.0, 4.0),
        "D": (0.0, 4.0),
    }
    distances = calculate_distances(cities)

    # Add some self-loops to throw off the MTZ formulation
    distances[("A", "A")] = 0.0
    distances[("C", "C")] = 0.0

    mtz_formulation = MTZFormulation()
    problem = mtz_formulation.formulate(distances)

    # Simply check sums of variables, objective, and constraints

    # 6 bidirectional edges + 3 auxiliary variables
    assert len(problem.variables()) == 12 + 3
    assert len(problem.objective) == 12  # 6 bidirectional edges

    # Check that all edges are present, but not self-loops
    variable_names = set(problem.variablesDict().keys())
    for city_i in cities.keys():
        for city_j in cities.keys():
            if city_i != city_j:
                assert f"y_('{city_i}',_'{city_j}')" in variable_names
            if city_i == city_j:
                assert f"y_('{city_i}',_'{city_j}')" not in variable_names

    # 4 incoming for each city + 4 outgoing for each city
    #   + 3 auxiliary variable bounds + 6 subtour elimination constraints
    assert len(problem.constraints) == 4 + 4 + 3 + 6
