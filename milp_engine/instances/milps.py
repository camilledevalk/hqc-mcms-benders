# examples/milps.py
"""This module contains example MILP problems for use in tests and benchmarks."""
from typing import Dict, List, Tuple

import pulp

from milp_engine.instances.tsp import (
    DFJFormulation,
    MTZFormulation,
    TSPFormulation,
    calculate_distances,
)


def binary_choice_milp() -> (
    Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]
):
    """Produce a simple MILP with a binary choice to make.

    The problem has two continous variables (x1, x2) and two binary variables (y1, y2).
    The constraints that act on x1 and x2 depend on the binary variables y1 and y2.
    The problem has a known solution. Namely: x1 = 5, x2 = 10, y1 = 0, y2 = 1.

    :return: The problem as a pulp.LpProblem instance, the list of complicating
    """
    x1, x2 = [pulp.LpVariable(f"x_{i}", 0, cat=pulp.LpContinuous) for i in range(1, 3)]
    y1, y2 = [pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in range(1, 3)]

    problem = pulp.LpProblem("binary_choice_milp", pulp.LpMinimize)

    problem += x1 + x2

    # Only one of y1 or y2 can be 1
    problem += -y1 - y2 >= -1

    # x1 >= 5 and x2 >= 10 if y1 = 0
    problem += x1 + 100 * y1 >= 5
    problem += x2 + 100 * y1 >= 10

    # x1 >= 8 and x2 >= 8 if y2 = 0
    problem += x1 + 100 * y2 >= 8
    problem += x2 + 100 * y2 >= 8

    optimal_solution = {
        "x_1": 5.0,
        "x_2": 10.0,
        "y_1": 0.0,
        "y_2": 1.0,
    }

    return problem, [y1, y2], optimal_solution


def savings_milp() -> Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]:
    """Produce a MILP representing a savings problem.

    The problem has a known solution. Namely: x_10, ..., x_5 = 100;
    x4, ..., x_1 = 0; y = 400.

    :return: The problem as a pulp.LpProblem instance, the list of complicating
        variables, and the optimal solution.
    """
    x = [pulp.LpVariable(f"x_{i}", 0, cat=pulp.LpContinuous) for i in range(1, 11)]
    y = pulp.LpVariable("y", lowBound=0, cat=pulp.LpInteger)
    c = [1.0 + i * 0.01 for i in range(1, 11)]

    problem = pulp.LpProblem("savings_milp", pulp.LpMinimize)

    # Objective function maximize savings (negated to work with minimization)
    problem += pulp.lpSum(-c_i * x_i for c_i, x_i in zip(c, x)) - 1.045 * y

    # Constraints
    for i, x_i in enumerate(x):
        problem += x_i <= 100, f"can't_spend_more_than_100_on_fund_({i})"

    problem += (pulp.lpSum(x) + y) <= 1000, "can't_spend_more_than_1000_in_total"

    optimal_solution = {
        "x_1": 0.0,
        "x_2": 0.0,
        "x_3": 0.0,
        "x_4": 0.0,
        "x_5": 100.0,
        "x_6": 100.0,
        "x_7": 100.0,
        "x_8": 100.0,
        "x_9": 100.0,
        "x_10": 100.0,
        "y": 400.0,
    }

    return problem, [y], optimal_solution


def _tsp_milp(
    formulation: TSPFormulation,
) -> Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]:
    """Create an 11-city TSP linear problem using the given TSP formulation.

    :param formulation: the type of TSP-to-LP formulation to use.
    :return: A tuple containing the TSP problem as a pulp.LpProblem instance
        and the optimal solution as a dictionary.
    """
    cities = {
        "A": (0.0, 0.0),
        "B": (1.0, 3.0),
        "C": (4.0, 3.0),
        "D": (6.0, 1.0),
        "E": (3.0, 0.0),
        "F": (2.0, 2.0),
        "G": (5.0, 2.0),
        "H": (7.0, 4.0),
        "I": (-1.0, 1.0),
        "J": (-1.0, 5.0),
        "K": (-1.0, 1.0),
    }
    distances = calculate_distances(cities)

    problem = formulation.formulate(distances)

    # Solve the problem
    problem.solve()

    complicating_vars = [v for v in problem.variables() if v.cat == pulp.LpBinary]

    # Extract the solution
    solution = {v.name: v.varValue for v in problem.variables()}

    return problem, complicating_vars, solution


def dfj_tsp_milp() -> Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]:
    """Create an 11-city TSP linear problem using the DFJ formulation.

    Note: this is not really a MILP as it only uses integer variables.

    :return: A tuple containing the TSP problem as a pulp.LpProblem instance
        and the optimal solution as a dictionary.
    """
    return _tsp_milp(DFJFormulation())


def mtz_tsp_milp() -> Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]:
    """Create an 11-city TSP linear problem using the MTZ formulation.

    :return: A tuple containing the TSP problem as a pulp.LpProblem instance
        and the optimal solution as a dictionary.
    """
    return _tsp_milp(MTZFormulation())


def basic_milp(cat_x1: str, cat_x2: str) -> pulp.LpProblem:
    """Produce a basic MILP problem with two binary variables.

    :param cat_x1: Category of the first variable.
    :param cat_x2: Category of the second variable.
    :return: A pulp.LpProblem instance representing the MILP problem.
    """
    # Define a maximization problem with binary (0/1) variables
    problem = pulp.LpProblem("mock_problem", pulp.LpMaximize)
    if cat_x1 == "Integer":
        x1 = pulp.LpVariable("x1", lowBound=0, upBound=10, cat=cat_x1)
    else:
        x1 = pulp.LpVariable("x1", lowBound=0, upBound=1, cat=cat_x1)
    if cat_x2 == "Integer":
        x2 = pulp.LpVariable("x2", lowBound=0, upBound=10, cat=cat_x2)
    else:
        x2 = pulp.LpVariable("x2", lowBound=0, upBound=1, cat=cat_x2)

    problem += x1 + x2, "Maximize"
    problem += x1 + x2 <= 8, "Constraint1"
    return problem
