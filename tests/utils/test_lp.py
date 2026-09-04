# tests/utils/test_lp.py
import math
from typing import List, Tuple

import numpy as np
import pulp
import pytest

from milp_engine.utils.lp import (
    get_B,
    get_b,
    get_d,
    get_dual,
    get_objective_val,
    get_z,
    is_feasible,
    substitute_vars,
    to_standard_form,
)


def test_get_dual(continuous_min_problem: pulp.LpProblem) -> None:
    dual = get_dual(continuous_min_problem)

    # Check optimality criterion
    assert dual.sense == pulp.LpMaximize

    # Check name
    assert dual.name == "continuous_min_problem_dual"

    # Check variables
    y1, y2 = sorted(dual.variables(), key=lambda v: v.name)
    for j, y_j in enumerate([y1, y2]):
        assert y_j.name == f"constraint_{j + 1}"
        assert y_j.lowBound == 0
        assert y_j.upBound is None

    # Check objective
    assert dual.objective[y1] == 1.0
    assert dual.objective[y2] == 2.0

    # Check constraints
    assert len(dual.constraints) == 3

    A_T = np.array(
        [
            [0.4, 0.7],
            [0.5, 0.8],
            [0.6, 0.9],
        ]
    )

    c = [0.1, 0.2, 0.3]

    for i, constraint_i in enumerate(dual.constraints.values()):
        assert -constraint_i.constant == c[i]
        assert constraint_i.name == f"x_{i + 1}"
        for j, y_j in enumerate([y1, y2]):
            assert constraint_i.expr[y_j] == A_T[i, j]


def test_substitute_vars(continuous_min_problem: pulp.LpProblem) -> None:
    substitution = {"x_2": 2.0, "x_3": 3.0}

    subst_problem = substitute_vars(continuous_min_problem, substitution)

    # Check name
    assert f"{continuous_min_problem.name}_substituted(" in subst_problem.name

    # Check optimality criterion
    assert subst_problem.sense == continuous_min_problem.sense

    # Check number of variables
    assert len(subst_problem.variables()) == 1

    # Check objective
    assert len(subst_problem.objective) == 1
    var_1, c_1 = list(subst_problem.objective.items())[0]
    assert var_1.name == "x_1"
    assert c_1 == 0.1

    # Check constraints
    assert len(subst_problem.constraints) == 2
    c1, c2 = subst_problem.constraints.values()
    assert len(c1.expr) == len(c2.expr) == 1
    assert math.isclose(-c1.constant, 1.0 - 2.0 * 0.5 - 3.0 * 0.6)
    assert math.isclose(-c2.constant, 2.0 - 2.0 * 0.8 - 3.0 * 0.9)


def test_get_b(continuous_min_problem: pulp.LpProblem) -> None:
    b = get_b(continuous_min_problem)
    assert b.shape == (2,)
    b_expected = np.array([1.0, 2.0])
    assert b == pytest.approx(b_expected)


def test_get_b_sorting(continuous_min_problem: pulp.LpProblem) -> None:
    # invert alphabetical order of constraints
    for i, (old_name, c) in enumerate(list(continuous_min_problem.constraints.items())):
        del continuous_min_problem.constraints[old_name]
        c.name = f"newname_{2 - i}"
        continuous_min_problem.constraints[c.name] = c

    b = get_b(continuous_min_problem)
    b_expected = np.array([2.0, 1.0])  # flipped from the original order
    assert b == pytest.approx(b_expected)


def test_get_B(basic_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable]]) -> None:
    milp, y = basic_milp
    B = get_B(milp, y)
    assert B.shape == (2, 2)
    B_expected = np.array([[-1000, 0], [0, -2000]])
    assert B == pytest.approx(B_expected)


def test_get_B_sorting(
    basic_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable]],
) -> None:
    milp, y = basic_milp

    # invert alphabetical order of constraints
    for i, (old_name, c) in enumerate(list(milp.constraints.items())):
        del milp.constraints[old_name]
        c.name = f"newname_{2 - i}"
        milp.constraints[c.name] = c

    # invert alphabetical order of y variables
    for i, y_i in enumerate(y):
        y_i.name = f"y_{2 - i}"

    B = get_B(milp, y)
    B_expected = np.array([[-2000, 0], [0, -1000]])  # flipped in var and c direction
    assert B == pytest.approx(B_expected)


def test_get_z(continuous_min_problem: pulp.LpProblem) -> None:
    milp = continuous_min_problem
    z = pulp.LpVariable("z", lowBound=0, cat=pulp.LpInteger)
    milp.objective = continuous_min_problem.objective + z

    z_retrieved = get_z(milp)

    assert z == z_retrieved


def test_get_d(basic_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable]]) -> None:
    milp, y = basic_milp
    d = get_d(milp, y)
    assert d.shape == (2,)
    d_expected = np.array([0.4, 0.5])
    assert d == pytest.approx(d_expected)


def test_get_d_sorting(
    basic_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable]],
) -> None:
    milp, y = basic_milp

    # invert alphabetical order of y variables
    for i, y_i in enumerate(y):
        y_i.name = f"y_{2 - i}"

    d = get_d(milp, y)
    d_expected = np.array([0.5, 0.4])  # flipped from the original order
    assert d == pytest.approx(d_expected)


def constraint_eq(
    constraint1: pulp.LpConstraint,
    constraint2: pulp.LpConstraint,
) -> bool:
    """Compare two constraints for equality using simple string equality.

    :param constraint1: The first constraint to compare.
    :param constraint2: The second constraint to compare.
    :return: True if the constraints are equal, False otherwise.
    """
    return str(constraint1) == str(constraint2)


def test_to_standard_form() -> None:
    # Create a sample LP problem
    prob = pulp.LpProblem("TestProblem", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10, cat=pulp.LpInteger)
    y = pulp.LpVariable("y", lowBound=0, upBound=5, cat=pulp.LpInteger)

    # Add constraints
    prob += x + 2 * y >= 3, "Constraint1"
    prob += 2 * x - y <= 4, "Constraint2"
    prob += x + y == 5, "Constraint3"

    prob_stf = to_standard_form(prob)

    # Check the problem is in standard form
    expected_constraints = [
        (x + 2 * y >= 3, "Constraint1"),
        (-2 * x + y >= -4, "Constraint2_flipped"),
        (x + y >= 5, "Constraint3_ge"),
        (-x - y >= -5, "Constraint3_le_flipped"),
    ]

    for expected_constraint, name in expected_constraints:
        expected_constraint.name = name

    for expected_constraint, name in expected_constraints:
        assert name in prob_stf.constraints
        assert constraint_eq(expected_constraint, prob_stf.constraints[name])


def test_get_objective_val_pulp_linear() -> None:
    # Create a simple LP problem
    prob = pulp.LpProblem("ObjValTest", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10, cat=pulp.LpInteger)
    y = pulp.LpVariable("y", lowBound=0, upBound=5, cat=pulp.LpInteger)
    prob += 2 * x + 3 * y, "Objective"
    prob += x + y <= 8, "Constraint"

    # Provide a feasible solution
    solution = {"x": 3.0, "y": 2.0}
    val = get_objective_val(prob, solution)
    assert val == 2 * 3 + 3 * 2

    # Provide another feasible solution
    solution = {"x": 0.0, "y": 5.0}
    val = get_objective_val(prob, solution)
    assert val == 2 * 0 + 3 * 5

    # Provide a solution with missing variable (should raise KeyError)
    with pytest.raises(KeyError):
        get_objective_val(prob, {"x": 1})


def test_is_feasible_pulp() -> None:
    # Create a simple LP problem
    prob = pulp.LpProblem("FeasibilityTest", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10, cat=pulp.LpInteger)
    y = pulp.LpVariable("y", lowBound=0, upBound=5, cat=pulp.LpInteger)
    prob += 2 * x + 3 * y, "Objective"
    prob += x + y <= 8, "Constraint"

    # Check feasible solution
    solution = {"x": 3.0, "y": 2.0}
    assert is_feasible(prob, solution)

    # Check infeasible solution
    solution = {"x": 10.0, "y": 10.0}
    assert not is_feasible(prob, solution)


def test_get_objective_val_qiskit_qp() -> None:
    # Create a simple qiskit QP problem
    from qiskit_optimization.problems import QuadraticProgram

    qp = QuadraticProgram()
    qp.binary_var("x")
    qp.binary_var("y")
    qp.maximize(linear={"x": 2, "y": 3}, quadratic={("x", "y"): 1})
    qp.linear_constraint({"x": 1, "y": 1}, "<=", 8, "Constraint")
    qp.linear_constraint({"x": 1}, "==", 3, "EqualityConstraint")
    qp.linear_constraint({"y": 1}, "==", 2, "AnotherEqualityConstraint")
    solution = {"x": 1.0, "y": 0.0}
    val = get_objective_val(qp, solution)
    assert val == 2 * 1 + 3 * 0 + 1 * 1 * 0  # linear + quadratic term


def test_is_feasible_qiskit_qp() -> None:
    # Create a simple qiskit QP problem
    from qiskit_optimization.problems import QuadraticProgram

    qp = QuadraticProgram()
    qp.binary_var("x")
    qp.binary_var("y")
    qp.maximize(linear={"x": 2, "y": 3}, quadratic={("x", "y"): 1})
    qp.linear_constraint({"x": 1, "y": 1}, "<=", 8, "Constraint")

    # Check feasible solution
    solution = {"x": 1.0, "y": 0.0}
    assert is_feasible(qp, solution)

    # Check infeasible solution
    solution = {"x": 10.0, "y": 10.0}
    assert not is_feasible(qp, solution)
