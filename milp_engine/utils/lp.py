# milp_engine/utils/lp.py
"""This module contains utility functions for linear programming (LP) problems."""

import math
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pulp
import sympy as sm
from pulp import LpAffineExpression
from qiskit_optimization import QuadraticProgram


def get_dual(primal: pulp.LpProblem) -> pulp.LpProblem:
    """Produce the dual of the given primal problem.

    Uses the algorithm for finding the dual described in:
    https://en.wikipedia.org/wiki/Dual_linear_program#Constructing_the_dual_LP.

    :param primal: The primal LP problem to produce the dual of.
    :return: The dual as an LpProblem instance.
    :raises ValueError: If the primal problem does not meet the requirements
        for dualization, such as having non-continuous variables or unsupported bounds.
    """
    # Assert all variables are continuous and have valid bounds
    for p_var in primal.variables():
        if p_var.cat != pulp.LpContinuous:
            raise ValueError(
                f"Variable {p_var.name} is not continuous (category: {p_var.cat})"
            )
        if not (
            (p_var.lowBound is None and p_var.upBound is None)  # Unbounded
            or (
                p_var.lowBound == 0 and (p_var.upBound is None or p_var.upBound >= 0)
            )  # ≥0
            or (
                p_var.upBound == 0 and (p_var.lowBound is None or p_var.lowBound <= 0)
            )  # ≤0
        ):
            raise ValueError(
                f"Variable {p_var.name} has unsupported bounds: [{p_var.lowBound}, "
                f"{p_var.upBound}]"
            )

    # Determine the sense of the dual problem
    dual_sense = pulp.LpMinimize if primal.sense == pulp.LpMaximize else pulp.LpMaximize
    dual = pulp.LpProblem(f"{primal.name}_dual", dual_sense)

    # Create dual variables corresponding to primal constraints
    dual_vars = []
    primal_constraints = list(primal.constraints.values())
    for i, constraint in enumerate(primal_constraints):
        if constraint.sense == pulp.LpConstraintLE:
            u = pulp.LpVariable(constraint.name, upBound=0)
        elif constraint.sense == pulp.LpConstraintGE:
            u = pulp.LpVariable(constraint.name, lowBound=0)
        else:
            u = pulp.LpVariable(constraint.name)
        dual_vars.append(u)

    # Set the dual objective function
    objective = LpAffineExpression(
        [
            (dual_var, -constraint.constant)
            for dual_var, constraint in zip(dual_vars, primal_constraints)
        ]
    )
    dual += objective

    # Create dual constraints corresponding to primal variables
    primal_vars = primal.variables()
    for j, p_var in enumerate(primal_vars):
        coeffs = [constraint.get(p_var, 0) for constraint in primal_constraints]
        expr = pulp.lpSum([coeffs[i] * dual_vars[i] for i in range(len(dual_vars))])
        c_j = primal.objective.get(p_var, 0)

        if p_var.lowBound == 0 and p_var.upBound is None:
            dual += (expr <= c_j, p_var.name)
        elif p_var.upBound == 0 and p_var.lowBound is None:
            dual += (expr >= c_j, p_var.name)
        else:
            dual += (expr == c_j, p_var.name)

    return dual


def substitute_vars(
    problem: pulp.LpProblem, substitutions: Dict[str, float]
) -> pulp.LpProblem:
    """Substitute a subset of the variables inside a problem with constant values.

    For this, a new LpProblem is created. The original problem is not modified.

    :param problem: the LpProblem to base the substitution on.
    :param substitutions: a dictionary mapping variable names to constants.
    :return: a new LpProblem instance with substitutions applied.
    """
    subst_problem = pulp.LpProblem(
        f"{problem.name}_substituted({substitutions})".replace(" ", ""), problem.sense
    )

    # Create an objective without the substituted variables
    subst_problem.objective = pulp.LpAffineExpression(
        [
            (var_j, c_j)
            for var_j, c_j in problem.objective.items()
            if var_j.name not in substitutions.keys()
        ]
    )

    # Create constraints
    for name, constraint in problem.constraints.items():
        # Substituted variables are omitted from the expression
        expr = pulp.LpAffineExpression(
            [
                (var_j, A_ij)
                for var_j, A_ij in constraint.expr.items()
                if var_j.name not in substitutions.keys()
            ]
        )

        # Substituted variables are subtracted from the right-hand side
        # Note: b_i is defined as -constant in PuLP.
        b_i = -constraint.constant - sum(
            [
                substitutions[var_j.name] * A_ij
                for var_j, A_ij in constraint.expr.items()
                if var_j.name in substitutions.keys()
            ]
        )

        subst_problem += pulp.LpConstraint(expr, constraint.sense, name, b_i)

    return subst_problem


def get_b(problem: pulp.LpProblem) -> np.ndarray:
    """Get the b vector of the problem.

    :param problem: The LP problem to extract b from.
    :return: A numpy array containing the b values.
    """
    b = np.empty(len(problem.constraints))
    constraints_sorted = sorted(problem.constraints.items(), key=lambda c: c[0])
    for i, (_, constraint) in enumerate(constraints_sorted):
        b[i] = -constraint.constant
    return b


def get_B(milp: pulp.LpProblem, y: List[pulp.LpVariable]) -> np.ndarray:
    """Get the B matrix of the MILP.

    This is the constraint matrix for the y variables.

    :param milp: The MILP problem to extract B from.
    :param y: The y variables of the problem.
    :return: A numpy array containing the B matrix.
    """
    B = np.empty((len(milp.constraints), len(y)))
    constraints_sorted = sorted(milp.constraints.items(), key=lambda c: c[0])
    for i, (_, constraint) in enumerate(constraints_sorted):
        y_sorted = sorted(y, key=lambda y_j: y_j.name)
        for j, y_j in enumerate(y_sorted):
            B[i, j] = constraint.expr[y_j] if y_j in constraint.expr else 0

    return B


def get_z(milp: pulp.LpProblem) -> pulp.LpVariable:
    """Get the z (objective) variable form the MILP.

    This assumes the name of this variable is 'z'.

    :param milp: The MILP problem to extract z from.
    :return: The LpVariable z.
    :raises ValueError: If no z variable is found.
    """
    for var in milp.variables():
        if var.name == "z":
            return var
    raise ValueError("No z variable found in the MILP problem.")


def get_d(milp: pulp.LpProblem, y: List[pulp.LpVariable]) -> np.ndarray:
    """Get the d vector from the MILP.

    This is the vector of objective coefficients for the y variables.

    :param milp: The MILP problem to extract d from.
    :param y: The y variables of the problem.
    :return: A numpy array containing the d vector.
    """
    d = np.empty(len(y))
    y_sorted = sorted(y, key=lambda y_i: y_i.name)
    for i, y_i in enumerate(y_sorted):
        d[i] = milp.objective.get(y_i, 0)
    return d


def get_objective_val(
    problem: pulp.LpProblem | QuadraticProgram, solution: Dict[str, float]
) -> float:
    """Get the objective value of the problem, given the particular solution.

    :param problem: The optimisation problem to extract objective from.
    :param solution: The solution dictionary.
    :raises TypeError: If the problem type is not supported.
    :return: The objective value.
    """
    if isinstance(problem, pulp.LpProblem):
        return sum(
            coeff * solution[var.name] for var, coeff in problem.objective.items()
        )
    elif isinstance(problem, QuadraticProgram):
        return problem.objective.evaluate(list(solution.values()))
    else:
        raise TypeError("Unsupported problem type.")


def is_feasible(
    problem: pulp.LpProblem | QuadraticProgram, solution: Dict[str, float]
) -> bool:
    """Evaluate if the given solution is feasible for the optimization problem.

    Note: the SymPy package is imported specifically for this function.

    :param problem: the optimization problem to evaluate.
    :param solution: a possible solution to the optimization problem.
    :raises ValueError: If the dimensions of the inputs are incompatible
        or if no extreme ray can be found.
    :return: a bool indicating feasibility.
    """
    if isinstance(problem, QuadraticProgram):
        return problem.is_feasible(list(solution.values()))
    # Define symbolic variables based on solution keys
    optimization_problem_vars = {key: sm.Symbol(key) for key in solution.keys()}

    # Check feasibility by evaluating the constraints
    for constraint in problem.constraints.values():
        # Build the left-hand side
        lhs = sum(
            coeff * optimization_problem_vars[var.name]
            for var, coeff in constraint.expr.items()
        )

        # Extract the inequality type
        sense = constraint.sense
        rhs = -constraint.constant

        # Create the inequality expression
        # Note: < or > constraints are not handled in PuLP. Therefore, this
        # capability is not included in this function
        if sense == -1:
            expr = lhs <= rhs
        elif sense == 0:
            expr = sm.Eq(lhs, rhs)
        elif sense == 1:
            expr = lhs >= rhs
        else:
            raise ValueError("Unknown sense value.")

        # Evaluate the constraint function (True if satisfied, False otherwise)
        constraint_satisfied = expr.subs(solution)
        if not constraint_satisfied:
            return False

    # Check variable bounds (binary, integer, continuous)
    for var in problem.variables():
        var_value = solution.get(var.name, None)

        # Check if the variable value is provided in the solution
        if var_value is None:
            return False  # If no value is provided for a variable, return False

        ub = var.upBound if var.upBound is not None else math.inf
        lb = var.lowBound if var.lowBound is not None else -math.inf

        # Integer variable check: is an integer (i.e., divisible by 1) and within
        # the bounds [lowBound, upBound]
        if var.cat == pulp.LpInteger:
            if (
                var_value % 1 != 0
            ):  # Check if the value is not divisible by 1 (i.e., not integer)
                return False  # Not an integer
            elif not (lb <= var_value <= ub):
                return False  # Integer bound violated

        # Continuous variable check: within the bounds [lowBound, upBound]
        elif var.cat == pulp.LpContinuous:
            if not (lb <= var_value <= ub):
                return False  # Continuous bound violated

    return True


def to_standard_form(problem: pulp.LpProblem) -> pulp.LpProblem:
    """Transform the given problem to a standard form accepted by `to_dual`.

    Specifically, this function ensures that:
    - All constraints are inequalities (no equalities).
    - All inequalities are in the same direction (either all >= or all <=).
    - If the problem is a minimization problem, all inequalities are >=.
    - If the problem is a maximization problem, all inequalities are <=.

    :param problem: The LP problem to transform.
    :return: A new LpProblem instance in standard form.
    """
    problem_copy = deepcopy(problem)
    # Transform the problem to standard form: no equalities, and all inequalities in
    #   the same direction.

    # Equality: replace with two inequalities
    for name, constraint in list(problem_copy.constraints.items()):
        if constraint.sense == 0:
            expr = constraint.expr
            rhs = -constraint.constant
            # Remove the equality constraint
            del problem_copy.constraints[name]
            # Add two inequalities
            problem_copy += (expr >= rhs), f"{name}_ge"
            problem_copy += (expr <= rhs), f"{name}_le"

    # Flip inequalities if needed
    for name, constraint in list(problem_copy.constraints.items()):
        if problem_copy.sense == pulp.LpMinimize and constraint.sense == -1:
            # For minimization, want all >=, so flip <= to >= by multiplying by -1
            expr = -1 * constraint.expr
            rhs = constraint.constant
            del problem_copy.constraints[name]
            problem_copy += (expr >= rhs), f"{name}_flipped"
        elif problem_copy.sense == pulp.LpMaximize and constraint.sense == 1:
            # For maximization, want all <=, so flip >= to <= by multiplying by -1
            expr = -1 * constraint.expr
            rhs = constraint.constant
            del problem_copy.constraints[name]
            problem_copy += (expr <= rhs), f"{name}_flipped"
    return problem_copy
