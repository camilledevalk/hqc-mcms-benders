# milp_engine/solvers/base_solver.py
"""This module contains solver interfaces.

The abstract class 'Solver' defines the interface for any solver that can
solve and evaluate optimization problems. It enables consistent integration
of various solver types.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, cast

import numpy as np
import pulp


class Solver(ABC):
    """Abstract class defining a solver."""

    @abstractmethod
    def solve(self, problem: pulp.LpProblem) -> pulp.LpProblem:
        """Solve the linear problem and return the results.

        :param problem: The PuLP problem instance to be solved.
        :return: The solved PuLP problem.
        """
        pass

    def enumerate_feasible_solutions(
        self, problem: pulp.LpProblem, max_solutions: int, large_num: float = 1e5
    ) -> list[dict[str, float]]:
        """Enumerate multiple feasible solutions for the given problem.

        :param problem: the optimization problem to find feasible solutions for.
        :param max_solutions: the maximum number of feasible solutions to find.
        :param large_num: a large number used in the enumeration process. This should
            be significantly larger than the expected variable values in the problem.
        :return: a list of feasible solutions as dictionaries of variable names and
            values.
        """
        return _enumerate_feasible_solutions_iterative(
            problem, self, max_solutions, large_num
        )


class ExtremeRaySolver(Solver):
    """Abstract class defining a solver with extra features.

    Specifically:
    - Find extreme rays when a problem is unbounded.
    """

    @abstractmethod
    def solve_and_find_extreme_ray(
        self, problem: pulp.LpProblem
    ) -> Tuple[pulp.LpProblem, np.ndarray | None]:
        """Solve the optimization problem and find extreme rays if unbounded.

        :param problem: The PuLP problem instance to be solved.
        :return: The solved PuLP problem and the extreme ray if unbounded.
        """
        pass


class CutSelectionSolver(ABC):
    """Abstract base class for solvers that address cut selection optimization problems.

    This solver is designed for problems that require custom strategies,
    binary indicator matrices, and additional parameters such as coverage limits.
    Subclasses must implement the `solve_with_binary_indicator_matrix` method.
    """

    @abstractmethod
    def solve_with_binary_indicator_matrix(
        self,
        problem: pulp.LpProblem,
        binary_indicator_matrix: np.ndarray,
        max_number_covered: int | None,
    ) -> pulp.LpProblem:
        """Solves the given cut selection problem.

        :param problem: the optimization problem to be solved.
        :param binary_indicator_matrix: matrix representing the binary indicator
            matrix of the selected criteria of the problem.
        :param max_number_covered: maximum number of elements allowed to be
            selected or covered.
        :return: the solved optimization problem.
        """
        pass


def _enumerate_feasible_solutions_iterative(
    problem: pulp.LpProblem,
    solver: "Solver",
    max_solutions: int = 10,
    large_num: float = 1e5,
) -> list[dict[str, float]]:
    """Enumerate multiple feasible solutions of an LP using PuLP.

    :param problem: An LP for which to find multiple feasible solutions.
    :param solver: A solver used to solve the LP.
    :param max_solutions: Maximum number of feasible solutions to find.
    :param large_num: A large float value used for unbounded variables.
    :raises ValueError: If max_solutions is a negative integer.

    :returns: A list of solutions, each as a dict of variable names and values.

    Note: these enumeration problem can get big with increasing number of
    delta variables. License is needed for many feasible solutions to be found.
    """
    solutions: list[dict[str, float]] = []
    problem_copy = deepcopy(problem)  # Don't modify the original problem

    original_vars = problem_copy.variables().copy()

    if max_solutions < 0:
        raise ValueError("Number of solutions should not be a negative integer.")

    for i in range(max_solutions):
        match solver:
            case ExtremeRaySolver():
                problem_copy, ray_none = solver.solve_and_find_extreme_ray(problem_copy)
            case _:
                problem_copy, ray_none = solver.solve(problem_copy), None

        # Check infeasibility
        if problem_copy.status == pulp.LpStatusInfeasible:
            return solutions

        current_vars = problem_copy.variables()

        # if the problem was unbounded, set unbounded variables to (+/-)large_num,
        # depending on their coefficient in the objective and the optimality criterion.
        if problem_copy.status == pulp.LpStatusUnbounded:
            if ray_none is None:
                return solutions  # No ray found -> cannot provide feasible solution

            ray = cast(np.ndarray, ray_none)
            ray = ray / np.min(ray)  # Attempt to make ray integer
            solution_vector = large_num * ray  # Get a large, yet bounded solution
            for j, var in enumerate(original_vars):

                # Check if the solution vector adheres to integer constraints
                if var.cat == pulp.LpInteger and solution_vector[i] % 1.0 != 0:
                    return solutions

                var.varValue = solution_vector[j]

        # Store current solution
        solution = {v.name: v.varValue for v in original_vars}
        solutions.append(solution)
        (
            problem_copy,
            exclusion_binary,
            exclusion_combs,
            number_of_integer_variables,
        ) = _build_exclusion_constraints(problem_copy, current_vars, solution_index=i)

        # Exclude the binary part (at least one bit is different)
        if len(exclusion_binary) > 0:
            problem_copy += pulp.lpSum(exclusion_binary) >= 1, f"exclusion_binary_{i}"

        # Make sure only one of the existing solution constraints is excluded
        if len(exclusion_combs) > 0:
            problem_copy += (
                pulp.lpSum(exclusion_combs) <= number_of_integer_variables - 1,
                f"exclusion_integer_{i}",
            )

    return solutions


def _build_exclusion_constraints(
    problem: pulp.LpProblem,
    original_vars: list[pulp.LpVariable],
    solution_index: int = 0,
) -> Tuple[pulp.LpProblem, list[pulp.LpAffineExpression], list[pulp.LpVariable], int]:
    """Construct exclusion constraints to eliminate the current solution.

    :param problem: The LP problem to which constraints are added.
    :param original_vars: The original variables from the problem (non-auxiliary).
    :param solution_index: Optional index to ensure unique constraint names.
    :returns: Tuple of (problem, exclusion_binary, exclusion_combs,
        number_of_integer_variables).

    Big-M trick:
    The Big-M method is used to conditionally enforce constraints by introducing binary
    variables and a sufficiently large constant `M`. In this context, it's used to
    formulate "not equal to" constraints for integer variables.

    For a variable `x` with a known value `x_val` in a solution, we add a constraint
    that forces `x ≠ x_val` in the next solution. This is done by introducing a binary
    variable `delta` and writing two inequalities:

        x - x_val >= 1 - M * delta - 2 * M * comb
        x - x_val <= -1 + M * (1 - delta) + 2 * M * comb

    The `comb` binary variable allows for softening one constraint in a set to
    ensure only one integer variable changes in a new solution.

    This trick allows us to iteratively generate diverse feasible solutions
    by excluding previously found integer solutions while still solving the
    modified LP.

    See: https://en.wikipedia.org/wiki/Big_M_method
    """
    exclusion_binary = []
    exclusion_combs = []
    number_of_integer_variables = 0
    big_M = 1e5  # big-M trick

    for v in original_vars:
        if v.cat == pulp.LpInteger and v.lowBound == 0 and v.upBound == 1:
            if v.varValue == 1:
                exclusion_binary.append(1 - v)
            else:
                exclusion_binary.append(v)
        elif v.cat == pulp.LpInteger:
            delta = pulp.LpVariable(
                f"delta_{v.name}_{solution_index}", cat=pulp.LpBinary
            )
            comb = pulp.LpVariable(f"comb_{v.name}_{solution_index}", cat=pulp.LpBinary)
            exclusion_combs.append(comb)

            problem += v - v.varValue >= 1 - big_M * delta - 2 * big_M * comb
            problem += v - v.varValue <= -1 + big_M * (1 - delta) + 2 * big_M * comb

            number_of_integer_variables += 1

    return problem, exclusion_binary, exclusion_combs, number_of_integer_variables
