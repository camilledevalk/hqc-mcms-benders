# milp_engine/solvers/scip_solver.py
"""This module contains a solver interface for pyscipopt.

The 'ScipSolver' class implements this interface using the classical pyscipopt
library. A benefit of this solver is its capability to find multiple feasible solutions
in a single solve call, enhancing performance for enumeration tasks.
"""
import os
import warnings
from typing import Any, Dict, List, Tuple

import pulp
from pyscipopt import Model, quicksum

from milp_engine.solvers.base_solver import Solver


class ScipSolver(Solver):
    """A solver class relying on pyscipopt.optimize."""

    def solve(self, problem: pulp.LpProblem) -> pulp.LpProblem:
        """Solve the linear problem and return the results.

        :param problem: The PuLP problem instance to be solved.
        :return: The solved PuLP problem.
        """
        scip_model = _convert_pulp_to_scip(problem)
        scip_model.optimize()

        # Load the optimal variable values back into the PuLP problem
        scip_vars = scip_model.getVars()
        problem_vars_dict = problem.variablesDict()
        for v in scip_vars:
            if v is not None:
                problem_vars_dict[v.name].varValue = scip_model.getSolVal(
                    scip_model.getBestSol(), v
                )

        # Update the problem status
        scip_status = scip_model.getStatus()
        if scip_status == "optimal":
            problem.status = pulp.LpStatusOptimal
        elif scip_status == "infeasible":
            problem.status = pulp.LpStatusInfeasible
        elif scip_status == "unbounded":
            problem.status = pulp.LpStatusUnbounded
        else:
            problem.status = pulp.LpStatusUndefined

        return problem

    def solve_scip(self, scip_model: Model) -> Model:
        """Solve the given pyscipopt Model.

        :param scip_model: The pyscipopt Model to be solved.
        :return: The solved pyscipopt Model.
        """
        scip_model.optimize()
        return scip_model

    def enumerate_feasible_solutions(
        self, problem: pulp.LpProblem, max_solutions: int, large_num: float = 1e5
    ) -> list[dict[str, float]]:
        """Enumerate multiple feasible solutions for the given problem.

        This is an override of the standard method in the base Solver class,
        leveraging pyscipopt's capabilities at finding multiple solutions in one solve.
        This potentially offers performance benefits over iterative enumeration.

        :param problem: the optimization problem to find feasible solutions for.
        :param max_solutions: the maximum number of feasible solutions to find.
        :param large_num: a large number used in the enumeration process. This should
            be significantly larger than the expected variable values in the problem.
        :return: a list of feasible solutions as dictionaries of variable names and
            values.
        """
        return _enumerate_feasible_solutions_scip(
            problem, self, max_solutions, large_num
        )


def _convert_pulp_to_scip(problem: pulp.LpProblem) -> Model:
    """Convert a PuLP problem to a pyscipopt Model.

    :param problem: The PuLP problem instance to be converted.
    :return: The corresponding pyscipopt Model.
    """
    problem.writeLP("temp_problem.lp")
    scip_model = Model()
    scip_model.readProblem("temp_problem.lp")

    # Remove the file right after reading
    os.remove("temp_problem.lp")
    return scip_model


def _enumerate_feasible_solutions_scip(
    problem: pulp.LpProblem,
    solver: ScipSolver,
    max_solutions: int = 10,
    large_num: float = 1e5,
) -> list[dict[str, float]]:
    """
    Enumerate multiple feasible solutions using pySCIPOpt and exclusion constraints.

    This enumeration of feasible solutions offer the benefit of finding multiple
    solutions in a single solve call, which can be more efficient than an iterative
    approach.

    :param problem: An LP for which to find multiple feasible solutions.
    :param solver: A solver used to solve the LP.
    :param max_solutions: Maximum number of feasible solutions to find.
    :param large_num: a large number used in the enumeration process. This should
        be significantly larger than the expected variable values in the problem.
    :raises ValueError: If the number of solutions is negative.
    :return: List of solutions as dicts {varname: value}.
    """
    solutions: list[Dict[str, float]] = []
    if max_solutions < 0:
        raise ValueError("Number of solutions should not be a negative integer.")

    scip_model = _convert_pulp_to_scip(problem)

    original_var_names = [var.name for var in problem.variables()]
    var_list = [var for var in scip_model.getVars() if var.name in original_var_names]
    while True:
        current_length = len(solutions)
        scip_model = solver.solve_scip(scip_model)
        status = scip_model.getStatus()
        if status not in ["optimal", "feasible", "unbounded"]:
            # No more feasible solutions
            warnings.warn(
                (
                    f"Solver status is '{status}', which means there are no more "
                    "feasible solutions."
                ),
                UserWarning,
            )
            return solutions

        scip_solutions = sorted(
            scip_model.getSols(),
            key=lambda sol: scip_model.getSolObjVal(sol),
            reverse=(
                scip_model.getObjectiveSense() == "maximize"
            ),  # True for maximization
        )

        # Convert each solution into a plain Python dict
        extracted_solution_list = []
        for sol in scip_solutions:
            extracted_solution = {
                v.name: scip_model.getSolVal(sol, v) for v in var_list
            }
            extracted_solution_list.append(extracted_solution)
            # Extract variable values
            solution = {}
            for v in var_list:
                if any(v.name.startswith(prefix) for prefix in original_var_names):
                    solution[v.name] = extracted_solution[v.name]
            solutions.append(solution)
        if len(solutions) >= max_solutions:
            return solutions[
                :max_solutions
            ]  # don't return more solutions than max_solutions

        # Before adding constraints, exit solve mode:
        scip_model.freeTransform()
        for j, scip_solution in enumerate(extracted_solution_list):
            (
                scip_model,
                exclusion_binary,
                exclusion_combs,
                number_of_integer_variables,
            ) = _build_exclusion_constraints_scip(
                var_list,
                scip_model,
                scip_solution,
                j=j,
                current_length=current_length,
                large_num=large_num,
            )

            # Exclude the binary part (at least one bit is different)
            if len(exclusion_binary) > 0:
                scip_model.addCons(
                    quicksum(exclusion_binary) >= 1,
                    f"exclusion_binary_{j+current_length}",
                )

            # Make sure only one of the existing solution constraints is excluded
            if len(exclusion_combs) > 0:
                scip_model.addCons(
                    quicksum(exclusion_combs) <= number_of_integer_variables - 1,
                    f"exclusion_combination_{j+current_length}",
                )


def _build_exclusion_constraints_scip(
    var_list: List[Any],
    scip_model: Model,
    scip_solution: Dict[Any, Any],
    j: int = 0,
    current_length: int = 0,
    large_num: float = 1e5,
) -> Tuple[Model, list[Any], list[Any], int]:
    """Construct exclusion constraints to eliminate the current solution.

    :param var_list: List of variables from the SCIP model.
    :param scip_model: The SCIP model to which constraints are added.
    :param scip_solution: The current solution from the SCIP model.
    :param j: Optional index to ensure unique constraint names.
    :param current_length: Current length of the solutions list.
    :param large_num: a large number used in the enumeration process. This should
        be significantly larger than the expected variable values in the problem.
    :returns: Tuple of (scip_model, exclusion_binary, exclusion_combs,
        number_of_integer_variables).
    """
    exclusion_binary = []
    exclusion_combs = []
    number_of_integer_variables = 0
    big_M = large_num  # big-M trick

    var_list_for_constraints = var_list
    for v in var_list_for_constraints:
        if v.vtype() == "BINARY":
            try:
                value = scip_solution[v.name]
            except KeyError:
                value = 0
            if value == 1:
                exclusion_binary.append(1 - v)
            else:
                exclusion_binary.append(v)
        elif v.vtype() == "INTEGER":
            delta = scip_model.addVar(
                f"delta_{v.name}_{j+current_length}", vtype="BINARY"
            )
            comb = scip_model.addVar(
                f"comb_{v.name}_{j+current_length}", vtype="BINARY"
            )
            var_list_for_constraints.append(delta)
            var_list_for_constraints.append(comb)
            exclusion_combs.append(comb)

            scip_model.addCons(
                v - scip_solution[v.name] >= 1 - big_M * delta - 2 * big_M * comb
            )
            scip_model.addCons(
                v - scip_solution[v.name] <= -1 + big_M * (1 - delta) + 2 * big_M * comb
            )

            number_of_integer_variables += 1
    return scip_model, exclusion_binary, exclusion_combs, number_of_integer_variables
