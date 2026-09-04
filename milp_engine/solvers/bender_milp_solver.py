# milp_engine/solvers/bender_milp_solver.py
"""This module contains Bender MILP solver interfaces and implementations.

The 'BenderMILPSolver' class applies Bender's Decomposition to decompose and
solve large-scale MILPs iteratively. It separates the problem into a master
problem and one or more subproblems, then iteratively refines the master
problem by generating and selecting cuts based on user-defined criteria and
strategies. It supports classical and quantum solvers as backends
(for cut selection strategies) and provides flexible configuration for
optimization parameters such as convergence bounds, maximum iterations, and
solution limits.
"""
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, cast

import numpy as np
import pulp

from milp_engine.criteria.base_criterion import Criterion
from milp_engine.data.bender_state import BenderState
from milp_engine.data.cut import Cut, CutType
from milp_engine.solvers.base_solver import CutSelectionSolver, ExtremeRaySolver, Solver
from milp_engine.strategies.base_strategy import Strategy
from milp_engine.utils.lp import (
    get_B,
    get_b,
    get_d,
    get_dual,
    get_objective_val,
    substitute_vars,
)
from milp_engine.utils.parallel import run_tasks_parallel


def create_feasibility_cut(
    milp: pulp.LpProblem,
    y: list[pulp.LpVariable],
    extreme_ray: np.ndarray,
) -> Cut:
    """Construct a feasibility cut.

    :param milp: The MILP as a pulp.LpProblem instance.
    :param y: List of master problem decision variables.
    :param extreme_ray: The extreme ray direction.
    :return: A feasibility cut.
    """
    b = get_b(milp)
    B = get_B(milp, y)

    coeff_vector = (-B.T) @ extreme_ray
    constant = (b.T @ extreme_ray).item()

    # Append a 0 for the z variable, as it does not appear a feasibility cut
    coeff_vector = np.append(coeff_vector, 0)

    return Cut(coeff_vector, constant, CutType.FEASIBILITY)


def create_optimality_cut(
    milp: pulp.LpProblem,
    y: list[pulp.LpVariable],
    u_tilde: np.ndarray,
) -> Cut:
    """Construct an optimality cut.

    :param milp: The MILP as a pulp.LpProblem instance.
    :param y: List of master problem decision variables.
    :param u_tilde: Optimal values of dual decision variables.
    :return: An optimality cut.
    """
    b = get_b(milp)
    B = get_B(milp, y)
    d = get_d(milp, y)
    constant = (b.T @ u_tilde).item()
    coeff_vector = (-B.T) @ u_tilde + np.transpose(d)

    # Add a -1 for the z variable, as it appears on the right-hand side of the
    #   optimality cut
    coeff_vector = np.append(coeff_vector, -1)

    return Cut(coeff_vector, constant, CutType.OPTIMALITY)


def update_master_problem(
    mp: pulp.LpProblem,
    cuts: List[Cut],
) -> pulp.LpProblem:
    """Update the master problem with a set of new cuts.

    :param mp: The master problem to be updated.
    :param cuts: A list of Cut instances.
    :raises AssertionError: If not all constraints were added to the master problem.
    :return: The updated master problem with the new constraints added.
    """
    mp_copy = deepcopy(mp)
    mp_var_dict = mp_copy.variablesDict()
    mp_vars_sorted = [v for name, v in sorted(mp_var_dict.items()) if name != "z"] + [
        mp_var_dict["z"]
    ]  # sorted by variable names and with z in the back

    # Loop over all coefficients and variables and create pulp constraints
    for cut in cuts:
        mp_copy += (
            pulp.lpSum([cut.coeff_vector[i] * v for i, v in enumerate(mp_vars_sorted)])
            + cut.constant
            <= 0
        )  # A cut is always of the form "<= 0"

    constraints_added = len(mp_copy.constraints) - len(mp.constraints)

    assert constraints_added == len(
        cuts
    ), f"Expect {len(cuts)} constraints, {constraints_added} added."

    return mp_copy


def construct_master_problem(
    milp: pulp.LpProblem, variable_prefix: str
) -> pulp.LpProblem:
    """Construct the master problem from the given MILP problem.

    The master problem minimizes the sum of selected variables,
    keeping only their bounds as constraints.

    :param milp: The full MILP problem to extract the master problem from.
    :param variable_prefix: Prefix used to select which variables to include.
    :return: The constructed master problem.
    """
    # Create master_problem_constructed such that mypy allows
    mp = pulp.LpProblem("master_problem", pulp.LpMinimize)

    # Get copies of the y variables from the MILP
    y = []
    for y_i in milp.variables():
        if y_i.name.startswith(variable_prefix):
            y_copy = deepcopy(y_i)
            # Add lower and upper bounds if not present
            if y_copy.lowBound is None:
                y_copy.lowBound = -1e6
            if y_copy.upBound is None:
                y_copy.upBound = 1e6
            y.append(y_copy)

    # Create a new integer variable 'z' to represent x-part of the MILP within the MP
    z = pulp.LpVariable("z", lowBound=-1e6, cat=pulp.LpContinuous)

    # Objective function: Minimize the sum of the binary variables
    mp += (
        pulp.LpAffineExpression(
            [(v, val) for v, val in zip([z] + y, [1.0] + [0.0] * (len(y) + 1))]
        ),
        "objective",
    )

    return mp


def solve_milp_given_y_hat(
    solver: Solver, milp: pulp.LpProblem, y_hat: Dict[str, float]
) -> pulp.LpProblem:
    """Solve an MILP by substituting the y and then doing a regular solve.

    :param solver: The solver instance to use for solving the problem that remains
        after substitution.
    :param milp: The MILP problem to solve.
    :param y_hat: The values of the y variables to substitute into the MILP.
    :return: The solved MILP problem.
    """
    final_sp = substitute_vars(milp, y_hat)
    final_sp = solver.solve(final_sp)

    solved_milp = deepcopy(milp)

    # Set milp variables to their optimal value
    for v_i in solved_milp.variables():
        if v_i.name in y_hat:
            v_i.varValue = y_hat[v_i.name]
        else:
            x_i = [x_j for x_j in final_sp.variables() if x_j.name == v_i.name][0]
            v_i.varValue = x_i.varValue

    # Set the status of the solved optimal subproblem
    solved_milp.status = final_sp.status

    return solved_milp


@dataclass
class DSPTaskResult:
    """A class to hold the result of a DSP task.

    :param status: The status of the DSP task after solving.
    :param cut: The cut generated from the DSP task, if applicable.
    :param objective_value: The objective value of the DSP task, if solved to
        optimality.
    :param upper_bound: The upper bound derived from of the DSP task, if solved to
        optimality.
    """

    status: int
    cut: Cut | None
    objective_value: float | None
    upper_bound: float | None = None


def create_and_solve_dsp(
    dsp_solver: ExtremeRaySolver,
    milp: pulp.LpProblem,
    y_hat: Dict[str, float],
) -> DSPTaskResult:
    """Create and solve a dual sub problem from the given MILP and y_hat.

    The provided solver instance is used for solving the dual.

    :param dsp_solver: The solver instance to use for solving the dual sub problem.
    :param milp: The MILP problem to extract the dual sub problem from.
    :param y_hat: Fixed values for decision variables 'y'.
    :return: The result of the DSP task as a DSPTaskResult instance.

    :raises ValueError: If the DSP solver status is unexpected.
    """
    # Step 1: Create sub problem by substituting y_hat into MILP
    sp = substitute_vars(milp, y_hat)

    # Step 2: Create dual of sub problem
    dsp = get_dual(sp)

    # Step 3: Solve dual sub problem
    dsp_solved, extreme_ray = dsp_solver.solve_and_find_extreme_ray(dsp)

    # Step 4: Create cuts based on the solution of the DSP (or declare infeasible)
    y = [var for var in milp.variables() if var.name in y_hat.keys()]

    match dsp_solved.status:
        # Step 4.1: If infeasible, simply return infeasible. Make no cut
        case pulp.LpStatusInfeasible:
            return DSPTaskResult(pulp.LpStatusInfeasible, None, None, None)

        # Step 4.2: Else, if unbounded, produce infeasibility cut
        case pulp.LpStatusUnbounded:
            extreme_ray = cast(np.ndarray, extreme_ray)
            cut = create_feasibility_cut(milp, y, extreme_ray)
            return DSPTaskResult(pulp.LpStatusUnbounded, cut, None, None)

        # Step 4.3: Else, if optimal, calculate UB and produce optimality cut
        case pulp.LpStatusOptimal:
            u_tilde = np.array([var.varValue for var in dsp_solved.variables()])
            cut = create_optimality_cut(milp, y, u_tilde)

            # Calculate the upper bound
            milp_obj_keyed_by_str = {  # needed because we only have var names in y_hat
                var.name: milp.objective.get(var, 0) for var in milp.variables()
            }
            ub_y = sum(
                milp_obj_keyed_by_str[y_hat_i_name] * y_hat_i_val
                for y_hat_i_name, y_hat_i_val in y_hat.items()
                if y_hat_i_name != "z"  # z does not appear in the MILP, only in MP
            )
            ub_x = pulp.value(dsp_solved.objective)
            upper_bound = ub_y + ub_x

            return DSPTaskResult(
                pulp.LpStatusOptimal, cut, pulp.value(dsp_solved.objective), upper_bound
            )

        case _:
            raise ValueError(f"Unexpected status from DSP solver {dsp_solved.status}.")


class BenderMILPSolver(Solver):
    """A solver class that applies Bender's Decomposition."""

    def __init__(
        self,
        criterion: Criterion,
        strategy: Strategy,
        mp_solver: Solver,
        dsp_solver: ExtremeRaySolver,
        cut_selection_solver: CutSelectionSolver,
        max_iterations: int | None = 100,
        convergence_bound: float = 1e-4,
        n_subproblems: int = 1,
        parallel_dsp: bool = False,
    ):
        """Initialize setting for Bender's Decomposition.

        :param criterion: The selection criterion to use.
        :param strategy: The selection strategy to use.
        :param mp_solver: The solver to use for the master problem.
        :param dsp_solver: The solver to use for the dual subproblem.
        :param cut_selection_solver: The solver to use for cut selection.
        :param max_iterations: Maximum number of iterations allowed.
        :param convergence_bound: Threshold for convergence.
        :param n_subproblems: The number of subproblems considered per iteration.
            Technically, this is an upper bound on the number of subproblems, as fewer
            may be created if the master problem returns fewer solutions.
        :param parallel_dsp: Boolean indicating whether to use parallel processing for
            creating and solving (dual) subproblems.
        """
        self.criterion = criterion
        self.strategy = strategy
        self.mp_solver = mp_solver
        self.dsp_solver = dsp_solver
        self.cut_selection_solver = cut_selection_solver
        self.max_iterations = max_iterations
        self.convergence_bound = convergence_bound
        self.n_subproblems = n_subproblems
        self.parallel_dsp = parallel_dsp

    def solve(
        self, milp: pulp.LpProblem, y_prefix: str = "y", verbose: bool = False
    ) -> pulp.LpProblem:
        """Solve optimization problem using Bender's Decomposition.

        :param milp: A MILP that needs to be solved.
        :param y_prefix: The prefix in the names of the complicating variables
            (the integer variables, typically).
        :param verbose: If True, print progress and convergence information.
        :returns: Solved MILP.
        """
        milp_solved, _ = self.solve_exposed(milp, y_prefix, verbose)
        return milp_solved

    def solve_exposed(
        self, milp: pulp.LpProblem, y_prefix: str = "y", verbose: bool = False
    ) -> Tuple[pulp.LpProblem, List[BenderState]]:
        """Solve optimization problem using Bender's Decomposition.

        This method additionally exposes the internal state of the solver,
        allowing for more detailed debugging and analysis.

        :param milp: A MILP that needs to be solved.
        :param y_prefix: The prefix in the names of the complicating variables
            (the integer variables, typically). By default, 'y'.
        :param verbose: If True, print progress and convergence information.
        :returns: Solved MILP.
        """
        states: list[BenderState] = []

        lower_bound, upper_bound = -float("inf"), float("inf")
        mp = construct_master_problem(milp, y_prefix)
        added_cuts: Set[Tuple[float, ...]] = set()  # tuple is hashable

        iteration = 1
        while self.max_iterations is None or iteration <= self.max_iterations:
            start_iteration_time = time.perf_counter()
            # Find and extract n_subproblems number of solutions.
            start_mp_solving = time.perf_counter()
            y_hats = self.mp_solver.enumerate_feasible_solutions(mp, self.n_subproblems)
            end_mp_solving = time.perf_counter()

            # Create a state object to hold the current state of Bender's Decomposition.
            #   This is useful for debugging and analysis.
            state = BenderState(
                iteration=iteration,
                mp=mp,
                y_hats=y_hats,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

            # Track the running time of the master problem solving procedure
            state.runtimes["mp_solving_time"] = end_mp_solving - start_mp_solving

            # If the master problem is now infeasible, declare the MILP infeasible.
            if not y_hats:
                milp.status = pulp.LpStatusInfeasible
                states.append(state)
                return milp, states

            y_star = y_hats[0]  # Take the first solution as y_star

            # Update lower bound based on best solution for the current MP
            mp_obj_val = get_objective_val(mp, y_star)
            lower_bound = max(lower_bound, mp_obj_val)
            state.lower_bound = lower_bound

            # Solve a maximum of self.n_subproblems dual sub problems
            start_sp_solving = time.perf_counter()
            if self.parallel_dsp:
                dsp_results, individual_sp_timings, individual_sp_timings_with_init = (
                    run_tasks_parallel(
                        lambda args: create_and_solve_dsp(*args),
                        [(self.dsp_solver, milp, y_hat) for y_hat in y_hats],
                        timing=True,
                    )
                )
            else:
                dsp_results = [
                    create_and_solve_dsp(self.dsp_solver, milp, y_hat)
                    for y_hat in y_hats
                ]
            end_sp_solving = time.perf_counter()

            # Track the running time of the subproblem solving procedure
            state.runtimes["sp_solving_time"] = end_sp_solving - start_sp_solving
            if self.parallel_dsp:
                state.runtimes["individual_sp_solving_time"] = individual_sp_timings
                state.runtimes["individual_sp_solving_time_with_init"] = (
                    individual_sp_timings_with_init
                )

            # Check feasibility of dual sub problems (solution)
            if any(result.status == pulp.LpStatusInfeasible for result in dsp_results):
                # Declare MILP infeasible if any DSP is infeasible
                milp.status = pulp.LpStatusInfeasible
                states.append(state)
                return milp, states

            # Update upper bound with min(UB, solution dual sub problem)
            dsp_upper_bounds = [
                dsp_result.upper_bound
                for dsp_result in dsp_results
                if dsp_result.status == pulp.LpStatusOptimal
            ]
            upper_bound = min([upper_bound] + cast(List[float], dsp_upper_bounds))
            state.upper_bound = upper_bound

            # If UB - LB < convergence_bound, we can stop iterating
            if (upper_bound - lower_bound) / max(
                abs(upper_bound), 1e-6
            ) < self.convergence_bound:
                solved_milp = solve_milp_given_y_hat(self.dsp_solver, milp, y_star)
                if verbose:
                    print(
                        f"Converged in {iteration:>5d} iterations: "
                        f"LB = {lower_bound:.5e}, "
                        f"UB = {upper_bound:.5e}, "
                        f"Gap = {upper_bound - lower_bound:.5e}, "
                        f"MP constraints = {len(mp.constraints):>5d}"
                    )
                return solved_milp, states

            # Select cuts based on combination criterion/strategy

            # If no DSP is infeasible, then all cuts must be well-defined.
            assert None not in [
                dsp_result.cut for dsp_result in dsp_results
            ], "All cuts must be well-defined."
            cuts: List[Cut] = cast(
                List[Cut], [dsp_result.cut for dsp_result in dsp_results]
            )

            # Exclude cuts that have already been added
            cuts = [
                cut
                for cut in cuts
                if tuple(cut.coeff_vector.tolist()) not in added_cuts
            ]
            state.candidate_cuts = cuts

            # Create binary indicator matrix
            binary_indicator_matrix, selection_cuts, passed_cuts = (
                self.criterion.construct_binary_indicator_matrix(state)
            )
            state.binary_indicator_matrix = binary_indicator_matrix

            start_cut_selection = time.perf_counter()
            selected_cuts = self.strategy.select_cuts(
                selection_cuts,
                binary_indicator_matrix,
                self.cut_selection_solver,
            )
            end_cut_selection = time.perf_counter()

            # Track the running time of the cut selection procedure
            state.runtimes["cut_selection_time"] = (
                end_cut_selection - start_cut_selection
            )

            mp_update_cuts = selected_cuts + passed_cuts
            state.mp_update_cuts = mp_update_cuts

            # Update master problem with selected cuts
            mp = update_master_problem(mp, mp_update_cuts)
            state.mp_updated = mp

            # Add selected cuts to the set of added cuts
            added_cuts.update(
                [tuple(cut.coeff_vector.tolist()) for cut in selected_cuts]
            )

            if verbose:
                print(
                    f"Iteration {iteration:>5d}, "
                    f"LB = {lower_bound:.5e}, "
                    f"UB = {upper_bound:.5e}, "
                    f"Gap = {upper_bound - lower_bound:.5e}, "
                    f"MP constraints = {len(mp.constraints):>5d}"
                )
            end_iteration_time = time.perf_counter()

            # Track the running time of the iteration
            state.runtimes["iteration_time"] = end_iteration_time - start_iteration_time

            # Save the BenderState
            states.append(state)

            iteration += 1

        # If we reach here, we did not converge within the maximum number of iterations.
        milp.status = pulp.LpStatusNotSolved
        return milp, states
