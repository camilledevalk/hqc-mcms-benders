from typing import Dict, List, Tuple

import numpy as np
import pulp
import pytest

from milp_engine.criteria.coverage_criterion import CoverageCriterion
from milp_engine.criteria.exclusion_criterion import ExclusionCriterion
from milp_engine.data.bender_state import BenderState
from milp_engine.data.cut import Cut, CutType
from milp_engine.solvers.bender_milp_solver import (
    BenderMILPSolver,
    construct_master_problem,
    create_feasibility_cut,
    create_optimality_cut,
    solve_milp_given_y_hat,
    update_master_problem,
)
from milp_engine.solvers.highs_solver import HiGHSSolver
from milp_engine.solvers.pulp_solver import PuLPSolver
from milp_engine.solvers.scip_solver import ScipSolver
from milp_engine.strategies.maximum_coverage_strategy import MaximumCoverageStrategy
from milp_engine.strategies.minimum_set_cover_strategy import MinimumSetCoverStrategy
from milp_engine.strategies.take_all_strategy import TakeAllStrategy
from milp_engine.utils.lp import get_objective_val


@pytest.fixture
def simple_milp() -> pulp.LpProblem:
    """Produce a simple MILP for testing.

    :return: The problem as a pulp.LpProblem instance.
    """
    milp = pulp.LpProblem("simple_milp", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0, upBound=5, cat="Continuous")
    x2 = pulp.LpVariable("x2", lowBound=1, upBound=4, cat="Continuous")
    y1 = pulp.LpVariable("y1", cat="Binary")
    y2 = pulp.LpVariable("y2", cat="Binary")

    milp += x1 + 2 * x2 + 3 * y1 + 4 * y2, "Maximize"
    milp += x1 + x2 <= 10, "Constraint1"
    milp += y1 + y2 >= 1, "Constraint2"

    return milp


@pytest.mark.parametrize(
    "variable_prefix",
    ["x", "y"],
)
def test_master_problem_constructor(
    variable_prefix: str, simple_milp: pulp.LpProblem
) -> None:
    mp = construct_master_problem(simple_milp, variable_prefix)

    assert all(
        (
            (var.name, coeff)
            in [(f"{variable_prefix}1", 0), (f"{variable_prefix}2", 0), ("z", 1)]
        )
        for var, coeff in mp.objective.items()
    )


def test_update_master_problem(simple_milp: pulp.LpProblem) -> None:

    mp = construct_master_problem(simple_milp, "x")

    # Define cuts
    cuts = [
        Cut(np.array([1.0, 0.0, 0.0]), -100.0, CutType.FEASIBILITY),
        Cut(np.array([0.0, 1.0, -1.0]), -200.0, CutType.OPTIMALITY),
    ]

    mp_updated = update_master_problem(mp, cuts)

    # Check if the number of constraints has increased by 2
    assert len(mp_updated.constraints) - len(mp.constraints) == 2

    # Check if the added constraints are equal to the cuts that needed to be added.
    for cut, constraint in zip(cuts, mp_updated.constraints.values()):
        # Check if the constraint is equal to the cut
        mp_vars = mp_updated.variables()
        assert all(
            cut.coeff_vector[j] == constraint.get(v, 0) for j, v in enumerate(mp_vars)
        )
        assert constraint.constant == cut.constant
        assert constraint.sense == pulp.LpConstraintLE

    # Check if the same-name variables in constraints and objective are the
    # same python objects.
    for c in mp_updated.constraints.values():
        for c_v in c.expr.keys():
            match = [v for v in mp_updated.objective.keys() if v.name == c_v.name]
            if match:
                obj_v = match[0]
                assert id(obj_v) == id(c_v)


def test_create_feasibility_cut(
    basic_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable]],
) -> None:
    milp, y = basic_milp
    extreme_ray = np.array([1, 0])

    feasibility_cut = create_feasibility_cut(milp, y, extreme_ray)

    expected_cut = Cut(np.array([1000.0, 0.0, 0.0]), 1.0, CutType.FEASIBILITY)
    assert feasibility_cut == expected_cut


def test_create_optimality_cut(
    basic_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable]],
) -> None:
    milp, y = basic_milp
    u_tilde = np.array([0.5, 2.0])

    optimality_cut = create_optimality_cut(milp, y, u_tilde)

    assert optimality_cut.type == CutType.OPTIMALITY

    expected_cut = Cut(np.array([500.4, 4000.5, -1.0]), 4.5, CutType.OPTIMALITY)
    assert optimality_cut == expected_cut


def test_solve_milp(basic_milp: Tuple[pulp.LpProblem, List[pulp.LpVariable]]) -> None:
    milp, _ = basic_milp
    solver = PuLPSolver()
    y_hat = {"y_1": 1.0, "y_2": 0.0}
    solved_milp = solve_milp_given_y_hat(solver, milp, y_hat)

    e = {
        "x_1": 1001.0 / 0.4,
        "x_2": 0.0,
        "x_3": 0.0,
        "y_1": 1.0,
        "y_2": 0.0,
        "z": 0.0,
    }

    expected_objective = (
        e["z"]
        + 0.1 * e["x_1"]
        + 0.2 * e["x_2"]
        + 0.3 * e["x_3"]
        + 0.4 * e["y_1"]
        + 0.5 * e["y_2"]
    )

    assert solved_milp.status == pulp.LpStatusOptimal
    assert pulp.value(solved_milp.objective) == pytest.approx(expected_objective)


@pytest.mark.parametrize(
    (
        "milp_name, n_subproblems, parallel, strategy_choice, "
        "criterion_choice, mp_solver_choice, var_match"
    ),
    list(
        set(
            (
                (
                    "mtz_tsp_milp",
                    1,
                    False,
                    "take_all_strategy",
                    "coverage_criterion",
                    mp_solver,
                    False,
                )
                if milp == "mtz_tsp_milp"
                else (
                    (
                        (
                            milp,
                            5,
                            True,
                            "take_all_strategy",
                            "coverage_criterion",
                            mp_solver,
                            True,
                        )
                        if parallel
                        else (
                            milp,
                            n_subproblems,
                            False,
                            "take_all_strategy",
                            "coverage_criterion",
                            mp_solver,
                            True,
                        )
                    )
                    if strategy == "take_all_strategy"
                    else (
                        milp,
                        5,
                        False,
                        strategy,
                        criterion,
                        mp_solver,
                        True,
                    )
                )
            )
            for mp_solver in ["pulp_solver", "scip_solver"]
            for milp in ["savings_milp", "binary_choice_milp", "mtz_tsp_milp"]
            for strategy in [
                "take_all_strategy",
                "maximum_coverage_strategy",
                "minimum_set_cover_strategy",
            ]
            for criterion in ["coverage_criterion", "exclusion_criterion"]
            for parallel in [False, True]
            for n_subproblems in [1, 5]
        )
    ),
)
def test_bender_milp_solver(
    milp_name: str,
    n_subproblems: int,
    parallel: bool,
    strategy_choice: str,
    criterion_choice: str,
    mp_solver_choice: str,
    var_match: bool,
    milps: Dict[str, Tuple[pulp.LpProblem, List[pulp.LpVariable], Dict[str, float]]],
    ray_shutdown: None,
) -> None:
    strategy_map = {
        "take_all_strategy": TakeAllStrategy(),
        "maximum_coverage_strategy": MaximumCoverageStrategy(max_cuts=2),
        "minimum_set_cover_strategy": MinimumSetCoverStrategy(),
    }
    criterion_map = {
        "coverage_criterion": CoverageCriterion(),
        "exclusion_criterion": ExclusionCriterion(),
    }
    mp_solver_map = {
        "pulp_solver": PuLPSolver(),
        "scip_solver": ScipSolver(),
    }
    strategy = strategy_map[strategy_choice]
    criterion = criterion_map[criterion_choice]
    mp_solver = mp_solver_map[mp_solver_choice]

    milp, y, expected_solution = milps[milp_name]

    # Create a solver instance
    classical_solver = PuLPSolver()
    rich_solver = HiGHSSolver()
    bender_solver = BenderMILPSolver(
        criterion,
        strategy,
        mp_solver,
        rich_solver,
        classical_solver,
        n_subproblems=n_subproblems,
        convergence_bound=1e-5,  # Effectively 0. This is for numerical stability.
        parallel_dsp=parallel,
    )

    # Solve the MILP using the Bender's decomposition solver
    milp_solved = bender_solver.solve(milp)

    assert milp_solved.status == pulp.LpStatusOptimal

    expected_obj_val = get_objective_val(milp_solved, expected_solution)
    assert milp_solved.objective.value() == pytest.approx(expected_obj_val)

    if var_match:
        for var_name, expected_value in expected_solution.items():
            actual = pulp.value(milp_solved.variablesDict()[var_name])
            assert actual == pytest.approx(expected_value)


@pytest.mark.skip(reason="Passes but runtime is too long: 2-3mins.")
def test_bender_milp_solver_multi_depot_vrp() -> None:
    _, vrp_problem = pulp.LpProblem.fromMPS("../res/MultiDepotVRP.mps")

    # Create a solver instance
    classical_solver = PuLPSolver()
    rich_solver = HiGHSSolver()
    criterion = CoverageCriterion()
    strategy = TakeAllStrategy()
    bender_solver = BenderMILPSolver(
        criterion,
        strategy,
        classical_solver,
        rich_solver,
        classical_solver,
        n_subproblems=1,
        convergence_bound=1e-3,
        max_iterations=500,
        parallel_dsp=False,
    )

    milp_solved = bender_solver.solve(vrp_problem, y_prefix="x")
    assert milp_solved.status == pulp.LpStatusOptimal
    assert milp_solved.objective.value() == pytest.approx(444.98329799, 1e-3)


def test_bender_milp_solver_solve_exposed() -> None:
    _, vrp_problem = pulp.LpProblem.fromMPS("../res/MultiDepotVRP.mps")

    # Create a solver instance
    classical_solver = PuLPSolver()
    rich_solver = HiGHSSolver()
    criterion = CoverageCriterion()
    strategy = TakeAllStrategy()
    bender_solver = BenderMILPSolver(
        criterion,
        strategy,
        classical_solver,
        rich_solver,
        classical_solver,
        n_subproblems=2,
        convergence_bound=1e-3,
        max_iterations=1,
        parallel_dsp=True,
    )

    _, states = bender_solver.solve_exposed(vrp_problem, y_prefix="x")

    # Check all required fields in each BenderState
    for state in states:
        assert isinstance(state, BenderState)
        assert state.iteration is not None, "Missing iteration"
        assert isinstance(state.mp, pulp.LpProblem), "mp should be a pulp.LpProblem"
        assert isinstance(state.y_hats, list) and all(
            isinstance(assignment, dict) for assignment in state.y_hats
        ), "Invalid y_hats"
        assert state.lower_bound is not None, "Missing lower_bound"
        assert state.upper_bound is not None, "Missing upper_bound"
        assert isinstance(state.runtimes, dict), "runtimes should be a dict"
        # Keys expected in runtimes and their expected types
        expected_runtime_keys = {
            "mp_solving_time": float,
            "sp_solving_time": float,
            "cut_selection_time": float,
            "iteration_time": float,
            "individual_sp_solving_time": list,
            "individual_sp_solving_time_with_init": list,
        }

        for key, expected_type in expected_runtime_keys.items():
            value = state.runtimes.get(key)
            assert value is not None, f"Missing runtime key '{key}'"

            if expected_type == float:
                assert isinstance(
                    value, (int, float)
                ), f"Runtime '{key}' should be a number, got {type(value)}"
                assert value > 0, f"Runtime '{key}' should be > 0, got {value}"
            elif expected_type == list:
                assert isinstance(
                    value, list
                ), f"Runtime '{key}' should be a list, got {type(value)}"
                assert all(
                    isinstance(v, (int, float)) and v > 0 for v in value
                ), f"Runtime list '{key}' contains invalid entries: {value}"
