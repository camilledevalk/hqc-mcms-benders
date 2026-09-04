# tests/test_run.py
"""This module provides tests for the command-line interface of the BenderMILPSolver."""
import json
import sys
from pathlib import Path
from typing import Any

import pulp
import pytest
from _pytest.monkeypatch import MonkeyPatch

import milp_engine.run as run
from milp_engine.criteria.coverage_criterion import CoverageCriterion
from milp_engine.criteria.exclusion_criterion import ExclusionCriterion
from milp_engine.solvers.highs_solver import HiGHSSolver
from milp_engine.solvers.pulp_solver import PuLPSolver
from milp_engine.solvers.quantum_solver import QuantumSolver
from milp_engine.strategies.maximum_coverage_strategy import MaximumCoverageStrategy
from milp_engine.strategies.minimum_set_cover_strategy import MinimumSetCoverStrategy


@pytest.fixture
def mock_token_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    # Write the tokens.json file inside it
    token_path = tmp_path / "tokens.json"
    # token_path = token_dir / "tokens.json"

    token_path.write_text(json.dumps({"id": "test_id", "secret": "test_secret"}))

    # Patch os.getcwd to return the temp path
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    return token_path


@pytest.mark.parametrize(
    "cli_args,expected_solver_args, expected_problem_name, expected_y_prefix",
    [
        (
            [
                "--criterion",
                "coverage",
                "--strategy",
                "min-set-cover",
                "--problem-name",
                "savings_milp",
            ],
            {
                "criterion": CoverageCriterion,
                "strategy": MinimumSetCoverStrategy,
                "mp_solver": PuLPSolver,
                "dsp_solver": HiGHSSolver,
                "cut_selection_solver": PuLPSolver,
                "max_iterations": None,
                "n_subproblems": 1,
                "convergence_bound": 1e-5,
                "parallel_dsp": True,
            },
            "savings_milp",
            "y",
        ),
        (
            [
                "--criterion",
                "exclusion",
                "--strategy",
                "max-coverage",
                "--max-cuts",
                "5",
                "--problem-name",
                "binary_choice_milp",
                "--y-prefix",
                "u",
                "--n-subproblems",
                "3",
                "--max-iterations",
                "10",
                "--convergence-bound",
                "0.01",
                "--no-parallel",
                "--use-quantum",
                "--backend",
                "qiskit",
                "--backend-options",
                "{}",
                "--solver-options",
                "{}",
            ],
            {
                "criterion": ExclusionCriterion,
                "strategy": MaximumCoverageStrategy,
                "mp_solver": PuLPSolver,
                "dsp_solver": HiGHSSolver,
                "cut_selection_solver": QuantumSolver,
                "max_iterations": 10,
                "n_subproblems": 3,
                "convergence_bound": 0.01,
                "parallel_dsp": False,
            },
            "binary_choice_milp",
            "u",
        ),
    ],
)
def test_argument_parsing_and_solver_construction(
    cli_args: list[str],
    expected_solver_args: dict[str, Any],
    expected_problem_name: str,
    expected_y_prefix: str,
    mocker: Any,
    mock_token_file: Path,
) -> None:
    # Inject the command line arguments
    mocker.patch.object(sys, "argv", ["run.py"] + cli_args)

    # Create mock objects for the solver constructor, the solver object,
    #   the solve method, and the solved problem.
    #   This way, we can avoid calling the actual solver (expensive).
    solved_problem_mock = pulp.LpProblem("mocked_problem", pulp.LpMinimize)
    solved_problem_mock.status = pulp.LpStatusOptimal
    solved_problem_mock.objective = 1.0 * pulp.LpVariable("x")

    solver_mock = mocker.Mock()
    solver_mock.solve.return_value = solved_problem_mock

    solver_constructor_mock = mocker.patch(
        "milp_engine.run.BenderMILPSolver", return_value=solver_mock
    )

    run.main()

    # Check if BenderMILPSolver was constructed with the expected arguments
    solver_constructor_mock.assert_called_once()
    _, kwargs = solver_constructor_mock.call_args
    for key, val in expected_solver_args.items():
        if key in [
            "criterion",
            "strategy",
            "mp_solver",
            "dsp_solver",
            "cut_selection_solver",
        ]:
            # These are objects, so we check their type
            assert isinstance(kwargs[key], val)
        else:
            assert kwargs[key] == pytest.approx(val)

    # Check if the correct MILP problem was retrieved and if the correct y_prefix
    #   was passed
    solver_mock.solve.assert_called_once()
    args, kwargs = solver_mock.solve.call_args
    assert args[0].name == expected_problem_name
    assert kwargs["y_prefix"] == expected_y_prefix


@pytest.mark.parametrize(
    "cli_args,error_message",
    [
        ([], "You must specify either --mps-file or --problem-name."),
        (
            ["--criterion", "coverage", "--strategy", "max-coverage"],
            "The --max-cuts argument is required when using the 'max-coverage' "
            "strategy.",
        ),
        (
            ["--criterion", "coverage", "--strategy", "take-all"],
            "You must specify either --mps-file or --problem-name.",
        ),
    ],
)
def test_parser_errors(cli_args: list[str], error_message: str, mocker: Any) -> None:
    mocker.patch.object(sys, "argv", ["run.py"] + cli_args)
    with pytest.raises(SystemExit) as _:
        run._parse_arguments()
