# milp_engine/run.py
"""This module provides a command-line interface for the BenderMILPSolver."""
import argparse
import json
import warnings
from typing import Any, Dict, Literal, cast

import pulp
from pydantic import BaseModel, model_validator

from milp_engine.criteria.base_criterion import Criterion
from milp_engine.criteria.coverage_criterion import CoverageCriterion
from milp_engine.criteria.exclusion_criterion import ExclusionCriterion
from milp_engine.solvers.base_solver import CutSelectionSolver
from milp_engine.solvers.bender_milp_solver import BenderMILPSolver
from milp_engine.solvers.highs_solver import HiGHSSolver
from milp_engine.solvers.pulp_solver import PuLPSolver
from milp_engine.solvers.quantum_solver import (
    FermioniqBackend,
    QiskitBackend,
    QuantumOptimisationBackend,
    QuantumSolver,
)
from milp_engine.strategies.base_strategy import Strategy
from milp_engine.strategies.maximum_coverage_strategy import MaximumCoverageStrategy
from milp_engine.strategies.minimum_set_cover_strategy import MinimumSetCoverStrategy
from milp_engine.strategies.take_all_strategy import TakeAllStrategy


def parse_json_dict(s: str) -> Dict[str, Any]:
    """
    Parse a JSON-formatted string into a Python dictionary.

    :param s: A string representing a JSON object.
    :return: The parsed JSON object as a Python dictionary.
    :raises ArgumentTypeError: If the input string is not valid JSON.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Options must be a valid JSON string")


class SolverArguments(BaseModel):
    """Subset of BenderMILPSolver arguments, just for constructing the solver."""

    n_subproblems: int = 1
    max_iterations: int | None = None
    convergence_bound: float = 1e-5
    no_parallel: bool = False
    use_quantum: bool = False
    criterion: Literal["coverage", "exclusion"]
    strategy: Literal["max-coverage", "min-set-cover", "take-all"]
    max_cuts: int | None = None
    backend: Literal["fermioniq", "qiskit"] | None = None
    backend_options: Dict[str, Any] = {}
    solver_options: Dict[str, Any] = {}
    verbose: bool = False


class MILPEngineArguments(SolverArguments):
    """Arguments for both the BenderMILPSolver command-line interface and the API."""

    mps_file: str | None = None
    problem_name: str | None = None
    y_prefix: str = "y"
    use_profiler: bool = False

    @model_validator(mode="after")
    def validate_dependent_fields(self) -> "MILPEngineArguments":
        """Validate input fields that depend on each other.

        PyDantic will call this method after the model is initialized to ensure
        that the required fields are set correctly based on the provided arguments.

        :return: The validated instance of MILPEngineArguments.
        :raises ValueError: If the required fields are not set correctly.
        """
        if not self.mps_file and not self.problem_name:
            raise ValueError("You must specify either 'mps_file', or 'problem_name'.")
        if self.strategy == "max-coverage" and self.max_cuts is None:
            raise ValueError(
                "The 'max_cuts' argument is required when using the 'max-coverage' "
                "strategy."
            )
        if self.use_profiler and not self.no_parallel:
            warnings.warn(
                "The 'use_profiler' option should not be used without the "
                "'no_parallel' option. The profiler may produce incorrect output if "
                "parallel processing "
                "is enabled."
            )
        if self.backend is None and self.use_quantum:
            raise ValueError(
                "The 'backend' argument is required when using the 'use_quantum' "
                "option."
            )
        if self.backend and not self.use_quantum:
            raise ValueError(
                "The 'use_quantum' option must be set to True when a 'backend' is "
                "specified."
            )
        return self


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Quantum-enabled BenderMILPSolver.

    :return: Parsed arguments as an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for the Quantum-enabled BenderMILPSolver."
    )

    # Input file or problem name
    parser.add_argument(
        "-f", "--mps-file", type=str, help="Path to the .mps file to solve."
    )
    parser.add_argument(
        "-n",
        "--problem-name",
        type=str,
        help="Name of a predefined problem in the test suite.",
    )
    parser.add_argument(
        "--y-prefix",
        type=str,
        default="y",
        help="Prefix for the variables in the MILP to be considered complicating "
        "variables (default: 'y').",
    )

    # Solver configuration
    parser.add_argument(
        "-sp",
        "--n-subproblems",
        type=int,
        default=1,
        help="Number of subproblems to solve per iteration (default: 1).",
    )
    parser.add_argument(
        "-i",
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations (default: infinity).",
    )
    parser.add_argument(
        "-b",
        "--convergence-bound",
        type=float,
        default=1e-5,
        help="Convergence bound (default: 1e-5).",
    )
    parser.add_argument(
        "-np",
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing of subproblems.",
    )
    parser.add_argument(
        "-p",
        "--use-profiler",
        action="store_true",
        help="Enable cProfiler to measure performance.",
    )
    parser.add_argument(
        "-q",
        "--use-quantum",
        action="store_true",
        help="Use quantum backend for the heuristic step.",
    )

    # Heuristic configuration
    parser.add_argument(
        "-c",
        "--criterion",
        choices=["coverage", "exclusion"],
        required=True,
        help="Criterion to use for cut selection. The possible values are 'coverage' "
        "or 'exclusion'.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["max-coverage", "min-set-cover", "take-all"],
        required=True,
        help="Strategy to use for cut selection. The possible values are "
        "'max-coverage', 'min-set-cover', or 'take-all'.",
    )
    parser.add_argument(
        "--max-cuts",
        type=int,
        help="The maximum number of cuts to select when using the maximum coverage "
        "strategy.",
    )
    parser.add_argument(
        "--backend",
        choices=["fermioniq", "qiskit"],
        help="Backend to use for quantum solver.",
    )
    parser.add_argument(
        "--backend-options",
        type=parse_json_dict,
        help="The options required for the quantum backend.",
        default={},
    )
    parser.add_argument(
        "--solver-options",
        type=parse_json_dict,
        help="The options required for the quantum solver.",
        default={},
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during solving.",
    )

    args = parser.parse_args()

    # Validate that either mps-file or problem-name is provided
    if not args.mps_file and not args.problem_name:
        parser.error("You must specify either --mps-file or --problem-name.")

    if args.strategy == "max-coverage" and args.max_cuts is None:
        parser.error(
            "The --max-cuts argument is required when using the "
            "'max-coverage' strategy."
        )

    if (not args.no_parallel) and args.use_profiler:
        warnings.warn(
            "The --use-profiler option should not be used without the --no-parallel "
            "option. The profiler may produce incorrect output if parallel processing "
            "is enabled."
        )

    return args


def construct_solver(args: SolverArguments) -> BenderMILPSolver:
    """Construct a BenderMILPSolver based on the given arguments.

    :param args: The arguments object that specifies a particular solver config.
    :return: The instantiated solver.
    :raises ValueError: An unknown criterion or strategy was specified.
    """
    criterion: Criterion
    match args.criterion:
        case "coverage":
            criterion = CoverageCriterion()
        case "exclusion":
            criterion = ExclusionCriterion()
        case _:
            raise ValueError(f"Unknown criterion: {args.criterion}")

    strategy: Strategy
    match args.strategy:
        case "max-coverage":
            strategy = MaximumCoverageStrategy(max_cuts=cast(int, args.max_cuts))
        case "min-set-cover":
            strategy = MinimumSetCoverStrategy()
        case "take-all":
            strategy = TakeAllStrategy()
        case _:
            raise ValueError(f"Unknown strategy: {args.strategy}")

    classical_solver = PuLPSolver()
    rich_solver = HiGHSSolver()

    cut_selection_solver: CutSelectionSolver
    if args.use_quantum:
        backend: QuantumOptimisationBackend
        match args.backend:
            case "fermioniq":
                backend = FermioniqBackend(fermioniq_options=args.backend_options)
            case "qiskit":
                backend = QiskitBackend(qiskit_options=args.backend_options)
            case _:
                raise ValueError(f"Unknown backend: {args.backend}")

        cut_selection_solver = QuantumSolver(
            backend=backend, solver_options=args.solver_options, solve_method="qaoa"
        )
    else:
        cut_selection_solver = classical_solver

    return BenderMILPSolver(
        criterion=criterion,
        strategy=strategy,
        mp_solver=classical_solver,
        dsp_solver=rich_solver,
        cut_selection_solver=cut_selection_solver,
        max_iterations=args.max_iterations,
        n_subproblems=args.n_subproblems,
        convergence_bound=args.convergence_bound,
        parallel_dsp=not args.no_parallel,
    )


def _get_local_milp(tag: str) -> pulp.LpProblem:
    """Fetch a MILP from the examples.milps package based on a tag.

    :param tag: The tag identifying the MILP.
    :return: The fetched MILP as a PuLP LpProblem.
    :raises ValueError: The specified tag was not found in examples.milps.
    """
    match tag:
        case "binary_choice_milp":
            from milp_engine.instances.milps import binary_choice_milp

            return binary_choice_milp()[0]
        case "savings_milp":
            from milp_engine.instances.milps import savings_milp

            return savings_milp()[0]
        case _:
            raise ValueError(f"Unknown problem tag: {tag}")


def _get_milp(args: MILPEngineArguments) -> pulp.LpProblem:
    """Retrieve a MILP based on the given arguments.

    Either a MILP is read from an MPS file or it is fetched from an internal library
    of named MILPs

    :param args: The arguments object.
    :return: The MILP that the arguments object points at.
    :raises ValueError: No MILP was specified in the arguments.
    """
    if args.mps_file:
        _, vrp_problem = pulp.LpProblem.fromMPS(args.mps_file)
        return vrp_problem
    elif args.problem_name:
        return _get_local_milp(args.problem_name)
    else:
        raise ValueError(
            "No valid problem specified. " "Provide either mps_file or problem_name."
        )


def _solve(args: MILPEngineArguments) -> pulp.LpProblem:
    """Run the BenderMILPSolver solve() function based on the given arguments.

    :param args: Arguments for the BenderMILPSolver.
    :return: The solved MILP.
    """
    solver = construct_solver(args)
    milp = _get_milp(args)

    milp_solved = solver.solve(milp, y_prefix=args.y_prefix, verbose=args.verbose)
    print(
        "\n"
        "============================================================\n"
        f"    Solver finished with status: {pulp.LpStatus[milp_solved.status]}\n"
        f"    Objective value: {milp_solved.objective.value()}\n"
        "============================================================\n"
    )
    return milp_solved


def run_solver(args: MILPEngineArguments) -> pulp.LpProblem:
    """Run the BenderMILPSolver with command-line arguments.

    Optionally, a profiler is also run to get insights in solver performance.

    :param args: Arguments for the BenderMILPSolver.
    :return: The solved MILP.
    """
    if args.use_profiler:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()

        milp_solved = _solve(args)

        profiler.disable()
        # Analyze the results
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats(30)

        return milp_solved
    else:
        return _solve(args)


def main() -> None:
    """Run the MILP engine from the command line."""
    args = _parse_arguments()
    milp_engine_args = MILPEngineArguments(**vars(args))
    run_solver(milp_engine_args)


if __name__ == "__main__":
    main()
