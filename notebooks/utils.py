"""Shared helpers for the MCMS Benders experiment notebooks (02 and 03).

`BendersSolverWithTimeout` wraps `BenderMILPSolver` to add a wall-clock time
budget and streaming iteration logs, without touching the solver's own
convergence logic. It works with any `cut_selection_solver` (classical PuLP
or the QAOA-based `QuantumSolver`), since it only relies on the generic
`Solver`/`CutSelectionSolver` interfaces plus an optional
`qaoa_all_infeasible` attribute for QAOA-specific fallback tracking.
"""
import json
import time
import pulp
from pathlib import Path
from typing import List, Tuple, Any, cast, Set

from milp_engine.solvers.bender_milp_solver import (
    BenderMILPSolver,
    construct_master_problem,
    create_and_solve_dsp,
    solve_milp_given_y_hat,
    update_master_problem,
)
from milp_engine.data.bender_state import BenderState
from milp_engine.data.cut import Cut
from milp_engine.utils.lp import get_objective_val


class TimeoutError(Exception):
    """Raised when a task exceeds its time budget."""
    pass


class BendersSolverWithTimeout:
    """Wrapper for BenderMILPSolver that enforces a time budget and streams iteration logs.

    Minimal version of run_benders_with_budget:
    - no SLURM/task metadata
    - no manifest/config handling
    - no signal handling
    - does run the actual Benders loop
    """

    def __init__(self, solver: BenderMILPSolver, time_budget_seconds: int):
        self.solver = solver
        self.time_budget = time_budget_seconds

    def solve(
        self,
        milp: pulp.LpProblem,
        y_prefix: str,
        output_dir: Path,
    ) -> Tuple[pulp.LpProblem, List[BenderState], str]:
        """Solve MILP with Benders decomposition and time budget.

        Args:
            milp: The MILP problem to solve.
            y_prefix: Prefix for complicating variables, e.g. "x_".
            output_dir: Directory to write iteration logs and result summary.

        Returns:
            Tuple of (solved_milp, iteration_states, exit_reason).

            exit_reason is one of:
            - "converged"
            - "infeasible"
            - "max_iterations"
            - "time_budget"
            - "qaoa_no_feasible"
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        iterations_file = output_dir / "iterations.jsonl"

        wall_start = time.perf_counter()
        states: List[BenderState] = []

        lower_bound = -float("inf")
        upper_bound = float("inf")

        mp = construct_master_problem(milp, y_prefix)
        added_cuts: Set[Tuple[float, ...]] = set()

        max_iterations = self.solver.max_iterations
        exit_reason = "max_iterations"

        consecutive_qaoa_infeasible = 0
        max_consecutive_qaoa_infeasible = 5

        with open(iterations_file, "w") as fh:
            iteration = 1

            while max_iterations is None or iteration <= max_iterations:
                elapsed = time.perf_counter() - wall_start
                remaining = self.time_budget - elapsed

                if remaining <= 0:
                    exit_reason = "time_budget"
                    break

                iter_start = time.perf_counter()

                # Step 1: solve master problem
                t0 = time.perf_counter()
                y_hats = self.solver.mp_solver.enumerate_feasible_solutions(
                    mp,
                    self.solver.n_subproblems,
                )
                mp_time = time.perf_counter() - t0

                state = BenderState(
                    iteration=iteration,
                    mp=mp,
                    y_hats=y_hats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                state.runtimes["mp_solving_time"] = mp_time

                if not y_hats:
                    milp.status = pulp.LpStatusInfeasible
                    exit_reason = "infeasible"
                    states.append(state)
                    self._write_iteration(fh, state, iter_start)
                    break

                y_star = y_hats[0]

                mp_obj_val = get_objective_val(mp, y_star)
                lower_bound = max(lower_bound, mp_obj_val)
                state.lower_bound = lower_bound

                # Step 2: solve dual subproblems
                t0 = time.perf_counter()
                if self.solver.parallel_dsp:
                    dsp_results, sp_timings = self._run_dsp_parallel(milp, y_hats)
                else:
                    dsp_results = [
                        create_and_solve_dsp(self.solver.dsp_solver, milp, y_hat)
                        for y_hat in y_hats
                    ]
                    sp_timings = None

                sp_time = time.perf_counter() - t0
                state.runtimes["sp_solving_time"] = sp_time

                if sp_timings is not None:
                    state.runtimes["individual_sp_solving_time"] = sp_timings

                if any(r.status == pulp.LpStatusInfeasible for r in dsp_results):
                    milp.status = pulp.LpStatusInfeasible
                    exit_reason = "infeasible"
                    states.append(state)
                    self._write_iteration(fh, state, iter_start)
                    break

                dsp_upper_bounds = [
                    r.upper_bound
                    for r in dsp_results
                    if r.status == pulp.LpStatusOptimal
                ]
                upper_bound = min([upper_bound] + cast(List[float], dsp_upper_bounds))
                state.upper_bound = upper_bound

                # Step 3: convergence check
                if self._has_converged(lower_bound, upper_bound):
                    solved_milp = solve_milp_given_y_hat(
                        self.solver.dsp_solver,
                        milp,
                        y_star,
                    )
                    exit_reason = "converged"
                    states.append(state)
                    self._write_iteration(fh, state, iter_start)
                    self._write_summary(
                        output_dir / "result.json",
                        states,
                        exit_reason,
                        time.perf_counter() - wall_start,
                    )
                    return solved_milp, states, exit_reason

                # Step 4: cut generation and selection
                assert None not in [r.cut for r in dsp_results]

                cuts: List[Cut] = cast(List[Cut], [r.cut for r in dsp_results])
                cuts = [
                    cut
                    for cut in cuts
                    if tuple(cut.coeff_vector.tolist()) not in added_cuts
                ]
                state.candidate_cuts = cuts

                binary_indicator_matrix, selection_cuts, passed_cuts = (
                    self.solver.criterion.construct_binary_indicator_matrix(state)
                )
                state.binary_indicator_matrix = binary_indicator_matrix

                t0 = time.perf_counter()
                selected_cuts = self.solver.strategy.select_cuts(
                    selection_cuts,
                    binary_indicator_matrix,
                    self.solver.cut_selection_solver,
                )
                cut_selection_time = time.perf_counter() - t0
                state.runtimes["cut_selection_time"] = cut_selection_time

                cut_selection_solver = self.solver.cut_selection_solver
                if (
                    hasattr(cut_selection_solver, "qaoa_all_infeasible")
                    and cut_selection_solver.qaoa_all_infeasible
                ):
                    state.qaoa_fallback_all_cuts = True
                    consecutive_qaoa_infeasible += 1
                else:
                    consecutive_qaoa_infeasible = 0

                mp_update_cuts = selected_cuts + passed_cuts
                state.mp_update_cuts = mp_update_cuts

                mp = update_master_problem(mp, mp_update_cuts)
                state.mp_updated = mp

                added_cuts.update(
                    tuple(cut.coeff_vector.tolist()) for cut in selected_cuts
                )

                iter_time = time.perf_counter() - iter_start
                state.runtimes["iteration_time"] = iter_time

                states.append(state)
                self._write_iteration(fh, state, iter_start)

                if consecutive_qaoa_infeasible >= max_consecutive_qaoa_infeasible:
                    exit_reason = "qaoa_no_feasible"
                    break

                elapsed = time.perf_counter() - wall_start
                if elapsed > self.time_budget:
                    exit_reason = "time_budget"
                    break

                gap = self._compute_gap(lower_bound, upper_bound)
                print(
                    f"  Iteration {iteration}: "
                    f"LB={lower_bound:.6e}, "
                    f"UB={upper_bound:.6e}, "
                    f"gap={gap:.6f}, "
                    f"elapsed={elapsed:.1f}s"
                )

                iteration += 1

        milp.status = pulp.LpStatusNotSolved
        self._write_summary(
            output_dir / "result.json",
            states,
            exit_reason,
            time.perf_counter() - wall_start,
        )
        return milp, states, exit_reason

    def _run_dsp_parallel(
        self,
        milp: pulp.LpProblem,
        y_hats: list,
    ) -> Tuple[list, List[float]]:
        """Run DSP solves in parallel using threads."""
        from concurrent.futures import ThreadPoolExecutor
        import os

        n = len(y_hats)
        max_workers = min(
            n,
            int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)),
        )

        results: List[Any] = [None] * n
        durations: List[float] = [0.0] * n

        def _solve(idx: int) -> None:
            t0 = time.perf_counter()
            results[idx] = create_and_solve_dsp(
                self.solver.dsp_solver,
                milp,
                y_hats[idx],
            )
            durations[idx] = time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_solve, i) for i in range(n)]
            for future in futures:
                future.result()

        return results, durations

    def _write_iteration(
        self,
        fh: Any,
        state: BenderState,
        iter_start: float,
    ) -> None:
        """Write iteration state to JSONL file."""
        data = {
            "iteration": state.iteration,
            "lower_bound": state.lower_bound,
            "upper_bound": state.upper_bound,
            "gap": self._compute_gap(state.lower_bound, state.upper_bound),
            "binary_indicator_matrix": (
                state.binary_indicator_matrix.tolist()
                if state.binary_indicator_matrix is not None
                else None
            ),
            "runtimes": {k: v for k, v in state.runtimes.items()},
            "number_of_candidate_cuts": (
                len(state.candidate_cuts) if state.candidate_cuts else 0
            ),
            "number_of_update_cuts": (
                len(state.mp_update_cuts) if state.mp_update_cuts else 0
            ),
            "mp_constraints": (
                len(state.mp_updated.constraints) if state.mp_updated else None
            ),
            "qaoa_fallback_all_cuts": state.qaoa_fallback_all_cuts,
            "wall_time_at_write": time.perf_counter() - iter_start,
        }
        fh.write(json.dumps(data) + "\n")
        fh.flush()

    def _write_summary(
        self,
        path: Path,
        states: List[BenderState],
        exit_reason: str,
        total_time: float,
    ) -> None:
        """Write final summary JSON."""
        final_state = states[-1] if states else None

        summary = {
            "exit_reason": exit_reason,
            "total_time_seconds": total_time,
            "n_iterations": len(states),
            "final_lower_bound": (
                final_state.lower_bound if final_state else None
            ),
            "final_upper_bound": (
                final_state.upper_bound if final_state else None
            ),
            "final_gap": (
                self._compute_gap(final_state.lower_bound, final_state.upper_bound)
                if final_state
                else None
            ),
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def _has_converged(self, lb: float, ub: float) -> bool:
        if ub == float("inf") or lb == -float("inf"):
            return False
        return (ub - lb) / max(abs(ub), 1e-6) < self.solver.convergence_bound

    @staticmethod
    def _compute_gap(lb: float, ub: float) -> float:
        """Compute optimality gap.

        This matches run_benders_with_budget more closely than the vibecoded version:
        denominator is max(abs(ub), 1e-6), not ub itself.
        """
        if ub == float("inf") or lb == -float("inf"):
            return float("inf")
        return (ub - lb) / max(abs(ub), 1e-6)
