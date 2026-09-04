"""QAOA Solver Implementation.

This module contains QAOA (Quantum Approximate Optimization Algorithm) solver
interfaces and implementations for quantum optimization problems. The QuantumSolver
class provides quantum-based optimization capabilities leveraging either Fermioniq
emulators or Qiskit quantum backends.

The module supports:
- Fermioniq quantum emulation with MPS (Matrix Product State) simulation
- Qiskit quantum backend integration
- QAOA circuit generation and execution
- Solution extraction from quantum measurement results

Classes:
    QuantumSolver: Main quantum solver class implementing QAOA optimization
"""

import json
import logging
import os
import subprocess
import tempfile
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Tuple, cast

import fermioniq
import numpy as np
import pulp
import qiskit_optimization
import qiskit_optimization.algorithms  # .algorithms import MinimumEigenOptimizer
from fermioniq.config.defaults import standard_config
from fermioniq.custom_logging.printing import Printer, StringMessage
from qiskit.primitives import Sampler as Sampler_sim
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.minimum_eigensolvers import QAOA as QAOA_algo
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Sampler as Sampler_hardware
from qiskit_ibm_runtime import Session
from qiskit_optimization.minimum_eigensolvers import QAOA as QAOA_opt

from milp_engine.solvers.base_solver import CutSelectionSolver
from milp_engine.utils.lp import get_objective_val
from milp_engine.utils.quantum import (
    create_qaoa_circuit,
    interpret_samples,
    lp_to_quadratic_program,
    lp_to_qubo,
    qubo_to_z_interactions,
)
from milp_engine.utils.set_cover import (
    determine_max_coverage_class,
    determine_set_cover_class,
)

logger = logging.getLogger(__name__)


def get_relevant_bits_from_output(
    samples: Dict[str, int],
    decision_variables: list[pulp.LpVariable],
) -> tuple[dict[str, list[str]], list[pulp.LpVariable]]:
    """Extract the relevant bitstring from QAOA output samples.

    :param samples: The output samples of a QAOA execution
    :param decision_variables: A list of decision variable identifiers
        corresponding to the problem formulation.
    :return: A tuple, containing a dictionary with the original bitstring as
        key and the relevant bitstrings as value and a list of LpVariable
        that represents the actual solution bits.
    """
    remaining_samples = samples.copy()

    # Filter out the decision variables that represent actual solution bits.
    x_decision_variables = [
        var for var in decision_variables if var.name.startswith("x_")
    ]

    # Extract the relevant bits from the bitstring corresponding to the
    # decision variables for each bitstring in remaining_sample
    filtered_bitstrings = {
        bitstring: [
            bitstring[i]
            for i, var in enumerate(decision_variables)
            if var in x_decision_variables
        ]
        for bitstring in remaining_samples
    }

    return filtered_bitstrings, x_decision_variables


def get_solution_class_from_output(
    max_number_covered: int | None,
    binary_indicator_matrix: np.ndarray,
    filtered_bits: list[str],
) -> str | None:
    """Extract the solution class.

    :param max_number_covered: Maximum number of elements that may be
        selected or covered during the solution process.
    :param binary_indicator_matrix: NxN numpy array representing the binary
        indicator matrix.
    :param filtered_bits: The relevant bits from the bitstring.
    :return: A string indicating which solution class the solution belongs to.
    """
    if max_number_covered is None:
        sol_class = determine_set_cover_class(binary_indicator_matrix, filtered_bits)
    else:
        sol_class = determine_max_coverage_class(
            binary_indicator_matrix, filtered_bits, max_number_covered
        )

    return sol_class


def map_solution_class_to_decision_variables(
    x_decision_variables: list[pulp.LpVariable],
    filtered_bits: list[str],
    sol_class: str | None,
) -> dict[str, float]:
    """Map the solution class to the decision variables in the relevant bitstring.

    :param x_decision_variables: list of LpVariable that represents the actual
        solution bits.
    :param filtered_bits: The relevant bits from the bitstring.
    :param sol_class: A string indicating which solution class the solution belongs
        to.
    :return: A dictionary containing the original bitstring as key and as value the
        solution class mapped to the values of the decision variables.
    """
    # Initialize a dictionary to hold the interpreted solution.
    solution_bitstring = {}

    for i in range(len(x_decision_variables)):
        solution_bitstring[cast(str, x_decision_variables[i].name)] = (
            1.0 if filtered_bits[i] == sol_class else 0.0
        )

    return solution_bitstring


class QuantumOptimisationBackend(ABC):
    """Abstract base class for quantum backends."""

    def __init__(self, optimizer_settings: Dict[str, Any] = {}) -> None:
        """Initialize the quantum backend.

        :param optimizer_settings: Configuration settings for the optimizer.
        """
        self.decision_variables: List[pulp.LpVariable]

        # Also save the optimizer settings for later use
        self.optimizer_settings = optimizer_settings or {
            "rhobeg": np.pi / 2,
            "max_iter": 250,
            "tol": 1e-4,
        }

    @abstractmethod
    def run_qaoa(
        self, optimization_problem: pulp.LpProblem, reps: int, *args: Any, **kwargs: Any
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Run QAOA using a specific backend.

        :param optimization_problem: The pulp optimization problem to be solved.
        :param reps: The number of repetitions of the QAOA circuit
        :param args: Additional positional arguments for the backend.
        :param kwargs: Additional keyword arguments for the backend.
        :return: A tuple containing the results and samples of the QAOA execution.
        """
        pass


class FermioniqBackend(QuantumOptimisationBackend):
    """Fermioniq backend for quantum optimization."""

    supported_solve_methods = ["qaoa"]

    def __init__(
        self,
        optimizer_settings: Dict[str, Any] = {},
        fermioniq_options: Dict[str, Any] = {},
    ) -> None:
        """
        Initialize the Fermioniq backend with the given configuration.

        :param optimizer_settings: Configuration settings for the optimizer.
        :param fermioniq_options: Configuration settings for the backend.

        **Fermioniq Backend Options**
        - **remote_config** (str, default: "cpu"):
            Name of the remote Fermioniq execution engine to use.
            Available backends: `"cpu-2"`, `"cpu-4"`, `"cpu-8"`, `"gpu-h100"`.
            See also client.remote_configs()
        - **n_shots** (int, default: 1000):
            Number of circuit executions (shots) per experiment.
        - **optimization_level** (int, default: 3):
            Compilation and transpilation optimization level (0–3).
            See https://quantum.cloud.ibm.com/docs/en/guides/set-optimization

        In a dict in fermioniq_options["optimizer"], you can specify the
            following options:
        - **enabled** (bool, default: False):
            Set to True to use the optimizer.
            The default is False, in which case a normal emulation is performed
            with parameters set to zero.
        - **observable**:
            The observable for which the expectation value is minimized using the
            VQE algorithm.
            Note: this can only be one observable.
        - **evaluation_mode** (str, default: "contract"):
            Specifies the evaluation method of the observable expectation value.
            Currently, the only option is "contract", which uses exact MPS contraction
            for evaluation.
        - **optimizer_name** (str, default: "cobyla"):
            Specifies the optimizer used.
            Currently, the SPSA and COBYLA optimizers are available.
        - **optimizer_settings** (dict):
            A dictionary containing the settings for the classical optimizer.
            The structure of this dictionary depends on the optimizer used.
            This is exactly the same as the optimizer_settings parameter of the
            QuantumSolver class if it's provided.
        - **initial_param_values** (dict, optional):
            Specifies the initial values for the parameters.
            If not provided, the optimizer randomly initializes the parameters.
        - **initial_param_noise** (float, default: 0.1):
            This parameter controls the amount of random noise to the
            initial parameter values.

        :raises FileNotFoundError: If the `tokens.json` file is not found in the
            current working directory. This file is required to authenticate with
            the Fermioniq API.
        """
        super().__init__(optimizer_settings)

        self.fermioniq_backend = fermioniq_options.get("remote_config", "cpu")
        self.fermioniq_config = {
            k: v for k, v in fermioniq_options.items() if k != "remote_config"
        }
        self.n_shots = fermioniq_options.get("n_shots", 1000)
        self.optimization_level = fermioniq_options.get("optimization_level", 3)

        token_path = os.path.join(os.getcwd(), "tokens.json")

        # Instantiate fermioniq client
        try:
            with open(token_path, "r") as f:
                token_json = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"""
                tokens.json not found in the current directory: {os.getcwd()}"""
            ) from exc

        self.client = fermioniq.Client(
            access_token_id=token_json["id"], access_token_secret=token_json["secret"]
        )

        self.fermioniq_optimizer_config = (
            fermioniq_options["optimizer"]
            if "optimizer" in fermioniq_options
            else {
                "enabled": True,
                "evaluation_mode": "contract",
                "optimizer_name": "cobyla",
                "initial_param_values": {},
                "initial_param_noise": 0.1,
                "optimizer_settings": self.optimizer_settings,
                # note that self.optimizer_settings is never empty, as it is
                # set in the parent class constructor
            }
        )

    def run_qaoa(
        self, optimization_problem: pulp.LpProblem, reps: int, fidelity_threshold: float
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Run QAOA using Fermioniq.

        :param optimization_problem: The pulp optimization problem to be solved.
        :param reps: The number of repitions of the QAOA circuit
        :param fidelity_threshold: The minimum fidelity threshold for the QAOA
            solution to be considered valid. If the achieved fidelity is below this
            threshold, the solution will not be accepted.
        :return: A tuple containing the results and samples of the QAOA execution.
        """
        logger = Printer().pprint
        # Create the QAOA circuit and observable
        qubo = lp_to_qubo(optimization_problem)
        circuit, qiskit_obs = create_qaoa_circuit(
            qubo,
            reps,
            optimization_level=self.optimization_level,
            basis_gates="qiskit",
        )

        # Create standard config
        config = standard_config(circuit, effort=1.0)

        # Deep update: recursively update nested dicts (e.g., config["dmrg"]["D"])
        def deep_update(d: dict, u: dict) -> None:
            """Recursively update dictionary d with values from u.

            :param d: The dictionary to be updated.
            :param u: The dictionary with new values.
            """
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v

        deep_update(config, self.fermioniq_config)

        # Log circuit metadata
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        num_gates = circuit.size()
        logger(
            str(
                StringMessage(
                    f"Quantum circuit metadata: {num_qubits} qubits | "
                    f"depth {depth} | {num_gates} gates | bond dimension: "
                    f"{config['dmrg']['D']}"
                )
            )
        )

        self.fermioniq_optimizer_config["observable"] = {"energy": qiskit_obs}

        config["optimizer"] = self.fermioniq_optimizer_config

        # Optionally enable sampling
        sampler_config = {"enabled": True, "n_shots": self.n_shots}

        config["output"][
            "sampling"
        ] = sampler_config  # Other output options: expectation value, amplitudes, mps

        # Run QAOA
        emulator_job = fermioniq.EmulatorJob(
            circuit=circuit,
            config=[config],
            remote_config=self.fermioniq_backend,
            # Available backends: cpu-2, cpu-4, cpu-8, gpu-h100.
            # See also client.remote_configs()
        )

        self.client.schedule_async(emulator_job)

        print(f"Emulator job ID: {emulator_job.job_id}")

        job_finished = False

        while not job_finished:
            try:
                while self.client.get_status(emulator_job.job_id) != "finished":
                    time.sleep(1)
                job_finished = True
            except ConnectionError:
                print("Connection error while checking job status. Retrying...")
                time.sleep(5)

        output = self.client.get_results(emulator_job.job_id)

        job_outputs = output.job_outputs

        achieved_fidelity = min(
            job_outputs[0]["metadata"]["extrapolated_2qubit_gate_fidelity"],
            job_outputs[0]["metadata"]["fidelity_product"],
        )

        # Check if the achieved fidelity is below the threshold and raise warning
        if achieved_fidelity < fidelity_threshold:
            warnings.warn(
                f"Achieved fidelity {achieved_fidelity} is below the threshold "
                f"{fidelity_threshold}. The solution may not be reliable.",
                UserWarning,
            )

        self.decision_variables = qubo.variables

        return (output, output.samples(0, 0))


class QiskitBackend(QuantumOptimisationBackend):
    """Qiskit backend for quantum optimization."""

    supported_solve_methods = ["qaoa"]

    def __init__(
        self,
        optimizer_settings: Dict[str, Any] = {},
        qiskit_options: Dict[str, Any] = {},
    ) -> None:
        """
        Initialize the simulator backend with the given configuration.

        :param optimizer_settings: A dictionary containing the settings
            for the classical optimizer.
        :param qiskit_options: Configuration settings for the Qiskit backend.
            The structure of this dictionary depends on the optimizer used.

        **Qiskit Options**
            - **n_shots** (int, default=1000):
            Number of shots (circuit executions) per experiment.
            Higher values reduce statistical noise but increase runtime.

            - **use_hardware** (bool, default=False):
            If True, submit jobs to IBM Quantum hardware instead of running locally.

            - **token_path** (str or None, default=None):
            Filesystem path to a stored IBM Quantum API token, required if
            ``use_hardware`` is True.

            - **ibm_hardware_name** (str, default="ibm_strasbourg"):
            Name of the IBM Quantum device to use when running on hardware.
        """
        super().__init__(optimizer_settings)
        # Stupid difference between Fermioniq and Qiskit way of writing "max iter"
        if "max_iter" in self.optimizer_settings:
            self.optimizer_settings["maxiter"] = self.optimizer_settings.pop("max_iter")
        self.n_shots = qiskit_options.get("n_shots", 1000)
        self.use_hardware = qiskit_options.get("use_hardware", False)
        self.token_path = qiskit_options.get("token_path", None)
        self.ibm_hardware_name = qiskit_options.get(
            "ibm_hardware_name", "ibm_strasbourg"
        )

    def run_qaoa(
        self, optimization_problem: pulp.LpProblem, reps: int, *args: Any, **kwargs: Any
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Run QAOA using Qiskit Aer.

        :param optimization_problem: The pulp optimization problem to be solved.
        :param reps: The number of repetitions of the QAOA circuit
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: A tuple containing the results and samples of the QAOA execution.
        """
        # Initialize COBYLA optimizer
        # Note that self.optimizer_settings is never empty, as it is set in the parent
        # class constructor
        optimizer = COBYLA(**self.optimizer_settings)

        # Create QAOA circuit and observable
        qp = lp_to_quadratic_program(optimization_problem)
        qubo = lp_to_qubo(optimization_problem)

        if self.use_hardware:
            # Read token from file
            with open(self.token_path, "r") as f:
                token = f.read().strip()

            # Save the account (only needs to be done once)
            QiskitRuntimeService.save_account(
                token=token, overwrite=True, channel="ibm_cloud", instance="logistiqs"
            )
            service = QiskitRuntimeService()
            ibm_backend = service.backend(self.ibm_hardware_name)
            pass_manager = generate_preset_pass_manager(
                optimization_level=3, backend=ibm_backend
            )
            with Session(backend=ibm_backend) as session:
                backend = Sampler_hardware(
                    mode=session, options={"default_shots": self.n_shots}
                )

                mes = QAOA_opt(
                    sampler=backend,
                    optimizer=optimizer,
                    reps=reps,
                    pass_manager=pass_manager,
                )
                meo = qiskit_optimization.algorithms.MinimumEigenOptimizer(
                    min_eigen_solver=mes
                )
                results = meo.solve(qubo)

        else:
            backend = Sampler_sim(options={"shots": self.n_shots})
            mes = QAOA_algo(sampler=backend, optimizer=optimizer, reps=reps)
            meo = qiskit_optimization.algorithms.MinimumEigenOptimizer(
                min_eigen_solver=mes
            )
            results = meo.solve(qp)
        prob_dict = {
            "".join(str(int(bit)) for bit in sample.x): sample.probability
            for sample in results.samples
        }
        # Prepare bitstrings and probabilities
        bitstrings = list(prob_dict.keys())
        probabilities = np.array(list(prob_dict.values()))
        probabilities /= probabilities.sum()  # normalize to sum to 1, just in case

        # Sample n_shots according to the probability distribution
        samples = np.random.choice(bitstrings, size=self.n_shots, p=probabilities)

        # Count frequencies of each bitstring
        frequency_counter = Counter(samples)

        # Convert to a dict if desired
        frequency_dict = dict(frequency_counter)

        self.decision_variables = qp.variables

        return (results, frequency_dict)


class JuliaMPSBackend(QuantumOptimisationBackend):
    """MPS-QAOA backend using JuliQAOA via subprocess."""

    supported_solve_methods = ["qaoa"]

    def __init__(
        self,
        optimizer_settings: Dict[str, Any] = {},
        julia_options: Dict[str, Any] = {},
    ) -> None:
        """Initialize the Julia MPS backend.

        :param optimizer_settings: Settings for the classical optimizer.
        :param julia_options: Configuration for the Julia MPS backend.

        **Julia Backend Options**

        - **julia_executable** (str, default: "julia"):
            Path to the Julia executable. Override for HPC environments.
        - **julia_project_path** (str, default: "julia/MPS_JuliQAOA"):
            Path to the Julia project containing the MPS-JuliQAOA environment.
        - **n_shots** (int, default: 1000):
            Number of bitstring samples to draw from the final MPS state.
        - **cutoff** (float, default: 1e-6):
            MPS truncation cutoff for the variational circuit.
        - **maxdim** (int, default: 64):
            Maximum MPS bond dimension.
        - **optimizer** (str, default: "cobyla"):
            Classical optimizer: ``"cobyla"``, ``"nelder_mead"``,
            or ``"particle_swarm"``.
        - **maxiter** (int, default: 1000):
            Maximum number of optimizer iterations.
        - **rhobeg** (float | list[float] | None, optional):
            COBYLA initial step size (NLopt `initial_step`). Only used when
            ``optimizer="cobyla"``.
        """
        super().__init__(optimizer_settings)
        self.julia_executable = julia_options.get("julia_executable", "julia")
        self.julia_project_path = julia_options.get(
            "julia_project_path", "julia/MPS_JuliQAOA"
        )
        self.n_shots = julia_options.get("n_shots", 1000)
        self.cutoff = julia_options.get("cutoff", 1e-6)
        self.maxdim = julia_options.get("maxdim", 64)
        self.optimizer = julia_options.get("optimizer", "cobyla")
        self.maxiter = julia_options.get("maxiter", 1000)
        self.rhobeg = julia_options.get("rhobeg")
        self.qaoa_log_dir: str | None = None
        self._qaoa_call_count = 0

    def run_qaoa(
        self,
        optimization_problem: pulp.LpProblem,
        reps: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, int]]:
        """Run QAOA using Julia MPS-JuliQAOA via subprocess.

        :param optimization_problem: The PuLP optimization problem.
        :param reps: Number of QAOA circuit repetitions (p).
        :param args: Additional positional arguments (unused).
        :param kwargs: Additional keyword arguments (unused).
        :return: A tuple ``(metadata_dict, samples_dict)`` where
            *samples_dict* maps bitstrings to integer frequencies.
        :raises RuntimeError: If the Julia subprocess returns a non-zero exit code.
        """
        qubo = lp_to_qubo(optimization_problem)
        z_interactions, offset = qubo_to_z_interactions(qubo)
        nqubits = qubo.get_num_vars()

        input_data = {
            "z_interactions": z_interactions,
            "nqubits": nqubits,
            "p": reps,
            "optimize": True,
            "optimizer": self.optimizer,
            "maxiter": self.maxiter,
            "cutoff": self.cutoff,
            "maxdim": self.maxdim,
            "n_shots": self.n_shots,
        }

        if self.rhobeg is not None:
            input_data["rhobeg"] = self.rhobeg

        self._qaoa_call_count += 1
        logger.info(
            "QAOA call #%d: nqubits=%d, z_interactions=%d, maxdim=%d, p=%d",
            self._qaoa_call_count,
            nqubits,
            len(z_interactions),
            self.maxdim,
            reps,
        )

        if self.qaoa_log_dir is not None:
            os.makedirs(self.qaoa_log_dir, exist_ok=True)
            log_path = os.path.join(
                self.qaoa_log_dir,
                f"qaoa_input_{self._qaoa_call_count}.json",
            )
            with open(log_path, "w") as f:
                json.dump(input_data, f, indent=2)
            logger.info("Saved QAOA input to %s", log_path)

        script_path = os.path.join(self.julia_project_path, "run_qaoa.jl")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output.json")

            with open(input_path, "w") as f:
                json.dump(input_data, f)

            julia_threads = os.environ.get("JULIA_NUM_THREADS", "auto")
            cmd = [
                self.julia_executable,
                f"--project={self.julia_project_path}",
                f"--threads={julia_threads}",
                script_path,
                input_path,
                output_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.stderr:
                logger.info("Julia stderr:\n%s", result.stderr.rstrip())

            if result.returncode != 0:
                raise RuntimeError(
                    f"Julia MPS-QAOA failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )

            with open(output_path, "r") as f:
                output = json.load(f)

        if self.qaoa_log_dir is not None:
            log_path = os.path.join(
                self.qaoa_log_dir,
                f"qaoa_output_{self._qaoa_call_count}.json",
            )
            with open(log_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info("Saved QAOA output to %s", log_path)

        samples = {k: int(v) for k, v in output.get("samples", {}).items()}

        # Correct for Ising offset: the expectation value returned by Julia
        # is the Ising Hamiltonian value; the true QUBO objective is
        #   E_qubo = E_ising + offset
        metadata = {
            "expectation_value": output["expectation_value"],
            "offset": offset,
            "qubo_objective": output["expectation_value"] + offset,
            "angles": output["angles"],
            "p": output["p"],
            "nqubits": output["nqubits"],
            "converged": output.get("converged"),
            "iterations": output.get("iterations"),
            "optimization_trace": output.get("optimization_trace"),
            "status": output["status"],
        }

        self.decision_variables = qubo.variables

        return (metadata, samples)


class QuantumSolver(CutSelectionSolver):
    """A solver class for a QAOA."""

    def __init__(
        self,
        backend: QuantumOptimisationBackend,
        solver_options: Dict[str, Any] = {},
        solve_method: str = "qaoa",
    ) -> None:
        """
        Initialize the quantum solver with the specified backend and options.

        :param backend: The quantum backend that is used in the quantum solver.
        :param solver_options: Configuration options for the Fermioniq backend.
        :param solve_method: The solution method to be used by the backend.
            Defaults to "qaoa". Must be supported by the selected backend.

        :raises ValueError: If an unsupported `solve_method` is provided for the
            chosen backend.
        """
        self.solve_method = solve_method
        self.backend = backend
        self.solve_options = solver_options
        if self.solve_method not in FermioniqBackend.supported_solve_methods:
            raise ValueError(f"Unsupported solve method for Fermioniq: {solve_method}")
        if self.solve_method not in QiskitBackend.supported_solve_methods:
            raise ValueError(f"Unsupported solve method for Qiskit: {solve_method}")
        if self.solve_method not in JuliaMPSBackend.supported_solve_methods:
            raise ValueError(f"Unsupported solve method for JuliaMPS: {solve_method}")

    def _get_solution_from_output(
        self,
        samples: Dict[str, int],
        optimization_problem: pulp.LpProblem,
        binary_indicator_matrix: np.ndarray,
        max_number_covered: int | None,
        decision_variables: list[pulp.LpVariable],
    ) -> pulp.LpProblem:
        """Extract a binary solution mapping from QAOA output samples.

        :param samples: The output samples of a QAOA execution
        :param optimization_problem: The pulp optimization problem to be solved.
        :param binary_indicator_matrix: NxN numpy array representing the binary
            indicator matrix.
        :param max_number_covered: Maximum number of elements that may be
            selected or covered during the solution process.
        :param decision_variables: A list of decision variable identifiers
            corresponding to the problem formulation.
        :return: An optimization problem that is solved for its decision variables,
            extracted from the most probable feasible bitstring sampled from the QAOA
            output. If no feasible sample is found, all cuts are selected by default.
        """
        self.qaoa_all_infeasible = False
        remaining_samples = samples.copy()
        # Initialize the solution class as None
        sol_class = None

        # Iterate while there are samples left to try and we haven’t found a
        # valid solution class corresponding with a feasible solution
        while remaining_samples and sol_class is None:

            # Extract the relevant bits from the bitstring corresponding to the
            # decision variables for each bitstring in remaining_sample
            filtered_bitstrings, x_decision_variables = get_relevant_bits_from_output(
                remaining_samples, decision_variables
            )

            # Initialize a dictionary to hold the objective value corresponding to
            # the bitstring
            obj_val_by_bitstring = {}

            # Iterate over the bitstrings to find the objective value
            for bitstring, filtered_bits in filtered_bitstrings.items():

                # Determine the solution class for this bitstring
                sol_class = get_solution_class_from_output(
                    max_number_covered, binary_indicator_matrix, filtered_bits
                )

                # Map the solution class back to the decision variables
                solution_bitstring = map_solution_class_to_decision_variables(
                    x_decision_variables, filtered_bits, sol_class
                )

                # Calculate the objective value for the solution
                obj_val = get_objective_val(optimization_problem, solution_bitstring)

                # Store the objective value using the original bitstring as the key
                obj_val_by_bitstring[bitstring] = obj_val

            # Select the bitstring resulting in the optimal objective value from the
            # QAOA output
            if optimization_problem.sense == pulp.LpMinimize:
                best_qaoa_bitstring = min(
                    obj_val_by_bitstring, key=lambda k: obj_val_by_bitstring[k]
                )
            else:
                best_qaoa_bitstring = max(
                    obj_val_by_bitstring, key=lambda k: obj_val_by_bitstring[k]
                )

            # Extract the relevant bits from the best bitstring corresponding to the
            # decision variables
            x_bitstring = filtered_bitstrings[best_qaoa_bitstring]

            # Initialize a dictionary to hold the interpreted solution.
            solution = {}

            sol_class = get_solution_class_from_output(
                max_number_covered, binary_indicator_matrix, x_bitstring
            )

            # Remove the tried bitstring so we don't repeat it in the next iteration
            del remaining_samples[best_qaoa_bitstring]

        # If no feasible solution was found after checking all samples, warn and
        # default to selecting all cuts.
        if sol_class is None:
            self.qaoa_all_infeasible = True
            msg = (
                "No feasible solution found in the provided QAOA samples. "
                "Defaulting to selecting all cuts."
            )
            logging.getLogger(__name__).warning(msg)
            warnings.warn(msg, UserWarning)
            for var in optimization_problem.variables():
                var.varValue = 1
        else:
            # Map the solution class bit (0 or 1) back to the individual decision
            # variables.
            for i in range(len(x_decision_variables)):
                solution[x_decision_variables[i]] = (
                    1 if x_bitstring[i] == sol_class else 0
                )

            # Now update the variables in the actual optimization problem with the
            # values from the solution.
            for var in optimization_problem.variables():
                matching_var = next((k for k in solution if k.name == var.name), None)
                if matching_var is not None:
                    var.varValue = solution[matching_var]

        # Mark the problem as having an optimal solution.
        if optimization_problem is not None:
            optimization_problem.status = pulp.constants.LpStatusOptimal

        return optimization_problem

    def solve_with_binary_indicator_matrix(
        self,
        optimization_problem: pulp.LpProblem,
        binary_indicator_matrix: np.ndarray,
        max_number_covered: int | None = None,
        fidelity_threshold: float = 0.99,
    ) -> pulp.LpProblem:
        """Solve a binary LP using QAOA.

        :param optimization_problem: The pulp optimization problem to be solved.
        :param binary_indicator_matrix: NxN numpy array representing the binary
            indicator matrix.
        :param max_number_covered: Maximum number of elements that may be
            selected or covered during the solution process. Defaults to 0.
        :param fidelity_threshold: The minimum fidelity threshold for the QAOA
            solution to be considered valid. If the achieved fidelity is below this
            threshold, the solution will not be accepted. Defaults to 0.99.
        :return: An optimization problem that is solved for its decision variables,
            extracted from the most probable bitstring sampled from the QAOA output.
        :raises ValueError: If the selected solve method is unsupported.
        """
        match self.solve_method:
            case "qaoa":
                self.output, samples = self.backend.run_qaoa(
                    optimization_problem,
                    self.solve_options.get("reps", 1),
                    fidelity_threshold,
                )
                interpreted_samples, _ = interpret_samples(
                    samples, optimization_problem
                )
                samples_frequencies = {
                    bitstring: int(data["frequency"])
                    for bitstring, data in interpreted_samples.items()
                }
                solved_optimization_problem = self._get_solution_from_output(
                    samples_frequencies,
                    optimization_problem,
                    binary_indicator_matrix,
                    max_number_covered,
                    self.backend.decision_variables,
                )
                return solved_optimization_problem
            case _:
                raise ValueError(f"Unknown solve method: {self.solve_method}")
