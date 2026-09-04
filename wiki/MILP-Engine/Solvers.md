# Solvers

Solvers are responsible for solving Binary, Integer and Mixed Integer Linear Problems in various parts of the code base. Not all solvers support every kind of linear program (LP) due to the nature of the decision variables which can be binary, integer or continuous. Additionally, different solvers have different features which are required in particular applications in the code base (features such as extreme ray finding or feasible solution extraction).

## Supported Solvers

The engine currently supports the following solvers:

- [**BenderMILPSolver**](#bendermilpsolver)
- [**HiGHSSolver**](#highssolver)
- [**PuLPSolver**](#pulpsolver)
- [**QuantumSolver**](#quantumsolver)
- [**SCIPSolver**](#scipsolver)
- [**SimulatedAnnealingSolver**](#simulatedannealingsolver)

To add a new solver, implement the required interface and register your solver with the engine. This approach makes it straightforward to extend the system with additional solving capabilities.

---

## BenderMILPSolver

The `BenderMILPSolver` implements Benders Decomposition, a method for solving large or complex Mixed Integer Linear Problems (MILPs) by splitting them into two interconnected problems:

- **Master Problem (MP):** Handles the complicating integer variables.
- **Dual Subproblems (DSPs):** Handle the remaining continuous variables or constraints.

### Procedure

The solver iteratively performs the following steps:

1. **Solve the Master Problem**
   Generate a candidate integer solution `y_hat`, often by solving a relaxed version of the full MILP (e.g., a restricted master problem).
   → **Update Lower Bound:**
   Use the objective value of the master problem as the new lower bound if it improves upon the current one.

2. **Solve Dual Subproblems**
   For each candidate `y_hat`, verify feasibility and solve associated dual problems (e.g., pricing or feasibility subproblems).
   These subproblems identify constraint violations or potential cost improvements.

3. **Generate and Select Cuts**
   Use the subproblem results to generate cutting planes (i.e., feasibility cuts, optimality cuts).
   Select the most impactful cuts based on dual violation, heuristic strength, or coverage.

4. **Update Upper Bound**
   If a feasible primal solution is found (e.g., from the subproblem or a heuristic), compute its objective value.
   → **Update Upper Bound:**
   Replace the upper bound if this solution improves upon the current best.

5. **Refine the Master Problem**
   Add the selected cuts to the master problem to further restrict the solution space and eliminate infeasible or suboptimal regions.

6. **Check for Convergence**
   Evaluate the relative gap between the upper and lower bounds.
   - If the gap is below a specified tolerance (e.g., 1%), or
   - If the maximum number of iterations or a time limit is reached,
   → **Terminate the procedure.**

This process continues until an optimal solution is found or the problem is declared infeasible.

### When to Use Benders Decomposition

- When the MILP can be naturally decomposed into complicating integer variables and other constraints.
- For very large-scale MILPs where solving is computationally infeasible.
- When subproblems can be solved efficiently and independently (possibly in parallel).
- To exploit problem structures such as two-stage stochastic programs, facility location, network design, or scheduling problems with decomposable constraints.

By focusing computational effort on smaller subproblems and iteratively refining the master problem, Benders Decomposition often achieves significant performance improvements and better scalability compared to direct MILP solvers. For more information about Benders Decomposition works and possible applications, see the following references:

Benders, J. F. (2005). *Partitioning procedures for solving mixed-variables programming problems*.
**Computational Management Science**, 2(1).
[https://doi.org/10.1007/s10287-004-0020-y](https://doi.org/10.1007/s10287-004-0020-y)

Rahmaniani, R., Crainic, T. G., Gendreau, M., & Rei, W. (2017). *The Benders decomposition algorithm: A literature review*.
**European Journal of Operational Research**, 259(3), 801–817.
[https://doi.org/10.1016/j.ejor.2016.12.005](https://doi.org/10.1016/j.ejor.2016.12.005)


Paterakis, N. G. (2023). *Hybrid quantum-classical multi-cut Benders approach with a power system application*.
**Computers & Chemical Engineering**, 172, 108161.
[https://doi.org/10.1016/j.compchemeng.2023.108161](https://doi.org/10.1016/j.compchemeng.2023.108161)

---

## HiGHSSolver

The `HiGHSSolver` integrates the high-performance HiGHS solver with the PuLP framework to solve LPs. This solver has the capability of finding and extracting extreme rays (if the LP is unbounded) during each solving procedure.

### Procedure

- Accepts a PuLP problem and solves it using the HiGHS backend.
- Detects and returns an extreme primal ray for unbounded LPs, providing insight into the direction of unboundedness.
- Supports advanced diagnostics beyond finding optimal solutions.

### When to Use HiGHSSolver

- For fast, reliable LP solving within PuLP.
- When analyzing unbounded problems and extracting extreme rays is important.
- When advanced solver features and parameter control (e.g., time limits, optimality gap) are required.
- Suitable for diagnostics and algorithmic workflows needing detailed solution analysis.

Complete documentation can be found [here](https://highs.dev/).

---

## PuLPSolver

The `PuLPSolver` provides an interface for solving general LPs (including MILPs) using the PuLP library with the classical CBC solver backend.

### Procedure

- Formulate LPs using PuLP and pass them to the solver.
- The solver uses the CBC backend to find optimal or feasible solutions.
- Supports setting solver parameters such as time limits and optimality gap tolerances.
- Includes a wrapper method to accept diverse inputs (e.g., binary indicator matrices), though this does not currently affect the solving process.

### When to Use PuLPSolver

- When you need a straightforward, well-established MILP solver integrated with PuLP and CBC.
- When you want to control solver runtime or solution quality via time limits or gap tolerance.
- For problems requiring compatibility with PuLP’s modeling and solver interfaces.

Complete documentation can be found [here](https://coin-or.github.io/pulp/).

---

## QuantumSolver
The `QuantumSolver` enables solving combinatorial optimization problems by leveraging the Quantum Approximate Optimization Algorithm (QAOA) via different backends.
The `QuantumSolver` implementation supports only specific LPs since they need to be able to be converted into Quadratic Unconstrained Binary Optimization (QUBO) problems. For examples, continuous LPs must be discretized before they can be converted into QUBO problems.

### Procedure
- Initialize the solver with customizable QAOA parameters and Fermioniq backend credentials.
- Convert classical MILP problems into QUBO (Quadratic Unconstrained Binary Optimization) form suitable for quantum processing.
- Run QAOA on a relevant backend (emulator or quantum-hardware) to sample solution bitstrings.
- Extract the most probable solution from QAOA output and map it back to the original problem variables.
- Optionally enforce problem-specific constraints like maximum elements covered in set cover problems.
- Check the fidelity of the quantum execution to validate solution reliability, issuing warnings if fidelity is low.

### Tunability and Backend Choice
The solver provides tunability of QAOA parameters such as optimizer type, learning rate, number of QAOA layers (repetitions), noise levels, and maximum iterations.

#### Default parameters
In this project, we did some research on which parameters and which optimizer to use for our case. The research can be found in the [lab journals](https://dev.azure.com/AIE-Utrecht/Logistiqs/_git/logistiqs?version=GBlabjournals). From this, we concluded that COBYLA is best for us to use, because it A) works with Fermioniq's VQE implementation and B) handles expectation value calculations better than the other supported optimiser (SPSA), we think. In any case, it performs better.
We did research on which parameters for COBYLA yielded the best performance, and the results weren't super clear. We chose $\rho_{beg}$ to be $\pi/2$.

#### Fermioniq
Users can select the Fermioniq backend to run on different platforms (e.g., "cpu" or "gpu" emulator), enabling flexible experimentation across simulators.

Parameter tuning allows adapting the solver to specific problem instances and quantum device characteristics, balancing solution quality, runtime, and noise resilience.

Complete documentation can be found [here](https://docs.fermioniq.com/).
#### Qiskit
The `QuantumSolver` can also leverage the Qiskit Aer simulator as a backend for quantum circuit execution. This enables fast, noise-free simulation of quantum algorithms locally without requiring cloud access.

Users can configure parameters such as the number of shots (sampling runs), optimization level for circuit compilation, and noise models (if desired) to tailor simulation fidelity and performance.
Complete documentation can be found [here](https://quantum.cloud.ibm.com/docs/en/api/qiskit/0.39/qiskit_aer.AerSimulator).

### Why and When to Use QuantumSolver

- When exploring quantum-inspired methods for approximate solutions to combinatorial optimization problems.
- For problems naturally expressible as QUBOs (e.g., set cover or max coverage).
- When result validation via fidelity measurement is important due to noise in quantum devices or emulators.

---

## SCIPSolver

The `SCIPSolver` provides an interface for solving LPs using the `pyscipopt` library with the classical SCIPOPT solver. This solver has the capability of finding and extracting multiple feasible solutions during each solving procedure.

### Procedure

- Formulate LPs using `pyscipopt` and pass them to the solver.
- The solver uses the SCIPOPT solver to find optimal or feasible solutions.
- Supports setting solver parameters such as time limits and optimality gap tolerances.

### When to Use SCIPSolver

- When you want to extract multiple feasible solutions per solve procedure.
- When you need an open-source alternative to commercial solvers.

Complete documentation can be found [here](https://pyscipopt.readthedocs.io/en/latest/).

---

## SimulatedAnnealingSolver

The `SimulatedAnnealingSolver` is a state-of-the-art metaheuristic solver designed specifically for the Multi-Depot Vehicle Routing Problem (MDVRP). Simulated annealing is a probabilistic technique for approximating the global optimum of a given function, particularly effective for large combinatorial optimization problems where exact methods become computationally infeasible.

### Procedure

- Initializes with a feasible solution for the MDVRP, typically generated using a constructive heuristic.
- Iteratively explores the solution space by making small random changes (neighbor solutions), such as swapping customer assignments between routes or depots.
- Accepts new solutions based on a probability that depends on the change in objective value and a temperature parameter, allowing occasional acceptance of worse solutions to escape local optima.
- Gradually reduces the temperature according to a cooling schedule, focusing the search as the algorithm progresses.
- Continues until a stopping criterion is met (e.g., minimum temperature, maximum iterations, or convergence).

### When to Use SimulatedAnnealingSolver

- When solving large or complex MDVRP instances where exact solvers are too slow or memory-intensive.
- For applications requiring near-optimal solutions within practical time limits.
- When exploring solution diversity and robustness is important.

For more details on simulated annealing and its application to vehicle routing, see:

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing*. Science, 220(4598), 671–680. [https://doi.org/10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)


Redi, A. A. N. P., et al. (2020). *Simulated annealing algorithm for solving the capacitated vehicle routing problem: a case study of pharmaceutical distribution*. Jurnal Sistem dan Manajemen Industri, 4(1), 41–49.
[https://doi.org/10.30656/jsmi.v4i1.2092](https://doi.org/10.30656/jsmi.v4i1.2092)
