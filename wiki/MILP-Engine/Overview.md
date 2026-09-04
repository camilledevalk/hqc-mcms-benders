# MILP Engine Overview

The MILP Engine is designed to help solve complex optimization problems—like vehicle routing or scheduling—by breaking them down into smaller, easier-to-solve pieces. The core idea is to use Benders Decomposition (BD), a method that splits a big problem into a master problem (MP) and one or more subproblems (SP). This separation lets you tackle the hardest decisions first (usually the ones involving yes/no or integer choices), and then handle the details (often continuous variables or constraints) in the subproblems.

One of the key strengths of the MILP Engine is its ability to solve multiple subproblems in parallel. By doing so, the engine can generate a large pool of candidate cuts in a single iteration, significantly speeding up the optimization process. However, selecting the most impactful subset of these cuts becomes a combinatorial challenge, which the engine addresses using advanced strategies and, optionally, quantum computing.

## Why Use Benders Decomposition?

Many real-world optimization problems are too large or complex to solve directly. BD helps by:

- **Decomposing** the problem: MP handles the main structure, while SPs check if those assignments are feasible or optimal.
- **Parallelizing subproblem solving**: SPs can be solved simultaneously, leveraging modern computational resources to generate many cuts efficiently.
- **Iteratively improving**: After each MP solution, SPs (and their separate solutions) either confirm it's feasible or optimal and provide feedback (called "cuts") to refine the MP, guiding it toward optimal solutions.

---

## Mathematical Context of Benders Decomposition

Benders Decomposition divides the original problem into a Master Problem (MP) and one or more Subproblems (SP). The general mathematical formulation is as follows:

## Original Problem

Consider a Mixed Integer Linear Programming (MILP) problem:

$$
\begin{aligned}
\text{minimize} \quad & \mathbf{c}^\top \mathbf{x} + \mathbf{d}^\top \mathbf{y} \\
\text{subject to} \quad & A \mathbf{x} + B \mathbf{y} \geq \mathbf{b}, \\
& \mathbf{y} \in Y, \\
& \mathbf{x} \geq 0,
\end{aligned}
$$

where:
- $\mathbf{x}$: Subproblem decision variables
- $\mathbf{y}$: Master Problem decision variables
- A, B: Constraint matrices
- $\mathbf{b}$: Right-hand side vector

---

## Master Problem

The Master Problem iteratively incorporates Benders cuts generated from the Subproblem. Initially, it is formulated as:

$$
\begin{aligned}
\text{minimize} \quad & z \\
\text{subject to} \quad & \{\text{Benders cuts}\}, \\
& \mathbf{y} \in Y.
\end{aligned}
$$

where:
- $\mathbf{y}$: The master problem decision variables
- $z$: An auxiliary variable representing the objective value


## Subproblem

For a fixed $\mathbf{\bar{y}} \in Y$, the subproblem becomes:

$$
\begin{aligned}
\text{minimize} \quad & \mathbf{c}^\top \mathbf{x} \\
\text{subject to} \quad & A \mathbf{x} \geq \mathbf{b} - B \mathbf{\bar{y}}, \\
& \mathbf{x} \geq 0.
\end{aligned}
$$

The dual of this subproblem is:

$$
\begin{aligned}
\text{maximize} \quad & (\mathbf{b} - B \mathbf{\bar{y}})^\top \mathbf{u} \\
\text{subject to} \quad & A^\top \mathbf{u} \leq \mathbf{c}, \\
& \mathbf{u} \geq 0,
\end{aligned}
$$

where $\mathbf{u}$ are the dual variables associated with the constraints.

---

## Parallel Subproblem Solving and Cut Selection

We can solve multiple subproblems in parallel to generate a large pool of candidate Benders cuts per iteration of the Bender Decomposition.
From this pool, the best cuts can be selected based on a protocol, which is defined as a combination of a cut selection criterion and a strategy.

Using multiple subproblems in parallel allows us to generate a richer set of candidate cuts in each iteration. Selecting cuts based on a protocol ensures that only the most effective and relevant cuts are added to the Master Problem.

### What Are Criteria and Strategies?

- **[Criteria](Criteria.md)** are scoring functions or rules that evaluate the usefulness of each cut that help the MP avoid bad or infeasible solutions. For example, a criterion might check if some subset of cuts might make other subset of cuts obsolete.
- **[Strategies](Strategies.md)** decide *which* cuts to add. Adding every possible cut can slow down solving procedures, so strategies help pick the most useful cuts. Some strategies add all cuts, while others select only the most impactful, balancing speed and solution quality.

### Where Does Quantum Computing Fit In?

Cut selection constitutes a challenging combinatorial optimization problem. To address this, the MILP Engine can leverage quantum computing techniques—most notably quantum approximate optimization algorithms (QAOA) executed on quantum backends—to explore candidate cut sets, with the potential to achieve improvements in efficiency over purely classical approaches. The engine reformulates the cut selection task into a representation suitable for quantum solvers, such as a Quadratic Unconstrained Binary Optimization (QUBO) problem, which can then be executed on backends provided by frameworks such as [Qiskit](Solvers.md#qiskit) or [Fermioniq](Solvers.md#fermioniq).

## How It All Works Together

The workflow of the Benders decomposition engine can be summarized in an iterative process:

1. **Define the Problem**: Specify your optimization task in the engine.
2. **Decompose**: Split the problem into a Master Problem (MP) and multiple Subproblems (SPs).
3. **Iterative Procedure**: Repeat the following steps until convergence:
   1. **Solve the Master Problem**: Obtain a candidate solution $(\mathbf{\bar{y}}, z)$.
   2. **Solve Subproblems in Parallel**: For the fixed $\mathbf{\bar{y}}$, solve each SP (or its dual) to:
      - Generate new Benders cuts if the subproblem solutions allow us to.
      - Update upper and lower bounds.
   3. **Select Cuts**: From the pool of generated cuts, select the most effective ones using a protocol—potentially leveraging quantum computing for efficiency.
   4. **Refine the Master Problem**: Incorporate the selected cuts into the MP.
4. **Repeat**: Continue iterations until the gap between the upper and lower bounds is sufficiently small, indicating convergence.

## Key Components in `milp_eninge`

The `milp_engine` module implements the full Benders decomposition workflow and related optimization procedures.  It provides tools to define problems, solve subproblems (potentially in parallel), generate and select cuts using configurable criteria and strategies, and iteratively refine the Master Problem. It consists of the following main components:

1. **[Criteria](Criteria.md):**
   Criteria define the rules for evaluating and selecting cuts during the Benders decomposition process. Examples include:
   - **[Exclusion Criterion](Criteria.md#criterion-i-cut-selection-based-on-the-exclusion-of-infeasible-solutions):** Registers cuts which exclude the same infeasible solutions.
   - **[Coverage Criterion](Criteria.md#criterion-ii-cut-selection-based-on-mp-variable-coverage):** Registers cuts that cover most of the variables in the master problem.

2. **[Strategies](Strategies.md):**
   Strategies guide the selection of cuts during the optimization process. Examples include:
   - **[Take All Strategy](Strategies.md#strategy-0-take-all-strategy):** Naive approach that selects all cuts without any optimization.
   - **[Minimum Set Cover Strategy](Strategies.md#strategy-i-minimum-set-cover-strategy):** Approach to select a subset of cuts that cover all relevant elements (constraints, infeasible solutions, etc.) while minimizing the number of cuts.
   - **[Maximum Coverage Strategy](Strategies.md#strategy-ii-maximum-coverage-strategy):** Approach to select a subset of cuts that maximizes coverage of infeasible solutions or decision variables, often under a cardinality limit.

3. **[Solvers](Solvers.md):**
   The engine supports multiple solvers, each tailored to specific problem types and computational requirements:
   - **[BenderMILPSolver](Solvers.md#bendermilpsolver):** Implements Benders decomposition for large-scale MILPs.
   - **[HiGHSSolver](Solvers.md#highssolver):** Integrates the high-performance HiGHSsolver for solving LPs and extracting extreme rays for unbounded problems.
   - **[PuLPSolver](Solvers.md#pulpsolver):** A classical solver using the PuLP library.
   - **[QuantumSolver](Solvers.md#quantumsolver):** Leverages different quantum backends and algorithms for combinatorial optimization, including [Qiskit](Solvers.md#qiskit) and [Fermioniq](Solvers.md#fermioniq).
   - **[SCIPSolver](Solvers.md#scipsolver):** Uses the SCIPOPT solver for extracting multiple feasible solutions.

4. **[API](API.md):**
    The API layer exposes programmatic access to the MILP Engine. Users can define problems, configure solvers, and retrieve solutions either through Python interfaces or HTTP endpoints. This allows integration with external systems or pipelines.

## Summary

The MILP Engine combines decomposition, parallel subproblem solving, and smart cut selection to solve large-scale optimization problems efficiently.
By generating multiple candidate cuts per iteration and selecting the best ones using configurable protocols—combinations of criteria and strategies—the engine accelerates convergence and improves solution quality.

Its modular design lets you experiment with different criteria, strategies, and solvers, whether classical or quantum, all through a unified API.


## References

[1] N. G. Paterakis, "Hybrid quantum-classical multi-cut Benders approach with a power system application", *Computers and Chemical Engineering*, 2023.
