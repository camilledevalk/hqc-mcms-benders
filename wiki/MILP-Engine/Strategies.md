# Strategies

Strategies define the process for selecting cuts in `BenderMILPSolver` execution. They are essential for guiding the solver to choose the most effective cuts, improving convergence and solution quality.

## What are Strategies?

Strategies are decision-making processes that determine which cuts to include in the master problem during Benders decomposition or other MILP-based approaches. Each strategy provides a unique approach to cut selection, such as selecting all cuts, minimizing the number of cuts while ensuring coverage, or maximizing the coverage of infeasible solutions.

Formally, suppose we have:

- A set of **cuts** $\mathcal{C} = \{c_1, c_2, \ldots, c_m\}$
- A set of **elements to cover**, which can be constraints, infeasible solutions, or decision variables, denoted as $\mathcal{E} = \{e_1, e_2, \ldots, e_n\}$
- A **binary indicator matrix** $M \in \{0,1\}^{m \times n}$ where $M$ is constructed by a certain criteria (see [Criteria](criteria)).

The binary indicator matrix $M$ encodes the relationship between cuts and elements to be covered. Each row corresponds to a constraint, and each column represents an element—such as a specific solution (feasible or infeasible) or decision variable—depending on the chosen criteria. An entry $M_{ij} = 1$ indicates that constraint $c_i$ covers element $e_j$, while $M_{ij} = 0$ means there is no coverage. The precise definition of elements and coverage depends on the criteria selected for the problem. Each strategy then formulates an optimization problem to decide which cuts to select.

Strategies are implemented as Python classes in the `milp_engine/strategies/` directory. Each strategy inherits from a common base class `Strategy` and implements the `create_optimization_problem` method to define the optimization problem specific to the strategy.

---

## Implemented Strategies

### Strategy 0: Take All Strategy

**File:** `milp_engine/strategies/take_all_strategy.py`

The **TakeAllStrategy** is a naive approach that selects all cuts without any optimization. Duplicate cuts are removed.

#### Key Features

- **Simple Implementation**: Returns all cuts immediately.
- **No Optimization**: Does not perform any calculations or filtering.

#### Mathematical Formulation

No MILP is solved in this strategy. Formally, the solution is simply:

$$
x_i = 1 \quad \forall i \in I^m
$$

where $x_i$ is a binary decision variable indicating whether cut $c_i$ is selected.

#### Potential Advantages and Drawbacks

**Advantages:**
- Acts as a baseline for evaluating the effectiveness of other strategies.
- Useful in small or trivial instances where optimization overhead and Bender Decomposition convergence is not a concern.

**Drawbacks:**
- May lead to inefficiencies due to the inclusion of redundant cuts.
- Does not scale well for large problems.

---

### Strategy I: Minimum Set Cover Strategy

**File:** `milp_engine/strategies/minimum_set_cover_strategy.py`

The **MinimumSetCoverStrategy** uses the **Minimum Set Cover** approach to select a subset of cuts that cover all relevant elements (constraints, infeasible solutions, etc.) while minimizing the number of cuts.

#### Mathematical Formulation

We introduce binary variables $x_i$ for each cut $c_i$, where:

$$
x_i =
\begin{cases}
1 & \text{if cut } c_i \text{ is selected} \\
0 & \text{otherwise}
\end{cases}
$$

The MILP formulation is:

$$
\begin{aligned}
\min \quad & \sum_{i=1}^m x_i \\
\text{s.t.} \quad & \sum_{i=1}^m M_{ij} x_i \ge 1, \quad \forall j \in J^n \\
& x_i \in \lbrace 0,1 \rbrace, \quad \forall i \in I^m
\end{aligned}
$$


This ensures that every element $e_j$ is covered by at least one selected cut.

#### Potential Advantages and Drawbacks

**Advantages:**
- Reduces the number of cuts that are added per iteration, improving solver efficiency.
- Guarantees full coverage of constraints.

**Drawbacks:**
- Solving the set cover MILP can be computationally expensive for large number of constraints.
- When this problem is solved using a QAOA using a QUBO formulation, the required number of qubits increases rapidly with the number of non-zero entries in the binary indicator matrix.

---

### Strategy II: Maximum Coverage Strategy

**File:** `milp_engine/strategies/maximum_coverage_strategy.py`

The **MaximumCoverageStrategy** focuses on selecting a subset of cuts that **maximizes coverage** of infeasible solutions or decision variables, often under a cardinality limit.

#### Mathematical Formulation

We introduce binary variables $x_i$ for each cut $c_i$. Given an optional budget $k$ for the maximum number of cuts to select:

$$
\begin{aligned}
\max \quad & \sum_{j=1}^n y_j \\
\text{s.t.} \quad & y_j \le \sum_{i=1}^m M_{ij} x_i, \quad \forall j \in J^n \\
& \sum_{i=1}^m x_i \le k \quad (\text{if budgeted}) \\
& x_i, y_j \in \lbrace 0,1 \rbrace, \quad \forall i,j \in I^m, J^n
\end{aligned}
$$

Here:

- $x_i$ indicates whether cut $c_i$ is selected.
- $y_j$ indicates whether element $e_j$ is covered by at least one selected cut.

This strategy is especially useful when there are many potential cuts, but evaluating all of them is too expensive.

#### Potential Advantages and Drawbacks

**Advantages:**
- Prioritizes cuts that cover the largest number of elements.
- Improves the practical efficiency of the Benders decomposition.

**Drawbacks:**
- Solving the maximum coverage MILP can become expensive as $m$ and $n$ grow.
- May not guarantee full coverage if budgeted selection is used.
- When this problem is solved using a QAOA using a QUBO formulation, the required number of qubits increases rapidly with the number of non-zero entries in the binary indicator matrix.

---

## Extending Strategies

To add a new strategy:

1. Create a new Python file in `milp_engine/strategies/`.
2. Inherit from the `Strategy` base class.
3. Implement the `create_optimization_problem` method to define the optimization problem specific to the strategy.

# References
[1] N. G. Paterakis, "Hybrid quantum-classical multi-cut Benders approach with a power system application", Computers and Chemical Engineering, 2023.
