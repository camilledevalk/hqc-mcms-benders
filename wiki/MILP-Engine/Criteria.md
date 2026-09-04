# Criteria

Criteria setup the selection process in the Multiple Cuts Multiple Solutions Benders MILP Engine. Particularly, criteria create the binary indicator matrix used in cut selection. They are essential for guiding the solver to choose the most effective cuts, improving convergence and solution quality.

## What are Criteria?

Criteria are scoring functions or rules that evaluate the usefulness of each cut generated during the Benders decomposition or other MILP-based approaches. Each criterion provides a different perspective on the value of a cut: the [Exclusion Criterion](#criterion-i-cut-selection-based-on-the-exclusion-of-infeasible-solutions) shows which cuts overlap each other and the [Coverage Criterion](#criterion-ii-cut-selection-based-on-mp-variable-coverage) shows how many of the MP variables are covered by each of the cuts.

Criteria are implemented as Python classes in the `milp_engine/criteria/` directory. Each criterion inherits from a common base and implements a `create_binary_indicator_matrix` method that takes the cuts and creates the binary indicator matrix.

## Implemented Criteria

### Criterion I: Cut selection based on the exclusion of infeasible solutions

Implemented in `milp_engine/criteria/exclusion_criterion.py`.

The idea is to indicate which cuts exclude the same infeasible solutions. Later, the strategy will choose the set of cuts that excludes (collectively) the most infeasible solutions, given the boundaries of the strategy.

#### Mathematical formulation
$|G_F^k|$ is how many feasible ($F$) solutions are created in iteration $k$. Now, the $|G_F^k| \times |G_F^k|$ binary indicator matrix $\mathbf{E}$ has information about how the cuts cover the feasible solutions. This is a square matrix, since every feasible solution generates exactly 1 cut. So that means $\mathbf{E}_{ij} = 1$ if the (feasibility) cut excludes the infeasible solution associated with the $j$th solution of the MP. And $\mathbf{E}_{ij}=0$ otherwise.

#### Scaling

This criterion will always create a square $|G_F^k| \times |G_F^k|$ binary indicator matrix. $F$ is at most `n_subproblems`. This means this criterion does not scale with the problem size.

#### Potential advantages and drawbacks

##### Drawbacks

- The criterion does not consider optimality cuts.

##### Advantages
- The criterion only scales with a parameter (not with problem size).

### Criterion II: Cut selection based on MP variable coverage

Implemented in `milp_engine/criteria/coverage_criterion.py`.

The idea is to choose cuts that cover most of the variables in the master problem (MP). The strategy will choose a set that covers the most variables, given the boundaries of the strategy.

#### Mathematical formulation

First, a $|G_F^k| \times m$ binary indicator matrix $\mathbf{D}^F$ is constructed. $m$ is the number of variables in the MP. $\mathbf{D}^F_{ij}=1$ if the $j$th variable is covered by the cut $i$ and $0$ otherwise. The same can be done for the optimality cuts (matrix $\mathbf{D}^O$).

Note that in our implementation, we don't make this distinction between feasibility cuts and optimalitity cuts, so we essentially make one matrix.

#### Scaling

As you can see, this criterion will create two binary indicator matrices ($\mathbf{D}^F$ and $\mathbf{D}^O$). They have size $|G_F^k| \times m$ and $|G_O^k| \times m$. The number of rows in the $\mathbf{D}^F$ matrix is at most `n_subproblems`. The number of columns is exactly $m$, the number of MP variables. The number of rows in the $\mathbf{D}^O$ matrix is also at most `n_subproblems`, but probably smaller, since optimality cuts are harder to find.

#### Potential advantages and drawbacks

##### Drawbacks

- The size of the indicator matrix scales with the number of MP variables $m$.

##### Advantages

- Both optimality and feasibility cuts are considered.

## Extending Criteria

To add a new criterion:
1. Create a new Python file in `milp_engine/criteria/`.
2. Inherit from `BaseCriterion`.
3. Implement the `construct_binary_indicator_matrix` method.

# References:
[1] N. G. Paterakis, ‘Hybrid quantum-classical multi-cut Benders approach with a power system application’, Computers and Chemical Engineering, 2023.
