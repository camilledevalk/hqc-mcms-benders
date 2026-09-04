# Vehicle Routing Problem Formulations

Vehicle routing problems are optimization problems in which the goal is to determine optimal routes for vehicles as they visit and supply customers in a network. This document describes ways of formulating such problems as linear problems, such that it can be solved by a linear solver such as Gurobi, CPLEX, or our own linear solver based on the Bender's Decomposition. The included formulations are:

- **[Multi-Depot Vehicle Routing Problem (MDVRP)](Multi-Depot-VRP.md)**: a variant of VRP in which vehicles have a particular home depot that needs to be part of their route and there exist multiple of such depots with different vehicles assigned to them.
- **[Stochastic Multi-Depot Vehicle Routing Problem (S-MDVRP)](Stochastic-Multi-Depot-VRP.md)**: an extension of MDVRP where, for each in the network, there exists an independent probability that vehicles fail to traverse it. Vehicle routes are constrained to have a survival probability that is higher than a particular threshold.
- **[Multi-Modal Multi-Depot Vehicle Routing Problem (MMMDVRP)](Multi-Modal-Multi-Depot-VRP.md)**: an extension of MDVRP where there is more than one type of vehicle. Vehicle types (referred to as a *modes* or *modalities*) can differ from one another in how much load they can carry or in how (time) costly it is for them to traverse particular edges in the network.
- **[Stochastic Multi-Modal Multi-Depot Vehicle Routing Problem (S-MMMDVRP)](Stochastic-Multi-Modal-Multi-Depot-VRP.md)**: an extension all of the above. This formulation combines stochasticity with multi-modality. In addition to differences in load and traversion cost, different vehicle modes can also have different survival probabilities when traversing particular edges.

## Defining The Simpler Variants In Terms Of S-MMDVRP
Given that MDVRP, S-MDVRP, and MMMDVRP are all subsets of S-MMMDVRP, we only implemented the latter variant and offer the other three as small wrappers around this variant. For the same reason, we recommend reading up on [S-MMMDVRP](Stochastic-Multi-Modal-Multi-Depot-VRP.md) first and then using the overview below to see how this leads to the other variants.
- **MMMDVRP** can be obtained by setting the survival probability $\text{prob}_{i,j,m} = 1$ for every edge $(i, j)$ and for every mode $m$ and by setting the survival threshold $\text{threshold}_m = 0$ for all modes $m$.
- **S-MDVRP** can be obtained by only having only one mode. That is, setting $\text{modes} = \{m_0\}$.
- **MDVRP** can be obtained by combining the above operations. That is, by setting the survival probability $\text{prob}_{i,j,m_0} = 1$ for every edge $(i, j)$, by setting the survival threshold $\text{threshold}_{m_0} = 0$, and by setting $\text{modes} = \{m_0\}$.
