## Introduction to Stochastic Multi-Modal Multi-Depot Vehicle Routing Problems

Stochastic Multi-Modal Multi-Depot Vehicle Routing Problems (S-MMMDVRPs) extend classical vehicle routing by integrating multiple transportation modes—such as trucks and drones—multiple depots, and explicit uncertainty in operations. In S-MMMDVRPs, each mode of transport is subject to a probability of survival on each edge. The objective is to design robust, coordinated routes for heterogeneous fleets that minimize risk, while satisfying service and operational constraints.

## Problem Formulation and Motivation

This VRP variant focuses on a stochastic, multi-modal, multi-depot vehicle routing problem where vehicles are equally distributed among and dispatched from depots to serve customer demands. Each mode has distinct operational characteristics and a probability of survival on each edge. The goal is to determine coordinated routes for transport vehicles, originating from multiple depots, that minimize distance, accounting for the risk of failure, while respecting vehicle capacities and service requirements.

The motivation is to capture the synergy between ground and aerial vehicles under uncertainty, enabling more resilient and cost-effective delivery networks. By explicitly modeling both the capabilities and risks of each mode, the approach supports robust planning in complex, stochastic environments.

## Mathematical Formulation of Stochastic Multi-Modal Multi-Depot VRP

### Sets and Indices
- $i, j$: Nodes in the graph ($i, j \in \text{nodes}$).
- $d$: Depots ($d \in \text{depots} \subset \text{nodes}$).
- $c$: Customers ($c \in \text{customers} \subset \text{nodes}$).
- $m$: Modes ($m \in \text{modes}$) where for example ($\text{modes} = \{\text{trucks}, \text{drones}\}$).
- $v$: Vehicles ($v \in \text{vehicles}$).

### Parameters
- $\text{distance}_{i,j,m}$: Distance between nodes $i$ and $j$ for mode $m$.
- $\text{demands}_i$: Demand at customer $i$.
- $\text{capacity}_m$: Capacity of each vehicle of mode $m$.
- $\text{prob}_{i,j,m}$: Survival probability for edge $(i, j)$ for a vehicle with mode $m$.
- $\text{threshold}_m$: Minimum survival probability threshold for routes taken by a vehicle of mode $m$.
- $\text{mode}_v \in \text{modes}$: the mode of a vehicle $v$.
- $\text{home}_v \in \text{depots}$: the home depot of a vehicle $v$.

### Decision Variables
- $x_{i,j,v} \in \mathcal{X}$: Binary variable indicating whether vehicle $v$ travels from node $i$ to node $j$. This essentially defines that an edge $(i, j)$ is available to a vehicle $v$. Edges are only included in $\mathcal{X}$ if the following requirements are met:
    - A distance for $(i, j)$ is defined for $\text{mode}_v$.
    - $i \neq j$. Self-edges are not allowed.
    - $i = \text{home}_v$ if $i \in \text{depots}$. A vehicle cannot leave from depots that are not its home depot.
    - $j = \text{home}_v$ if $j \in \text{depots}$. A vehicle cannot arrive at depots that are not its home depot.
    - $\neg(j \in \text{customers} \ \land\ \text{demand}_j > \text{capacity}_{\text{mode}_v})$. A vehicle cannot *arrive* at a customer $j$ that has a demand that is higher than the capacity of this vehicle's mode. This particular exclusion of edges is necessary to prevent an issue where the existence of a node that has a demand higher than the carrying capacity of a mode can cause the problem to become infeasible (related to constraint 3).
    - $\neg(i \in \text{customers} \ \land\ \text{demand}_i > \text{capacity}_{\text{mode}_v})$. The same as the above rule, except then for *leaving* a customer $i$. This further reduces the number of decision variables, making the problem easier to solve. It also prevents false infeasibility, but we don't quite understand how or why.
    - $\neg(j \in \text{customers} \ \land\ \text{demand}_j = 0)$. It is pointless for a vehicle to *arrive* at a customer that has no demand, assuming that the graph is fully connected (which we do assume).
    - $\neg(i \in \text{customers} \ \land\ \text{demand}_i = 0)$. The same rule as above, except then for *leaving* a customer $i$.


- $u_{i,m}$: Continuous variable representing the load of a vehicle with modality $m$ at customer $i$.

### Objective Function
Minimize the total distance traveled by all vehicles:
$$
\min \sum_{v \in \text{vehicles}} \sum_{i \in \text{nodes}} \sum_{j \in \text{nodes}} \text{distance}_{i,j,\text{mode}_v} \cdot x_{i,j,v}\quad, x_{i,j,v} \in \mathcal{X}
$$

### Constraints

#### 1. Each customer is visited at least once (if its demand is greater than zero)
$$
\sum_{v \in \text{vehicles}} \sum_{i \in \text{nodes}} x_{i,j,v}  \geq 1 \quad \forall j \in \text{customers}, x_{i,j,v} \in \mathcal{X}
$$

#### 2. Flow conservation
$$
\sum_{j \in \text{nodes}} x_{i,j,v} = \sum_{j \in \text{nodes}} x_{j,i,v} \quad \forall v \in \text{vehicles}, \forall i \in \text{nodes}, x_{i,j,v} \in \mathcal{X}, x_{j,i,v} \in \mathcal{X}
$$

#### 3. Subtour elimination (MTZ Constraints)

[Source for the MTZ formulation](https://dl.acm.org/doi/pdf/10.1145/321043.321046).
$$
u_{i,\text{mode}_v} - u_{j,\text{mode}_v} + \text{capacity}_{\text{mode}_v} \cdot x_{i,j,v} \leq \text{capacity}_{\text{mode}_v} - \text{demands}_j \quad \forall i, j \in \text{customers}, \forall v \in \text{vehicles}, x_{i,j,v} \in \mathcal{X}
$$

#### 4. Survival probability constraints

$$
\sum_{i \in \text{nodes}} \sum_{j \in \text{nodes}} \log(\text{prob}_{i,j,\text{mode}_v}) \cdot x_{i,j,v} \geq \log(\text{threshold}_{\text{mode}_v}) \quad \forall v \in \text{vehicles}, x_{i,j,v} \in \mathcal{X}
$$

#### 5. A vehicle visits its depot at most once
$$
\sum_{j \in \text{customers}} x_{\text{home}_v,j,v} \leq 1 \quad \forall v \in \text{vehicles}, x_{\text{home}_v,j,v} \in \mathcal{X}
$$
$$
\sum_{i \in \text{customers}} x_{i,\text{home}_v,v} \leq 1 \quad \forall v \in \text{vehicles}, x_{i,\text{home}_v,v} \in \mathcal{X}
$$


## Working with Log Probabilities and Model Constraints

To maintain linearity in our optimization model, we work with the logarithm of survival probabilities. This transformation makes the probabilities additive along a route, allowing us to avoid products of decision variables and enabling the use of linear programming techniques. By summing log probabilities, we can efficiently enforce constraints on the minimum allowable probability of survival for any route.

The probability of survival on an edge is modeled as a function of the distance from the destination node to the nearest depot. Edges that approach the "front line" or are farther from depots are assigned lower survival probabilities, reflecting increased operational risk in these areas.

It is important to note that if the survival threshold for a route is set too high, the model may become infeasible. In such cases, it may not be possible to construct routes that meet all demand requirements within the specified risk limits.

## Parameter Sensitivity and Model Flexibility

The model supports experimentation with several key parameters:

- **Vehicle Capacities:** Adjusting these values changes the feasible service area for each mode.
- **Fleet Size per Mode:** Limits the transport vehicle fleet size, affecting coverage and cost.
- **Edge Survival Probabilities:** Allows for different risk models transport vehicles, influencing route selection and robustness.
- **Depot Locations:** Changing depot locations impacts feasible assignments and overall efficiency.

By varying these parameters, users can explore trade-offs between cost, service level, operational complexity, and risk, and assess the benefits of integrating multiple modes under uncertainty.
