## Introduction to Multi-Depot Vehicle Routing Problems

Multi-Depot Vehicle Routing Problems (MDVRPs) extend the classical vehicle routing problem by introducing multiple depots from which vehicles are dispatched. In MDVRPs, the objective is to design optimal routes for a fleet of vehicles, starting and ending at the same depot where multiple depots exist, to serve a set of customers while minimizing total operational costs such as distance or time.

## Problem Formulation and Motivation

This VRP variant focusses on the deterministic multi-depot vehicle routing problem (MDVRP), where vehicles are equally distributed among depots and must serve a subset of customers before returning to its origin depot. The formulation assumes known and fixed travel distances and customer demands. The objective is to determine a set of routes, each starting and ending at a depot, that collectively serve all customers while minimizing the total distance traveled and respecting vehicle capacity constraints.

The motivation for this formulation is to capture the operational realities faced by logistics providers managing multiple distribution centers. By explicitly modeling multiple depots, the MDVRP enables more accurate planning of vehicle assignments, depot utilization, and customer allocation, leading to improved efficiency and service levels in complex distribution networks.

# Mathematical Formulation of Multi-Depot VRP

### Sets and Indices
- $i, j$: Nodes in the graph ($i, j \in \text{nodes}$).
- $d$: Depots ($d \in \text{depots} \subset \text{nodes}$).
- $c$: Customers ($c \in \text{customers} \subset \text{nodes}$).
- $v$: Vehicles ($v \in \text{vehicles}$).

### Parameters
- $\text{distance}_{i,j}$: Distance between nodes $i$ and $j$.
- $\text{demands}_i$: Demand at customer $i$.
- $\text{capacity}$: Capacity of each vehicle.
- $\text{home}_v \in \text{depots}$: the home depot of a vehicle $v$.

### Decision Variables
- $x_{i,j,v} \in \mathcal{X}$: Binary variable indicating whether vehicle $v$ travels from node $i$ to node $j$. This essentially defines that an edge $(i, j)$ is available to a vehicle $v$. Edges are only included in $\mathcal{X}$ if the following requirements are met:
    - A distance for $(i, j)$ is defined.
    - $i \neq j$. Self-edges are not allowed.
    - $i = \text{home}_v$ if $i \in \text{depots}$. A vehicle cannot leave from depots that are not its home depot.
    - $j = \text{home}_v$ if $j \in \text{depots}$. A vehicle cannot arrive at depots that are not its home depot.
    - $\neg(j \in \text{customers} \ \land\ \text{demand}_j = 0)$. It is pointless for a vehicle to *arrive* at a customer that has no demand, assuming that the graph is fully connected (which we do assume).
    - $\neg(i \in \text{customers} \ \land\ \text{demand}_i = 0)$. The same rule as above, except then for *leaving* a customer $i$.


- $u_{i,m}$: Continuous variable representing the load of a vehicle at customer $i$.

### Objective Function
Minimize the total distance traveled by all vehicles:
$$
\min \sum_{v \in \text{vehicles}} \sum_{i \in \text{nodes}} \sum_{j \in \text{nodes}} \text{distance}_{i,j} \cdot x_{i,j,v}\quad, x_{i,j,v} \in \mathcal{X}
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
u_{i} - u_{j} + \text{capacity} \cdot x_{i,j,v} \leq \text{capacity} - \text{demands}_j \quad \forall i, j \in \text{customers}, \forall v \in \text{vehicles}, x_{i,j,v} \in \mathcal{X}
$$

#### 4. A vehicle visits its depot at most once
$$
\sum_{j \in \text{customers}} x_{\text{home}_v,j,v} \leq 1 \quad \forall v \in \text{vehicles}, x_{\text{home}_v,j,v} \in \mathcal{X}
$$
$$
\sum_{i \in \text{customers}} x_{i,\text{home}_v,v} \leq 1 \quad \forall v \in \text{vehicles}, x_{i,\text{home}_v,v} \in \mathcal{X}
$$


## Parameter Sensitivity and Model Flexibility

The model allows for experimentation with several key parameters:

- **Maximum Number of Vehicles:** Limits the total fleet size available for routing, affecting the ability to meet demand under uncertainty.
- **Vehicle Capacity:** Determines the maximum load a vehicle can carry, influencing the number and length of routes required.

By varying these parameters, users can explore trade-offs between cost, and service level, and assess the robustness of solutions under different operational scenarios.
