## Introduction to Stochastic Vehicle Routing Problems

Stochastic (Multi Depot) Vehicle Routing Problems (SVRPs) are an extension of classical vehicle routing problems where some elements of the problem, such as failures on edges, are uncertain and modeled as random variables. This uncertainty reflects real-world logistics scenarios, where exact information is rarely available in advance. The goal in SVRPs is to design robust and efficient routes that can accommodate these uncertainties, minimizing risk or maximizing service levels.

## Problem Formulation and Motivation

This VRP variant focusses on a stochastic multi-depot vehicle routing problem (MDVRP) where vehicles are equally distributed among and dispatched from depots to serve customer demands. Whenever each vehicle travels over an edge, there exists a survival probability of this vehicle on this edge. Our chosen formulation models survival probability of a vehicle on an edge as independent random variables. The objective is to determine a set of routes originating from multiple depots that minimize the total distance of the routes, where the survival probability of each vehicle route should always be above a certain threshold.

The motivation for this formulation is to more accurately reflect the operational challenges encountered by logistics providers in uncertain and dynamic environments. We deliberately exclude the expected value of vehicle failures from the objective function, as the relatively small vehicle pool (i.e., limited sample size) would yield expected values that do not reliably represent the true risk profile in smaller-scale routing problems. Instead, by constraining the minimum allowable survival probability on any route, the model ensures robust performance even in scenarios where statistical averages may be misleading.

# Mathematical Formulation of Multi-Depot VRP with Survival Probabilities

### Sets and Indices
- $i, j$: Nodes in the graph ($i, j \in \text{nodes}$).
- $d$: Depots ($d \in \text{depots} \subset \text{nodes}$).
- $c$: Customers ($c \in \text{customers} \subset \text{nodes}$).
- $v$: Vehicles ($v \in \text{vehicles}$).

### Parameters
- $\text{distance}_{i,j}$: Distance between nodes $i$ and $j$.
- $\text{demands}_i$: Demand at customer $i$.
- $\text{capacity}$: Capacity of each vehicle.
- $\text{prob}_{i,j}$: Survival probability for edge $(i, j)$ for a vehicle.
- $\text{threshold}$: Minimum survival probability threshold for feasible vehicle routes.
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

#### 4. Survival probability constraints

$$
\sum_{i \in \text{nodes}} \sum_{j \in \text{nodes}} \log(\text{prob}_{i,j}) \cdot x_{i,j,v} \geq \log(\text{threshold}) \quad \forall v \in \text{vehicles}, x_{i,j,v} \in \mathcal{X}
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

The model allows for experimentation with several key parameters:

- **Maximum and Minimum Failure Probability:** Adjusting these values changes the risk profile of the network, influencing which routes are feasible.
- **Maximum Number of Vehicles:** Limits the total fleet size available for routing, affecting the ability to meet demand under uncertainty.
- **Vehicle Capacity:** Determines the maximum load a vehicle can carry, influencing the number and length of routes required.
- **Success threshold:** Determines the minimal acceptable success probability of a route of a single vehicle.

By varying these parameters, users can explore trade-offs between risk, and service level, and assess the robustness of solutions under different operational scenarios.
