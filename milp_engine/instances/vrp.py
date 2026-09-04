# instances/vrp.py
"""This module contains example MILP problems for use in tests and benchmarks."""
import math
import re
from typing import Dict, List, Tuple

import pulp
from pydantic import BaseModel


def stochastic_n_modal_multi_depot_vrp(
    n_nodes: int,
    n_vehicles: int,
    customers: List[int],
    depots: List[int],
    distances: Dict[Tuple[int, int, int], float],
    demands: Dict[int, int],
    capacities: List[int],
    homes: List[int],
    survival_probabilities: Dict[Tuple[int, int, int], float] | None = None,
    survival_thresholds: List[float] | None = None,
    modes: List[int] | None = None,
    name: str = "StochasticMultiModalMultiDepotVRP",
) -> pulp.LpProblem:
    """Formulate a stochastic multi-modal multi-depot VRP problem.

    :param n_nodes: Total number of nodes (customers + depots).
    :param n_vehicles: Total number of vehicles.
    :param customers: List of customer node indices. This is a subset of the nodes.
    :param depots: List of depot node indices. This is a subset of the nodes.
    :param distances: Dictionary of edges and their distances/costs for each mode. It
        is formatted (i, j, mode) -> distance, where i and j are node indices.
    :param demands: Demand for each customer.
    :param capacities: Capacity for each mode. The list index corresponds to the mode.
    :param homes: Home location for each vehicle. The list index corresponds to the
        vehicle.
    :param survival_probabilities: Survival probabilities for each edge and mode. It
        is a dictionary formatted (i, j, mode) -> survival_prob, where i and j are
        node indices.
    :param survival_thresholds: Survival thresholds for each mode. The list index
        corresponds to the mode. If None, no survival constraints are applied.
    :param modes: Transport mode for each vehicle. The list index corresponds to the
        vehicle.
    :param name: The name to assign to the problem.
    :return: A PuLP problem instance representing the stochastic multi-modal
        multi-depot VRP.
    :raises AssertionError: If the input parameters are incompatible with one another.
    """
    modes = modes or [0] * n_vehicles  # Default to mode 0 if not provided
    assert len(modes) == n_vehicles, "Number of modes must match number of vehicles"
    assert len(homes) == n_vehicles, "Number of homes must match number of vehicles"
    assert len(demands) == len(customers), "Demands must match number of customers"
    assert (
        len(customers) + len(depots) == n_nodes
    ), "Number of nodes must match number of customers and depots"
    assert set(customers).issubset(
        set(range(n_nodes))
    ), "Customers must be within the range of node indices"
    assert set(depots).issubset(
        set(range(n_nodes))
    ), "Depots must be within the range of node indices"
    n_modes = max(modes) + 1  # Assuming modes are indexed from 0 to n_modes-1
    assert len(capacities) >= n_modes, (
        f"There should be at least as many capacities defined ({len(capacities)}) "
        f"as that there are modes ({n_modes})"
    )
    assert (
        not survival_thresholds or len(survival_thresholds) >= n_modes
    ), "There should be at least as many survival thresholds as there are modes"
    # Why ">=" and not "=="? -> Overspecification is allowed (e.g. when a mode is
    # unused), underspecification is not.

    # Define problem
    problem = pulp.LpProblem(name, pulp.LpMinimize)

    # Create x[i, j, v]
    x = pulp.LpVariable.dicts(
        "x",
        (
            (i, j, v)
            for (i, j, m) in distances
            for v in range(n_vehicles)
            # disallow self-loops
            if i != j
            # ensure there is an edge for the vehicle's mode
            and modes[v] == m
            # remove edges to non-home depots for vehicles
            and (i not in depots or homes[v] == i)
            and (j not in depots or homes[v] == j)
            # vehicles can't visit customers for which they don't have enough capacity.
            #   Without this, the problem becomes infeasible when a mode's capacity is
            #   less than the demand of any customer in the problem.
            and not (i in customers and demands[i] > capacities[modes[v]])
            and not (j in customers and demands[j] > capacities[modes[v]])
            # it is pointless to visit a customer if it has no demand, assuming that
            #    the graph is fully connected (which we indeed assume).
            and not (i in customers and demands[i] == 0)
            and not (j in customers and demands[j] == 0)
        ),
        cat="Binary",
    )

    # u[i] for subtour elimination
    u = pulp.LpVariable.dicts(
        "u",
        ((i, m) for i in customers for m in range(n_modes)),
        lowBound=0,
        cat="Continuous",
    )

    # Objective function: total distance
    problem += pulp.lpSum(distances[i, j, modes[v]] * x[i, j, v] for i, j, v in x)

    # Each customer with demand>0 visited by at least one vehicle
    for j in customers:
        if demands[j] > 0:
            problem += (
                pulp.lpSum(
                    x[i, j, v]
                    for v in range(n_vehicles)
                    for i in range(n_nodes)
                    if (i, j, v) in x  # (i, j, v) needs to be a valid edge
                )
                >= 1
            ), f"visit_j={j}"

    # Flow conservation for all vehicles and modalities
    for v in range(n_vehicles):
        for i in range(n_nodes):
            outgoing = pulp.lpSum(x[i, j, v] for j in range(n_nodes) if (i, j, v) in x)
            incoming = pulp.lpSum(x[j, i, v] for j in range(n_nodes) if (j, i, v) in x)
            problem += outgoing == incoming, f"flow_cons_v={v}_i={i}"

    # MTZ Subtour elimination
    for i in customers:
        for j in customers:
            for v in range(n_vehicles):
                if (i, j, v) in x:
                    problem += (
                        u[i, modes[v]]
                        - u[j, modes[v]]
                        + capacities[modes[v]] * x[i, j, v]
                        <= capacities[modes[v]] - demands[j]
                    ), f"subtour_elim_i={i}_j={j}_v={v}"

    # Survival probability constraints
    if survival_probabilities is not None and survival_thresholds is not None:
        eps = 1e-9  # Small epsilon to avoid numerical issues
        for v in range(n_vehicles):
            survival = pulp.lpSum(
                math.log2(max(survival_probabilities[i, j, modes[v]], eps)) * x[i, j, v]
                for (i, j, m) in survival_probabilities
                if m == modes[v] and (i, j, v) in x
            )
            threshold = math.log2(max(survival_thresholds[modes[v]], eps))
            problem += survival >= threshold, f"surv_prob_v={v}"

    # Single outbound and inbound connection
    for v in range(n_vehicles):
        d = homes[v]  # Home depot for vehicle v
        # At most one outbound and inbound per vehicle
        problem += (
            pulp.lpSum(x[d, j, v] for j in customers if (d, j, v) in x) <= 1
        ), f"outbound_depot_{d}_v={v}"
        problem += (
            pulp.lpSum(x[i, d, v] for i in customers if (i, d, v) in x) <= 1
        ), f"inbound_depot_{d}_v={v}"

    return problem


def multi_depot_vrp(
    n_nodes: int,
    n_vehicles: int,
    customers: List[int],
    depots: List[int],
    distances: Dict[Tuple[int, int, int], float],
    demands: Dict[int, int],
    capacities: List[int],
    homes: List[int],
) -> pulp.LpProblem:
    """Formulate a multi-depot VRP problem.

    This is a wrapper around the stochastic_n_modal_multi_depot_vrp function
    that does not use modes, survival probabilities, or thresholds.

    :param n_nodes: Total number of nodes (customers + depots).
    :param n_vehicles: Total number of vehicles.
    :param customers: List of customer node indices. This is a subset of the nodes.
    :param depots: List of depot node indices. This is a subset of the nodes.
    :param distances: Dictionary of edges and their distances/costs for each mode. It
        is formatted (i, j, mode) -> distance, where i and j are node indices.
    :param demands: Demand for each customer.
    :param capacities: Capacity for each mode. The list index corresponds to the mode.
    :param homes: Home location for each vehicle. The list index corresponds to the
        vehicle.
    :return: A PuLP problem instance representing the multi-depot VRP.
    """
    return stochastic_n_modal_multi_depot_vrp(
        n_nodes=n_nodes,
        n_vehicles=n_vehicles,
        customers=customers,
        depots=depots,
        distances=distances,
        demands=demands,
        capacities=capacities,
        homes=homes,
        name="MultiDepotVRP",
    )


def stochastic_multi_depot_vrp(
    n_nodes: int,
    n_vehicles: int,
    customers: List[int],
    depots: List[int],
    distances: Dict[Tuple[int, int, int], float],
    demands: Dict[int, int],
    capacities: List[int],
    homes: List[int],
    survival_probabilities: Dict[Tuple[int, int, int], float] | None,
    survival_thresholds: List[float] | None,
) -> pulp.LpProblem:
    """Formulate a stochastic multi-depot VRP problem.

    This is a wrapper around the stochastic_n_modal_multi_depot_vrp function
    that does not use modes.

    :param n_nodes: Total number of nodes (customers + depots).
    :param n_vehicles: Total number of vehicles.
    :param customers: List of customer node indices. This is a subset of the nodes.
    :param depots: List of depot node indices. This is a subset of the nodes.
    :param distances: Dictionary of edges and their distances/costs for each mode. It
        is formatted (i, j, mode) -> distance, where i and j are node indices.
    :param demands: Demand for each customer.
    :param capacities: Capacity for each mode. The list index corresponds to the mode.
    :param homes: Home location for each vehicle. The list index corresponds to the
        vehicle.
    :param survival_probabilities: Survival probabilities for each edge and mode. It
        is a dictionary formatted (i, j, mode) -> survival_prob, where i and j are
        node indices.
    :param survival_thresholds: Survival thresholds for each mode. The list index
        corresponds to the mode. If None, no survival constraints are applied.
    :return: A PuLP problem instance representing the stochastic multi-depot VRP.
    """
    return stochastic_n_modal_multi_depot_vrp(
        n_nodes=n_nodes,
        n_vehicles=n_vehicles,
        customers=customers,
        depots=depots,
        distances=distances,
        demands=demands,
        capacities=capacities,
        homes=homes,
        survival_probabilities=survival_probabilities,
        survival_thresholds=survival_thresholds,
        name="StochasticMultiDepotVRP",
    )


def n_modal_multi_depot_vrp(
    n_nodes: int,
    n_vehicles: int,
    customers: List[int],
    depots: List[int],
    distances: Dict[Tuple[int, int, int], float],
    demands: Dict[int, int],
    capacities: List[int],
    homes: List[int],
    modes: List[int] | None,
) -> pulp.LpProblem:
    """Formulate a multi-modal multi-depot VRP problem.

    This is a wrapper around the stochastic_n_modal_multi_depot_vrp function
    that does not use survival probabilities or thresholds.

    :param n_nodes: Total number of nodes (customers + depots).
    :param n_vehicles: Total number of vehicles.
    :param customers: List of customer node indices. This is a subset of the nodes.
    :param depots: List of depot node indices. This is a subset of the nodes.
    :param distances: Dictionary of edges and their distances/costs for each mode. It
        is formatted (i, j, mode) -> distance, where i and j are node indices.
    :param demands: Demand for each customer.
    :param capacities: Capacity for each mode. The list index corresponds to the mode.
    :param homes: Home location for each vehicle. The list index corresponds to the
        vehicle.
    :param modes: Transport mode for each vehicle. The list index corresponds to the
        vehicle.
    :return: A PuLP problem instance representing the multi-modal multi-depot VRP.
    """
    return stochastic_n_modal_multi_depot_vrp(
        n_nodes=n_nodes,
        n_vehicles=n_vehicles,
        customers=customers,
        depots=depots,
        distances=distances,
        demands=demands,
        capacities=capacities,
        homes=homes,
        modes=modes,
        name="MultiModalMultiDepotVRP",
    )


class VehicleInfo(BaseModel):
    """A model to represent some vehicle info as part of the solution for a VRP."""

    route: List[Tuple[int, int]]
    load: int
    distance: float
    modality: int = 0
    survival_probability: float | None = None


def extract_vrp_solution(
    vrp: pulp.LpProblem,
    distances: Dict[Tuple[int, int, int], float],
    demands: Dict[int, int],
    modes: List[int] | None = None,
    survival_probabilities: Dict[Tuple[int, int, int], float] | None = None,
) -> List[VehicleInfo]:
    """
    Extract the solution of a solved VRP problem.

    It produces a list of VehicleInfo objects, each containing the route,
    load, modality, and distance for each vehicle.

    :param vrp: The solved VRP problem as a pulp.LpProblem instance.
    :param distances: The dictionary of distances between nodes used to construct
        the VRP.
    :param demands: The dictionary of demands for each customer used to construct
        the VRP.
    :param modes: The list of transport modalities for each vehicle used to
        construct the VRP.
    :param survival_probabilities: The dictionary of survival probabilities for each
        edge and mode used to construct the VRP. If None, survival probabilities are
        not included in the solution.
    :return: A list of VehicleInfo objects representing the solution.
    """
    solution = []

    # Get x variable dict
    x: Dict[Tuple[int, int, int], pulp.LpVariable] = {}

    for var_name, var in vrp.variablesDict().items():
        if var_name.startswith("x_"):
            parts = var_name.split("_")
            parts = [re.sub("[(),]", "", part) for part in parts]
            i, j, v = map(int, parts[1:])
            x[i, j, v] = var

    n_vehicles = max(v for _, _, v in x) + 1
    modes = modes or [0] * n_vehicles  # Default to mode 0
    for v in range(n_vehicles):
        route: List[Tuple[int, int]] = []
        load = 0
        total_distance = 0.0
        mode = modes[v]

        for i, j, m in distances:
            if (i, j, v) in x and x[i, j, v].varValue == 1.0 and m == mode:
                route.append((i, j))
                load += demands.get(j, 0)
                total_distance += distances[i, j, m]

        survival_probability: float | None = None
        if survival_probabilities is not None:
            survival_probability = 1.0
            for i, j, m in survival_probabilities:
                if (i, j, v) in x and x[i, j, v].varValue == 1.0 and m == mode:
                    survival_probability *= survival_probabilities[i, j, m]

        vehicle_info = VehicleInfo(
            route=route,
            load=load,
            modality=mode,
            distance=total_distance,
            survival_probability=survival_probability,
        )
        solution.append(vehicle_info)

    return solution
