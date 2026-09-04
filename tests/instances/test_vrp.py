from typing import Any, Dict, Tuple

import pulp
import pytest

from milp_engine.instances.vrp import (
    VehicleInfo,
    extract_vrp_solution,
    stochastic_n_modal_multi_depot_vrp,
)

VRPInstanceType = Tuple[pulp.LpProblem, Dict[str, Any]]


@pytest.fixture
def small_vrp_instance() -> VRPInstanceType:
    n_nodes = 4
    n_vehicles = 2
    depots = [0, 2]
    customers = [1, 3]
    node_coordinates = [
        (0.0, 0.0),  # 0=Depot 0
        (1.0, 1.0),  # 1=Customer 0
        (0.0, 1.0),  # 2=Depot 1
        (1.0, 0.0),  # 3=Customer 1
    ]
    n_modes = 2  # 0=truck, 1=drone
    modes = [0, 1]  # one truck and one drone
    homes = [0, 2]  # truck and drone at depot 0, drone at depot 1
    distances = {
        (i, j, m): (
            (node_coordinates[i][0] - node_coordinates[j][0]) ** 2
            + (node_coordinates[i][1] - node_coordinates[j][1]) ** 2
        )
        ** 0.5
        * (2 if m == 0 else 1)  # Different costs for modes
        for i in range(n_nodes)
        for j in range(n_nodes)
        for m in range(n_modes)
        if i != j and (i, j) not in [(1, 2), (2, 1)]  # Remove 1-2 edge
    }
    demands = {1: 1, 3: 1}  # Customer demands
    capacities = [2, 1]  # Mode capacities
    survival_probabilities = {
        (i, j, m): 1 - distances[i, j, m] / 10 for (i, j, m) in distances.keys()
    }
    survival_thresholds = [0.5, 0.3]

    args = {
        "n_nodes": n_nodes,
        "n_vehicles": n_vehicles,
        "depots": depots,
        "customers": customers,
        "distances": distances,
        "demands": demands,
        "capacities": capacities,
        "survival_probabilities": survival_probabilities,
        "survival_thresholds": survival_thresholds,
        "modes": modes,
        "homes": homes,
    }

    return (
        stochastic_n_modal_multi_depot_vrp(
            n_nodes=n_nodes,
            n_vehicles=n_vehicles,
            customers=customers,
            depots=depots,
            distances=distances,
            demands=demands,
            capacities=capacities,
            survival_probabilities=survival_probabilities,
            survival_thresholds=survival_thresholds,
            modes=modes,
            homes=homes,
        ),
        args,
    )


@pytest.fixture
def large_vrp_instance() -> VRPInstanceType:
    n_nodes = 10
    n_vehicles = 4
    depots = [0, 1]
    customers = list(range(2, n_nodes))
    node_coordinates = [
        (0.0, 0.0),  # 0=Depot
        (1.0, 1.0),  # 1=Depot
        (0.0, 1.0),  # 2=Customer
        (0.9, 0.0),  # 3=Customer
        (0.5, 0.4),  # 4=Customer
        (0.2, 0.8),  # 5=Customer
        (0.8, 0.2),  # 6=Customer
        (0.3, 0.3),  # 7=Customer
        (0.7, 0.7),  # 8=Customer
        (0.1, 0.9),  # 9=Customer
    ]
    n_modes = 2  # 0=truck, 1=drone
    modes = [0, 0, 1, 1]  # two trucks and two drones
    homes = [0, 1, 0, 1]
    distances = {
        (i, j, m): (
            (node_coordinates[i][0] - node_coordinates[j][0]) ** 2
            + (node_coordinates[i][1] - node_coordinates[j][1]) ** 2
        )
        ** 0.5  # Euclidean distance
        * (1.5 if m == 0 else 1)  # Different costs for modes
        for i in range(n_nodes)
        for j in range(n_nodes)
        for m in range(n_modes)
        if i != j
    }
    demands = {2: 1, 3: 1, 4: 3, 5: 3, 6: 2, 7: 2, 8: 0, 9: 1}
    capacities = [2, 6]  # Mode capacities
    survival_probabilities = {
        (i, j, m): 1 - distances[i, j, m] / 100 for (i, j, m) in distances.keys()
    }
    survival_thresholds = [0.5, 0.3]

    args = {
        "n_nodes": n_nodes,
        "n_vehicles": n_vehicles,
        "depots": depots,
        "customers": customers,
        "distances": distances,
        "demands": demands,
        "capacities": capacities,
        "survival_probabilities": survival_probabilities,
        "survival_thresholds": survival_thresholds,
        "modes": modes,
        "homes": homes,
    }

    return (
        stochastic_n_modal_multi_depot_vrp(
            n_nodes=n_nodes,
            n_vehicles=n_vehicles,
            customers=customers,
            depots=depots,
            distances=distances,
            demands=demands,
            capacities=capacities,
            survival_probabilities=survival_probabilities,
            survival_thresholds=survival_thresholds,
            modes=modes,
            homes=homes,
        ),
        args,
    )


def test_solve_small_vrp(small_vrp_instance: VRPInstanceType) -> None:
    vrp, args = small_vrp_instance

    # Solve the problem
    vrp.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if the solution is feasible
    assert vrp.status == pulp.LpStatusOptimal

    distances = args["distances"]
    demands = args["demands"]
    modes = args["modes"]
    surv_probs = args["survival_probabilities"]
    solution = extract_vrp_solution(vrp, distances, demands, modes, surv_probs)

    # The optimal solution in this problem is that the drone goes from depot 1 (=node 2)
    # to customer 1 (=node 3) and back, while the truck goes from depot 0 (=node 0)
    # to customer 0 (=node 1) and back.
    def is_the_drone_route(vehicle_info: VehicleInfo) -> bool:
        return (
            vehicle_info.modality == 1
            and set(vehicle_info.route) == {(2, 3), (3, 2)}
            and vehicle_info.load == 1
            and vehicle_info.survival_probability is not None
            and vehicle_info.survival_probability >= 0.3
        )

    assert any(map(is_the_drone_route, solution))

    def is_the_truck_route(vehicle_info: VehicleInfo) -> bool:
        return (
            vehicle_info.modality == 0
            and set(vehicle_info.route) == {(0, 1), (1, 0)}
            and vehicle_info.load == 1
            and vehicle_info.survival_probability is not None
            and vehicle_info.survival_probability >= 0.5
        )

    assert any(map(is_the_truck_route, solution))


def test_solve_large_vrp(large_vrp_instance: VRPInstanceType) -> None:
    vrp, args = large_vrp_instance

    # Solve the problem
    vrp.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if the solution is feasible
    assert vrp.status == pulp.LpStatusOptimal

    distances = args["distances"]
    demands = args["demands"]
    modes = args["modes"]
    solution = extract_vrp_solution(vrp, distances, demands, modes)

    expected_solution = [
        VehicleInfo(
            route=[(0, 7), (7, 0)], load=2, modality=0, distance=1.2727922061357855
        ),
        VehicleInfo(route=[], load=0, modality=0, distance=0.0),
        VehicleInfo(
            route=[(0, 4), (3, 0), (4, 6), (6, 3)],
            load=6,
            modality=1,
            distance=2.124474349039663,
        ),
        VehicleInfo(
            route=[(1, 5), (2, 1), (5, 9), (9, 2)],
            load=5,
            modality=1,
            distance=2.1074638375981514,
        ),
    ]

    assert len(solution) == len(expected_solution)

    # Check if the actual solution matches the expected solution
    #   Do it in such a way that the order of vehicles does not matter,
    #   that the routes can be reversed,
    #   and that the distance is within a small tolerance.
    unmatched_expected = expected_solution.copy()
    for actual in solution:
        match_found = False
        for expected in unmatched_expected:
            expected_route_reversed = [(j, i) for (i, j) in expected.route]
            if (
                set(actual.route) in [set(expected.route), set(expected_route_reversed)]
                and actual.load == expected.load
                and actual.modality == expected.modality
                and actual.distance == pytest.approx(expected.distance, rel=1e-6)
            ):
                unmatched_expected.remove(expected)
                match_found = True
                break
        assert match_found, f"No match found for actual vehicle: {actual}"


def test_infeasible_if_depot_unreachable() -> None:
    n_nodes = 3
    n_vehicles = 1
    customers = [1, 2]
    depots = [0]
    modes = [0]
    homes = [0]
    distances = {(1, 2, 0): 1.0, (2, 1, 0): 1.0}  # no edge to depot
    demands = {1: 1, 2: 1}
    capacities = [2]
    survival_probabilities = distances.copy()
    survival_thresholds = [1.0]

    vrp = stochastic_n_modal_multi_depot_vrp(
        n_nodes=n_nodes,
        n_vehicles=n_vehicles,
        customers=customers,
        depots=depots,
        distances=distances,
        demands=demands,
        capacities=capacities,
        survival_probabilities=survival_probabilities,
        survival_thresholds=survival_thresholds,
        modes=modes,
        homes=homes,
    )

    # Solve the problem
    vrp.solve(pulp.PULP_CBC_CMD(msg=False))

    solution = extract_vrp_solution(vrp, distances, demands, modes)
    print(solution)

    # Check if the solution is feasible
    assert vrp.status == pulp.LpStatusInfeasible


def test_extract_vrp_solution() -> None:
    truck1_route = [(0, 1), (1, 2), (2, 3), (3, 0)]
    truck2_route = [(0, 2), (2, 3), (3, 0)]
    drone1_route = [(1, 2), (2, 1)]
    routes = [truck1_route, truck2_route, drone1_route]
    edges = set(truck1_route + truck2_route + drone1_route)
    modes = [0, 0, 1]  # Truck, Truck, Drone

    xs = {
        (i, j, v): pulp.LpVariable(f"x_{i}_{j}_{v}", cat="Binary")
        for (i, j) in edges
        for v in range(3)
    }
    for i, j in edges:
        for v in range(3):
            xs[i, j, v].varValue = 1 if (i, j) in routes[v] else 0

    distances = {(i, j, m): 1.0 * (m + 1) for (i, j) in edges for m in modes}
    surv_probs = {(i, j, m): 0.9 for (i, j) in edges for m in modes}

    demands = {1: 1, 2: 2, 3: 3}

    vrp = pulp.LpProblem("VRP", pulp.LpMinimize)
    vrp.objective = pulp.lpSum(xs.values())

    vehicle_infos = extract_vrp_solution(vrp, distances, demands, modes, surv_probs)
    assert len(vehicle_infos) == 3  # 2 trucks + 1 drone

    # Check modes
    for vehicle_info, m in zip(vehicle_infos, modes):
        assert vehicle_info.modality == m

    # Check loads
    expected_loads = [6, 5, 3]
    for vehicle_info, expected_load in zip(vehicle_infos, expected_loads):
        assert vehicle_info.load == expected_load

    # Check distances traveled by each vehicle
    expected_distances = [4.0, 3.0, 4.0]
    for vehicle_info, expected_distance in zip(vehicle_infos, expected_distances):
        assert vehicle_info.distance == expected_distance

    # Check routes
    for vehicle_info, route in zip(vehicle_infos, routes):
        assert set(vehicle_info.route) == set(route)

    # Check the survival probabilities
    expected_surv_probs = [0.9 ** len(route) for route in routes]
    for vehicle_info, expected_surv_prob in zip(vehicle_infos, expected_surv_probs):
        assert pytest.approx(vehicle_info.survival_probability) == expected_surv_prob
