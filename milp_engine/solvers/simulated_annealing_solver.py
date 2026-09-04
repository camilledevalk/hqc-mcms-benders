# milp_engine/solvers/simulated_annealing_solver.py
"""This module contains a solver interface for simulated annealing.

This is a specialized solver for the Capacitated Vehicle Routing Problem (VRP) using
the Simulated Annealing metaheuristic provided by Google's OR-Tools. This can serve as a
baseline for comparison with MILP-based approaches.
"""
from dataclasses import dataclass
from typing import Any, Dict, List

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class VRPInput:
    """Dataclass representing the input parameters for a Capacitated VRP.

    Attributes
    ----------
    problem_type : str
        Type of the problem, e.g., "Capacitated VRP".
    n_customers : int
        Number of customers to serve (excluding depot).
    demands : List[int]
        List of customer demands, length must equal ``n_customers``.
    num_vehicles : int
        Number of vehicles available in the fleet.
    vehicle_capacity : int
        Maximum capacity per vehicle.
    distance_matrix : List[List[int]]
        Precomputed distance matrix between customers and depot.
    """

    problem_type: str
    n_customers: int
    demands: List[int]
    num_vehicles: int
    vehicle_capacity: int
    distance_matrix: List[List[int]]
    depots: List[int]


class SimulatedAnnealingSolver:
    """Solver for the Capacitated Vehicle Routing Problem using Simulated Annealing.

    Note: this solver does not extend the base Solver class because it is not a MILP
    solver. It only solves VRPs.
    """

    def __init__(self, time_limit: int = 10) -> None:
        """Initialize the solver with a problem type name.

        :param time_limit: Maximum time limit for the solver in seconds
        """
        self.time_limit: int = time_limit

    def _create_data_model(self, problem: VRPInput) -> Dict[str, Any]:
        """Build the OR-Tools data model for the VRP.

        :param problem: Problem definition dataclass.
        :returns: Dictionary containing model data including coordinates,
            demands, and capacities.
        : raises ValueError: If the number of demands does not match `n_customers`.
        """
        if len(problem.demands) != problem.n_customers:
            raise ValueError(
                f"Demands must be of length {problem.n_customers}, "
                f"got {len(problem.demands)}"
            )

        number_of_depots = len(set(problem.depots))
        demands = [0] * number_of_depots + problem.demands  # one 0 per unique depot
        vehicle_capacities = [problem.vehicle_capacity] * problem.num_vehicles

        return {
            "num_vehicles": problem.num_vehicles,
            "depots": problem.depots,
            "demands": demands,
            "vehicle_capacities": vehicle_capacities,
            "distance_matrix": problem.distance_matrix,
        }

    def _solve_capacitated_vrp(self, problem: VRPInput) -> Dict[str, Any]:
        """Solve the Capacitated VRP using Simulated Annealing.

        :param problem: Problem definition dataclass containing customers,
            demands, and vehicles.
        :returns: Dictionary with solution status, routes, and total distance.
            - ``status`` : str
                Solver status ("Success" or "No solution found").
            - ``problem`` : str
                Name of the problem type.
            - ``routes`` : List[Dict[str, Any]]
                A list of route dictionaries, each containing:
                ``vehicle`` (int), ``path`` (List[int]), and ``distance`` (int).
            - ``total_distance`` : int
                Total distance of all routes.
        """
        data = self._create_data_model(problem)
        distance_matrix = data["distance_matrix"]

        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix),
            data["num_vehicles"],
            data["depots"],  # start indices for vehicles
            data["depots"],  # or different ends if needed
        )
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_cb(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity (demand) callback
        def demand_cb(from_index: int) -> int:
            node = manager.IndexToNode(from_index)
            return data["demands"][node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            data["vehicle_capacities"],
            True,
            "Capacity",
        )

        # Add Distance dimension
        routing.AddDimension(
            transit_callback_index,
            0,
            10**6,
            True,
            "Distance",
        )
        distance_dim = routing.GetDimensionOrDie("Distance")
        distance_dim.SetGlobalSpanCostCoefficient(1)

        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        )
        search_params.time_limit.FromSeconds(self.time_limit)
        search_params.log_search = True

        # Solve
        solution = routing.SolveWithParameters(search_params)
        if not solution:
            return {"status": "No solution found", "routes": []}

        # Extract solution
        results: Dict[str, Any] = {
            "problem": problem.problem_type,
            "routes": [],
            "total_distance": 0,
        }
        total_distance = 0

        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                prev = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev, index, vehicle_id)
            route.append(manager.IndexToNode(index))  # end depot
            results["routes"].append(
                {"vehicle": vehicle_id, "path": route, "distance": route_distance}
            )
            total_distance += route_distance

        results["total_distance"] = total_distance
        results["status"] = "Success"
        return results

    def solve(self, problem: VRPInput) -> Dict[str, Any]:
        """Solve the Capacitated VRP using Simulated Annealing.

        :param problem: Problem definition dataclass containing customers,
            demands, and vehicles.
        :returns: Dictionary with solution status, routes, and total distance.
        :raises ValueError: If the problem type is unsupported.
        """
        match problem.problem_type:
            case "Capacitated VRP":
                return self._solve_capacitated_vrp(problem)
            case _:
                raise ValueError(f"Unsupported problem type: {problem.problem_type}")
