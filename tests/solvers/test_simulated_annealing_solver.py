import pytest

from milp_engine.solvers.simulated_annealing_solver import (
    SimulatedAnnealingSolver,
    VRPInput,
)


@pytest.fixture
def sample_problem() -> VRPInput:
    """A simple VRP instance with 2 depots and 5 customers.

    :returns: A VRPInput instance with predefined parameters.
    """
    return VRPInput(
        problem_type="Capacitated VRP",
        n_customers=5,
        demands=[2, 4, 6, 3, 7],
        num_vehicles=2,
        vehicle_capacity=40,
        distance_matrix=[
            [0, 12, 18, 23, 27, 31, 20],  # depot 0
            [12, 0, 22, 19, 29, 24, 15],  # depot 1
            [18, 22, 0, 26, 21, 28, 17],  # customer 1
            [23, 19, 26, 0, 17, 13, 22],  # customer 2
            [27, 29, 21, 17, 0, 9, 14],  # customer 3
            [31, 24, 28, 13, 9, 0, 19],  # customer 4
            [20, 15, 17, 22, 14, 19, 0],  # customer 5
        ],
        depots=[0, 1],
    )


def test_create_data_model_valid(sample_problem: VRPInput) -> None:
    solver = SimulatedAnnealingSolver()
    data = solver._create_data_model(sample_problem)

    assert data["num_vehicles"] == 2
    # Demands should include one zero per unique depot
    assert data["demands"][0] == 0
    assert data["demands"][1] == 0
    assert len(data["demands"]) == sample_problem.n_customers + len(
        set(sample_problem.depots)
    )
    assert all(
        cap == sample_problem.vehicle_capacity for cap in data["vehicle_capacities"]
    )
    assert data["distance_matrix"] == sample_problem.distance_matrix


def test_create_data_model_invalid_demands() -> None:
    """Number of demands must match n_customers."""
    problem = VRPInput(
        problem_type="Capacitated VRP",
        n_customers=3,
        demands=[1, 2],  # too short
        num_vehicles=1,
        vehicle_capacity=10,
        distance_matrix=[[0]],
        depots=[0],
    )
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError):
        solver._create_data_model(problem)


def test_solve_supported_problem(sample_problem: VRPInput) -> None:
    solver = SimulatedAnnealingSolver(time_limit=1)
    solution = solver.solve(sample_problem)

    assert solution["status"] == "Success"
    assert solution["problem"] == "Capacitated VRP"
    assert "routes" in solution
    assert isinstance(solution["total_distance"], int)
    # At least one vehicle should have a non-empty path
    assert any(route["path"] for route in solution["routes"])


def test_solve_unsupported_problem_type(sample_problem: VRPInput) -> None:
    problem = sample_problem
    problem.problem_type = "Invalid Type"
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError, match="Unsupported problem type"):
        solver.solve(problem)
