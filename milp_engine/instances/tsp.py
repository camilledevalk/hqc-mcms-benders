# examples/tsp.py
"""Traveling Salesman Problem (TSP) formulations using PuLP."""
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import chain, combinations
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pulp
from matplotlib import pyplot as plt


def plot_tsp(cities: Dict[str, Tuple[float, float]]) -> None:
    """Plot the cities for TSP visualization.

    :param cities: Dictionary of city names and their coordinates.
    """
    # Plot the cities
    plt.figure(figsize=(8, 6))
    for city, coord in cities.items():
        plt.scatter(*coord, label=city)
        plt.text(coord[0], coord[1], f" {city}", fontsize=12)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Cities for TSP")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_distances(
    cities: Dict[str, Tuple[float, float]],
) -> Dict[Tuple[str, str], float]:
    """
    Calculate distances between each pair of cities.

    :param cities: Dictionary of city names and their coordinates.
    :return: Dictionary of distances between each pair of cities.
    """
    distances: Dict[Tuple[str, str], float] = {}
    for city1, coord1 in cities.items():
        for city2, coord2 in cities.items():
            if city1 != city2:
                distances[(city1, city2)] = np.linalg.norm(
                    np.array(coord1) - np.array(coord2)
                ).item()
    return distances


def _distances_without_self_loops(
    distances: Dict[Tuple[str, str], float],
    cities: Iterable[str],
) -> Dict[Tuple[str, str], float]:
    """Remove self-loops from a given distance dictionary.

    :param distances: Dictionary of distances between city pairs.
    :param cities: Iterable of city names.
    :return: Dictionary of distances without self-loops.
    """
    distances_copy = deepcopy(distances)
    for city in cities:
        if (city, city) in distances_copy:
            del distances_copy[(city, city)]
    return distances_copy


class TSPFormulation(ABC):
    """Abstract base class for TSP formulations."""

    @abstractmethod
    def formulate(self, distances: Dict[Tuple[str, str], float]) -> pulp.LpProblem:
        """Abstract method to formulate the TSP problem.

        :param distances: A dictionary of city pairs and their distances.
        :return: A PuLP problem instance.
        """
        pass


class DFJFormulation(TSPFormulation):
    """Dantzig–Fulkerson–Johnson (DFJ) formulation for the TSP."""

    def formulate(self, distances: Dict[Tuple[str, str], float]) -> pulp.LpProblem:
        """Formulate the TSP problem using the DFJ formulation.

        It follows the definition for the Dantzig–Fulkerson–Johnson formulation in
        https://en.wikipedia.org/wiki/Travelling_salesman_problem

        :param distances: A dictionary of city pairs and their distances.
        :return: A PuLP problem instance.
        """
        cities_set = set().union(*[{i, j} for (i, j) in distances.keys()])
        cities = list(sorted(list(cities_set)))

        # Remove self-loops from the distances as otherwise a degenerate solution
        #   will be found where the path goes from a city to itself.
        distances_wsl = _distances_without_self_loops(distances, cities)

        prob = pulp.LpProblem("TSP", pulp.LpMinimize)

        # Decision variables: x[i,j] is 1 if the path goes from city i to city j, 0
        #   otherwise.
        x = pulp.LpVariable.dicts("y", distances_wsl, cat=pulp.LpBinary)

        # Objective function: minimize the total distance
        prob += pulp.lpSum(
            [distances_wsl[(i, j)] * x[(i, j)] for (i, j) in distances_wsl]
        )

        # Constraints: each city must be entered and left exactly once
        for k in cities:
            prob += pulp.lpSum([x[(i, j)] for (i, j) in distances_wsl if i == k]) == 1
            prob += pulp.lpSum([x[(i, j)] for (i, j) in distances_wsl if j == k]) == 1

        def powerset(iterable: Iterable[Any]) -> chain[tuple[Any, ...]]:
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        # Subtour elimination constraints
        def subtour_elimination_constraints(
            prob: pulp.LpProblem, subset: tuple[str, ...]
        ) -> None:
            if 1 < len(subset) < len(cities):
                prob += (
                    pulp.lpSum([x[(i, j)] for i in subset for j in subset if i != j])
                    <= len(subset) - 1
                )

        for subset in powerset(cities):
            subtour_elimination_constraints(prob, subset)

        return prob


class MTZFormulation(TSPFormulation):
    """Miller-Tucker-Zemlin (MTZ) formulation for the TSP."""

    def formulate(self, distances: Dict[Tuple[str, str], float]) -> pulp.LpProblem:
        """Formulate the TSP problem using the MTZ formulation.

        It follows the definition for the Miller-Tucker-Zemlin formulation in
        https://en.wikipedia.org/wiki/Travelling_salesman_problem

        :param distances: A dictionary of city pairs and their distances.
        :return: A PuLP problem instance.
        """
        cities_set = set().union(*[{i, j} for (i, j) in distances.keys()])
        cities = list(sorted(list(cities_set)))

        # Remove self-loops from the distances as otherwise a degenerate solution
        #   will be found where the path goes from a city to itself.
        distances_wsl = _distances_without_self_loops(distances, cities)

        prob = pulp.LpProblem("TSP", pulp.LpMinimize)

        # Decision variables: x[i,j] is 1 if the path goes from city i to city j,
        #   0 otherwise
        y = pulp.LpVariable.dicts("y", distances_wsl, cat=pulp.LpBinary)

        # Objective function: minimize the total distance
        prob += pulp.lpSum(
            [distances_wsl[(i, j)] * y[(i, j)] for (i, j) in distances_wsl]
        )

        # Constraints: each city must be entered and left exactly once
        for k in cities:
            prob += pulp.lpSum([y[(i, j)] for (i, j) in distances_wsl if i == k]) == 1
            prob += pulp.lpSum([y[(i, j)] for (i, j) in distances_wsl if j == k]) == 1

        # MTZ constraints
        x = pulp.LpVariable.dicts(
            "x", list(cities)[1:], lowBound=0, cat=pulp.LpContinuous
        )  # could be continuous?

        for x_i in x.values():
            prob += x_i <= len(cities) - 2

        for i in list(cities)[1:]:  # Excluding one city is fine.
            for j in list(cities)[1:]:
                if i != j:
                    prob += x[i] - x[j] + 1 <= (len(cities) - 1) * (1 - y[(i, j)])

        return prob
