import pytest
import numpy as np
from knapsack_solvers.utils.fitness import calculate_fitness


def knapsack_fitness(individual, weights, values, max_weight):
    total_weight = np.dot(individual, weights)
    total_value = np.dot(individual, values)
    return total_value if total_weight <= max_weight else 0


# Unit tests for knapsack_fitness
def test_knapsack_fitness_valid_solution():
    individual = [1, 0, 1]
    weights = [2, 3, 5]
    values = [10, 20, 15]
    max_weight = 10
    result = calculate_fitness(individual, weights, values, max_weight)
    expected = 25
    assert result == expected


def test_knapsack_fitness_overweight_solution():
    individual = [1, 1, 1]
    weights = [2, 3, 5]
    values = [10, 20, 15]
    max_weight = 9
    result = calculate_fitness(individual, weights, values, max_weight)
    assert result == 0


def test_knapsack_fitness_empty_solution():
    individual = [0, 0, 0]
    weights = [2, 3, 5]
    values = [10, 20, 15]
    max_weight = 10
    result = calculate_fitness(individual, weights, values, max_weight)
    assert result == 0


def test_knapsack_fitness_exact_capacity():
    individual = [1, 1, 0]
    weights = [2, 3, 5]
    values = [10, 20, 15]
    max_weight = 5
    result = calculate_fitness(individual, weights, values, max_weight)
    expected = 30
    assert result == expected
