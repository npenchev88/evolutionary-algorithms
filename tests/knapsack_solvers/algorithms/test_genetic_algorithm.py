import pytest
from knapsack_solvers.algorithms.genetic_algorithm import GeneticAlgorithm
import numpy as np
from knapsack_solvers.utils.risk_adjustment import compute_sharpe_adjusted_values


@pytest.fixture
def simple_data():
    weights = [1, 2, 3]
    values = [10, 20, 30]
    return weights, values


@pytest.fixture
def complex_data():
    weights = [2, 3, 4, 5, 9, 7, 8, 9, 10, 12, 1, 2, 2, 3, 8, 7, 8, 4, 6, 5, 14, 13, 11, 6, 7, 9, 12, 15, 17, 19]
    values = [3, 4, 8, 8, 10, 11, 14, 13, 6, 7, 9, 3, 4, 7, 20, 12, 13, 8, 9, 6, 21, 22, 20, 5, 8, 8, 12, 25, 24, 13]
    max_weight = 50
    return weights, values, max_weight


def test_genetic_algorithm_without_timeout(complex_data):
    np.random.seed(42)
    # random.seed(None)
    # np.random.seed(None)
    weights, values, max_weight = complex_data
    ga = GeneticAlgorithm(weights, values, max_weight=max_weight)
    result = ga.solve()
    assert isinstance(result, list)
    assert result[0] == "GENETICALGORITHM"
    assert isinstance(result[1], np.int64)
    assert result[1] == 85


def test_genetic_algorithm_without_timeout_but_with_risk(simple_data):
    np.random.seed(42)
    # random.seed(None)
    # np.random.seed(None)
    weights, values = simple_data
    risk = [1, 2, 3]
    adjusted_values = compute_sharpe_adjusted_values(values, risk)
    ga = GeneticAlgorithm(weights, values, max_weight=5, adjusted_values=adjusted_values)
    result = ga.solve()
    assert isinstance(result, list)
    assert result[0] == "GENETICALGORITHM"
    assert isinstance(result[1], np.float64)
    assert result[1] == 20.0


def test_genetic_algorithm_with_timeout(complex_data):
    np.random.seed(42)
    timeout = 0.3
    weights, values, max_weight = complex_data
    ga = GeneticAlgorithm(weights, values, max_weight=max_weight)
    result = ga.solve(timeout=timeout)
    assert isinstance(result, list)
    assert result[0] == "GENETICALGORITHM"
    assert isinstance(result[1], np.int64)
    assert result[1] == 85
    # total time should not exceed 0.4 seconds approx
    assert result[2] <= 0.4
