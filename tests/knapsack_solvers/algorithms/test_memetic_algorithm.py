import pytest

import numpy as np
from knapsack_solvers.algorithms.memetic_algorithm import MemeticAlgorithm


@pytest.fixture
def complex_data():
    weights = [2, 3, 4, 5, 9, 7, 8, 9, 10, 12, 1, 2, 2, 3, 8, 7, 8, 4, 6, 5, 14, 13, 11, 6, 7, 9, 12, 15, 17, 19]
    values = [3, 4, 8, 8, 10, 11, 14, 13, 6, 7, 9, 3, 4, 7, 20, 12, 13, 8, 9, 6, 21, 22, 20, 5, 8, 8, 12, 25, 24, 13]
    max_weight = 50
    return weights, values, max_weight


def test_memetic_algorithm_returns_valid_result(complex_data):
    np.random.seed(42)
    # random.seed(None)
    # np.random.seed(None)
    weights, values, max_weight = complex_data
    ma = MemeticAlgorithm(weights, values, max_weight=50)
    result = ma.solve(timeout=2)
    assert isinstance(result, list)
    assert result[0] == "MEMETICALGORITHM"
    assert isinstance(result[1], np.int64)
    assert result[1] == 101
    assert result[2] <= 3
