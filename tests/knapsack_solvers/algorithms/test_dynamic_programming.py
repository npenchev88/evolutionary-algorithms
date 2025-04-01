from typing import List

import numpy as np
import pytest
from knapsack_solvers.algorithms.dynamic_programming import DP
from knapsack_solvers.utils.risk_adjustment import compute_sharpe_adjusted_values


# Wrapping the DP tests for pytest

@pytest.fixture
def simple_data():
    weights = [1, 2, 3]
    values = [10, 20, 30]
    return weights, values


def test_basic_knapsack(simple_data):
    weights, values = simple_data
    dp = DP(weights, values, max_weight=5)
    result = dp.solve()
    assert result[1] == 50


def test_known_input_knapsack():
    # global optimum 105
    weights = [2, 3, 4, 5, 9, 7, 8, 9, 10, 12, 1, 2, 2, 3, 8, 7, 8, 4, 6, 5, 14, 13, 11, 6, 7, 9, 12, 15, 17, 19]
    values = [3, 4, 8, 8, 10, 11, 14, 13, 6, 7, 9, 3, 4, 7, 20, 12, 13, 8, 9, 6, 21, 22, 20, 5, 8, 8, 12, 25, 24, 13]
    max_weight = 50
    dp = DP(weights, values, max_weight)
    result = dp.solve()
    # ['DYNAMIC PROGRAMMING (Sharpe-Aware)', 105, 0.0003631114959716797]
    assert result[1] == 105


def test_sharpe_adjustment_effect(simple_data):
    weights, values = simple_data
    risk = [1, 2, 3]
    dp_risk = DP(weights, values, max_weight=5, risk=risk)
    result_risk = dp_risk.solve()

    dp_plain = DP(weights, values, max_weight=5)
    result_plain = dp_plain.solve()
    assert result_risk[1] != result_plain[1]

def test_sharpe_adjustment_effect_passed_from_outside(simple_data):
    weights, values = simple_data
    risk = [1, 2, 3]
    adjusted_values = compute_sharpe_adjusted_values(values,risk)
    dp_risk = DP(weights, values, max_weight=5, risk=risk, adjusted_values=adjusted_values)
    result_risk = dp_risk.solve()

    dp_plain = DP(weights, values, max_weight=5)
    result_plain = dp_plain.solve()
    assert result_risk[1] != result_plain[1]


def test_zero_risk_handling(simple_data):
    weights, values = simple_data
    risk = [0, 0, 0]  # Zero risk values
    dp = DP(weights, values, max_weight=5, risk=risk)
    result = dp.solve()
    assert isinstance(result[1], float)


def test_large_input_performance():
    np.random.seed(42)
    weights = [int(x) for x in np.random.normal(10, 3, 300)]
    values = [int(x) for x in np.random.normal(13, 3, 300)]
    risk = np.abs(np.random.normal(2.5, 0.8, 300))
    max_weight = 1000
    dp = DP(weights, values, max_weight, risk=risk)
    result = dp.solve()
    assert result[1] > 0
