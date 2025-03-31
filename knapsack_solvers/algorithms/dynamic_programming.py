import time

import numpy as np
from knapsack_solvers.utils.risk_adjustment import compute_sharpe_adjusted_values
from typing import Optional


class DP:
    def __init__(
            self,
            weights: list,
            values: list,
            max_weight: int,
            risk: Optional[list] = None,
            risk_free_rate: float = 0.0,
            adjusted_values: Optional[list] = None
    ):
        """
        :param weights: List of weights (costs)
        :param values: List of expected returns
        :param max_weight: Maximum allowed weight (capacity)
        :param risk: Optional list of risk (std deviation)
        :param risk_free_rate: Used only if computing Sharpe-adjusted values
        :param adjusted_values: Precomputed adjusted values (overrides risk-based computation)
        """
        self.weights = weights
        self.values = values
        self.capacity = max_weight

        if adjusted_values is not None:
            self.adjusted_values = adjusted_values
        elif risk is not None:
            self.adjusted_values = compute_sharpe_adjusted_values(values, risk, risk_free_rate)
        else:
            self.adjusted_values = values

    def solve(self):
        start_time = time.time()

        n = len(self.adjusted_values)
        dp = [[0 for _ in range(self.capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(self.capacity + 1):
                if self.weights[i - 1] <= w:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        dp[i - 1][w - self.weights[i - 1]] + self.adjusted_values[i - 1]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        end_time = time.time()
        total_time = end_time - start_time

        max_value = dp[n][self.capacity]
        print(f"[DP] Max value: {max_value:.2f}, Total time: {total_time:.4f}s")
        return ["DYNAMIC PROGRAMMING", max_value, total_time]
