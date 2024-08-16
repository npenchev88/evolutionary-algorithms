import time

import numpy as np


class DP:
    def __init__(self, weights, values, max_weight):
        self.weights = weights
        self.values = values
        self.capacity = max_weight

    def solve(self):
        start_time = time.time()

        # Number of items
        n = len(self.values)

        # Create a 2D DP array to store the maximum value at each n and capacity
        dp = [[0 for _ in range(self.capacity + 1)] for _ in range(n + 1)]

        # Build the dp array from bottom up
        for i in range(1, n + 1):
            for w in range(self.capacity + 1):
                if self.weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - self.weights[i - 1]] + self.values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]

        end_time = time.time()
        total_time = end_time - start_time

        print(f"DYNAMIC PROGRAMMING The maximum value that can be carried in the knapsack is: {dp[n][self.capacity]}, total time:  {total_time}")
        print(dp[n][self.capacity])
        return ["DYNAMIC PROGRAMMING", dp[n][self.capacity], total_time]
