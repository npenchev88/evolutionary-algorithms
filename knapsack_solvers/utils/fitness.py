import numpy as np


def calculate_fitness(weights, values, individual, max_weight):
    total_weight = np.dot(individual, weights)
    total_value = np.dot(individual, values)
    if total_weight > max_weight:
        return 0  # Penalize over-weight solutions
    return total_value
