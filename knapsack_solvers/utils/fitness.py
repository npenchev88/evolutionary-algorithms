import numpy as np


def calculate_fitness(individual, weights, values, max_weight):
    total_weight = np.dot(individual, weights)
    total_value = np.dot(individual, values)
    return total_value if total_weight <= max_weight else 0
