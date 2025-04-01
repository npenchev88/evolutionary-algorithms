import numpy as np


def single_point_crossover(parent1, parent2, crossover_rate=0.7):
    """
    Single-point crossover for binary individuals.
    """
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1.copy(), parent2.copy()
