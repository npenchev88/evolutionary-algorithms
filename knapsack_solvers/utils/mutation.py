import numpy as np


def bit_flip_mutation(individual, mutation_rate=0.01):
    """
    Bit-flip mutation for binary individuals.
    """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
