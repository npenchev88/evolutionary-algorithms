import numpy as np


def tournament_selection(population, k, fitness_func):
    """
    Selects the best individual among k randomly chosen using tournament selection.
    :param population: List of individuals
    :param k: Tournament size
    :param fitness_func: Function to evaluate fitness of individuals
    :return: Selected individual
    """
    best_idx = np.random.randint(len(population))
    for i in np.random.randint(0, len(population), k - 1):
        if fitness_func(population[i]) > fitness_func(population[best_idx]):
            best_idx = i
    return population[best_idx]
