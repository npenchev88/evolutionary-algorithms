import time

import numpy as np

from knapsack_solvers.utils.decorators import evolutionary_solver
from knapsack_solvers.utils.fitness import calculate_fitness


class Base:
    def __init__(self, weights, values, max_weight, iterations, adjusted_values=None, fitness_func=calculate_fitness):
        self.weights = weights
        self.values = values
        self.adjusted_values = adjusted_values or values
        self.max_weight = max_weight
        self.population_size = 50
        self.iterations = iterations
        self.fitness_func = fitness_func
        self._population = []

    @property
    def population(self):
        return self._population

    def fitness(self, individual):
        return self.fitness_func(individual, self.weights, self.adjusted_values, self.max_weight)

    @evolutionary_solver
    def solve(self):
        all_best = []

        for _ in range(self.iterations):
            population = np.random.randint(2, size=(self.population_size, len(self.weights)))
            best_solution = max(population, key=self.fitness)
            all_best.append(best_solution)

        self._population = all_best
        return self._population
