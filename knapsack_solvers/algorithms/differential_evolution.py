# DE
import time

import numpy as np

from knapsack_solvers.utils.decorators import evolutionary_solver
from knapsack_solvers.utils.fitness import calculate_fitness


class DifferentialEvolution:
    def __init__(self, weights, values, max_weight, adjusted_values=None, fitness_func=calculate_fitness):
        self.weights = weights
        self.values = values
        self.adjusted_values = adjusted_values or values
        self.max_weight = max_weight
        self.population_size = 50
        self.number_of_generations = 100
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover rate
        self.fitness_func = fitness_func
        self._population = []

    @property
    def population(self):
        return self._population

    def fitness(self, individual):
        return self.fitness_func(individual, self.weights, self.adjusted_values, self.max_weight)

    def crossover(self, target, donor):
        trial = target.copy()
        for i in range(len(target)):
            if np.random.rand() < self.CR:
                trial[i] = donor[i]
        return trial

    @evolutionary_solver
    def solve(self):
        self._population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        for _ in range(self.number_of_generations):
            new_population = []
            for i in range(self.population_size):
                target = self._population[i]
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                a, b, c = self._population[indices]
                # Differential mutation using modulo 2 to keep binary
                donor = np.where(np.random.rand(len(target)) < self.CR, (a + self.F * (b - c)) % 2, target).astype(int)
                trial = self.crossover(target, donor)

                if self.fitness(trial) > self.fitness(target):
                    new_population.append(trial)
                else:
                    new_population.append(target)
            self._population = np.array(new_population)


        return self._population
