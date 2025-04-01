# ES Steps
# Initialization: Generate an initial population of solutions randomly.
# Mutation and Recombination
# Selection: Select individuals for reproduction based on their fitness.
import time

import numpy as np

from knapsack_solvers.utils.decorators import evolutionary_solver
from knapsack_solvers.utils.fitness import calculate_fitness


class EvolutionStrategy:
    def __init__(self, weights, values, max_weight, adjusted_values=None, fitness_func=calculate_fitness):
        self.weights = weights
        self.values = values
        self.adjusted_values = adjusted_values or values
        self.max_weight = max_weight
        self.population_size = 50
        self.number_of_generations = 100
        self.sigma = 1
        self.fitness_func = fitness_func
        self._population = []

    @property
    def population(self):
        return self._population

    def fitness(self, individual):
        return self.fitness_func(individual, self.weights, self.adjusted_values, self.max_weight)

    def mutate(self, individual, sigma):
        for _ in range(sigma):
            idx = np.random.randint(0, len(individual))
            individual[idx] = 1 - individual[idx]
        return individual

    @evolutionary_solver
    def solve(self):
        self._population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        for _ in range(self.number_of_generations):
            new_population = []
            for individual in self._population:
                offspring = self.mutate(individual.copy(), self.sigma)
                if self.fitness(offspring) > self.fitness(individual):
                    new_population.append(offspring)
                else:
                    new_population.append(individual)
            self._population = new_population

        return self._population
