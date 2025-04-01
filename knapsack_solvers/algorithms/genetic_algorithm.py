# Genetic Algorithm Steps
# Initialization: Generate an initial population of solutions randomly.
# Evaluation: Calculate the fitness of each individual.
# Selection: Select individuals for reproduction based on their fitness.
# Crossover: Create new individuals by combining parts of two parents.
# Mutation: Introduce random changes to new individuals.
# Replacement: Form a new generation.
import time
from knapsack_solvers.utils.fitness import calculate_fitness
from knapsack_solvers.utils.selection import tournament_selection
from knapsack_solvers.utils.mutation import bit_flip_mutation
from knapsack_solvers.utils.crossover import single_point_crossover
from knapsack_solvers.utils.decorators import evolutionary_solver
import numpy as np


class GeneticAlgorithm:
    def __init__(self, weights, values, max_weight, adjusted_values=None,
                 fitness_func=calculate_fitness, selection_func=tournament_selection,
                 crossover_func=single_point_crossover, mutation_func=bit_flip_mutation):
        self.weights = weights
        self.values = values
        self.max_weight = max_weight
        self.adjusted_values = adjusted_values or values
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func

        # Hyperparameters
        self.population_size = 50
        self.number_of_generations = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.7
        self.tournament_size = 5

        self._population = []  # Internal storage

    @property
    def population(self):
        return self._population

    def fitness(self, individual):
        return self.fitness_func(individual, self.weights, self.adjusted_values, self.max_weight)

    @evolutionary_solver
    def solve(self):
        self._population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        for _ in range(self.number_of_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.selection_func(self._population, self.tournament_size, self.fitness)
                parent2 = self.selection_func(self._population, self.tournament_size, self.fitness)
                child1, child2 = self.crossover_func(parent1, parent2, self.crossover_rate)
                self.mutation_func(child1, self.mutation_rate)
                self.mutation_func(child2, self.mutation_rate)
                new_population.extend([child1, child2])
            self._population = new_population

        return self._population
