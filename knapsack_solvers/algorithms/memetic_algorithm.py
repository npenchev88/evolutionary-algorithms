# Memetic Algorithm Steps
# Initialization: Generate an initial population of solutions randomly.
# Evaluation: Calculate the fitness of each individual.
# Selection: Select individuals for reproduction based on their fitness.
# Crossover and Mutation: Apply genetic operators to create new individuals.
# Local Search: Apply a local search algorithm to each new individual to find nearby, potentially better solutions.
# Replacement: Form a new generation, replacing some of the less fit individuals with the new ones.
import time

from knapsack_solvers.algorithms.genetic_algorithm import GeneticAlgorithm
from knapsack_solvers.utils.decorators import evolutionary_solver
import numpy as np


class MemeticAlgorithm(GeneticAlgorithm):
    def local_search(self, individual):
        for i in range(len(individual)):
            new_individual = individual.copy()
            new_individual[i] = 1 - new_individual[i]
            if self.fitness(new_individual) > self.fitness(individual):
                individual = new_individual
        return individual

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

                # 🔁 Add local search before inserting to population
                child1 = self.local_search(child1)
                child2 = self.local_search(child2)

                new_population.extend([child1, child2])
            self._population = new_population

        return self._population
