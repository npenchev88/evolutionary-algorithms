# Genetic Algorithm Steps
# Initialization: Generate an initial population of solutions randomly.
# Evaluation: Calculate the fitness of each individual.
# Selection: Select individuals for reproduction based on their fitness.
# Crossover: Create new individuals by combining parts of two parents.
# Mutation: Introduce random changes to new individuals.
# Replacement: Form a new generation.
import time

import numpy as np


class GeneticAlgorithm:
    def __init__(self, weights, values, max_weight):
        self.weights = weights
        self.values = values
        self.max_weight = max_weight
        self.population_size = 50
        self.number_of_generations = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.7
        self.tournament_size = 5

    def fitness(self, individual):
        total_weight = np.dot(individual, self.weights)
        total_value = np.dot(individual, self.values)
        if total_weight > self.max_weight:
            return 0  # Penalize over-weight solutions
        return total_value
# sredna predeglena
    def tournament_selection(self, pop, k):
        best = np.random.randint(len(pop))
        for i in np.random.randint(0, len(pop), k - 1):
            if self.fitness(pop[i]) > self.fitness(pop[best]):
                best = i
        return pop[best]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(self.weights) - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]

    def solve(self):
        start_time = time.time()
        population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        for generation in range(self.number_of_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.tournament_selection(population, self.tournament_size)
                parent2 = self.tournament_selection(population, self.tournament_size)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population

            # Optionally, print best solution of each generation
            # best_solution = max(population, key=self.fitness)
            # print(f"Generation {generation + 1}: Best value = {self.fitness(best_solution)}, Solution = {best_solution}")
        end_time = time.time()
        total_time = end_time - start_time
        # Final best solution
        best_solution = max(population, key=self.fitness)
        print(f"GENETIC ALGORITHM Final Best value = {self.fitness(best_solution)}, Solution = N/A, total time: {total_time}")
        return ["GENETIC ALGORITHM", self.fitness(best_solution), total_time]
