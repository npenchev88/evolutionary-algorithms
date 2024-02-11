# Memetic Algorithm Steps
# Initialization: Generate an initial population of solutions randomly.
# Evaluation: Calculate the fitness of each individual.
# Selection: Select individuals for reproduction based on their fitness.
# Crossover and Mutation: Apply genetic operators to create new individuals.
# Local Search: Apply a local search algorithm to each new individual to find nearby, potentially better solutions.
# Replacement: Form a new generation, replacing some of the less fit individuals with the new ones.
import time

import numpy as np


class MemeticAlgorithmParameterTuning:
    def __init__(self, weights, values, max_weight):
        self.weights = weights
        self.values = values
        self.max_weight = max_weight
        self.population_sizes = [50, 100, 200]
        self.number_of_generations = 100
        self.mutation_rates = [0.01, 0.05, 0.1]
        self.crossover_rate = 0.7
        self.tournament_size = 5
        self.local_search_frequencies = [0.1, 0.25, 0.5]
        self.best_parameters = {}
        self.highest_fitness = 0

    def fitness(self, individual):
        total_weight = np.dot(individual, self.weights)
        total_value = np.dot(individual, self.values)
        if total_weight > self.max_weight:
            return 0  # Penalize over-weight solutions
        return total_value

    def local_search(self, individual):
        for i in range(len(individual)):
            # Try flipping each bit and check if it improves the fitness
            new_individual = individual.copy()
            new_individual[i] = 1 - new_individual[i]
            if self.fitness(new_individual) > self.fitness(individual):
                individual = new_individual  # Accept the new individual
        return individual

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

    def mutate(self, individual, mutation_rate):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = 1 - individual[i]

    def solve(self):
        start_time = time.time()
        population = []
        for pop_size in self.population_sizes:
            for mutation_rate in self.mutation_rates:
                for ls_frequency in self.local_search_frequencies:
                    print(pop_size,mutation_rate,ls_frequency)
                    population = np.random.randint(2, size=(pop_size, len(self.weights)))
                    for generation in range(self.number_of_generations):
                        new_population = []
                        for _ in range(pop_size // 2):
                            parent1 = self.tournament_selection(population, self.tournament_size)
                            parent2 = self.tournament_selection(population, self.tournament_size)
                            child1, child2 = self.crossover(parent1, parent2)
                            self.mutate(child1, mutation_rate)
                            self.mutate(child2, mutation_rate)
                            # Apply local search with the specified frequency
                            for i in range(int(pop_size * ls_frequency)):
                                child1 = self.local_search(child1)
                                child2 = self.local_search(child2)
                            new_population.extend([child1, child2])
                        population = new_population
                    # Example: Evaluate the solution quality based on the best individual in the final population
                    best_fitness = max(self.fitness(ind) for ind in population)

                    # Update best parameters if this run is the best so far
                    if best_fitness > self.highest_fitness:
                        self.highest_fitness = best_fitness
                        self.best_parameters = {
                            'population_size': pop_size,
                            'mutation_rate': mutation_rate,
                            'local_search_frequency': ls_frequency
                        }

        end_time = time.time()
        total_time = end_time - start_time
        # Final best solution
        best_solution = max(population, key=self.fitness)
        print(f"MEMETIC ALGORITHM WITH PARAMETERS TUNING best parameters {self.best_parameters}, Solution = {self.highest_fitness}, Total time: {total_time}")
        print(f"MEMETIC ALGORITHM WITH PARAMETERS TUNING Final Best value = {self.fitness(best_solution)}, Solution = {best_solution}, Total time: {total_time}")
