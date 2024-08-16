# Memetic Algorithm with hybrid local search

import time

import numpy as np
from knapsack_solvers.utils.timeout import with_timeout


class MemeticAlgorithmHybridLocalSearch:
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

    def hill_climbing(self, individual):
        current_fitness = self.fitness(individual)
        for i in range(len(individual)):
            neighbor = individual.copy()
            neighbor[i] = 1 - neighbor[i]  # Flip the bit
            neighbor_fitness = self.fitness(neighbor)
            if neighbor_fitness > current_fitness:
                individual = neighbor
                current_fitness = neighbor_fitness
        return individual

    def simulated_annealing(self, individual, initial_temp, cooling_rate):
        current = individual
        current_fitness = self.fitness(current)
        temp = initial_temp
        while temp > 1:
            neighbor = current.copy()
            i = np.random.randint(0, len(current) - 1)
            neighbor[i] = 1 - neighbor[i]  # Flip a bit
            neighbor_fitness = self.fitness(neighbor)
            if neighbor_fitness > current_fitness or np.random.rand() < np.exp(
                    (neighbor_fitness - current_fitness) / temp):
                current = neighbor
                current_fitness = neighbor_fitness
            temp *= cooling_rate
        return current

    def local_search(self, individual, iteration, max_iterations):
        if iteration < max_iterations // 2:
            # Use hill climbing in the first half of the iterations
            return self.hill_climbing(individual)
        else:
            # Use simulated annealing in the second half
            return self.simulated_annealing(individual, initial_temp=100, cooling_rate=0.99)

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

    def runner(self, population):
        def closure_fn():
            nonlocal population
            for generation in range(self.number_of_generations):
                new_population = []
                for _ in range(self.population_size // 2):
                    parent1 = self.tournament_selection(population, self.tournament_size)
                    parent2 = self.tournament_selection(population, self.tournament_size)
                    child1, child2 = self.crossover(parent1, parent2)
                    self.mutate(child1)
                    self.mutate(child2)
                    # Apply hybrid local search to each child
                    child1 = self.local_search(child1, generation, self.number_of_generations)
                    child2 = self.local_search(child2, generation, self.number_of_generations)
                    new_population.extend([child1, child2])
                population = new_population
        return closure_fn

    def solve(self, timeout):
        start_time = time.time()
        population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        with_timeout(timeout, self.runner(population), "MEMETIC ALGORITHM HYBRID LOCAL SEARCH")

        end_time = time.time()
        total_time = end_time - start_time
        # Final best solution
        best_solution = max(population, key=self.fitness)
        print(
            f"MEMETIC ALGORITHM HYBRID LOCAL SEARCH Final Best value = {self.fitness(best_solution)}, Solution = N/A, Total time: {total_time}")
        return ["MEMETIC ALGORITHM HYBRID LOCAL SEARCH", self.fitness(best_solution), total_time]
