# Memetic Algorithm with hybrid local search

import numpy as np
from knapsack_solvers.algorithms.genetic_algorithm import GeneticAlgorithm
from knapsack_solvers.utils.decorators import evolutionary_solver


class MemeticAlgorithmHybridLocalSearch(GeneticAlgorithm):
    def hill_climbing(self, individual):
        current_fitness = self.fitness(individual)
        for i in range(len(individual)):
            neighbor = individual.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbor_fitness = self.fitness(neighbor)
            if neighbor_fitness > current_fitness:
                individual = neighbor
                current_fitness = neighbor_fitness
        return individual

    def simulated_annealing(self, individual, initial_temp=100, cooling_rate=0.99):
        current = individual
        current_fitness = self.fitness(current)
        temp = initial_temp
        while temp > 1:
            neighbor = current.copy()
            i = np.random.randint(0, len(current))
            neighbor[i] = 1 - neighbor[i]
            neighbor_fitness = self.fitness(neighbor)
            if neighbor_fitness > current_fitness or np.random.rand() < np.exp((neighbor_fitness - current_fitness) / temp):
                current = neighbor
                current_fitness = neighbor_fitness
            temp *= cooling_rate
        return current

    def local_search(self, individual, iteration, max_iterations):
        return self.hill_climbing(individual) if iteration < max_iterations // 2 else self.simulated_annealing(individual)

    @evolutionary_solver
    def solve(self):
        self._population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        for generation in range(self.number_of_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.selection_func(self._population, self.tournament_size, self.fitness)
                parent2 = self.selection_func(self._population, self.tournament_size, self.fitness)
                child1, child2 = self.crossover_func(parent1, parent2, self.crossover_rate)
                self.mutation_func(child1, self.mutation_rate)
                self.mutation_func(child2, self.mutation_rate)
                child1 = self.local_search(child1, generation, self.number_of_generations)
                child2 = self.local_search(child2, generation, self.number_of_generations)
                new_population.extend([child1, child2])
            self._population = new_population

        return self._population
