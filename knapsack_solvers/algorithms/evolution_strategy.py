# ES Steps
# Initialization: Generate an initial population of solutions randomly.
# Mutation and Recombination
# Selection: Select individuals for reproduction based on their fitness.
import time

import numpy as np


class EvolutionStrategy:
    def __init__(self, weights, values, max_weight):
        self.weights = weights
        self.values = values
        self.max_weight = max_weight
        self.population_size = 50
        self.number_of_generations = 100
        self.sigma = 1

    def fitness(self, individual):
        total_weight = np.dot(individual, self.weights)
        total_value = np.dot(individual, self.values)
        if total_weight > self.max_weight:
            return 0  # Penalize over-weight solutions
        return total_value

    def mutate(self, individual, sigma):
        for _ in range(sigma):  # Mutate 'sigma' bits per individual
            idx = np.random.randint(0, len(individual))
            individual[idx] = 1 - individual[idx]
        return individual

    def solve(self):
        start_time = time.time()
        population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        for generation in range(self.number_of_generations):
            new_population = []
            for individual in population:
                # Create offspring by mutating the individual
                offspring = self.mutate(individual.copy(), self.sigma)
                # Select the better individual for the next generation
                if self.fitness(offspring) > self.fitness(individual):
                    new_population.append(offspring)
                else:
                    new_population.append(individual)
            population = new_population

            # Optionally, print best solution of each generation
            # best_solution = max(population, key=self.fitness)
            # print(f"Generation {generation}: Best Value = {fitness(best_individual)}")
        end_time = time.time()
        total_time = end_time - start_time
        # Final best solution
        best_solution = max(population, key=self.fitness)
        print(f"EVOLUTION STRATEGY Final Best value = {self.fitness(best_solution)}, Solution = N/A, Total time: {total_time}")
        return ["EVOLUTION STRATEGY", self.fitness(best_solution), total_time]
