# DE
import time

import numpy as np


class DifferentialEvolution:
    def __init__(self, weights, values, max_weight):
        self.weights = weights
        self.values = values
        self.max_weight = max_weight
        self.population_size = 50
        self.number_of_generations = 100
        self.F = 0.8
        self.CR = 0.9

    def fitness(self, individual):
        total_weight = np.dot(individual, self.weights)
        total_value = np.dot(individual, self.values)
        if total_weight > self.max_weight:
            return 0  # Penalize over-weight solutions
        return total_value

    def crossover(self, target, donor):
        trial = target.copy()
        for i in range(len(target)):
            if np.random.rand() < self.CR:
                trial[i] = donor[i]
        return trial

    def solve(self):
        start_time = time.time()
        population = np.random.randint(2, size=(self.population_size, len(self.weights)))

        for generation in range(self.number_of_generations):
            new_population = []
            for i in range(self.population_size):
                target = population[i]
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                # Differential mutation
                donor = np.where(np.random.rand(len(target)) < self.CR, (a + self.F * (b - c)) % 2, target).astype(int)
                # Discrete recombination
                trial = self.crossover(target, donor)
                # Selection
                if self.fitness(trial) > self.fitness(target):
                    new_population.append(trial)
                else:
                    new_population.append(target)
            population = np.array(new_population)

            # Optionally, print best solution of each generation
            # best_solution = max(population, key=self.fitness)
            # print(f"Generation {generation}: Best Value = {fitness(best_individual)}")
        end_time = time.time()
        total_time = end_time - start_time
        # Final best solution
        best_solution = max(population, key=self.fitness)
        print(f"DIFFERIENTIAL EVOLUTION Final Best value = {self.fitness(best_solution)}, Solution = {best_solution}, Total time: {total_time}")
