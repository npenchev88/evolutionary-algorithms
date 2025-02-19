import time

import numpy as np

from knapsack_solvers.utils.helpers import print_best_solution


class Base:
    def __init__(self, weights, values, max_weight, iterations):
        self.weights = weights
        self.values = values
        self.max_weight = max_weight
        self.population_size = 50
        self.iterations = iterations

    def fitness(self, individual):
        total_weight = np.dot(individual, self.weights)
        total_value = np.dot(individual, self.values)
        if total_weight > self.max_weight:
            return 0
        return total_value

    def solve(self):
        start_time = time.time()
        all_best = []

        for iteration in range(self.iterations):
            population = np.random.randint(2, size=(self.population_size, len(self.weights)))

            best_solution = max(population, key=self.fitness)
            all_best.append(best_solution)

        end_time = time.time()
        total_time = end_time - start_time
        print_best_solution("BASE", all_best, self.fitness, total_time)
