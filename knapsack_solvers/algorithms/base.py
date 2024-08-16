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
            return 0  # Penalize over-weight solutions
        return total_value

    # def mutate(self, individual):
    #     idx = np.random.randint(0, len(individual))
    #     individual[idx] = 1 - individual[idx]
    #
    #     return individual

    def solve(self):
        start_time = time.time()
        # all_best = []
        best_solution = []
        for iteration in range(self.iterations):
            population = np.random.randint(2, size=(self.population_size, len(self.weights)))
            # new_population = []
            # for individual in population:
            #     # # Create offspring by mutating the individual
            #     # # offspring = self.mutate(individual.copy())
        https://drive.google.com/file/d/1QM6BeFStniaJCnkJaTJQsTn4ASdtO4_i/view?usp=sharing    #     # # Select the better individual for the next generation
            #     # if self.fitness(offspring) > self.fitness(individual):
            #     #     new_population.append(offspring)
            #     # else:
            #     #     new_population.append(individual)

            # Final best solution per iteration
            best_solution = max(population, key=self.fitness)
            # all_best.append(best_solution)

        end_time = time.time()
        total_time = end_time - start_time
        print_best_solution("BASE", best_solution, self.fitness, total_time)
