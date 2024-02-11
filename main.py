import pygad

from knapsack_solvers.algorithms.genetic_algorithm import GeneticAlgorithm
from knapsack_solvers.algorithms.memetic_algorithm import MemeticAlgorithm
from knapsack_solvers.algorithms.memetic_algorithm_hybrid_local_search import MemeticAlgorithmHybridLocalSearch
from knapsack_solvers.algorithms.memetic_algorithm_parameter_tuning import MemeticAlgorithmParameterTuning
from knapsack_solvers.algorithms.evolution_strategy import EvolutionStrategy
from knapsack_solvers.algorithms.differential_evolution import DifferentialEvolution

# Knapsack problem parameters
# Global optimum 105
weights = [2, 3, 4, 5, 9, 7, 8, 9, 10, 12, 1, 2, 2, 3, 8, 7, 8, 4, 6, 5, 14, 13, 11, 6, 7, 9, 12, 15, 17, 19]
values = [3, 4, 8, 8, 10, 11, 14, 13, 6, 7, 9, 3, 4, 7, 20, 12, 13, 8, 9, 6, 21, 22, 20, 5, 8, 8, 12, 25, 24, 13]
max_weight = 50

# weights = [350, 2200, 333, 160, 192, 80, 25, 200, 70, 38]
# values = [60, 500, 40, 150, 30, 15, 5, 500, 100, 10]
# max_weight = 3000

ga_solver = GeneticAlgorithm(weights, values, max_weight)
ga_solver.solve()

ma_solver = MemeticAlgorithm(weights, values, max_weight)
ma_solver.solve()
#
# mahls_solver = MemeticAlgorithmHybridLocalSearch(weights, values, max_weight)
# mahls_solver.solve()

# mapt_solver = MemeticAlgorithmParameterTuning(weights, values, max_weight)
# mapt_solver.solve()

es_solver = EvolutionStrategy(weights, values, max_weight)
es_solver.solve()

de_solver = DifferentialEvolution(weights, values, max_weight)
de_solver.solve()


#
# # print(sum([0, 1, 0] * [3, 4, 7]))
#
#
# # Fitness function
# def fitness(ga_instance, solution, solution_idx):
#     total_weight = sum(solution * weights)
#
#     if total_weight <= max_weight:
#         return sum(solution * values)
#     else:
#         return 0
#
#
# # Create the genetic algorithm
# ga_instance = pygad.GA(
#     num_generations=100,
#     num_parents_mating=4,
#     sol_per_pop=10,
#     num_genes=len(weights),
#     fitness_func=fitness,
#     gene_type=int,
#     gene_space=[0, 1],
# )
#
# # Run the genetic algorithm
# ga_instance.run()
#
# # Get the best solution and its fitness value
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Best Solution:", solution)
# print("Fitness Value:", solution_fitness)
