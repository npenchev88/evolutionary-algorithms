from datetime import datetime

import pygad

import pandas as pd
from knapsack_solvers.utils.timeout import with_timeout
from knapsack_solvers.algorithms.genetic_algorithm import GeneticAlgorithm
from knapsack_solvers.algorithms.memetic_algorithm import MemeticAlgorithm
from knapsack_solvers.algorithms.dynamic_programming import DP
from knapsack_solvers.algorithms.memetic_algorithm_hybrid_local_search import MemeticAlgorithmHybridLocalSearch
from knapsack_solvers.algorithms.memetic_algorithm_parameter_tuning import MemeticAlgorithmParameterTuning
from knapsack_solvers.algorithms.evolution_strategy import EvolutionStrategy
from knapsack_solvers.algorithms.differential_evolution import DifferentialEvolution
from knapsack_solvers.algorithms.base import Base
from knapsack_solvers.data.knapsack_inputs import weights
from knapsack_solvers.data.knapsack_inputs import adjusted_values

# ITEMS_FOR_EXPLORATION = [3,5,10,15,20,30,50,100,200,300,400,500,600,700,800,900,1000]
ITEMS_FOR_EXPLORATION = [10, 100, 1000, 10000]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#
# Knapsack problem parameters
# Global optimum 105
# weights = [2, 3, 4, 5, 9, 7, 8, 9, 10, 12, 1, 2, 2, 3, 8, 7, 8, 4, 6, 5, 14, 13, 11, 6, 7, 9, 12, 15, 17, 19]
# values = [3, 4, 8, 8, 10, 11, 14, 13, 6, 7, 9, 3, 4, 7, 20, 12, 13, 8, 9, 6, 21, 22, 20, 5, 8, 8, 12, 25, 24, 13]
# max_weight = 50

# Define the data for the table
data = {
    'Algorithm': [],
    'Sample': [],
    'Result': [],
    'Time': []
}


def fill_data(data, result, count, timeout):
    if isinstance(result, str):
        cname = result
        data['Algorithm'].append(cname)
        data['Sample'].append(count)
        data['Result'].append("N/A")
        data['Time'].append(f"> {timeout} seconds")
        return data
    else:
        cname, best_result, time_result = result
        data['Algorithm'].append(cname)
        data['Sample'].append(count)
        data['Result'].append(best_result)
        data['Time'].append(time_result)
        return data


TIMEOUT_MAP = {10: 50,
               100: 200,
               1000: 350,
               10000: 500}

for number_of_items in ITEMS_FOR_EXPLORATION:
    print(f"NUMBER OF ITEMS {number_of_items}")
    timeout = TIMEOUT_MAP[number_of_items]
    print(f"Allowed timeout {timeout}")

    weights_cut = weights[:number_of_items]
    values_cut = adjusted_values[:number_of_items]
    max_weight = int(sum(weights_cut) / 2)

    dp_solver = DP(weights_cut, values_cut, max_weight)
    dp_result = with_timeout(timeout, dp_solver.solve, "DYNAMIC PROGRAMMING")
    fill_data(data, dp_result, number_of_items, timeout)

    base_solver = Base(weights_cut, values_cut, max_weight, 100)
    base_result = base_solver.solve(timeout=timeout)
    fill_data(data, base_result, number_of_items, timeout)

    ga_solver = GeneticAlgorithm(weights_cut, values_cut, max_weight)
    ga_result = ga_solver.solve(timeout=timeout)
    fill_data(data, ga_result, number_of_items, timeout)

    ma_solver = MemeticAlgorithm(weights_cut, values_cut, max_weight)
    ma_result = ma_solver.solve(timeout=timeout)
    fill_data(data, ma_result, number_of_items, timeout)

    mahls_solver = MemeticAlgorithmHybridLocalSearch(weights_cut, values_cut, max_weight)
    mahls_result = mahls_solver.solve(timeout=timeout)
    fill_data(data, mahls_result, number_of_items, timeout)

    es_solver = EvolutionStrategy(weights_cut, values_cut, max_weight)
    es_result = es_solver.solve(timeout=timeout)
    fill_data(data, es_result, number_of_items, timeout)

    de_solver = DifferentialEvolution(weights_cut, values_cut, max_weight)
    de_result = de_solver.solve(timeout=timeout)
    fill_data(data, de_result, number_of_items, timeout)

df = pd.DataFrame(data)

df.to_csv(f"table_{timestamp}.csv", index=False)
