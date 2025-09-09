# Multi-Objectiveqq Knapsack-like Portfolio Optimization
**Authors:** N. Penchev and A. Marchev jr.

## Introduction to the Problem
In finance, portfolio optimization aims to balance risk and return. For large portfolios with many assets, traditional methods (e.g., exhaustive search) become computationally expensive.
This report presents an experimental comparison of algorithms for a multi-objective knapsack-like portfolio optimization problem. The goal is to select a portfolio of assets that simultaneously maximizes the expected 'Value' (return) and minimizes the associated 'Risk'. These two objectives are often in conflict, requiring a trade-off. Multi-objective optimization is well-suited for this problem as it does not require a single, arbitrary weighting of risk versus return, but instead identifies a set of optimal trade-off solutions, known as the Pareto front.

## What is the Knapsack Problem?
A classic combinatorial optimization problem
- You have a knapsack with a capacity W.
- There are N items, each with value and weight
- The Goal is to Maximize total value without exceeding the knapsack capacity. 

<img src="knapsack.png" alt="Knapsack" width="300"/>


In a portfolio context:

- Capacity -> budget/limit

- Item -> asset (e.g., stock, bond, etc.)

- Item(Weight) -> investment cost

- Item(Value) -> expected return

- Item(Risk) -> Volatility/Uncertainty

<img src="portfolio.png" alt="Portfolio" width="300"/>


## Evolutionary Algorithms (EA)
- Key Idea: Inspired by natural selection.
- Main Steps:
  - Initialization of a population (random solutions).
  - Evaluation (Fitness) of each solution.
  - Selection of the best solutions.
  - Recombination (Crossover) and Mutation to create new solutions.
  - Repeat until stopping criteria are met.

<img src="darwin_2.png" alt="Darwin" width="300"/>


## Algorithms

Two algorithms were compared in this study:

- **NSGA-II (Non-dominated Sorting Genetic Algorithm II):** A widely-used evolutionary algorithm for multi-objective optimization. It employs mechanisms of selection, crossover, and mutation to iteratively evolve a population of solutions toward the true Pareto front. Its key features include a fast non-dominated sorting procedure and a crowding distance mechanism to maintain diversity among solutions.

- **Random Search:** This method serves as a baseline for comparison. It generates solutions randomly within the search space for a fixed time budget equivalent to that of NSGA-II. This helps to assess whether the sophisticated mechanisms of NSGA-II provide a significant advantage over simple, undirected search.

## Data Description

The experiments were conducted on a set of synthetically generated problem instances with varying sizes, where N represents the number of available assets. The problem sizes are N ∈ { ['10', '100', '1000', '10000'] }. To ensure statistical robustness, each experimental run was repeated across multiple random seeds.

The performance of each algorithm was evaluated using several standard metrics for multi-objective optimization:
- **Pareto Fronts:** The set of non-dominated solutions found by each algorithm.
- **Hypervolume (HV):** Measures the volume of the objective space dominated by the obtained front. Higher is better.
- **Inverted Generational Distance Plus (IGD+):** Measures the average distance from points in a true or reference Pareto front to the obtained front. Lower is better.
- **Number of Non-Dominated Points (|ND|):** The number of solutions in the final Pareto front.

## 1. Setup

This report summarizes the performance of multi-objective optimization methods.

- **Problem Sizes (N):** ['10', '100', '1000', '10000']
- **Methods:** ['NSGA2', 'RANDOM']
- **Seeds:** [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
- **Total Runs:** 240
- **Time Caps (s):** [25, 100, 175, 250]

**Objective Interpretation:**
- **Value:** `-f1` (higher is better)
- **Risk:** `f2` (lower is better)

The goal is to find solutions that maximize Value while minimizing Risk, representing a classic Pareto trade-off.

## 2. Performance Metrics

Metrics are aggregated across seeds (mean ± 95% CI).

- **HV (Hypervolume) ↑:** Measures the volume of the dominated portion of the objective space. Higher is better.
- **IGD+ (Inverted Generational Distance Plus) ↓:** Measures the average distance from each point in the reference front to the obtained front. Lower is better.
- **|ND| (Number of Non-Dominated Points) ↑:** The number of points in the final Pareto front. Higher is generally better, indicating more choices.

### HV (↑) mean ± 95% CI

| N | NSGA2 | RANDOM |
|---|---|---|
| 10 | 626.327 (626.327 - 626.327) | 626.327 (626.327 - 626.327) |
| 100 | 88210.700 (88169.537 - 88251.864) | 57972.241 (57579.739 - 58364.743) |
| 1000 | 3672542.847 (3669368.677 - 3675717.017) | 892217.123 (879423.465 - 905010.781) |
| 10000 | 2779940.308 (2629782.103 - 2930098.513) | 137958.226 (128440.809 - 147475.642) |

### IGD+ (↓) mean ± 95% CI

| N | NSGA2 | RANDOM |
|---|---|---|
| 10 | 0.000 (0.000 - 0.000) | 0.000 (0.000 - 0.000) |
| 100 | 0.327 (0.319 - 0.335) | 39.023 (38.445 - 39.600) |
| 1000 | 7.114 (6.791 - 7.438) | 558.306 (555.985 - 560.627) |
| 10000 | 149.190 (129.544 - 168.835) | 1453.729 (1449.668 - 1457.789) |

### |ND| (↑) mean ± 95% CI

| N | NSGA2 | RANDOM |
|---|---|---|
| 10 | 14 (14 - 14) | 14 (14 - 14) |
| 100 | 200 (200 - 200) | 52 (50 - 55) |
| 1000 | 200 (200 - 200) | 54 (52 - 57) |
| 10000 | 200 (200 - 200) | 47 (45 - 49) |

## 3. Pareto Fronts

Scatter plots of **Risk vs. Value**.  
The ideal region is the top-left (low risk, high value).  
NSGA-II is expected to produce fronts that dominate the random search, demonstrating its effectiveness.

### N = 10
![Pareto Front for N=10](pareto_N10.png)
### N = 100
![Pareto Front for N=100](pareto_N100.png)
### N = 1000
![Pareto Front for N=1000](pareto_N1000.png)
### N = 10000
![Pareto Front for N=10000](pareto_N10000.png)

## 4. Combined Boxplots (HV / IGD / |ND|)

### N = 10
![N=10 Combined](box_multi_N10.png)
### N = 100
![N=100 Combined](box_multi_N100.png)
### N = 1000
![N=1000 Combined](box_multi_N1000.png)
### N = 10000
![N=10000 Combined](box_multi_N10000.png)

## 5. Runtime Overview

![Runtime Overview](runtime.png)

