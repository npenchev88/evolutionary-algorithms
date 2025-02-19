# Knapsack Portfolio Optimization Using Evolutionary Algorithms

**Authors**:  
Angel Marchev Jr.
Nikolay Penchev

---

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Data](#data)
- [Methods](#methods)
  - [Baseline Methods](#baseline-methods)
  - [Evolutionary Algorithms](#evolutionary-algorithms)
  - [Timeout Constraints](#timeout-constraints)
  - [Algorithm Execution and Data Collection](#algorithm-execution-and-data-collection)
- [Empirical Results](#empirical-results)
  - [Algorithms and Timeout Constraints](#algorithms-and-timeout-constraints)
  - [Performance Comparison](#performance-comparison)
- [Conclusion](#conclusion)
- [References](#references)

---

## Abstract
This study examines optimizing asset allocation within a portfolio to achieve a cost-efficient solution. The optimization process focuses on two primary characteristics: the weight of each asset, representing the investment cost, and the value of each asset, representing the expected return. The analysis compares several evolutionary algorithms, including genetic algorithms, evolution strategies, and memetic algorithms. A dynamic programming approach and a random chromosome strategy (base) are also employed as baseline solutions. The performance of these algorithms is evaluated using datasets with varying sizes (10, 100, 1,000, and 10,000 assets).

**Keywords**: Evolutionary algorithms, genetic algorithms, evolution strategy, memetic algorithms, dynamic programming, differential evolution, knapsack problem, portfolio optimization

---

## Introduction
Portfolio optimization is a fundamental challenge in finance, focusing on allocating assets in a manner that balances risk and return. Traditional optimization methods often face difficulties dealing with large-scale, complex portfolios with multiple constraints. By framing the problem as a knapsack problem—where each asset has a weight (cost) and a value (expected return)—we explore evolutionary algorithms as a viable solution approach.

We compare several techniques:
- **Genetic Algorithm (GA)**
- **Memetic Algorithm (MA)**
- **Memetic Algorithm with Hybrid Local Search (MAHLS)**
- **Evolution Strategy (ES)**
- **Differential Evolution (DE)**
- **Dynamic Programming (DP)** (as an optimal but sometimes computationally expensive baseline)
- **Random Chromosome Strategy (BASE)** (as a minimal-baseline approach)

All algorithms are tested on datasets of varying sizes to assess their scalability and performance.

---

## Data
We generate asset weights and values using the `np.random.normal` function from the NumPy library:

- **Weights**: Mean = 10, Standard Deviation = 3 (`np.random.normal(10, 3, 10000)`)
- **Values**: Mean = 13, Standard Deviation = 3 (`np.random.normal(13, 3, 10000)`)

### Why `np.random.normal`?
- **Realistic Data Distribution**: The normal (Gaussian) distribution is commonly used to model financial data.
- **Controlled Variability**: Means and standard deviations can be precisely set.
- **Scalability**: Efficient generation of large datasets (up to 10,000 assets).
- **Reproducibility**: Seeding ensures experimental reproducibility.

---

## Methods

### Baseline Methods
1. **Random Chromosome Strategy (BASE)**  
   - Generates a population of binary vectors (chromosomes) without crossover or mutation.  
   - After 100 random solutions, the best chromosome (by fitness) is selected.

2. **Dynamic Programming (DP)**  
   - A classical knapsack approach that guarantees the optimal solution if it completes.  
   - Builds a table of maximum achievable values for each weight limit.  
   - Prone to timeouts on large datasets due to high computational costs.

### Evolutionary Algorithms

1. **Genetic Algorithm (GA)**
   - **Initialization**: Random binary vectors representing asset inclusion/exclusion.  
   - **Selection**: Tournament selection to preserve diversity.  
   - **Crossover**: Single-point crossover to produce offspring.  
   - **Mutation**: Bit-flip mutation at a specified rate.

2. **Evolution Strategy (ES)**
   - **Focus**: Mutation (small random perturbations) and less emphasis on crossover.  
   - **Sigma Parameter**: Controls mutation magnitude.  
   - **Selection**: Offspring replace parents if fitness improves.

3. **Memetic Algorithm (MA)**
   - Extends GA by adding **local search** (bit-flip improvements) after crossover/mutation.  
   - Balances global exploration (GA) with local exploitation (bit-flip search).

4. **Memetic Algorithm with Hybrid Local Search (MAHLS)**
   - Similar to MA, but local search alternates:
     - **Hill Climbing** in early generations.  
     - **Simulated Annealing** in later generations (controlled random acceptance of worse moves).  

5. **Differential Evolution (DE)**
   - **Population-Based**: Maintains a population of solutions.  
   - **Mutation (Differential Mutation)**: Creates a donor vector by adding a weighted difference of two random solutions to a third.  
   - **Crossover**: Combines target and donor vectors.  
   - **Selection**: Offspring replaces the target if it has better fitness.

### Timeout Constraints
To ensure fairness, each dataset size has a maximum runtime:

| Dataset Size | Timeout (seconds) |
|--------------|-------------------|
| 10 items     | 50                |
| 100 items    | 200               |
| 1,000 items  | 350               |
| 10,000 items | 500               |

If an algorithm hits the timeout, the best-known solution at that moment is recorded.

### Algorithm Execution and Data Collection
- Each algorithm is run under the specified time limit for each dataset size.  
- The best fitness score and execution time are saved.  
- Results are compiled into a DataFrame and exported to CSV for analysis.

---

## Empirical Results

### Algorithms and Timeout Constraints
Algorithms:
- **Dynamic Programming (DP)**
- **Base Algorithm (BASE)**
- **Genetic Algorithm (GA)**
- **Memetic Algorithm (MA)**
- **Memetic Algorithm with Hybrid Local Search (MAHLS)**
- **Evolution Strategy (ES)**
- **Differential Evolution (DE)**

Timeout settings are shown above.

### Performance Comparison
1. **Best Fitness Score**  
   - DP often finds the optimal solution for smaller datasets.  
   - For larger datasets (1,000 or 10,000 items), DP may take too long or reach the timeout.  
   - GA and ES often achieve high-quality solutions within reasonable time.

2. **Execution Time**  
   - DP is exceptionally fast for small problems but may exceed time limits on larger datasets.  
   - BASE is trivial and thus very fast but offers weaker solutions.  
   - GA, ES, and Memetic approaches strike a balance between execution speed and solution quality.

3. **Combined Results: Best Fitness / Execution Time**  
   - Illustrates efficiency per second.  
   - DP dominates on small datasets.  
   - For large datasets, ES, GA, or simpler baselines might yield higher fitness/time ratios due to time constraints.

A sample plot (log-log scale) of **Best Fitness Score vs. Execution Time** illustrates:
- **Efficient Frontier**: GA and ES often lead this frontier.  
- **DP’s Limitations**: Grows expensive computationally for large datasets.  
- **Variability**: Some algorithms show outlier results, emphasizing the importance of tuning and selecting the right method for the dataset size.

---

## Conclusion
This study highlights the trade-offs between **Dynamic Programming** (optimal, but less scalable) and **Evolutionary Algorithms** (robust, scalable, near-optimal solutions). For small datasets, **DP** is recommended due to guaranteed optimality. However, for larger, more complex portfolios, **Evolutionary Algorithms** (especially **Genetic Algorithm** or **Memetic** variants) are more practical. 

Future work may involve:
- Hybridizing DP with evolutionary methods.
- Refining local search steps (e.g., using adaptive or parameter tuning).
- Applying real-world constraints (risk measures, transaction costs) to validate performance further.

---

## References
1. Penchev, N. (2024). *Evolutionary Algorithms [GitHub repository]*. Retrieved from [https://github.com/npenchev88/evolutionary-algorithms](https://github.com/npenchev88/evolutionary-algorithms)  
2. NumPy Developers. (2021). *NumPy: The fundamental package for scientific computing with Python.* Retrieved from [https://numpy.org/](https://numpy.org/)  
3. Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing (2nd ed.)*. Springer.

---

**© 2025 Nikolay Penchev**  
Feel free to modify and distribute as permitted under the repository’s license.
