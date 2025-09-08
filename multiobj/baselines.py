
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def random_feasible_baseline(N, W, V, R, capacity, n_samples, seed):
    """Generates random solutions, repairs them, and returns the non-dominated front."""
    rng = np.random.default_rng(seed)
    solutions = rng.integers(2, size=(n_samples, N))
    feasible_solutions = []

    for sol in solutions:
        weight = np.dot(W, sol)
        while weight > capacity:
            item_indices = np.where(sol == 1)[0]
            if not len(item_indices):
                break
            remove_idx = rng.choice(item_indices)
            sol[remove_idx] = 0
            weight = np.dot(W, sol)
        feasible_solutions.append(sol)

    feasible_solutions = np.array(feasible_solutions)
    objectives = np.array([
        [-np.dot(V, sol), np.dot(R, sol)] for sol in feasible_solutions
    ])

    # Get the non-dominated front
    nds = NonDominatedSorting()
    nd_indices = nds.do(objectives, only_non_dominated_front=True)

    return feasible_solutions[nd_indices], objectives[nd_indices]
