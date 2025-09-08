import time
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def random_feasible_timeboxed(N, W, V, R, capacity, time_cap_s, seed, batch_size=2048, max_archive_size=10000):
    """
    Generates and repairs random solutions until time cap, maintaining only the
    non-dominated front to conserve memory.
    """
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    n_samples_evaluated = 0

    # Archive for non-dominated solutions and their objectives
    nd_solutions = np.empty((0, N), dtype=np.uint8)
    nd_objectives = np.empty((0, 2), dtype=np.float32)

    while time.perf_counter() - start_time < time_cap_s:
        # Generate a batch of solutions
        solutions = rng.integers(2, size=(batch_size, N), dtype=np.uint8)
        n_samples_evaluated += batch_size

        # Repair and evaluate objectives for the batch
        objectives = np.empty((batch_size, 2), dtype=np.float32)
        for i, sol in enumerate(solutions):
            weight = np.dot(W, sol)
            while weight > capacity:
                item_indices = np.where(sol == 1)[0]
                if not item_indices.size:
                    break
                remove_idx = rng.choice(item_indices)
                sol[remove_idx] = 0
                weight = np.dot(W, sol)
            
            objectives[i, 0] = -np.dot(V, sol)
            objectives[i, 1] = np.dot(R, sol)

        # Combine new results with the current non-dominated front
        combined_solutions = np.vstack([nd_solutions, solutions])
        combined_objectives = np.vstack([nd_objectives, objectives])

        # Find the new non-dominated front
        nd_indices = NonDominatedSorting().do(combined_objectives, only_non_dominated_front=True)
        
        nd_solutions = combined_solutions[nd_indices]
        nd_objectives = combined_objectives[nd_indices]

        # Optional: Thin the archive if it grows too large
        if len(nd_solutions) > max_archive_size:
            # Simple thinning: randomly sample a subset
            indices_to_keep = rng.choice(len(nd_solutions), size=max_archive_size, replace=False)
            nd_solutions = nd_solutions[indices_to_keep]
            nd_objectives = nd_objectives[indices_to_keep]

    return nd_solutions, nd_objectives, n_samples_evaluated