import signal
import functools
import time


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def with_timeout(timeout, func, cname, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = func(*args, **kwargs)
        signal.setitimer(signal.ITIMER_REAL, 0)  # cancel timer
        return result
    except TimeoutException:
        print(f"Timeout: {cname}")
        return "TIMEOUT"


# Decorator with integrated timeout handling
def evolutionary_solver(solve_func):
    @functools.wraps(solve_func)
    def wrapper(self, *args, **kwargs):
        timeout = kwargs.pop("timeout", None)
        start_time = time.time()

        def runner():
            return solve_func(self, *args, **kwargs)

        if timeout:
            result = with_timeout(timeout, runner, self.__class__.__name__)
            if isinstance(result, str) and result == "TIMEOUT":
                population = self.population if hasattr(self, "population") else []
            else:
                population = result
        else:
            population = runner()

        end_time = time.time()
        total_time = end_time - start_time

        if len(population) > 0:
            best_solution = max(population, key=self.fitness)
            best_value = self.fitness(best_solution)
        else:
            best_solution = None
            best_value = 0

        print(f"{self.__class__.__name__.upper()} Final Best value = {best_value:.2f}, total time: {total_time:.4f}s")
        return [self.__class__.__name__.replace('_', ' ').upper(), best_value, total_time]

    return wrapper
