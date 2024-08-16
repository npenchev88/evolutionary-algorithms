def print_best_solution(cname, best_solution, fitness, ttime):
    result = max(best_solution, key=fitness)

    print(f"{cname} Final Best value = {self.fitness(result)}, Solution = N/A, Total time: {ttime}")
    return [cname, fitness(result), ttime]
