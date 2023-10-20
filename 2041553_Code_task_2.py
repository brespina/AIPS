"""
Brandon Espina
09/29/2023
COSC_4368
Dr. Lin
"""

import random


# Define the objective function f(x, y)
def f(x, y):
    if -4.2 <= x <= 4.2 and -4.2 <= y <= 4.2:
        return (1.5 + x + x * y) ** 2 + (2.25 + x - x * y * y) ** 2 + (2.625 + x - x * y * y * y) ** 2
    else:
        return 0


# Randomized Hill Climbing (RHC) function
def RHC(sp, p, z, seed):
    random.seed(seed)
    current_solution = sp
    current_value = f(*current_solution)  # f(x, y)
    num_generated_solutions = 1

    while True:
        neighbors = []

        for i in range(p):
            z1 = random.uniform(-z, z)
            z2 = random.uniform(-z, z)
            neighbor = (current_solution[0] + z1, current_solution[1] + z2)  # x + z1, y+z2
            neighbors.append(neighbor)

        # this syntax computes the max on neighbors and returns the x that generates the max number of
        best_neighbor = max(neighbors, key=lambda x: f(*x))
        # print(best_neighbor)
        BN_value = f(*best_neighbor)

        if BN_value <= current_value:
            return current_solution, current_value, num_generated_solutions

        current_solution = best_neighbor
        current_value = BN_value
        num_generated_solutions += p


# Define the parameter sets
arguments = [
    # 65 and 0.2 and 42
    ((2, 2), 65, 0.2, 42),
    ((1, 4), 65, 0.2, 42),
    ((-2, -3), 65, 0.2, 42),
    ((1, -2), 65, 0.2, 42),

    # 65 and 0.2 and 43
    ((2, 2), 65, 0.2, 43),
    ((1, 4), 65, 0.2, 43),
    ((-2, -3), 65, 0.2, 43),
    ((1, -2), 65, 0.2, 43),

    # 400 and 0.2 and 42
    ((2, 2), 400, 0.2, 42),
    ((1, 4), 400, 0.2, 42),
    ((-2, -3), 400, 0.2, 42),
    ((1, -2), 400, 0.2, 42),

    # 400 and 0.2 and 43
    ((2, 2), 400, 0.2, 43),
    ((1, 4), 400, 0.2, 43),
    ((-2, -3), 400, 0.2, 43),
    ((1, -2), 400, 0.2, 43),

    # 65 and 0.01 and 42
    ((2, 2), 65, 0.01, 42),
    ((1, 4), 65, 0.01, 42),
    ((-2, -3), 65, 0.01, 42),
    ((1, -2), 65, 0.01, 42),

    # 65 and 0.01 and 43
    ((2, 2), 65, 0.01, 43),
    ((1, 4), 65, 0.01, 43),
    ((-2, -3), 65, 0.01, 43),
    ((1, -2), 65, 0.01, 43),

    # 400 and 0.01 and 42
    ((2, 2), 400, 0.01, 42),
    ((1, 4), 400, 0.01, 42),
    ((-2, -3), 400, 0.01, 42),
    ((1, -2), 400, 0.01, 42),

    # 400 and 0.01 and 43
    ((2, 2), 400, 0.01, 43),
    ((1, 4), 400, 0.01, 43),
    ((-2, -3), 400, 0.01, 43),
    ((1, -2), 400, 0.01, 43),

    # p = 10000
    ((1, -2), 10000, 0.01, 43)
]
count = 1
# Run RHC for each parameter set
for arg in arguments:
    sp, p, z, seed = arg
    print(count)
    count += 1
    result = RHC(sp, p, z, seed)
    print(f"Starting point: {sp}, p: {p}, z: {z}, seed: {seed}")
    print(f"Optimal solution: {result[0]}, Optimal value: {result[1]}, Number of generated solutions: {result[2]}\n")
