from scipy.optimize import fsolve

# Define the functions representing the equations
def equations(vars):
    v0, v1, v2 = vars
    eq1 = v0 - 0.8 * (1 + 0.9 * v0) - 0.2 * (-1 + 0.9 * v0)
    eq2 = v1 - 0.8 * (1 + 0.9 * v1) - 0.2 * (-1 + 0.9 * v1)
    eq3 = v2 - 0.8 * (1 + 0.9 * v2) - 0.2 * (-1 + 0.9 * v2)
    return [eq1, eq2, eq3]

# Solve the equations using fsolve (initial guesses: 0)
solution = fsolve(equations, [0, 0, 0])

print(f"Solution: {solution}")
