import numpy as np
from newton_method_main import newton_solver

# Define the system of nonlinear equations
def f(x):
    return np.array([x[0]**2 + x[1]**2 - 1, x[0]**2 - x[1]])

# Define the Jacobian of the system
def J(x):
    return np.array([[2*x[0], 2*x[1]], [2*x[0], -1]])

# Initial guess
x0 = np.array([0.5, 0.5])

# Solve using Newton's method
solution = newton_solver(f, J, x0)
print("Solution:", solution)
