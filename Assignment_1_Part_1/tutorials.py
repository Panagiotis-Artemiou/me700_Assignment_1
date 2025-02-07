import numpy as np
from newtons_method_functions import newton_solver
from sympy import symbols, Function, lambdify

# TUTORIAL 1: Simple system of nonlinear equations
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


# TUTORIAL 2: System of nonlinear equations with exponential

def f(x):
    return np.array([4*x[0]**3 + x[1]**2 - 3, np.exp(x[0]) - x[1] - 2])

def J(x):
    return np.array([[12*x[0]**2, 2*x[1]], [np.exp(x[0]), -1]])

x0 = np.array([10.0, 10.0])

solution = newton_solver(f, J, x0)
print("Solution:", solution)

# TUTORIAL 3: One polynomial equation
def f(x):
    return np.array([x[0]**5 - 2*x[0]**3 + 7*x[0] - 11])

def J(x):
    return np.array([[5*x[0]**4 - 6*x[0]**2 + 7]])

x0 = np.array([2.0])

solution = newton_solver(f, J, x0)
print("Solution:", solution)


# TUTORIAL 4: System of 3 nonlinear equations

# Define the system of nonlinear equations
def f(x):
    return np.array([
        np.exp(x[0]) + np.sin(x[1]),              # f1(x, y, z)
        x[0] + x[1] + x[2] - 3,                  # f2(x, y, z)
        -x[0]**2 - 3*x[1] + 2*x[2] - 4            # f3(x, y, z)
    ])

# Define the Jacobian of the system
def J(x):
    return np.array([
        [np.exp(x[0]), np.cos(x[1]), 0],          # df1/dx, df1/dy, df1/dz
        [1, 1, 1],                                # df2/dx, df2/dy, df2/dz
        [-2*x[0], -3, 2]                          # df3/dx, df3/dy, df3/dz
    ])

# Define the Newton's method solver
def newton_solver(f, J, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        Fx = f(x)
        Jx = J(x)
        delta_x = np.linalg.solve(Jx, -Fx)
        x = x + delta_x
        
        if np.linalg.norm(delta_x) < tol and np.linalg.norm(Fx) < tol:
            print(f"Converged in {i+1} iterations.")
            return x
    raise ValueError("Newton's method did not converge within the maximum number of iterations.")

# Initial guess
x0 = np.array([1.0, 1.0, 1.0])

# Solve using Newton's method
solution = newton_solver(f, J, x0)
print("Solution:", solution)


# TUTORIAL 5
