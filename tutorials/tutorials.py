import numpy as np
from newtons_method_functions import newton_solver # again how do i specify the path???
from sympy import symbols, Function, lambdify
import sympy as sp

# ========================= TUTORIAL 1: Simple system of nonlinear equations =========================
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
print("Solution1:", solution)


# ==================== TUTORIAL 2: System of nonlinear equations with exponential ====================

def f(x):
    return np.array([4*x[0]**3 + x[1]**2 - 3, np.exp(x[0]) - x[1] - 2])

def J(x):
    return np.array([[12*x[0]**2, 2*x[1]], [np.exp(x[0]), -1]])

x0 = np.array([10.0, 10.0])

solution = newton_solver(f, J, x0)
print("Solution2:", solution)

# ================================ TUTORIAL 3: One polynomial equation ================================
def f(x):
    return np.array([x[0]**5 - 2*x[0]**3 + 7*x[0] - 11])

def J(x):
    return np.array([[5*x[0]**4 - 6*x[0]**2 + 7]])

x0 = np.array([2.0])

soultion = newton_solver(f, J, x0)
print("Solution3:", solution)


# ============================ TUTORIAL 4: System of 3 nonlinear equations ============================

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

# Initial guess
x0 = np.array([1.0, 1.0, 1.0])

# Solve using Newton's method
solution = newton_solver(f, J, x0)
print("Solution4:", solution)

# =========================== TUTORIAL 5: Two different springs 2 dof system ===========================
def f(x):
    F = 0.1
    K_AB = 100
    K_BC = 200
    l = 10
    u = x[0]
    v = x[1]

    term1 = -(l + u) / np.sqrt(v**2 + (l + u)**2) * K_AB * (np.sqrt(v**2 + (l + u)**2) - l)
    term2 = (l - u) / np.sqrt(v**2 + (l - u)**2) * K_BC * (np.sqrt(v**2 + (l - u)**2) - l)

    Fx = term1 + term2

    term3 = v / np.sqrt(v**2 + (l + u)**2) * K_AB * (np.sqrt(v**2 + (l + u)**2) - l)
    term4 = v / np.sqrt(v**2 + (l - u)**2) * K_BC * (np.sqrt(v**2 + (l - u)**2) - l)

    Fy = term3 + term4 - F
    return np.array([Fx, Fy])

def J(x):
    # Define symbolic variables
    u_sym, v_sym = sp.symbols('u v')
    l, K_AB, K_BC, F = 10, 100, 200, 0.1

    term1 = -(l + u_sym) / sp.sqrt(v_sym**2 + (l + u_sym)**2) * K_AB * (sp.sqrt(v_sym**2 + (l + u_sym)**2) - l)
    term2 = (l - u_sym) / sp.sqrt(v_sym**2 + (l - u_sym)**2) * K_BC * (sp.sqrt(v_sym**2 + (l - u_sym)**2) - l)

    Fx = term1 + term2

    term3 = v_sym / sp.sqrt(v_sym**2 + (l + u_sym)**2) * K_AB * (sp.sqrt(v_sym**2 + (l + u_sym)**2) - l)
    term4 = v_sym / sp.sqrt(v_sym**2 + (l - u_sym)**2) * K_BC * (sp.sqrt(v_sym**2 + (l - u_sym)**2) - l)

    Fy = term3 + term4 - F

    # Compute Jacobian symbolically
    J11 = sp.diff(Fx, u_sym)
    J12 = sp.diff(Fx, v_sym)
    J21 = sp.diff(Fy, u_sym)
    J22 = sp.diff(Fy, v_sym)

    # Convert symbolic expressions to numerical functions
    J11_func = sp.lambdify((u_sym, v_sym), J11, 'numpy')
    J12_func = sp.lambdify((u_sym, v_sym), J12, 'numpy')
    J21_func = sp.lambdify((u_sym, v_sym), J21, 'numpy')
    J22_func = sp.lambdify((u_sym, v_sym), J22, 'numpy')

    # Evaluate at the given x
    u_val, v_val = x
    return np.array([
        [J11_func(u_val, v_val), J12_func(u_val, v_val)],
        [J21_func(u_val, v_val), J22_func(u_val, v_val)]
    ])

x0 = np.array([0.1, 1.0])
solution5 = newton_solver(f, J, x0)
print("Solution5:", solution5)



# ============================ TUTORIAL 6: Three spring two masses system equilibrium ============================
K1 = 10
K2 = 15
K3 = 8
m1 = 3
m2 = 4
g = 10
def f(x):
    f1 = -K1 * x[0] + K2 * (x[1] - x[0]) + m1 * g
    f2 = -K2 * (x[1] - x[0]) - K3 * x[1] + m2 * g
    return ([f1, f2])

def J(x):
    J11 = -K1 -K2
    J12 = K2
    J21 = K2
    J22 = -K2 -K3
    return ([J11, J12], [J21, J22])

X0 = [10.0, -1.0]
solution6 = newton_solver(f, J, x0)
print("Solution6:", solution6)