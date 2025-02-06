import numpy as np

def evaluate_function(f, x):
    return np.array(f(x))


def evaluate_jacobian(J, x):
    return np.array(J(x))


def converged(Fx, delta_x, tol):
    norm_Fx = np.linalg.norm(Fx)
    norm_delta_x = np.linalg.norm(delta_x)

    if norm_delta_x < tol and norm_Fx > tol: # For stricter control use np.max(np.abs(Fx)) < tol
        raise ValueError("Newton's method has stagnated: The change in solution is small, but the function value is still large. This may indicate a near-singular region or a local minimum.")
    
    if norm_Fx < tol and norm_delta_x > tol:
        raise ValueError("Newton's method is fluctuating: The function has converged to a small value, but the solution is still fluctuating and not stabilizing.")
    
    return norm_Fx < tol and norm_delta_x < tol


def solve_linear_system(Jx, Fx):
    try:
        delta_x=np.linalg.solve(Jx, -Fx)
    except np.linalg.LinAlgError:
        raise ValueError("Jacobian is singular or near singular: Newton's method failed.")
    return delta_x


def update_solution(x, delta_x):
    return x + delta_x


# -------------------- Main function for Newton's method --------------------

def newton_solver(f, J, x0, tol=1e-6, max_iter=100):

    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        Fx = evaluate_function(f, x)
        Jx = evaluate_jacobian(J, x)
        delta_x = solve_linear_system(Jx, Fx)
        x = update_solution(x, delta_x)
        
        if converged(Fx, delta_x, tol):
            print(f"Converged in {i+1} iterations.")
            return x
    
    raise ValueError("Newton's method did not converge within the maximum number of iterations.")