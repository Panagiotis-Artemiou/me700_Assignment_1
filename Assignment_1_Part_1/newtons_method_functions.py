import numpy as np

def newtons_method(f, jacobian, x0, tol=1e-6, max_iter=100):

    x = np.array(x0, dtype=float)
    
    for _ in range(max_iter):
        fx = f(x)
        Jx = jacobian(x)
        
        # Check if the Jacobian is singular (det(J) = 0)
        if np.linalg.det(Jx) == 0:
            raise ValueError("Jacobian is singular, cannot proceed")
        
        # Compute the next guess
        dx = np.linalg.solve(Jx, fx)
        x_new = x - dx
        
        # Check for convergence
        if np.linalg.norm(dx) < tol:
            return x_new
        
        x = x_new
    
    raise ValueError("Failed to converge within the maximum number of iterations")
