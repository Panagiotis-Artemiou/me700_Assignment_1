import pytest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import src.newtonsmethod.newtonsmethod_functions as nmf
from numpy.testing import assert_array_equal
from src.newtonsmethod import newtonsmethod_functions as nmf #from src/newtonsmethod_functions import ...???

# --- Test Function Evaluations ---
def test_evaluate_function():
    # Define a simple function for testing
    def f(x):
        return x**2 - 4

    x = np.array([2.0])
    result = nmf.evaluate_function(f, x)
    expected = np.array([0.0])
    
    assert_array_equal(result, expected)

def test_evaluate_jacobian():
    # Define a simple Jacobian for testing
    def J(x):
        return 2*x

    x = np.array([2.0])
    result = nmf.evaluate_jacobian(J, x)
    expected = np.array([4.0])
    
    assert_array_equal(result, expected)


# --- Test Convergence Function ---
def test_converged_success():
    Fx = np.array([1e-4])
    delta_x = np.array([1e-4])
    tol = 1e-3
    
    # Should return True, as  Fx and delta_x is below the tolerance
    assert nmf.converged(Fx, delta_x, tol)


def test_converged_stagnation():
    Fx = np.array([10.0])
    delta_x = np.array([0.00001])
    tol = 1e-3
    
    # Should raise an error, since Fx > tol but delta_x is small
    with pytest.raises(ValueError, match="Newton's method has stagnated: The change in solution is small, but the function value is still large. This may indicate a near-singular region or a local minimum."):
        nmf.converged(Fx, delta_x, tol)


def test_converged_fluctuation():
    Fx = np.array([0.00001])
    delta_x = np.array([0.1])
    tol = 1e-3
    
    # Should raise an error, since delta_x > tol but Fx is small
    with pytest.raises(ValueError, match="Newton's method is fluctuating: The function has converged to a small value, but the solution is still fluctuating and not stabilizing."):
        nmf.converged(Fx, delta_x, tol)


# --- Test Solve Linear System ---
def test_solve_linear_system():
    # Test a simple system: Jx * delta_x = -Fx
    Jx = np.array([[2.0, 0], [0, 2.0]])  # 2x2 Jacobian
    Fx = np.array([4.0, 6.0])
    
    delta_x = nmf.solve_linear_system(Jx, Fx)
    expected_delta_x = np.array([-2.0, -3.0])
    
    assert_array_equal(delta_x, expected_delta_x)

def test_solve_linear_system_singular():
    # Test a singular Jacobian
    Jx = np.array([[1.0, 1.0], [1.0, 1.0]])  # Singular matrix
    Fx = np.array([2.0, 2.0])
    
    with pytest.raises(ValueError, match="Jacobian is singular or near singular: Newton's method failed."):
        nmf.solve_linear_system(Jx, Fx)


# --- Test Update Solution ---
def test_update_solution():
    x = np.array([1.0, 2.0])
    delta_x = np.array([-0.5, 0.5])
    
    updated_x = nmf.update_solution(x, delta_x)
    expected = np.array([0.5, 2.5])
    
    assert_array_equal(updated_x, expected)


# --- Test Newton Solver ---
def test_newton_solver_convergence():
    def f(x):
        return x**2 - 4
    
    def J(x):
        return np.array([2*x])
    
    x0 = np.array([3.0])
    tol = 1e-6
    
    result = nmf.newton_solver(f, J, x0, tol)
    expected = np.array([2.0])
    
    assert np.linalg.norm(result - expected) < tol


def test_newton_solver_stagnation():
    def f(x):
        return x**2 - 4
    
    def J(x):
        return np.array([2*x])
    
    x0 = np.array([1000.0])
    tol = 1e-6
    
    with pytest.raises(ValueError, match="Newton's method has stagnated: The change in solution is small, but the function value is still large. This may indicate a near-singular region or a local minimum."):
        nmf.newton_solver(f, J, x0, tol)

'''
Can'f find a case where the solution is fluctuating :'(
def test_newton_solver_fluctuation():
    def f(x):
        return np.array([np.sin(x[0])])
    
    def J(x):
        return np.array([[np.cos(x[0])]])
    
    x0 = np.array([3.0])
    tol = 1e-6
    
    with pytest.raises(ValueError, match="Newton's method is fluctuating: The function has converged to a small value, but the solution is still fluctuating and not stabilizing."):
        nmf.newton_solver(f, J, x0, tol)
'''

def test_newton_solver_max_iterations():
    def f(x):
        return np.array([x[0]**2 - 4])
    
    def J(x):
        return np.array([[2*x[0]]])
    
    x0 = np.array([10.0])
    tol = 1e-6
    
    with pytest.raises(ValueError, match="Newton's method did not converge within the maximum number of iterations."):
        nmf.newton_solver(f, J, x0, tol, max_iter=1)
