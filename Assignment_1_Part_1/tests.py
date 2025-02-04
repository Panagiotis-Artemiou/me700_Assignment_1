import numpy as np
import pytest

def test_newtons_method_multidimensional_convergence():
    # Test for a system of nonlinear equations
    # f(x, y) = [x^2 + y^2 - 1, x^2 - y - 1]
    def f(x):
        return np.array([x[0]**2 + x[1]**2 - 1, x[0]**2 - x[1] - 1])

    def jacobian(x):
        return np.array([[2*x[0], 2*x[1]], [2*x[0], -1]])

    initial_guess = np.array([1.0, 1.0])
    result = newtons_method(f, jacobian, initial_guess)
    assert np.allclose(result, [0.70710678, 0.70710678], atol=1e-6), "Test failed for convergence"

def test_newtons_method_multidimensional_non_convergence():
    # f(x, y) = [x^2 + y^2 - 1, x^3 - 2x]
    def f(x):
        return np.array([x[0]**2 + x[1]**2 - 1, x[0]**3 - 2*x[0]])

    def jacobian(x):
        return np.array([[2*x[0], 2*x[1]], [3*x[0]**2 - 2, 0]])

    initial_guess = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="Failed to converge"):
        newtons_method(f, jacobian, initial_guess)

def test_newtons_method_multidimensional_derivative_zero():
    # f(x, y) = [x^3 - 2x, y^3 - 2y]
    def f(x):
        return np.array([x[0]**3 - 2*x[0], x[1]**3 - 2*x[1]])

    def jacobian(x):
        return np.array([[3*x[0]**2 - 2, 0], [0, 3*x[1]**2 - 2]])

    initial_guess = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="Jacobian is singular, cannot proceed"):
        newtons_method(f, jacobian, initial_guess)

def test_newtons_method_multidimensional_edge_case():
    # f(x, y) = [2x - 4, 3y - 9], should converge to (2, 3)
    def f(x):
        return np.array([2*x[0] - 4, 3*x[1] - 9])

    def jacobian(x):
        return np.array([[2, 0], [0, 3]])

    initial_guess = np.array([3.0, 4.0])
    result = newtons_method(f, jacobian, initial_guess)
    assert np.allclose(result, [2.0, 3.0], atol=1e-6), "Test failed for simple linear system"
