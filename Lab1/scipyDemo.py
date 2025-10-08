import numpy as np
from scipy.optimize import least_squares

def test_scipy_least_squares():
    # Test 1: Define a function for least squares optimization
    def func(x):
        return x[0] + 2 * x[1] - 3

    # Test 2: Provide initial guess and perform least squares optimization
    initial_guess = [1, 2]
    result = least_squares(func, initial_guess)

    # Test 3: Check the optimized values
    optimized_values = result.x
    expected_values = np.array([0.6, 1.2])
    assert np.allclose(optimized_values, expected_values), "Least squares optimization failed"

    print("All SciPy (least_squares) tests passed!")

# Run the SciPy (least_squares) tests
test_scipy_least_squares()