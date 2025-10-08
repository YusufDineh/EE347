import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from sympy.utilities.lambdify import lambdify

# Verification for NumPy
print("\nVerifying NumPy:")
arr = np.array([1, 2, 3, 4, 5])
print("Original Array:", arr)
arr_squared = np.square(arr)
print("Squared Array:", arr_squared)

# Verification for SymPy
print("\nVerifying SymPy:")
x = sp.Symbol('x')
expr = x**2 + 2*x + 1
derivative = sp.diff(expr, x)
integral = sp.integrate(expr, x)
print("Expression:", expr)
print("Derivative:", derivative)
print("Integral:", integral)

# Verification for SciPy
print("\nVerifying SciPy:")
def fun(x):
    return [x[0] + 0.5 * (x[0] - 3)**3 - 1.0]

result = least_squares(fun, [0.0])
print("Least Squares Result:", result)

# Verification for lambdify from SymPy
print("\nVerifying lambdify:")
func = lambdify(x, expr)
print("Evaluated Expression at x=2:", func(2))

print("\nVerification completed successfully.")