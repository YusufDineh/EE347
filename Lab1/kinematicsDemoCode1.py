import numpy as np
import sympy as sp
from scipy.optimize import least_squares

def test_kinematics():
   # Test 1: Forward Kinematics
   theta1 = np.pi / 4
   theta2 = np.pi / 3
   link1_length = 1.0
   link2_length = 0.8

   x_pos = link1_length * np.cos(theta1) + link2_length * np.cos(theta1 + theta2)
   y_pos = link1_length * np.sin(theta1) + link2_length * np.sin(theta1 + theta2)

   assert np.isclose(x_pos, 0.5000515451045311), "Forward kinematics - x position calculation failed"
   assert np.isclose(y_pos, 1.4798474422178023), "Forward kinematics - y position calculation failed"

   # Test 2: Inverse Kinematics
   x_desired = 1.0
   y_desired = 0.8

   theta1_sym = sp.Symbol('theta1')
   theta2_sym = sp.Symbol('theta2')

   x_eq = link1_length * sp.cos(theta1_sym) + link2_length * sp.cos(theta1_sym + theta2_sym) - x_desired
   y_eq = link1_length * sp.sin(theta1_sym) + link2_length * sp.sin(theta1_sym + theta2_sym) - y_desired

   solutions = sp.solve([x_eq, y_eq], [theta1_sym, theta2_sym])

   # Print and check all real solutions for inspection
   print("Inverse Kinematics Solutions:")
   valid_solution_found = False
   for sol in solutions:
       # Check if the solutions are real numbers
       if sol[0].is_real and sol[1].is_real:
           theta1_solution = float(sol[0].evalf())
           theta2_solution = float(sol[1].evalf())
           theta1_solution_deg = np.degrees(theta1_solution)
           theta2_solution_deg = np.degrees(theta2_solution)
           print(f"theta1: {theta1_solution_deg} degrees, theta2: {theta2_solution_deg} degrees")
          
 
   # Test 3: Least Squares Optimization for Inverse Kinematics
   def inverse_kinematics_least_squares(x):
       theta1, theta2 = x
       return [
           link1_length * np.cos(theta1) + link2_length * np.cos(theta1 + theta2) - x_desired,
           link1_length * np.sin(theta1) + link2_length * np.sin(theta1 + theta2) - y_desired
       ]

   initial_guess = [0.5, 0.5]
   result = least_squares(inverse_kinematics_least_squares, initial_guess)

   theta1_optimized = np.degrees(result.x[0])
   theta2_optimized = np.degrees(result.x[1])

   print(f"Least Squares Optimized Solution: theta1: {theta1_optimized} degrees, theta2: {theta2_optimized} degrees")

   print("All kinematics tests processed!")

# Run the kinematics tests
test_kinematics()
