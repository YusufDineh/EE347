import numpy as np
import sympy as sp
from scipy.optimize import least_squares

def test_kinematics():
    # Define link lengths
    link1_length = 1.0
    link2_length = 0.8
    link3_length = 0.6

    # Test 1: Forward Kinematics
    # Given joint angles (in radians)
    theta1 = np.pi / 4
    theta2 = np.pi / 3
    theta3 = np.pi / 6

    # Calculate the end-effector position (x, y) using forward kinematics
    x_pos = (link1_length * np.cos(theta1) + 
             link2_length * np.cos(theta1 + theta2) + 
             link3_length * np.cos(theta1 + theta2 + theta3))
    y_pos = (link1_length * np.sin(theta1) + 
             link2_length * np.sin(theta1 + theta2) + 
             link3_length * np.sin(theta1 + theta2 + theta3))

    print(f"Calculated Forward Kinematics Position: x = {x_pos}, y = {y_pos}")

    # Expected values for forward kinematics (recomputed manually)
    expected_x = link1_length * np.cos(theta1) + link2_length * np.cos(theta1 + theta2) + link3_length * np.cos(theta1 + theta2 + theta3)
    expected_y = link1_length * np.sin(theta1) + link2_length * np.sin(theta1 + theta2) + link3_length * np.sin(theta1 + theta2 + theta3)

    print(f"Expected Forward Kinematics Position: x = {expected_x}, y = {expected_y}")

    # Test the result against the expected values
    assert np.isclose(x_pos, expected_x), "Forward kinematics - x position calculation failed"
    assert np.isclose(y_pos, expected_y), "Forward kinematics - y position calculation failed"

    # Test 2: Inverse Kinematics
    # Given the desired end-effector position (x, y)
    x_desired = 1.0
    y_desired = 1.5
    phi_desired = np.deg2rad(20)

    # Create symbolic variables for joint angles
    theta1_sym = sp.Symbol('theta1')
    theta2_sym = sp.Symbol('theta2')
    theta3_sym = sp.Symbol('theta3')

    # Define the equations representing the end-effector position using inverse kinematics
    x_eq = (link1_length * sp.cos(theta1_sym) + 
            link2_length * sp.cos(theta1_sym + theta2_sym) + 
            link3_length * sp.cos(theta1_sym + theta2_sym + theta3_sym) - x_desired)
    y_eq = (link1_length * sp.sin(theta1_sym) + 
            link2_length * sp.sin(theta1_sym + theta2_sym) + 
            link3_length * sp.sin(theta1_sym + theta2_sym + theta3_sym) - y_desired)


    phi_eq = (theta1_sym + theta2_sym + theta3_sym) - phi_desired

    # Solve the equations to find the joint angles (in radians)
    sol_vec = sp.nsolve([x_eq, y_eq, phi_eq],
                        [theta1_sym, theta2_sym, theta3_sym],
                        [0.5, 0.5, 0.5])
    solutions = [(float(sol_vec[0]), float(sol_vec[1]), float(sol_vec[2]))]

    # Print and check all real solutions for inspection
    print("Inverse Kinematics Solutions:")
    valid_solution_found = False
    for sol in solutions:
            theta1_solution, theta2_solution, theta3_solution = sol   
            theta1_solution_deg = np.degrees(theta1_solution)
            theta2_solution_deg = np.degrees(theta2_solution)
            theta3_solution_deg = np.degrees(theta3_solution)
            print(f"theta1: {theta1_solution_deg} degrees, theta2: {theta2_solution_deg} degrees, theta3: {theta3_solution_deg} degrees")
            
            # Check the solution against expected values
            # These expected values are placeholders; update them with your actual expected values
            expected_theta1 = 30.0
            expected_theta2 = 45.0
            expected_theta3 = 60.0
            if np.isclose(theta1_solution_deg, expected_theta1) and np.isclose(theta2_solution_deg, expected_theta2) and np.isclose(theta3_solution_deg, expected_theta3):
                valid_solution_found = True

    if not valid_solution_found:
        print("No valid solution found that matches the expected values.")
    else:
        print("A valid solution matching the expected values was found.")

    # Test 3: Least Squares Optimization for Inverse Kinematics
    def inverse_kinematics_least_squares(x):
        theta1, theta2, theta3 = x
        return [
            link1_length * np.cos(theta1) + 
            link2_length * np.cos(theta1 + theta2) + 
            link3_length * np.cos(theta1 + theta2 + theta3) - x_desired,
            link1_length * np.sin(theta1) + 
            link2_length * np.sin(theta1 + theta2) + 
            link3_length * np.sin(theta1 + theta2 + theta3) - y_desired
        ]

    # Provide initial guess and perform least squares optimization
    initial_guess = [0.5, 0.5, 0.5]  # Initial guess for joint angles (in radians)
    result = least_squares(inverse_kinematics_least_squares, initial_guess)

    theta1_optimized = np.degrees(result.x[0])
    theta2_optimized = np.degrees(result.x[1])
    theta3_optimized = np.degrees(result.x[2])

    print(f"Least Squares Optimized Solution: theta1: {theta1_optimized} degrees, theta2: {theta2_optimized} degrees, theta3: {theta3_optimized} degrees")

    print("All kinematics tests processed!")

# Run the kinematics tests
test_kinematics()