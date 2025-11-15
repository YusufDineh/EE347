#!/usr/bin/env python3
"""
MechArm Inverse Kinematics Script

Computes inverse kinematics for target positions and sends commands to the robot.
Includes automatic error correction, position verification, gripper control, and comprehensive safety checks.

Usage:
  # First time: Record start/end positions manually
  python mecharm_INVK_advanced_group_5.py --output results.csv --positions positions.csv --manual
  
  # Run pick-and-place with gripper
  python mecharm_INVK_advanced_group_5.py --output results.csv --positions positions.csv
  
  # Custom gripper values
  python mecharm_INVK_advanced_group_5.py --output results.csv --positions positions.csv --grip-open 0 --grip-close 80
  
  # Test gripper only
  python mecharm_INVK_advanced_group_5.py --output results.csv --positions positions.csv --test-gripper
"""

import math
import time
import csv
import argparse
import logging
import sys
import threading
from typing import List, Tuple, Optional, Dict
import numpy as np
from sympy.matrices import Matrix
from scipy.optimize import least_squares
from sympy import symbols, cos, sin, atan2, pi, lambdify 
from pathlib import Path

# Try to import robot library
try:
    from pymycobot.mycobot import MyCobot
    from pymycobot import PI_PORT, PI_BAUD
    PYMCOBOT_AVAILABLE = True
except ImportError:
    PYMCOBOT_AVAILABLE = False
    logging.warning("pymycobot not available - only dry-run mode supported")

# Try to import keyboard library for spacebar pause/resume
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    logging.warning("keyboard library not available - spacebar pause/resume disabled")
    logging.warning("Install with: pip install keyboard")

# Joint angle limits (degrees) - will be updated from robot if available
JOINT_LIMITS = [
    (-165, 165),  # J1
    (-90, 90),    # J2
    (-180, 65),   # J3
    (-160, 160),  # J4
    (-115, 115),  # J5
    (-175, 175)   # J6
]

# Workspace safety limits (mm)
MIN_REACH = 50.0      # Minimum distance from base
MAX_Z = 400.0         # Maximum height
MIN_Z = -50.0         # Minimum height (safety margin above ground)

# Error correction settings
MAX_RETRIES = 3       # Number of retry attempts for movements
ANGLE_TOLERANCE = 2.0    # Degrees (unused - we verify coords instead)
COORD_TOLERANCE = 5.0    # Position tolerance in mm
ORIENT_TOLERANCE = 5.0   # Orientation tolerance in degrees

# Gripper settings
GRIPPER_OPEN = 0      # Gripper fully open (0-100)
GRIPPER_CLOSE = 1   # Gripper fully closed (0-100)
GRIPPER_SLEEP = 2.0   # Time to wait for gripper action (seconds)

# Speed settings
SPEED_PRECISION = 15  # Slow speed for pick/place operations
SPEED_NORMAL = 30     # Normal speed for travel movements

# Global pause flag for spacebar control
robot_paused = False
pause_lock = threading.Lock()

# Initialize robot (only if not in dry-run mode)
mc = None

# Global symbolic joint variables
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')
right = pi/2
origin = [0, 0, 0, 0, 0, 0]  # Origin position for servos

dh_table = [
    [0.049,          0,     135.926,        q1],                    # joint 1
    [0,         -right,           0,        q2 - right],            # joint 2
    [99.973,         0,           0,        q3],                    # joint 3
    [10.012,    -right,     107.011,        q4],                    # joint 4
    [0.003,      right,       0.006,        q5],                    # joint 5
    [0.054,     -right,      64.973,        q6]                     # joint 6
]


def setup_spacebar_control():
    """
    Setup spacebar listener for pause/resume control.
    Press SPACE to toggle pause/resume during operation.
    """
    global robot_paused
    
    if not KEYBOARD_AVAILABLE:
        logging.warning("Keyboard library not available - spacebar control disabled")
        return
    
    def on_space():
        global robot_paused
        with pause_lock:
            robot_paused = not robot_paused
            
            if robot_paused:
                logging.warning("⏸ PAUSED by spacebar - Press SPACE again to resume")
                if mc is not None:
                    try:
                        mc.pause()
                    except Exception as e:
                        logging.error(f"Failed to pause robot: {e}")
            else:
                logging.info("▶ RESUMED by spacebar")
                if mc is not None:
                    try:
                        mc.resume()
                    except Exception as e:
                        logging.error(f"Failed to resume robot: {e}")
    
    # Register spacebar hotkey
    keyboard.on_press_key('space', lambda _: on_space())
    logging.info("Spacebar control enabled: Press SPACE to pause/resume")


def check_and_handle_pause():
    """
    Check if robot is paused and wait until resumed.
    Call this before each critical operation.
    """
    global robot_paused
    
    with pause_lock:
        if robot_paused:
            logging.info("Waiting for resume...")
    
    # Wait while paused
    while robot_paused:
        time.sleep(0.1)


def get_robot_joint_limits() -> List[Tuple[float, float]]:
    """
    Query actual joint limits from the robot.
    
    Returns:
        List of (min_angle, max_angle) tuples for each joint
    """
    if mc is None:
        logging.warning("Cannot query joint limits - robot not initialized")
        return JOINT_LIMITS
    
    try:
        limits = []
        for joint_id in range(1, 7):
            min_ang = mc.get_joint_min_angle(joint_id)
            max_ang = mc.get_joint_max_angle(joint_id)
            limits.append((min_ang, max_ang))
            logging.debug(f"Joint {joint_id} limits: [{min_ang:.1f}, {max_ang:.1f}]")
        
        logging.info("Successfully queried joint limits from robot")
        return limits
        
    except Exception as e:
        logging.warning(f"Failed to query joint limits from robot: {e}")
        logging.warning("Using default joint limits")
        return JOINT_LIMITS


def check_robot_errors() -> bool:
    """
    Check for robot errors and clear them if found.
    
    Returns:
        True if errors were found and cleared, False otherwise
    """
    if mc is None:
        return False
    
    try:
        error_msg = mc.get_error_information()
        if error_msg:
            logging.error(f"Robot error detected: {error_msg}")
            mc.clear_error_information()
            logging.info("Error information cleared")
            return True
        return False
        
    except Exception as e:
        logging.warning(f"Failed to check robot errors: {e}")
        return False


def verify_robot_status() -> bool:
    """
    Comprehensive robot status verification.
    
    Checks:
    - Power status
    - Servo enable status
    - Pause status
    - Error messages
    
    Returns:
        True if robot is ready, False otherwise
    """
    if mc is None:
        return False
    
    try:
        # Check power status
        if not mc.is_power_on():
            logging.error("Robot is not powered on")
            return False
        
        # Check if all servos are enabled
        if not mc.is_all_servo_enable():
            logging.error("Not all servos are enabled")
            return False
        
        # Check if paused
        if mc.is_paused():
            logging.warning("Robot is in paused state, resuming...")
            mc.resume()
            time.sleep(0.5)
        
        # Check for errors
        check_robot_errors()
        
        logging.debug("Robot status verification passed")
        return True
        
    except Exception as e:
        logging.error(f"Failed to verify robot status: {e}")
        return False

def init_robot(dry_run: bool = False) -> bool:
    """
    Initialize robot connection and power on.
    
    Performs comprehensive initialization:
    - Connects to robot
    - Powers on
    - Verifies all servos enabled
    - Queries actual joint limits
    - Sets up spacebar control
    
    Args:
        dry_run: If True, skip actual robot initialization
        
    Returns:
        True if successful or in dry-run mode, False otherwise
    """
    global mc, JOINT_LIMITS
    
    if dry_run:
        logging.info("Dry-run mode: skipping robot initialization")
        return True
    
    if not PYMCOBOT_AVAILABLE:
        logging.error("pymycobot library not available - cannot connect to robot")
        return False
    
    try:
        # Connect and power on
        mc = MyCobot(PI_PORT, PI_BAUD)
        mc.power_on()
        time.sleep(1.0)

        logging.info("Robot connected and powered on")

        # Verify robot status
        if not verify_robot_status():
            logging.error("Robot status verification failed")
            return False

        # NOTE: Do NOT overwrite the hard-coded JOINT_LIMITS below.
        # Keep the fixed joint limits defined in the script for safety.
        logging.debug("Using fixed JOINT_LIMITS from script; not querying robot for limits.")

        # Setup spacebar control
        setup_spacebar_control()

        # Clear any existing errors
        check_robot_errors()

        logging.info("Robot initialization complete")
        return True

    except Exception as e:
        logging.error(f"Failed to initialize robot: {e}")
        return False


def get_transformation_matrix(a, alpha, d, theta):
    """
    Compute DH transformation matrix.
    
    Standard DH convention: T = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
    """
    return Matrix([
        [cos(theta),                          -sin(theta),                   0,                     a],
        [sin(theta)*cos(alpha),     cos(theta)*cos(alpha),         -sin(alpha),         -sin(alpha)*d],
        [sin(theta)*sin(alpha),     cos(theta)*sin(alpha),          cos(alpha),          cos(alpha)*d],
        [0,                                             0,                   0,                     1]
    ])


def overall_transformation(dh_table):
    """
    Compute forward kinematics transformation matrix from DH parameters.
    
    Returns 4x4 homogeneous transformation matrix from base to end-effector.
    """
    T = Matrix(np.identity(4))
    for i in range(len(dh_table)):
        a, alpha, d, theta = dh_table[i]
        T_i = get_transformation_matrix(a, alpha, d, theta)
        T = T * T_i
    return T


# Precompute symbolic FK and convert to numerical functions
T_symbolic = overall_transformation(dh_table)
fk_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, 3], modules='numpy')
rot_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, :3], modules='numpy')


def validate_joint_angles(q: np.ndarray) -> None:
    """
    Validate joint angles are within safe mechanical limits.
    
    Args:
        q: Array of 6 joint angles in degrees
        
    Raises:
        ValueError: If any joint exceeds its limits
    """
    for i, (angle, (min_ang, max_ang)) in enumerate(zip(q, JOINT_LIMITS)):
        if not (min_ang <= angle <= max_ang):
            raise ValueError(
                f"Joint {i+1} angle {angle:.1f}° outside limits [{min_ang}, {max_ang}]"
            )


def validate_workspace(x: float, y: float, z: float) -> None:
    """
    Validate target position is within safe workspace.
    
    Args:
        x, y, z: Target position in mm
        
    Raises:
        ValueError: If position is outside safe workspace
    """
    distance = math.sqrt(x*x + y*y + z*z)
    if distance < MIN_REACH:
        raise ValueError(f"Target too close to base: {distance:.1f}mm < {MIN_REACH}mm")
    
    if z > MAX_Z:
        raise ValueError(f"Target too high: {z:.1f}mm > {MAX_Z}mm")
    
    if z < MIN_Z:
        raise ValueError(f"Target too low: {z:.1f}mm < {MIN_Z}mm")


def fk(q_vals: np.ndarray) -> np.ndarray:
    """
    Forward kinematics: compute end-effector position from joint angles.
    
    Args:
        q_vals: Array of 6 joint angles in degrees
        
    Returns:
        3D position [x, y, z] in mm
    """
    q = np.asarray(q_vals, dtype=float)
    q = np.radians(q)
    xyz = np.asarray(fk_num(*q), dtype=float).ravel()
    return xyz


def rotation_matrix_to_euler_zyx(R: np.ndarray, transpose: bool = False) -> Tuple[float, float, float]:
    """
    Extract Z-Y-X Euler angles (roll, pitch, yaw) from rotation matrix.
    
    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    
    Args:
        R: 3x3 rotation matrix
        transpose: If True, use R.T instead of R
        
    Returns:
        (roll, pitch, yaw) in radians
    """
    if transpose:
        R = R.T
    
    # Extract pitch from R[2,0] = -sin(pitch)
    sin_beta = -R[2, 0]
    sin_beta = np.clip(sin_beta, -1.0, 1.0)
    pitch = np.arcsin(sin_beta)
    
    cos_beta = np.cos(pitch)
    
    # Check for gimbal lock (pitch ≈ ±90°)
    if np.abs(cos_beta) < 1e-10:
        # Gimbal lock: set yaw = 0, solve for roll
        yaw = 0.0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    else:
        # Normal case
        roll = np.arctan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
        yaw = np.arctan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
    
    return roll, pitch, yaw


def position_error(q_position: np.ndarray, x_target: float, y_target: float, z_target: float) -> np.ndarray:
    """
    Compute position error between FK result and target.
    
    Args:
        q_position: Joint angles in degrees (6-element array)
        x_target, y_target, z_target: Target position in mm
        
    Returns:
        3-element error vector [dx, dy, dz] in mm
    """
    pos = fk(q_position)
    return np.array([
        pos[0] - float(x_target),
        pos[1] - float(y_target),
        pos[2] - float(z_target)
    ])


def orientation_error(q_orientation: np.ndarray, rx_d: float, ry_d: float, rz_d: float) -> np.ndarray:
    """
    Compute orientation error between FK result and target.
    
    Args:
        q_orientation: Joint angles in degrees
        rx_d, ry_d, rz_d: Target orientation (roll, pitch, yaw) in degrees
        
    Returns:
        3-element error vector [d_roll, d_pitch, d_yaw] in radians
    """
    def _wrap_angle(a):
        """Wrap angle to [-π, π]"""
        return (a + np.pi) % (2*np.pi) - np.pi

    q = np.asarray(q_orientation, dtype=float)
    q_rad = np.radians(q)
    rx_d_rad = np.radians(rx_d)
    ry_d_rad = np.radians(ry_d)
    rz_d_rad = np.radians(rz_d)

    R = np.array(rot_num(*q_rad), dtype=float)
    roll, pitch, yaw = rotation_matrix_to_euler_zyx(R, transpose=False)

    err_roll = _wrap_angle(roll - rx_d_rad)
    err_pitch = _wrap_angle(pitch - ry_d_rad)
    err_yaw = _wrap_angle(yaw - rz_d_rad)

    return np.array([err_roll, err_pitch, err_yaw])


def link_lengths(dh_table) -> float:
    """
    Compute upper bound on robot reach from DH parameters.
    
    Returns:
        Maximum reach distance in mm
    """
    total_length = 0.0
    for a, alpha, d, theta in dh_table:
        total_length += abs(float(a)) + abs(float(d))
    return total_length


def inverse_kinematics(x_target: float, y_target: float, z_target: float, 
                       rx_d: float, ry_d: float, rz_d: float, 
                       q_init: np.ndarray, max_reach: float,
                       max_iterations: int = 5000, tolerance: float = 1e-6) -> np.ndarray:
    """
    Compute inverse kinematics using numerical optimization.
    
    Simultaneously optimizes position and orientation to find joint angles
    that achieve the target pose.
    
    Args:
        x_target, y_target, z_target: Target position in mm
        rx_d, ry_d, rz_d: Target orientation (roll, pitch, yaw) in degrees
        q_init: Initial guess for joint angles in degrees
        max_reach: Maximum robot reach in mm
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        
    Returns:
        Array of 6 joint angles in degrees
        
    Raises:
        ValueError: If target is unreachable or IK fails to converge
    """
    # Validate target is within workspace
    validate_workspace(x_target, y_target, z_target)
    
    distance_from_origin = math.hypot(x_target, y_target, z_target)
    if distance_from_origin > max_reach:
        raise ValueError(
            f"Target outside robot reach: {distance_from_origin:.1f}mm > {max_reach:.1f}mm"
        )
    
    q_init = np.asarray(q_init, dtype=float)
    
    def combined_residual(q_all, pos_weight=1.0, ori_weight=10.0):
        """
        Compute combined position and orientation residual.
        
        Returns 6-element vector: [pos_x, pos_y, pos_z, ori_roll, ori_pitch, ori_yaw]
        """
        pos_err = position_error(q_all, x_target, y_target, z_target)
        ori_err = orientation_error(q_all, rx_d, ry_d, rz_d)
        return np.concatenate([pos_err * pos_weight, ori_err * ori_weight])
    
    # Extract bounds from JOINT_LIMITS
    lower_bounds = np.array([limit[0] for limit in JOINT_LIMITS])
    upper_bounds = np.array([limit[1] for limit in JOINT_LIMITS])
    bounds = (lower_bounds, upper_bounds)
    
    # Use trust region reflective method for better non-linear handling with bounds
    result = least_squares(
        combined_residual, 
        q_init,
        bounds=bounds,  # Apply joint angle limits as bounds
        method='trf',   # 'trf' supports bounds
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        verbose=0
    )
    
    if not result.success:
        logging.warning(f"IK optimization warning: {result.message}")
    
    q_solution = result.x
    
    # Validate solution is within joint limits
    validate_joint_angles(q_solution)
    
    # Verify solution quality
    final_pos_error = np.linalg.norm(position_error(q_solution, x_target, y_target, z_target))
    final_ori_error = np.linalg.norm(orientation_error(q_solution, rx_d, ry_d, rz_d))
    
    if final_pos_error > 10.0:  # 10mm position error threshold
        logging.warning(f"Large position error: {final_pos_error:.2f}mm")
    
    if final_ori_error > np.radians(10):  # 10° orientation error threshold
        logging.warning(f"Large orientation error: {np.degrees(final_ori_error):.2f}°")
    
    return q_solution


def control_gripper(value: int, sleep_time: float = GRIPPER_SLEEP) -> bool:
    """
    Control the robot gripper.
    
    Args:
        value: Gripper value 0-100 (0=fully open, 100=fully closed)
        sleep_time: Wait time after gripper action
        
    Returns:
        True if successful, False otherwise
    """
    if mc is None:
        logging.warning("Cannot control gripper - robot not initialized")
        return False
    
    # Check for pause before gripper action
    check_and_handle_pause()
    
    try:
        # Clamp value to valid range
        value = max(0, min(100, int(value)))
        
        mc.set_gripper_state(value, 70)
        time.sleep(sleep_time)
        
        action = "CLOSED" if value > 50 else "OPEN"
        logging.info(f"Gripper {action} (value={value})")
        
        # Check for errors after gripper action
        check_robot_errors()
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to control gripper: {e}")
        check_robot_errors()
        return False


def get_current_positions() -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Get current joint angles and Cartesian coordinates from robot.
    
    Returns:
        Tuple of (angles, coords) where:
        - angles: List of 6 joint angles in degrees
        - coords: List of 6 values [x, y, z, rx, ry, rz]
        Returns (None, None) on error
    """
    if mc is None:
        return None, None
    
    try:
        coords = mc.get_coords()
        angles = mc.get_angles()
        
        if coords is None or angles is None:
            logging.error("Robot returned None for position data")
            return None, None
        
        return angles, coords
    except Exception as e:
        logging.error(f"Failed to read robot positions: {e}")
        return None, None


def compare_coords(target: List[float], actual: List[float], 
                   pos_tol: float = COORD_TOLERANCE, 
                   ori_tol: float = ORIENT_TOLERANCE) -> Tuple[bool, dict]:
    """
    Compare target and actual coordinates with tolerances.
    
    Args:
        target: Target coords [x, y, z, rx, ry, rz]
        actual: Actual coords [x, y, z, rx, ry, rz]
        pos_tol: Position tolerance in mm
        ori_tol: Orientation tolerance in degrees
        
    Returns:
        Tuple of (match, errors) where:
        - match: True if within tolerances
        - errors: Dict of error values for each coordinate
    """
    if len(target) < 6 or len(actual) < 6:
        return False, {}
    
    errors = {
        'x': abs(target[0] - actual[0]),
        'y': abs(target[1] - actual[1]),
        'z': abs(target[2] - actual[2]),
        'rx': abs(target[3] - actual[3]),
        'ry': abs(target[4] - actual[4]),
        'rz': abs(target[5] - actual[5])
    }
    
    pos_match = (errors['x'] <= pos_tol and 
                 errors['y'] <= pos_tol and 
                 errors['z'] <= pos_tol)
    
    ori_match = (errors['rx'] <= ori_tol and 
                 errors['ry'] <= ori_tol and 
                 errors['rz'] <= ori_tol)
    
    return pos_match and ori_match, errors


def send_angles_with_correction(target_angles: List[float], target_coords: List[float],
                                speed: int, sleep_time: float, 
                                position_name: str) -> Tuple[bool, List[float]]:
    """
    Send joint angles to robot with automatic error correction.
    
    Verifies the robot reaches the target Cartesian position (not just joint angles)
    and retries if necessary.
    
    Args:
        target_angles: List of 6 joint angles in degrees
        target_coords: Target Cartesian coords [x, y, z, rx, ry, rz]
        speed: Robot movement speed (0-100)
        sleep_time: Wait time after movement in seconds
        position_name: Name for logging
        
    Returns:
        Tuple of (success, actual_coords)
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            # Send angles to robot
            mc.send_angles(target_angles, speed)
            time.sleep(sleep_time)
            
            # Get actual position for verification
            current_angles, current_coords = get_current_positions()
            
            if current_coords is None:
                logging.warning(f"Cannot verify position for {position_name} (attempt {attempt+1}/{MAX_RETRIES})")
                continue
            
            # Verify coordinates match target (not joint angles - multiple solutions exist)
            coords_match, errors = compare_coords(target_coords, current_coords)
            
            if coords_match:
                logging.info(f"Successfully reached {position_name} (attempt {attempt+1}/{MAX_RETRIES})")
                return True, current_coords
            else:
                logging.warning(
                    f"Position mismatch for {position_name} (attempt {attempt+1}/{MAX_RETRIES}):\n"
                    f"  Position errors: x={errors['x']:.1f}mm, y={errors['y']:.1f}mm, z={errors['z']:.1f}mm\n"
                    f"  Orientation errors: rx={errors['rx']:.1f}°, ry={errors['ry']:.1f}°, rz={errors['rz']:.1f}°"
                )
                
        except Exception as e:
            logging.warning(f"Movement failed for {position_name} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
        
        # Wait before retry
        if attempt < MAX_RETRIES - 1:
            time.sleep(1.0)
    
    logging.error(f"Failed to reach {position_name} after {MAX_RETRIES} attempts")
    return False, target_coords


def reset(sleep_time: float, speed: int = 30) -> None:
    """
    Safely reset robot to origin position.
    
    Performs power cycle and moves to configured origin angles.
    
    Args:
        sleep_time: Wait time in seconds between operations
        speed: Movement speed (0-100)
    """
    
    if mc is None:
        logging.warning("Cannot reset - robot not initialized")
        return
    
    try:
        mc.power_off()
        time.sleep(sleep_time)
        mc.power_on()
        time.sleep(sleep_time)
        mc.send_angles(origin, speed)
        time.sleep(sleep_time)
    except Exception as e:
        logging.error(f"Reset failed: {e}")


def request_to_send(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read target positions from CSV file.
    
    CSV format: Index,J1,J2,J3,J4,J5,J6,X,Y,Z,RX,RY,RZ
    
    Args:
        path: Path to CSV file
        
    Returns:
        Tuple of (joint_angles, positions, orientations) as numpy arrays
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except ValueError as e:
        raise ValueError(f"Invalid CSV format in {path}: {e}")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    if data.shape[1] < 13:
        raise ValueError(f"CSV must have at least 13 columns, found {data.shape[1]}")
    
    # Extract columns (skip Index column 0)
    joint_angles = data[:, 1:7]    # J1-J6
    positions = data[:, 7:10]      # X, Y, Z
    orientations = data[:, 10:13]  # RX, RY, RZ
    
    return joint_angles, positions, orientations


def pick_and_place_sequence(positions: List[Tuple[int, List[float]]], 
                           output_csv: str,
                           speed: int = 30, 
                           sleep_time: float = 3.0,
                           gripper_open: int = GRIPPER_OPEN,
                           gripper_close: int = GRIPPER_CLOSE,
                           precision_mode: bool = True) -> None:
    """
    Execute pick-and-place sequence with IK, position verification, and gripper control.
    
    Uses cached IK solutions for repeated positions (START and END) to avoid recomputation.
    
    Sequence:
    1. Move to START position (slow speed if precision_mode)
    2. CLOSE gripper (pick up object)
    3. LIFT up from START
    4. Move to END position (elevated)
    5. Lower to END position (slow speed if precision_mode)
    6. OPEN gripper (release object)
    
    Args:
        positions: List of (index, [x, y, z, rx, ry, rz]) tuples
        output_csv: Output CSV file path for results
        speed: Robot movement speed (0-100) for travel
        sleep_time: Wait time after each move in seconds
        gripper_open: Gripper open value (0-100, default 0=fully open)
        gripper_close: Gripper close value (0-100, default 100=fully closed)
        precision_mode: If True, use slower speed for pick/place positions
    """
    logging.info(f"Starting pick and place sequence: {len(positions)} positions")
    
    if KEYBOARD_AVAILABLE:
        logging.info("⌨ Spacebar control active: Press SPACE to pause/resume")
    
    # Verify robot is ready
    if not verify_robot_status():
        logging.error("Robot not ready - aborting sequence")
        return
    
    # Start with gripper open
    logging.info("Opening gripper before sequence...")
    control_gripper(gripper_open)
    
    logging.info("Resetting robot to origin...")
    reset(sleep_time, speed)
    logging.info("Robot reset complete")
    
    max_reach = link_lengths(dh_table)
    
    # IK solution cache - keyed by position index
    ik_cache: Dict[int, np.ndarray] = {}
    
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "Index",
            "Req_X", "Req_Y", "Req_Z", "Req_RX", "Req_RY", "Req_RZ",
            "J1", "J2", "J3", "J4", "J5", "J6",
            "Meas_X", "Meas_Y", "Meas_Z", "Meas_RX", "Meas_RY", "Meas_RZ"
        ]
        writer.writerow(header)
        
        for idx, target_coords in positions:
            # Determine position name
            if idx == -1:
                position_name = "START"
            elif idx == -2:
                position_name = "END"
            elif idx == -3:
                position_name = "LIFT"
            elif idx == -4:
                position_name = "LIFT_TO_END"
            elif idx == -5:
                position_name = "HOME"
            elif idx == -6:
                position_name = "MIDDLE"
            else:
                position_name = f"Target {idx}"
            
            x, y, z, rx, ry, rz = target_coords
            
            # Check for pause before IK computation
            check_and_handle_pause()
            
            # Determine speed based on position (slower for pick/place)
            if precision_mode and position_name in ["START", "END"]:
                current_speed = SPEED_PRECISION
                logging.info(f"Using precision speed: {current_speed}")
            else:
                current_speed = speed
                logging.info(f"Using normal speed: {current_speed}")
            
            # Check if IK solution is cached
            if idx in ik_cache:
                logging.info(f"Using cached IK solution for {position_name}")
                q_solution = ik_cache[idx]
            else:
                logging.info(f"Computing IK for {position_name}: [{x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f}]")
                
                # Get current angles for initial guess
                get_init_angles, _ = get_current_positions()
                if get_init_angles is not None:
                    q_init = get_init_angles
                else:
                    q_init = origin
                
                try:
                    # Compute IK solution
                    q_solution = inverse_kinematics(x, y, z, rx, ry, rz, q_init, max_reach)
                    
                    # Cache solution for START (-1) and END (-2) positions
                    if idx in [-1, -2]:
                        ik_cache[idx] = q_solution
                        logging.info(f"Cached IK solution for {position_name}")
                    
                except Exception as e:
                    logging.error(f"IK failed for {position_name}: {e}")
                    check_robot_errors()
                    # Write row with zeros for failed IK
                    row = [idx, x, y, z, rx, ry, rz] + [0.0] * 12
                    writer.writerow(row)
                    continue
            
            # Send to robot with verification
            try:
                success, meas_coords = send_angles_with_correction(
                    q_solution.tolist(), target_coords, current_speed, sleep_time, 
                    position_name
                )
                
                if not success:
                    logging.error(f"Failed to reach {position_name}")
                    meas_coords = [0.0] * 6
                
                # GRIPPER CONTROL based on position (FIXED)
                if idx == -1:  # After reaching START position
                    logging.info("Closing gripper to pick up object...")
                    control_gripper(gripper_close)  # ✓ FIXED
                elif idx == -2:  # After reaching END position
                    logging.info("Opening gripper to release object...")
                    control_gripper(gripper_open)  # ✓ FIXED
                
                # Ensure meas_coords is length 6
                meas_coords = list(meas_coords) + [0.0] * (6 - len(meas_coords))
                
                # Write results
                row = [
                    idx,
                    x, y, z, rx, ry, rz,
                    q_solution[0], q_solution[1], q_solution[2], 
                    q_solution[3], q_solution[4], q_solution[5],
                    meas_coords[0], meas_coords[1], meas_coords[2],
                    meas_coords[3], meas_coords[4], meas_coords[5]
                ]
                writer.writerow(row)
                logging.info(f"Recorded data for {position_name}")
                
            except Exception as e:
                logging.error(f"Movement failed for {position_name}: {e}")
                check_robot_errors()
                # Write row with computed angles but failed movement
                row = [
                    idx, x, y, z, rx, ry, rz,
                    q_solution[0], q_solution[1], q_solution[2], 
                    q_solution[3], q_solution[4], q_solution[5]
                ] + [0.0] * 6
                writer.writerow(row)
    
    logging.info(f"Sequence complete. Results: {output_csv}")
    logging.info(f"IK cache statistics: {len(ik_cache)} positions cached")
    
    # Final status check
    if not verify_robot_status():
        logging.warning("Robot status check failed after sequence")
    
    # Check for any final errors
    check_robot_errors()


def test_gripper():
    """Test gripper open/close cycle independently"""
    if mc is None:
        print("❌ Robot not initialized")
        return False
    
    print("\n" + "="*60)
    print("  GRIPPER TEST MODE")
    print("="*60)
    
    try:
        print("Testing gripper...")
        print("1. Opening gripper...")
        control_gripper(0)  # Open
        time.sleep(2)
        
        print("2. Closing gripper...")
        control_gripper(100)  # Close
        time.sleep(2)
        
        print("3. Opening gripper...")
        control_gripper(0)  # Open
        time.sleep(1)
        
        print("✓ Gripper test complete")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Gripper test failed: {e}")
        return False


def parse_position(pos_str: str) -> List[float]:
    """
    Parse position from comma-separated string or CSV file.
    
    Args:
        pos_str: Either "x,y,z,rx,ry,rz" or path to CSV file
        
    Returns:
        List of 6 floats [x, y, z, rx, ry, rz]
        
    Raises:
        ValueError: If format is invalid or file not found
    """
    # Try comma-separated values first
    try:
        coords = [float(x.strip()) for x in pos_str.split(',')]
        if len(coords) != 6:
            raise ValueError("Position must have 6 values (X,Y,Z,RX,RY,RZ)")
        return coords
    except ValueError:
        pass
    
    # Try as CSV file
    try:
        _, positions, orientations = request_to_send(pos_str)
        if len(positions) == 0:
            raise ValueError("No positions found in CSV")
        # Use first position
        return [
            positions[0, 0], positions[0, 1], positions[0, 2],
            orientations[0, 0], orientations[0, 1], orientations[0, 2]
        ]
    except FileNotFoundError:
        raise ValueError(f"Position string '{pos_str}' is not valid coordinates or CSV file")
    except Exception as e:
        raise ValueError(f"Could not parse position '{pos_str}': {e}")


def record_manual_positions(positions_csv: str, sleep_time: float = 2.0) -> Tuple[List[float], List[float]]:
    """
    Manually teach start and end positions by moving the robot.
    
    Releases servos to allow free manual movement, then records positions
    when user is ready.
    
    Args:
        positions_csv: CSV file to save positions
        sleep_time: Wait time between readings
        
    Returns:
        Tuple of (start_coords, end_coords)
    """
    if mc is None:
        raise RuntimeError("Robot not initialized - cannot record positions")
    
    print("\n" + "="*60)
    print("  MANUAL POSITION RECORDING MODE")
    print("="*60)
    print("\nInstructions:")
    print("1. Servos will be released - you can manually move the robot")
    print("2. Move the arm to the START position")
    print("3. Press Enter to lock servos and record START")
    print("4. Servos will be released again")
    print("5. Move the arm to the END position")
    print("6. Press Enter to lock servos and record END")
    print("="*60 + "\n")
    
    try:
        # Record START position
        print("Releasing servos for manual positioning...")
        mc.release_all_servos()
        
        input("Move robot to START position, then press Enter...")
        
        # Re-enable servos to lock position and read coordinates
        print("Locking servos and reading position...")
        mc.power_on()
        time.sleep(sleep_time)
        
        _, start_coords = get_current_positions()
        if start_coords is None:
            raise RuntimeError("Failed to read START position from robot")
        
        print(f"\n✓ START recorded: [{start_coords[0]:.1f}, {start_coords[1]:.1f}, {start_coords[2]:.1f}, "
              f"{start_coords[3]:.1f}, {start_coords[4]:.1f}, {start_coords[5]:.1f}]")
        
        # Record END position
        print("\nReleasing servos again for END position...")
        mc.release_all_servos()
        
        input("\nMove robot to END position, then press Enter...")
        
        # Re-enable servos to lock position and read coordinates
        print("Locking servos and reading position...")
        mc.power_on()
        time.sleep(sleep_time)
        
        _, end_coords = get_current_positions()
        if end_coords is None:
            raise RuntimeError("Failed to read END position from robot")
        
        print(f"\n✓ END recorded: [{end_coords[0]:.1f}, {end_coords[1]:.1f}, {end_coords[2]:.1f}, "
              f"{end_coords[3]:.1f}, {end_coords[4]:.1f}, {end_coords[5]:.1f}]")
        
        # Save to CSV
        with open(positions_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Position", "X", "Y", "Z", "RX", "RY", "RZ"])
            writer.writerow(["START"] + start_coords)
            writer.writerow(["END"] + end_coords)
        
        print(f"\n✓ Positions saved to: {positions_csv}\n")
        
        return start_coords, end_coords
        
    except Exception as e:
        # Make sure to re-enable servos even if error occurs
        try:
            mc.power_on()
        except:
            pass
        raise e


def load_manual_positions(positions_csv: str) -> Tuple[List[float], List[float]]:
    """
    Load previously recorded start and end positions from CSV.
    
    Args:
        positions_csv: CSV file containing positions
        
    Returns:
        Tuple of (start_coords, end_coords)
        
    Raises:
        FileNotFoundError: If positions file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not Path(positions_csv).exists():
        raise FileNotFoundError(
            f"Positions file not found: {positions_csv}\n"
            f"Run with --manual to record positions first"
        )
    
    try:
        with open(positions_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            start_row = next(reader)
            end_row = next(reader)
            
            start_coords = [float(x) for x in start_row[1:7]]
            end_coords = [float(x) for x in end_row[1:7]]
            
            return start_coords, end_coords
            
    except Exception as e:
        raise ValueError(f"Failed to parse positions CSV: {e}")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="MechArm IK Pick-and-Place Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time: Record start/end positions manually
  %(prog)s --output results.csv --positions positions.csv --manual
  
  # Subsequent runs: Use recorded positions
  %(prog)s --output results.csv --positions positions.csv
  
  # Custom lift height
  %(prog)s --output results.csv --positions positions.csv --lift-height 80
  
  # Test gripper only
  %(prog)s --output results.csv --positions positions.csv --test-gripper
        """
    )
    p.add_argument("--output", "-o", required=True, 
                   help="Output CSV file for IK results")
    p.add_argument("--positions", "-p", required=True,
                   help="CSV file for start/end positions (read or write)")
    p.add_argument("--manual", action="store_true",
                   help="Manual mode: physically move robot to record positions")
    p.add_argument("--test-gripper", action="store_true",
                   help="Test gripper open/close cycle and exit")
    p.add_argument("--lift-height", type=float, default=150.0, 
                   help="Lift height above pick/place positions in mm (default: 50)")
    p.add_argument("--speed", type=int, default=30, 
                   help="Robot movement speed 0-100 (default: 30)")
    p.add_argument("--sleep", type=float, default=3.0, 
                   help="Wait time after each move in seconds (default: 3.0)")
    p.add_argument("--grip-open", type=int, default=GRIPPER_OPEN,
                   help=f"Gripper open value 0-100 (default: {GRIPPER_OPEN})")
    p.add_argument("--grip-close", type=int, default=GRIPPER_CLOSE,
                   help=f"Gripper close value 0-100 (default: {GRIPPER_CLOSE})")
    p.add_argument("--no-precision", action="store_true",
                   help="Disable precision mode (use same speed for all movements)")
    p.add_argument("--no-confirm", action="store_true", 
                   help="Skip initial confirmation prompt")
    p.add_argument("--verbose", action="store_true", 
                   help="Enable debug logging")
    p.add_argument("--pos-tolerance", type=float, default=COORD_TOLERANCE,
                   help=f"Position tolerance in mm (default: {COORD_TOLERANCE})")
    p.add_argument("--ori-tolerance", type=float, default=ORIENT_TOLERANCE,
                   help=f"Orientation tolerance in degrees (default: {ORIENT_TOLERANCE})")
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                   help=f"Max retry attempts per position (default: {MAX_RETRIES})")
    return p.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    # Quiet noisy external libraries (robot telemetry) while keeping our important logs
    try:
        logging.getLogger('pymycobot').setLevel(logging.WARNING)
        logging.getLogger('pymycobot.mycobot').setLevel(logging.WARNING)
    except Exception:
        pass
    
    # Update global tolerances if provided
    global COORD_TOLERANCE, ORIENT_TOLERANCE, MAX_RETRIES
    COORD_TOLERANCE = args.pos_tolerance
    ORIENT_TOLERANCE = args.ori_tolerance
    MAX_RETRIES = args.max_retries
    
    # Initialize robot (always needed)
    if not init_robot(dry_run=False):
        logging.error("Failed to initialize robot")
        return 1
    
    # TEST GRIPPER MODE
    if args.test_gripper:
        success = test_gripper()
        return 0 if success else 1
    
    # MANUAL MODE: Record positions by moving robot
    if args.manual:
        try:
            start_coords, end_coords = record_manual_positions(args.positions, args.sleep)
        except Exception as e:
            logging.error(f"Failed to record positions: {e}")
            return 1
    
    # AUTOMATIC MODE: Load previously recorded positions
    else:
        try:
            start_coords, end_coords = load_manual_positions(args.positions)
            logging.info(f"Loaded positions from: {args.positions}")
        except FileNotFoundError as e:
            logging.error(str(e))
            return 1
        except Exception as e:
            logging.error(f"Failed to load positions: {e}")
            return 1
    
    # Show confirmation prompt
    if not args.no_confirm:
        print(f"\n{'='*60}")
        print(f"  MechArm Pick-and-Place Sequence")
        print(f"{'='*60}")
        print(f"  Output file:    {args.output}")
        print(f"  Positions file: {args.positions}")
        print(f"  Mode:           {'MANUAL RECORD' if args.manual else 'AUTOMATIC'}")
        print(f"  Lift height:    {args.lift_height} mm")
        print(f"  Speed:          {args.speed} (normal), {SPEED_PRECISION} (precision)")
        print(f"  Precision mode: {'Disabled' if args.no_precision else 'Enabled'}")
        print(f"  Sleep time:     {args.sleep} s")
        print(f"  Gripper open:   {args.grip_open}")
        print(f"  Gripper close:  {args.grip_close}")
        if KEYBOARD_AVAILABLE:
            print(f"  Spacebar:       Enabled (pause/resume)")
        print(f"{'='*60}")
        print(f"\nStart: [{start_coords[0]:.1f}, {start_coords[1]:.1f}, {start_coords[2]:.1f}, "
              f"{start_coords[3]:.1f}, {start_coords[4]:.1f}, {start_coords[5]:.1f}]")
        print(f"End:   [{end_coords[0]:.1f}, {end_coords[1]:.1f}, {end_coords[2]:.1f}, "
              f"{end_coords[3]:.1f}, {end_coords[4]:.1f}, {end_coords[5]:.1f}]")
        print(f"{'='*60}\n")
        
        # If manual mode, positions are already recorded, ask if user wants to run sequence
        if args.manual:
            resp = input("Positions recorded. Run pick-and-place sequence now? [y/N]: ")
            if resp.lower() != 'y':
                logging.info("Positions saved. Run again without --manual to execute sequence.")
                return 0
        else:
            resp = input("Continue with pick-and-place sequence? [y/N]: ")
            if resp.lower() != 'y':
                logging.info("User cancelled")
                return 0
    
    # Build enhanced pick-and-place path
    all_positions = []

    # 0. Home (compute from origin joint angles)
    try:
        origin_joints = list(origin)
        origin_pos = fk(origin_joints)
        R_origin = np.array(rot_num(*np.radians(origin_joints)), dtype=float)
        r_o, p_o, y_o = rotation_matrix_to_euler_zyx(R_origin)
        origin_coords = [float(origin_pos[0]), float(origin_pos[1]), float(origin_pos[2]),
                         float(np.degrees(r_o)), float(np.degrees(p_o)), float(np.degrees(y_o))]
    except Exception:
        # Fallback: zeros if fk/rot fail
        origin_coords = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Above-start and above-end waypoints
    above_start = start_coords.copy()
    above_start[2] = float(above_start[2]) + args.lift_height

    above_end = end_coords.copy()
    above_end[2] = float(above_end[2]) + args.lift_height

    # Middle waypoint (midpoint in Cartesian space and average orientation)
    mid_coords = [
        float((start_coords[0] + end_coords[0]) / 2.0),
        float((start_coords[1] + end_coords[1]) / 2.0),
        float((start_coords[2] + end_coords[2]) / 2.0),
        float((start_coords[3] + end_coords[3]) / 2.0),
        float((start_coords[4] + end_coords[4]) / 2.0),
        float((start_coords[5] + end_coords[5]) / 2.0)
    ]

    # Sequence assembly
    # 1) Home (all zeros)
    all_positions.append((-5, origin_coords))

    # 2) Move above start (safe approach)
    all_positions.append((-3, above_start))

    # 3) Move down to start (pick)
    all_positions.append((-1, start_coords))

    # 4) Lift up from start
    all_positions.append((-3, above_start))

    # 5) Transit via middle waypoint
    all_positions.append((-6, mid_coords))

    # 6) Move above end
    all_positions.append((-4, above_end))

    # 7) Move down to end (place)
    all_positions.append((-2, end_coords))

    # 8) Lift up from end (safe retreat)
    all_positions.append((-4, above_end))

    # 9) Return home
    all_positions.append((-5, origin_coords))
    
    logging.info(f"Sequence: START -> CLOSE GRIPPER -> LIFT (+{args.lift_height}mm) -> "
                f"MOVE_TO_END (+{args.lift_height}mm) -> END -> OPEN GRIPPER")
    
    # Execute sequence
    try:
        pick_and_place_sequence(
            all_positions, 
            args.output, 
            speed=args.speed, 
            sleep_time=args.sleep,
            gripper_open=args.grip_open,
            gripper_close=args.grip_close,
            precision_mode=not args.no_precision  # ✓ FIXED
        )
        logging.info("Sequence completed successfully")
        return 0
    except KeyboardInterrupt:
        logging.error("\n⚠ EMERGENCY STOP - Keyboard interrupt detected!")
        if mc is not None:
            try:
                mc.stop()  # Emergency stop
                logging.info("Robot stopped")
            except Exception as e:
                logging.error(f"Failed to stop robot: {e}")
        return 130
    except Exception as e:
        logging.error(f"Sequence failed: {e}", exc_info=args.verbose)
        if mc is not None:
            try:
                mc.stop()  # Stop on error
                check_robot_errors()
            except:
                pass
        return 1


if __name__ == '__main__':
    sys.exit(main())
