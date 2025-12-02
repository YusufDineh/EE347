#!/usr/bin/env python3
"""
Independent Robot IK Data Collection System

Each robot runs completely independently - no communication needed.
Test cases are pre-divided and assigned by robot ID.
Data is collected locally, then merged offline later.

Usage on each Raspberry Pi:
  # Robot 0 (of 4 total robots)
  python IK_Data_Collection_and_Validation.py --output robot0_data.csv --robot-id 0 --total-robots 4 --mode full
  
  # Robot 1 (of 4 total robots)
  python IK_Data_Collection_and_Validation.py --output robot1_data.csv --robot-id 1 --total-robots 4 --mode full
  
  # Robot 2 (of 4 total robots)
  python IK_Data_Collection_and_Validation.py --output robot2_data.csv --robot-id 2 --total-robots 4 --mode full
  
  # Robot 3 (of 4 total robots)
  python IK_Data_Collection_and_Validation.py --output robot3_data.csv --robot-id 3 --total-robots 4 --mode full

After collection, merge on any computer:
  python merge_robot_data.py --input "robot*_data.csv" --output combined_data.csv
"""

import math
import time
import csv
import argparse
import logging
import sys
from typing import List, Tuple, Optional, Dict
import numpy as np
from sympy.matrices import Matrix
from scipy.optimize import least_squares
from sympy import symbols, cos, sin, pi, lambdify
from pathlib import Path
from datetime import datetime
import hashlib
import tempfile
import shutil
import os

# Import robot library
try:
    from pymycobot.mycobot import MyCobot
    from pymycobot import PI_PORT, PI_BAUD
    PYMCOBOT_AVAILABLE = True
except ImportError:
    PYMCOBOT_AVAILABLE = False
    logging.warning("pymycobot not available - dry-run mode only")

# ============================================================================
# CONFIGURATION
# ============================================================================

JOINT_LIMITS = [
    (-155, 155), (-80, 80), (-170, 55),
    (-150, 150), (-105, 105), (-165, 165)
]

WORKSPACE_ZONES = {
    'near': (50, 150), 'mid': (150, 250),
    'far': (250, 350), 'extreme': (350, 400)
}

# Safety boundaries from recorded workspace points
# Based on manual measurements: floor and base collision points
SAFETY_MIN_Z = 80.0        # Minimum Z height (above floor at 63.8mm + safety margin)
SAFETY_MIN_RADIUS = 135.0  # Minimum radial distance (beyond base collision at ~125mm)

# ============================================================================
# FORWARD/INVERSE KINEMATICS
# ============================================================================

q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

def get_transformation_matrix(a, alpha, d, theta):
    return Matrix([
        [cos(theta), -sin(theta), 0, a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
        [0, 0, 0, 1]
    ])

def build_fk():
    symbolic_dh = [
        [0.049, 0, 135.926, q1],
        [0, -pi/2, 0, q2 - pi/2],
        [99.973, 0, 0, q3],
        [10.012, -pi/2, 107.011, q4],
        [0.003, pi/2, 0.006, q5],
        [0.054, -pi/2, 64.973, q6]
    ]
    
    T = Matrix(np.identity(4))
    for params in symbolic_dh:
        a, alpha, d, theta = params
        T = T * get_transformation_matrix(a, alpha, d, theta)
    
    fk_pos = lambdify((q1, q2, q3, q4, q5, q6), T[:3, 3], modules='numpy')
    fk_rot = lambdify((q1, q2, q3, q4, q5, q6), T[:3, :3], modules='numpy')
    return fk_pos, fk_rot

fk_pos_func, fk_rot_func = build_fk()

def fk(q_vals: np.ndarray) -> np.ndarray:
    q_rad = np.radians(np.asarray(q_vals, dtype=float))
    return np.asarray(fk_pos_func(*q_rad), dtype=float).ravel()

def fk_full(q_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_rad = np.radians(np.asarray(q_vals, dtype=float))
    xyz = np.asarray(fk_pos_func(*q_rad), dtype=float).ravel()
    R = np.array(fk_rot_func(*q_rad), dtype=float)
    return xyz, R

def rotation_matrix_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    sin_beta = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_beta)
    cos_beta = np.cos(pitch)
    
    if np.abs(cos_beta) < 1e-10:
        yaw = 0.0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
        yaw = np.arctan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
    
    return roll, pitch, yaw

def position_error(q: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    return fk(q) - target_xyz

def orientation_error(q: np.ndarray, target_rpy: np.ndarray) -> np.ndarray:
    _, R = fk_full(q)
    roll, pitch, yaw = rotation_matrix_to_euler_zyx(R)
    current_rpy = np.array([roll, pitch, yaw])
    error = current_rpy - target_rpy
    return (error + np.pi) % (2*np.pi) - np.pi

def inverse_kinematics(target_xyz: np.ndarray, target_rpy: np.ndarray, 
                      q_init: np.ndarray) -> Dict:
    start_time = time.time()
    
    # Validate target is within safe workspace (floor and base collision check)
    x, y, z = target_xyz[0], target_xyz[1], target_xyz[2]
    radial_dist = math.sqrt(x*x + y*y + z*z)
    
    if z < SAFETY_MIN_Z:
        logging.warning(f"Target Z={z:.1f}mm below safety limit {SAFETY_MIN_Z}mm (floor at ~64mm)")
        return {
            'success': False,
            'q_solution': q_init,
            'iterations': 0,
            'time': time.time() - start_time,
            'residual': float('inf')
        }
    
    if radial_dist < SAFETY_MIN_RADIUS:
        logging.warning(f"Target radius={radial_dist:.1f}mm below safety limit {SAFETY_MIN_RADIUS}mm (base at ~125mm)")
        return {
            'success': False,
            'q_solution': q_init,
            'iterations': 0,
            'time': time.time() - start_time,
            'residual': float('inf')
        }
    
    # Validate initial guess is within bounds (with margin for safety)
    for i, (angle, (min_ang, max_ang)) in enumerate(zip(q_init, JOINT_LIMITS)):
        # Check if angle is too close to limits (10 degree safety margin)
        if angle < (min_ang + 10) or angle > (max_ang - 10):
            logging.warning(f"Initial guess joint {i+1} near limit: {angle:.1f}° (safe range: [{min_ang+10}, {max_ang-10}])")
            # Fall back to neutral position
            q_init = np.array([0, 0, 0, 0, 0, 0], dtype=float)
            logging.info("Switching to neutral initial guess [0,0,0,0,0,0]")
            break
    
    def combined_residual(q):
        pos_err = position_error(q, target_xyz)
        ori_err = orientation_error(q, target_rpy)
        return np.concatenate([pos_err * 1.0, ori_err * 10.0])
    
    lower_bounds = np.array([limit[0] for limit in JOINT_LIMITS])
    upper_bounds = np.array([limit[1] for limit in JOINT_LIMITS])
    
    try:
        result = least_squares(
            combined_residual, q_init,
            bounds=(lower_bounds, upper_bounds),
            method='trf', max_nfev=5000, ftol=1e-6, xtol=1e-6, verbose=0
        )
        
        return {
            'success': result.success,
            'q_solution': result.x,
            'iterations': result.nfev,
            'time': time.time() - start_time,
            'residual': result.cost
        }
    except ValueError as e:
        if "infeasible" in str(e).lower():
            logging.error(f"IK failed: Initial guess infeasible. q_init={q_init.tolist()}")
            return {
                'success': False,
                'q_solution': q_init,
                'iterations': 0,
                'time': time.time() - start_time,
                'residual': float('inf')
            }
        else:
            # Re-raise other ValueErrors
            raise

# ============================================================================
# WORKSPACE UTILITIES
# ============================================================================

def classify_workspace_zone(x: float, y: float, z: float) -> str:
    distance = math.sqrt(x*x + y*y + z*z)
    for zone_name, (min_d, max_d) in WORKSPACE_ZONES.items():
        if min_d <= distance < max_d:
            return zone_name
    return 'unknown'

def proximity_to_limits(q: np.ndarray) -> float:
    min_dist = float('inf')
    for angle, (min_ang, max_ang) in zip(q, JOINT_LIMITS):
        min_dist = min(min_dist, abs(angle - min_ang), abs(angle - max_ang))
    return min_dist

def compute_joint_utilization(q: np.ndarray) -> List[float]:
    utilization = []
    for angle, (min_ang, max_ang) in zip(q, JOINT_LIMITS):
        range_size = max_ang - min_ang
        position_in_range = (angle - min_ang) / range_size
        utilization.append(position_in_range * 100.0)
    return utilization

def spherical_to_cartesian(radius: float, elevation_deg: float, azimuth_deg: float):
    elev_rad = math.radians(elevation_deg)
    azim_rad = math.radians(azimuth_deg)
    x = radius * math.cos(elev_rad) * math.cos(azim_rad)
    y = radius * math.cos(elev_rad) * math.sin(azim_rad)
    z = radius * math.sin(elev_rad)
    return x, y, z

# ============================================================================
# DETERMINISTIC TEST CASE GENERATION
# ============================================================================

def generate_all_test_cases(mode: str, density: int = 5) -> List[Dict]:
    """
    Generate complete test suite deterministically.
    All robots will generate the EXACT same test cases.
    """
    test_cases = []
    case_id = 0
    
    # Use fixed seeds and parameters so all robots generate identical tests
    if mode == 'quick':
        density = 3
    elif mode == 'full':
        density = 7
    
    radii = np.linspace(100, 300, density)
    elevations = np.linspace(-30, 60, density)
    azimuths = np.linspace(0, 360, density, endpoint=False)
    
    # Standard orientations to test
    orientations = [
        (0, 0, 0),      # Neutral
        (0, 45, 0),     # Pitch up
        (0, -45, 0),    # Pitch down
        (45, 0, 0),     # Roll right
        (-45, 0, 0),    # Roll left
        (0, 0, 45),     # Yaw CW
        (0, 0, -45),    # Yaw CCW
    ]
    
    # Generate all combinations deterministically
    for r in radii:
        for elev in elevations:
            for azim in azimuths:
                x, y, z = spherical_to_cartesian(r, elev, azim)
                
                # Workspace validation using recorded safety boundaries
                # Floor safety: Z must be above 80mm (floor at 63.8mm + margin)
                if z < SAFETY_MIN_Z:
                    continue
                
                # Height limit
                if z > 400:
                    continue
                
                # Base collision safety: radial distance must be > 135mm (base at ~125mm + margin)
                radial_dist = math.sqrt(x*x + y*y + z*z)
                if radial_dist < SAFETY_MIN_RADIUS:
                    continue
                
                # Test each orientation
                for rx, ry, rz in orientations:
                    test_cases.append({
                        'id': case_id,
                        'x': x, 'y': y, 'z': z,
                        'rx': rx, 'ry': ry, 'rz': rz,
                        'zone': classify_workspace_zone(x, y, z)
                    })
                    case_id += 1
    
    return test_cases

def partition_test_cases(all_cases: List[Dict], robot_id: int, total_robots: int) -> List[Dict]:
    """
    Partition test cases for this specific robot using modulo distribution.
    
    This ensures:
    - No overlap between robots
    - Even distribution of test cases
    - Deterministic (same result every time)
    
    Example with 4 robots and 12 test cases:
      Robot 0: cases [0, 4, 8]     (IDs % 4 == 0)
      Robot 1: cases [1, 5, 9]     (IDs % 4 == 1)
      Robot 2: cases [2, 6, 10]    (IDs % 4 == 2)
      Robot 3: cases [3, 7, 11]    (IDs % 4 == 3)
    """
    my_cases = [case for case in all_cases if case['id'] % total_robots == robot_id]
    return my_cases

# ============================================================================
# ROBOT DATA COLLECTION
# ============================================================================

def is_safe_position(q_solution: np.ndarray, min_z_safe: float = SAFETY_MIN_Z) -> Tuple[bool, str]:
    """
    Check if the joint solution will result in a safe position.
    
    Uses recorded workspace boundaries:
    - Floor collision at Z=63.8mm -> safety limit at 80mm
    - Base collision at radius=125mm -> safety limit at 135mm
    
    Returns (is_safe, reason_if_not_safe)
    """
    # Check FK position
    fk_xyz, _ = fk_full(q_solution)
    x_pos, y_pos, z_pos = fk_xyz[0], fk_xyz[1], fk_xyz[2]
    
    # Check minimum Z height (floor safety)
    if z_pos < min_z_safe:
        return False, f"Z position too low: {z_pos:.1f}mm (min safe: {min_z_safe}mm, floor at ~64mm)"
    
    # Check radial distance (base collision safety)
    radial_dist = math.sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos)
    if radial_dist < SAFETY_MIN_RADIUS:
        return False, f"Too close to base: {radial_dist:.1f}mm (min safe: {SAFETY_MIN_RADIUS}mm, base at ~125mm)"
    
    # Check joint limits
    for i, (angle, (min_ang, max_ang)) in enumerate(zip(q_solution, JOINT_LIMITS)):
        if angle < min_ang or angle > max_ang:
            return False, f"Joint {i+1} out of limits: {angle:.1f}° (range: [{min_ang}, {max_ang}])"
    
    return True, ""

def reset_robot_servos(mc: Optional[MyCobot], sleep_time: float = 2.0) -> bool:
    """
    Reset robot by releasing and re-enabling servos.
    
    This can help recover from stuck states where servos are locked
    but not responding to movement commands.
    
    Args:
        mc: Robot instance
        sleep_time: Wait time between release and re-enable
        
    Returns:
        True if reset successful, False otherwise
    """
    if mc is None:
        return False
    
    try:
        logging.info("Resetting robot servos...")
        
        # Release all servos
        mc.release_all_servos()
        logging.info("  Servos released")
        time.sleep(sleep_time)
        
        # Re-enable servos
        mc.power_on()
        logging.info("  Servos re-enabled")
        time.sleep(sleep_time)
        
        # Verify robot is ready
        if verify_robot_status(mc):
            logging.info("✓ Robot reset successful")
            return True
        else:
            logging.warning("Robot reset completed but status check failed")
            return False
            
    except Exception as e:
        logging.error(f"Failed to reset robot servos: {e}")
        return False

def check_robot_errors(mc: Optional[MyCobot]) -> bool:
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

def verify_robot_status(mc: Optional[MyCobot]) -> bool:
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
        check_robot_errors(mc)
        
        logging.debug("Robot status verification passed")
        return True
        
    except Exception as e:
        logging.error(f"Failed to verify robot status: {e}")
        return False

def init_robot() -> Optional[MyCobot]:
    """Initialize robot connection"""
    if not PYMCOBOT_AVAILABLE:
        logging.error("pymycobot not available")
        return None
    
    try:
        mc = MyCobot(PI_PORT, PI_BAUD)
        mc.power_on()
        time.sleep(1.0)
        
        if not mc.is_power_on():
            logging.error("Robot failed to power on")
            return None
        
        # Verify robot status
        if not verify_robot_status(mc):
            logging.error("Robot status verification failed")
            return None
        
        # Clear any existing errors
        check_robot_errors(mc)
        
        logging.info("Robot initialized successfully")
        return mc
    except Exception as e:
        logging.error(f"Robot initialization failed: {e}")
        return None

def get_current_position(mc: Optional[MyCobot]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get current joint angles and Cartesian coordinates from robot.
    
    Returns:
        Tuple of (angles_array, coords_array) where:
        - angles_array: numpy array of 6 joint angles in degrees
        - coords_array: numpy array of 6 values [x, y, z, rx, ry, rz]
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
        
        # Convert to numpy arrays
        angles_array = np.array(angles, dtype=float)
        coords_array = np.array(coords, dtype=float)
        
        return angles_array, coords_array
    except Exception as e:
        logging.error(f"Failed to read robot positions: {e}")
        return None, None

def execute_test_case(mc: Optional[MyCobot], test_case: Dict, 
                     fast_mode: bool, robot_id: int) -> Dict:
    """Execute single test case and collect all metrics"""
    
    start_time = time.time()
    
    # Extract target
    target_xyz = np.array([test_case['x'], test_case['y'], test_case['z']])
    target_rpy = np.radians(np.array([test_case['rx'], test_case['ry'], test_case['rz']]))
    
    # Initialize tracking variables
    measured_coords = None
    measured_angles = None
    movement_success = False
    movement_attempted = False
    retry_attempted = False
    warning_flags = []  # Track all warnings
    
    # Get initial guess with safety validation
    if mc is not None:
        current = mc.get_angles()
        if current is not None:
            q_current = np.array(current, dtype=float)
            
            # Check if current position is safe to use as initial guess
            is_safe_init = True
            for i, (angle, (min_ang, max_ang)) in enumerate(zip(q_current, JOINT_LIMITS)):
                # Give 10 degree margin from limits
                if angle < (min_ang + 10) or angle > (max_ang - 10):
                    logging.warning(f"Test {test_case['id']}: Joint {i+1} too close to limit: {angle:.1f}° (safe range: [{min_ang+10}, {max_ang-10}])")
                    is_safe_init = False
                    warning_flags.append(f"init_j{i+1}_near_limit")
                    break
            
            if is_safe_init:
                q_init = q_current
                logging.debug(f"Test {test_case['id']}: Using current position as initial guess")
            else:
                # Use neutral position as fallback
                q_init = np.array([0, 0, 0, 0, 0, 0], dtype=float)
                logging.warning(f"Test {test_case['id']}: Current position unsafe - using neutral initial guess [0,0,0,0,0,0]")
                warning_flags.append("using_neutral_init")
        else:
            q_init = np.array([0, 0, 0, 0, 0, 0], dtype=float)
            logging.warning(f"Test {test_case['id']}: Could not read current angles - using neutral initial guess")
            warning_flags.append("no_current_angles")
    else:
        q_init = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    
    # Compute IK
    ik_result = inverse_kinematics(target_xyz, target_rpy, q_init)
    
    if not ik_result['success']:
        warning_flags.append("ik_failed")
        # CHANGED: Still return data even if IK failed
        return {
            'robot_id': robot_id,
            'test_id': test_case['id'],
            'zone': test_case.get('zone', 'unknown'),
            'target_x': test_case['x'],
            'target_y': test_case['y'],
            'target_z': test_case['z'],
            'target_rx': test_case['rx'],
            'target_ry': test_case['ry'],
            'target_rz': test_case['rz'],
            'ik_success': False,
            'ik_iterations': ik_result['iterations'],
            'ik_time_sec': ik_result['time'],
            'executed': False,
            'movement_detected': False,
            'movement_attempted': False,
            'retry_attempted': False,
            'warnings': ','.join(warning_flags),
            'timestamp': datetime.now().isoformat(),
            # Fill in zeros for missing fields
            'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0, 'q5': 0, 'q6': 0,
            'q_init_1': q_init[0], 'q_init_2': q_init[1], 'q_init_3': q_init[2],
            'q_init_4': q_init[3], 'q_init_5': q_init[4], 'q_init_6': q_init[5],
            'fk_x': 0, 'fk_y': 0, 'fk_z': 0, 'fk_rx': 0, 'fk_ry': 0, 'fk_rz': 0,
            'fk_pos_error': 0, 'fk_ori_error': 0,
            'meas_x': 0, 'meas_y': 0, 'meas_z': 0,
            'meas_rx': 0, 'meas_ry': 0, 'meas_rz': 0,
            'meas_j1': 0, 'meas_j2': 0, 'meas_j3': 0,
            'meas_j4': 0, 'meas_j5': 0, 'meas_j6': 0,
            'meas_pos_error': 0, 'meas_ori_error': 0,
            'radial_distance': math.sqrt(test_case['x']**2 + test_case['y']**2 + test_case['z']**2),
            'proximity_to_limits': 0,
            'total_time_sec': time.time() - start_time,
            'joint_1_util': 0, 'joint_2_util': 0, 'joint_3_util': 0,
            'joint_4_util': 0, 'joint_5_util': 0, 'joint_6_util': 0
        }
    
    q_solution = ik_result['q_solution']
    
    # FK verification
    fk_xyz, fk_R = fk_full(q_solution)
    fk_rpy = np.array(rotation_matrix_to_euler_zyx(fk_R))
    
    # Safety check before execution
    is_safe, safety_reason = is_safe_position(q_solution)
    if not is_safe:
        logging.warning(f"Test {test_case['id']} SKIPPED - Safety check failed: {safety_reason}")
        warning_flags.append(f"unsafe:{safety_reason.split(':')[0]}")
    
    # Execute on robot with movement verification
    if mc is not None and is_safe:
        movement_attempted = True
        try:
            speed = 50 if fast_mode else 30 
            False 
            sleep_time = 1.5 if fast_mode else 3.0
            
            # Check robot status before movement
            if not verify_robot_status(mc):
                logging.error(f"Test {test_case['id']}: Robot not ready for movement")
                warning_flags.append("robot_not_ready")
                raise Exception("Robot status check failed")
            
            # Get initial position for movement verification
            initial_angles, initial_coords = get_current_position(mc)
            if initial_angles is None:
                logging.error(f"Test {test_case['id']}: Cannot read initial position")
                warning_flags.append("no_initial_position")
                raise Exception("Robot not responding")
            
            # Send movement command
            logging.debug(f"Test {test_case['id']}: Sending angles {q_solution.tolist()}")
            mc.send_angles(q_solution.tolist(), speed)
            
            # Wait for movement to start
            time.sleep(sleep_time)
            
            # Wait for robot to finish moving (check if still moving)
            max_wait = 10  # seconds
            start_wait = time.time()
            while time.time() - start_wait < max_wait:
                try:
                    if not mc.is_moving():
                        break
                except:
                    # is_moving() might not be available, just break
                    break
                time.sleep(0.1)
            
            # Additional settling time
            time.sleep(0.5)
            
            # Read final position
            measured_angles, measured_coords = get_current_position(mc)
            
            # Verify movement actually occurred
            if measured_angles is not None and initial_angles is not None:
                movement = np.abs(measured_angles - initial_angles)
                max_movement = float(np.max(movement))
                
                if max_movement < 0.5:  # Less than 0.5 degrees on all joints
                    logging.warning(f"Test {test_case['id']}: Robot did not move! Max movement: {max_movement:.2f}°")
                    logging.warning(f"  Target angles: {q_solution.tolist()}")
                    logging.warning(f"  Initial angles: {initial_angles.tolist()}")
                    logging.warning(f"  Final angles: {measured_angles.tolist()}")
                    warning_flags.append(f"no_movement:{max_movement:.2f}deg")
                    
                    # Attempt to reset robot servos
                    logging.warning("  Attempting servo reset to recover from stuck state...")
                    retry_attempted = True
                    if reset_robot_servos(mc, sleep_time=1.5):
                        logging.info("  Servo reset successful - retrying movement...")
                        warning_flags.append("servo_reset_success")
                        
                        # Retry movement after reset
                        mc.send_angles(q_solution.tolist(), speed)
                        time.sleep(sleep_time)
                        
                        # Wait for movement completion
                        start_wait = time.time()
                        while time.time() - start_wait < max_wait:
                            try:
                                if not mc.is_moving():
                                    break
                            except:
                                break
                            time.sleep(0.1)
                        
                        time.sleep(0.5)
                        
                        # Check if retry worked
                        retry_angles, retry_coords = get_current_position(mc)
                        if retry_angles is not None:
                            retry_movement = np.abs(retry_angles - initial_angles)
                            retry_max = float(np.max(retry_movement))
                            
                            if retry_max >= 0.5:
                                logging.info(f"  ✓ Retry successful! Movement: {retry_max:.2f}°")
                                measured_angles = retry_angles
                                measured_coords = retry_coords
                                movement_success = True
                                warning_flags.append(f"retry_success:{retry_max:.2f}deg")
                            else:
                                logging.error(f"  ✗ Retry failed - robot still stuck (movement: {retry_max:.2f}°)")
                                warning_flags.append(f"retry_failed:{retry_max:.2f}deg")
                                movement_success = False
                        else:
                            logging.error("  ✗ Cannot read position after retry")
                            warning_flags.append("retry_no_position")
                            movement_success = False
                    else:
                        logging.error("  ✗ Servo reset failed")
                        warning_flags.append("servo_reset_failed")
                        movement_success = False
                else:
                    logging.debug(f"Test {test_case['id']}: Movement detected: {max_movement:.2f}° max")
                    movement_success = True
            
            # Check for robot errors after movement
            if check_robot_errors(mc):
                logging.warning(f"Test {test_case['id']}: Errors detected after movement")
                warning_flags.append("robot_errors_after_move")
            
        except Exception as e:
            logging.error(f"Movement failed: {e}")
            warning_flags.append(f"movement_exception:{str(e)[:30]}")
            check_robot_errors(mc)
    
    # Build complete result - ALWAYS RECORD DATA
    result = {
        'robot_id': robot_id,
        'test_id': test_case['id'],
        'zone': test_case['zone'],
        'timestamp': datetime.now().isoformat(),
        
        # Target
        'target_x': test_case['x'],
        'target_y': test_case['y'],
        'target_z': test_case['z'],
        'target_rx': test_case['rx'],
        'target_ry': test_case['ry'],
        'target_rz': test_case['rz'],
        
        # IK solution
        'q1': q_solution[0],
        'q2': q_solution[1],
        'q3': q_solution[2],
        'q4': q_solution[3],
        'q5': q_solution[4],
        'q6': q_solution[5],
        
        # Initial guess
        'q_init_1': q_init[0],
        'q_init_2': q_init[1],
        'q_init_3': q_init[2],
        'q_init_4': q_init[3],
        'q_init_5': q_init[4],
        'q_init_6': q_init[5],
        
        # FK verification
        'fk_x': fk_xyz[0],
        'fk_y': fk_xyz[1],
        'fk_z': fk_xyz[2],
        'fk_rx': np.degrees(fk_rpy[0]),
        'fk_ry': np.degrees(fk_rpy[1]),
        'fk_rz': np.degrees(fk_rpy[2]),
        
        # Model errors
        'fk_pos_error': np.linalg.norm(fk_xyz - target_xyz),
        'fk_ori_error': np.degrees(np.linalg.norm(fk_rpy - target_rpy)),
        
        # IK metrics
        'ik_success': True,
        'ik_iterations': ik_result['iterations'],
        'ik_time_sec': ik_result['time'],
        'ik_residual': ik_result['residual'],
        
        # Workspace
        'radial_distance': math.sqrt(test_case['x']**2 + test_case['y']**2 + test_case['z']**2),
        'proximity_to_limits': proximity_to_limits(q_solution),
        
        # Execution - CHANGED: Record even if failed
        'executed': measured_coords is not None and movement_success,
        'movement_detected': movement_success,
        'movement_attempted': movement_attempted,
        'retry_attempted': retry_attempted,
        'warnings': ','.join(warning_flags) if warning_flags else '',
        'total_time_sec': time.time() - start_time
    }
    
    # Measured values (using ndarray from get_current_position)
    if measured_coords is not None and measured_angles is not None:
        result.update({
            # Measured Cartesian coordinates
            'meas_x': float(measured_coords[0]),
            'meas_y': float(measured_coords[1]),
            'meas_z': float(measured_coords[2]),
            'meas_rx': float(measured_coords[3]),
            'meas_ry': float(measured_coords[4]),
            'meas_rz': float(measured_coords[5]),
            # Measured joint angles
            'meas_j1': float(measured_angles[0]),
            'meas_j2': float(measured_angles[1]),
            'meas_j3': float(measured_angles[2]),
            'meas_j4': float(measured_angles[3]),
            'meas_j5': float(measured_angles[4]),
            'meas_j6': float(measured_angles[5]),
            # Position error
            'meas_pos_error': float(np.linalg.norm(measured_coords[:3] - target_xyz)),
            # Orientation error
            'meas_ori_error': math.sqrt(
                (measured_coords[3] - test_case['rx'])**2 +
                (measured_coords[4] - test_case['ry'])**2 +
                (measured_coords[5] - test_case['rz'])**2
            )
        })
    else:
        result.update({
            'meas_x': 0, 'meas_y': 0, 'meas_z': 0,
            'meas_rx': 0, 'meas_ry': 0, 'meas_rz': 0,
            'meas_j1': 0, 'meas_j2': 0, 'meas_j3': 0,
            'meas_j4': 0, 'meas_j5': 0, 'meas_j6': 0,
            'meas_pos_error': 0, 'meas_ori_error': 0
        })
    
    # Joint utilization
    utilization = compute_joint_utilization(q_solution)
    for i, util in enumerate(utilization, 1):
        result[f'joint_{i}_util'] = util
    
    return result

def run_collection(robot_id: int, total_robots: int, mode: str, 
                  density: int, output_csv: str, fast_mode: bool,
                  dry_run: bool = False):
    """Main collection loop for this robot"""
    
    # Generate ALL test cases (deterministically)
    logging.info("Generating complete test suite...")
    all_test_cases = generate_all_test_cases(mode, density)
    total_cases = len(all_test_cases)
    logging.info(f"Total test suite: {total_cases} cases")
    
    if total_cases == 0:
        logging.error("No test cases generated - aborting")
        return
    
    # Partition: Get only THIS robot's test cases
    my_test_cases = partition_test_cases(all_test_cases, robot_id, total_robots)
    my_count = len(my_test_cases)
    pct = 100.0 * my_count / total_cases if total_cases > 0 else 0.0
    logging.info(f"Robot {robot_id} assigned: {my_count} cases ({pct:.1f}% of total)")
    
    # Show which test IDs this robot will handle
    test_ids = [tc['id'] for tc in my_test_cases]
    logging.info(f"Test IDs: {test_ids[:10]}{'...' if len(test_ids) > 10 else ''}")
    
    # Initialize robot
    mc = None
    if not dry_run:
        mc = init_robot()
        if mc is None:
            logging.error("Cannot initialize robot - aborting")
            return
        
        # Perform initial servo reset to ensure robot is in good state
        logging.info("Performing initial robot reset...")
        if not reset_robot_servos(mc, sleep_time=2.0):
            logging.warning("Initial servo reset failed - continuing anyway")
        
        # Move to home position (all zeros)
        logging.info("Moving robot to home position (all zeros)...")
        try:
            home_position = [0, 0, 0, 0, 0, 0]
            mc.send_angles(home_position, 30)
            time.sleep(3.0)
            
            # Verify we reached home
            current_angles, _ = get_current_position(mc)
            if current_angles is not None:
                max_error = float(np.max(np.abs(current_angles)))
                if max_error < 2.0:
                    logging.info(f"✓ Robot at home position (max error: {max_error:.2f}°)")
                else:
                    logging.warning(f"Robot not quite at home (max error: {max_error:.2f}°)")
            else:
                logging.warning("Cannot verify home position")
        except Exception as e:
            logging.error(f"Failed to move to home position: {e}")
            logging.warning("Continuing with collection anyway...")
    
    # Ensure output directory exists
    out_path = Path(output_csv)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV header
    fieldnames = [
        'robot_id', 'test_id', 'zone', 'timestamp',
        'target_x', 'target_y', 'target_z', 'target_rx', 'target_ry', 'target_rz',
        'q1', 'q2', 'q3', 'q4', 'q5', 'q6',
        'q_init_1', 'q_init_2', 'q_init_3', 'q_init_4', 'q_init_5', 'q_init_6',
        'fk_x', 'fk_y', 'fk_z', 'fk_rx', 'fk_ry', 'fk_rz',
        'fk_pos_error', 'fk_ori_error',
        'ik_success', 'ik_iterations', 'ik_time_sec', 'ik_residual',
        'radial_distance', 'proximity_to_limits',
        'executed', 'movement_detected', 'movement_attempted', 'retry_attempted', 'warnings',
        'meas_x', 'meas_y', 'meas_z', 'meas_rx', 'meas_ry', 'meas_rz',
        'meas_j1', 'meas_j2', 'meas_j3', 'meas_j4', 'meas_j5', 'meas_j6',
        'meas_pos_error', 'meas_ori_error', 'total_time_sec',
        'joint_1_util', 'joint_2_util', 'joint_3_util',
        'joint_4_util', 'joint_5_util', 'joint_6_util'
    ]
    
    # Use atomic write to temp file then move
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=out_path.name + '.', dir=str(out_path.parent))
    os.close(tmp_fd)
    
    # Execute test cases
    start_time = time.time()
    success_count = 0
    writer = None
    f = None
    
    try:
        f = open(tmp_path, 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for i, test_case in enumerate(my_test_cases):
            logging.info(f"Processing {i+1}/{len(my_test_cases)}: Test ID {test_case['id']}")
            
            try:
                result = execute_test_case(mc, test_case, fast_mode, robot_id)
                writer.writerow(result)
                f.flush()
                os.fsync(f.fileno())
                
                if result.get('ik_success', False):
                    success_count += 1
            except KeyboardInterrupt:
                logging.info("Interrupted by user - finishing and saving partial results")
                break
            except Exception as e:
                logging.exception(f"Unexpected error processing test {test_case['id']}: {e}")
                # Continue to next test so one failure doesn't abort whole run
                continue
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(my_test_cases) - i - 1) / rate if rate > 0 else 0
                logging.info(f"Progress: {i+1}/{len(my_test_cases)} "
                           f"({100*(i+1)/len(my_test_cases):.1f}%) "
                           f"| Success: {success_count} "
                           f"| Rate: {rate:.2f} tests/sec "
                           f"| ETA: {eta/60:.1f} min")
    finally:
        # Close file and move into place
        if f:
            f.close()
        try:
            shutil.move(tmp_path, str(out_path))
            logging.info(f"Successfully wrote results to {output_csv}")
        except Exception:
            # If move fails, leave temp file and log
            logging.exception(f"Failed to move temp CSV to final destination. Data saved in: {tmp_path}")
        
        # Shutdown robot
        if mc is not None:
            try:
                mc.power_off()
                logging.info("Robot powered off successfully")
            except Exception:
                logging.exception("Failed to power off robot cleanly")
    
    total_time = time.time() - start_time
    
    # Summary
    logging.info("="*60)
    logging.info(f"COLLECTION COMPLETE - Robot {robot_id}")
    logging.info(f"Total tests assigned: {len(my_test_cases)}")
    logging.info(f"Successful IK solves: {success_count}")
    logging.info(f"Failed: {len(my_test_cases) - success_count}")
    logging.info(f"Time: {total_time/60:.1f} minutes")
    if total_time > 0:
        logging.info(f"Rate: {len(my_test_cases)/total_time:.2f} tests/sec")
    logging.info(f"Output: {output_csv}")
    logging.info("="*60)

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent Robot IK Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # On Robot 0's Raspberry Pi (of 4 total robots):
  python independent_collection.py --robot-id 0 --total-robots 4 --output robot0.csv --mode full --fast
  
  # On Robot 1's Raspberry Pi:
  python independent_collection.py --robot-id 1 --total-robots 4 --output robot1.csv --mode full --fast
  
  # On Robot 2's Raspberry Pi:
  python independent_collection.py --robot-id 2 --total-robots 4 --output robot2.csv --mode full --fast
  
  # On Robot 3's Raspberry Pi:
  python independent_collection.py --robot-id 3 --total-robots 4 --output robot3.csv --mode full --fast

After collection, copy all CSV files to one computer and merge:
  python merge_robot_data.py --input "robot*.csv" --output combined_data.csv
        """
    )
    parser.add_argument('--robot-id', type=int, required=True,
                       help='This robot\'s ID (0-indexed, e.g., 0, 1, 2, 3)')
    parser.add_argument('--total-robots', type=int, required=True,
                       help='Total number of robots collecting data')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CSV file for this robot')
    parser.add_argument('--mode', choices=['full', 'quick', 'grid'],
                       default='quick',
                       help='Collection mode')
    parser.add_argument('--grid-density', type=int, default=5,
                       help='Grid density')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode (speed=50, sleep=1.5s)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test without robot')
    parser.add_argument('--verbose', action='store_true',
                       help='Debug logging')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate arguments
    if args.total_robots <= 0:
        logging.error(f"Invalid total_robots {args.total_robots} (must be > 0)")
        return 1
    
    if args.robot_id < 0 or args.robot_id >= args.total_robots:
        logging.error(f"Invalid robot ID {args.robot_id} (must be 0 to {args.total_robots-1})")
        return 1
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    logging.info("="*60)
    logging.info(f"INDEPENDENT ROBOT DATA COLLECTION")
    logging.info(f"Robot ID: {args.robot_id} of {args.total_robots} total robots")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Fast mode: {args.fast}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info(f"Output: {args.output}")
    logging.info("="*60)
    
    # Run collection
    try:
        run_collection(
            robot_id=args.robot_id,
            total_robots=args.total_robots,
            mode=args.mode,
            density=args.grid_density,
            output_csv=args.output,
            fast_mode=args.fast,
            dry_run=args.dry_run
        )
        return 0
    except KeyboardInterrupt:
        logging.info("\nCollection interrupted by user")
        return 130
    except Exception as e:
        logging.exception(f"Fatal error during collection: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())