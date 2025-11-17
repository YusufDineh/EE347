#!/usr/bin/env python3
"""
Independent Robot IK Data Collection System

Each robot runs completely independently - no communication needed.
Test cases are pre-divided and assigned by robot ID.
Data is collected locally, then merged offline later.

Usage on each Raspberry Pi:
  # Robot 0 (of 4 total robots)
  python independent_collection.py --output robot0_data.csv --robot-id 0 --total-robots 4 --mode full
  
  # Robot 1 (of 4 total robots)
  python independent_collection.py --output robot1_data.csv --robot-id 1 --total-robots 4 --mode full
  
  # Robot 2 (of 4 total robots)
  python independent_collection.py --output robot2_data.csv --robot-id 2 --total-robots 4 --mode full
  
  # Robot 3 (of 4 total robots)
  python independent_collection.py --output robot3_data.csv --robot-id 3 --total-robots 4 --mode full

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
    (-165, 165), (-90, 90), (-180, 65),
    (-160, 160), (-115, 115), (-175, 175)
]

WORKSPACE_ZONES = {
    'near': (50, 150), 'mid': (150, 250),
    'far': (250, 350), 'extreme': (350, 400)
}

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
    
    def combined_residual(q):
        pos_err = position_error(q, target_xyz)
        ori_err = orientation_error(q, target_rpy)
        return np.concatenate([pos_err * 1.0, ori_err * 10.0])
    
    lower_bounds = np.array([limit[0] for limit in JOINT_LIMITS])
    upper_bounds = np.array([limit[1] for limit in JOINT_LIMITS])
    
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
                
                # Workspace validation
                if z < -50 or z > 400:
                    continue
                if math.sqrt(x*x + y*y + z*z) < 50:
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
        
        logging.info("Robot initialized successfully")
        return mc
    except Exception as e:
        logging.error(f"Robot initialization failed: {e}")
        return None

def execute_test_case(mc: Optional[MyCobot], test_case: Dict, 
                     fast_mode: bool, robot_id: int) -> Dict:
    """Execute single test case and collect all metrics"""
    
    start_time = time.time()
    
    # Extract target
    target_xyz = np.array([test_case['x'], test_case['y'], test_case['z']])
    target_rpy = np.radians(np.array([test_case['rx'], test_case['ry'], test_case['rz']]))
    
    # Get initial guess
    if mc is not None:
        current = mc.get_angles()
        q_init = np.array(current if current else [0]*6)
    else:
        q_init = np.array([0, 0, 0, 0, 0, 0])
    
    # Compute IK
    ik_result = inverse_kinematics(target_xyz, target_rpy, q_init)
    
    if not ik_result['success']:
        return {
            'robot_id': robot_id,
            'test_id': test_case['id'],
            'ik_success': False,
            'ik_iterations': ik_result['iterations'],
            'ik_time_sec': ik_result['time'],
            'executed': False,
            'timestamp': datetime.now().isoformat()
        }
    
    q_solution = ik_result['q_solution']
    
    # FK verification
    fk_xyz, fk_R = fk_full(q_solution)
    fk_rpy = np.array(rotation_matrix_to_euler_zyx(fk_R))
    
    # Execute on robot
    measured_coords = None
    if mc is not None:
        try:
            speed = 50 if fast_mode else 30
            sleep_time = 1.5 if fast_mode else 3.0
            
            mc.send_angles(q_solution.tolist(), speed)
            time.sleep(sleep_time)
            
            measured_coords = mc.get_coords()
        except Exception as e:
            logging.error(f"Movement failed: {e}")
    
    # Build complete result
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
        
        # Execution
        'executed': measured_coords is not None,
        'total_time_sec': time.time() - start_time
    }
    
    # Measured values
    if measured_coords is not None:
        result.update({
            'meas_x': measured_coords[0],
            'meas_y': measured_coords[1],
            'meas_z': measured_coords[2],
            'meas_rx': measured_coords[3],
            'meas_ry': measured_coords[4],
            'meas_rz': measured_coords[5],
            'meas_pos_error': math.sqrt(
                (measured_coords[0] - test_case['x'])**2 +
                (measured_coords[1] - test_case['y'])**2 +
                (measured_coords[2] - test_case['z'])**2
            ),
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
    logging.info(f"Total test suite: {len(all_test_cases)} cases")
    
    # Partition: Get only THIS robot's test cases
    my_test_cases = partition_test_cases(all_test_cases, robot_id, total_robots)
    logging.info(f"Robot {robot_id} assigned: {len(my_test_cases)} cases "
                f"({100*len(my_test_cases)/len(all_test_cases):.1f}% of total)")
    
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
        'executed', 'meas_x', 'meas_y', 'meas_z', 'meas_rx', 'meas_ry', 'meas_rz',
        'meas_pos_error', 'meas_ori_error', 'total_time_sec',
        'joint_1_util', 'joint_2_util', 'joint_3_util',
        'joint_4_util', 'joint_5_util', 'joint_6_util'
    ]
    
    # Execute test cases
    start_time = time.time()
    success_count = 0
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for i, test_case in enumerate(my_test_cases):
            logging.info(f"Processing {i+1}/{len(my_test_cases)}: Test ID {test_case['id']}")
            
            result = execute_test_case(mc, test_case, fast_mode, robot_id)
            writer.writerow(result)
            
            if result.get('ik_success', False):
                success_count += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(my_test_cases) - i - 1) / rate if rate > 0 else 0
                logging.info(f"Progress: {i+1}/{len(my_test_cases)} "
                           f"({100*(i+1)/len(my_test_cases):.1f}%) "
                           f"| Success: {success_count} "
                           f"| Rate: {rate:.2f} tests/sec "
                           f"| ETA: {eta/60:.1f} min")
    
    total_time = time.time() - start_time
    
    # Shutdown
    if mc is not None:
        try:
            mc.power_off()
        except:
            pass
    
    # Summary
    logging.info("="*60)
    logging.info(f"COLLECTION COMPLETE - Robot {robot_id}")
    logging.info(f"Total tests: {len(my_test_cases)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {len(my_test_cases) - success_count}")
    logging.info(f"Time: {total_time/60:.1f} minutes")
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
    
    # Validate robot ID
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
    logging.info(f"Output: {args.output}")
    logging.info("="*60)
    
    # Run collection
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

if __name__ == '__main__':
    sys.exit(main())