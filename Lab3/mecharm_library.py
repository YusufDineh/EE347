import math
import time
import csv
import logging
import numpy as np
from sympy.matrices import Matrix
from scipy.optimize import least_squares
from sympy import symbols, cos, sin, pi, lambdify
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Try to import robot library
try:
    from pymycobot.mycobot import MyCobot
    from pymycobot import PI_PORT, PI_BAUD
    PYMCOBOT_AVAILABLE = True
except ImportError:
    PYMCOBOT_AVAILABLE = False
    logging.warning("pymycobot not available - only dry-run mode supported")

# Joint angle limits (degrees)
JOINT_LIMITS = [
    (-165, 165),  # J1
    (-90, 90),    # J2
    (-180, 65),   # J3
    (-160, 160),  # J4
    (-115, 115),  # J5
    (-175, 175)   # J6
]

# Workspace safety limits (mm)
MIN_REACH = 50.0
MAX_Z = 400.0
MIN_Z = -50.0

# Error correction settings
MAX_RETRIES = 3
COORD_TOLERANCE = 5.0
ORIENT_TOLERANCE = 5.0

# Gripper settings
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1
GRIPPER_SLEEP = 2.0

# Speed settings
SPEED_PRECISION = 15
SPEED_NORMAL = 30

# Global symbolic joint variables
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')
right = pi/2
origin = [0, 0, 0, 0, 0, 0]

dh_table = [
    [0.049, 0, 135.926, q1],
    [0, -right, 0, q2 - right],
    [99.973, 0, 0, q3],
    [10.012, -right, 107.011, q4],
    [0.003, right, 0.006, q5],
    [0.054, -right, 64.973, q6]
]

def get_transformation_matrix(a, alpha, d, theta):
    return Matrix([
        [cos(theta),                          -sin(theta),                   0,                     a],
        [sin(theta)*cos(alpha),     cos(theta)*cos(alpha),         -sin(alpha),         -sin(alpha)*d],
        [sin(theta)*sin(alpha),     cos(theta)*sin(alpha),          cos(alpha),          cos(alpha)*d],
        [0,                                             0,                   0,                     1]
    ])

def overall_transformation(dh_table):
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

class MechArmLibrary:
    def __init__(self, dry_run: bool = False):
        self.mc = None
        self.dry_run = dry_run
        self.init_robot()

    def init_robot(self) -> bool:
        if self.dry_run:
            return True
        
        if not PYMCOBOT_AVAILABLE:
            logging.error("pymycobot library not available")
            return False
        
        try:
            self.mc = MyCobot(PI_PORT, PI_BAUD)
            self.mc.power_on()
            time.sleep(1.0)
            
            if not self.verify_robot_status():
                logging.error("Robot status verification failed")
                return False

            self.check_robot_errors()
            return True

        except Exception as e:
            logging.error(f"Failed to initialize robot: {e}")
            return False

    def translate_error_code(self, code: int) -> str:
        if code == 0:
            return "No error message"
        elif 1 <= code <= 6:
            return f"Joint {code} exceeds the limit position"
        elif 16 <= code <= 19:
            return "Collision protection"
        elif code == 32:
            return "Kinematics inverse solution has no solution"
        elif 33 <= code <= 34:
            return "Linear motion has no adjacent solution"
        else:
            return None

    def check_robot_errors(self) -> bool:
        if self.mc is None:
            return False
        
        try:
            error_msg = self.mc.get_error_information()
            if error_msg and error_msg != 0:
                translated_msg = self.translate_error_code(error_msg)
                if translated_msg:
                    logging.error(f"Robot error detected: {translated_msg}")
                    self.mc.clear_error_information()
                    return True
                else:
                    # Remove logging for untranslated error codes as requested
                    self.mc.clear_error_information()
                    return True
            return False
            
        except Exception as e:
            logging.warning(f"Failed to check robot errors: {e}")
            return False

    def verify_robot_status(self) -> bool:
        if self.mc is None:
            return False
        
        try:
            if not self.mc.is_power_on():
                logging.error("Robot is not powered on")
                return False
            
            if not self.mc.is_all_servo_enable():
                logging.error("Not all servos are enabled")
                return False
            
            if self.mc.is_paused():
                self.mc.resume()
                time.sleep(0.5)
            
            self.check_robot_errors()
            return True
            
        except Exception as e:
            logging.error(f"Failed to verify robot status: {e}")
            return False

    def validate_joint_angles(self, q: np.ndarray) -> None:
        for i, (angle, (min_ang, max_ang)) in enumerate(zip(q, JOINT_LIMITS)):
            if not (min_ang <= angle <= max_ang):
                raise ValueError(f"Joint {i+1} angle {angle:.1f} outside limits")

    def validate_workspace(self, x: float, y: float, z: float) -> None:
        distance = math.sqrt(x*x + y*y + z*z)
        if distance < MIN_REACH:
            raise ValueError(f"Target too close to base: {distance:.1f}mm")
        if z > MAX_Z:
            raise ValueError(f"Target too high: {z:.1f}mm")
        if z < MIN_Z:
            raise ValueError(f"Target too low: {z:.1f}mm")

    def fk(self, q_vals: np.ndarray) -> np.ndarray:
        q = np.asarray(q_vals, dtype=float)
        q = np.radians(q)
        xyz = np.asarray(fk_num(*q), dtype=float).ravel()
        return xyz

    def rotation_matrix_to_euler_zyx(self, R: np.ndarray, transpose: bool = False) -> Tuple[float, float, float]:
        if transpose:
            R = R.T
        
        sin_beta = -R[2, 0]
        sin_beta = np.clip(sin_beta, -1.0, 1.0)
        pitch = np.arcsin(sin_beta)
        
        cos_beta = np.cos(pitch)
        
        if np.abs(cos_beta) < 1e-10:
            yaw = 0.0
            roll = np.arctan2(-R[0, 1], R[1, 1])
        else:
            roll = np.arctan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
            yaw = np.arctan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
        
        return roll, pitch, yaw

    def position_error(self, q_position: np.ndarray, x_target: float, y_target: float, z_target: float) -> np.ndarray:
        pos = self.fk(q_position)
        return np.array([
            pos[0] - float(x_target),
            pos[1] - float(y_target),
            pos[2] - float(z_target)
        ])

    def orientation_error(self, q_orientation: np.ndarray, rx_d: float, ry_d: float, rz_d: float) -> np.ndarray:
        def _wrap_angle(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        q = np.asarray(q_orientation, dtype=float)
        q_rad = np.radians(q)
        rx_d_rad = np.radians(rx_d)
        ry_d_rad = np.radians(ry_d)
        rz_d_rad = np.radians(rz_d)

        R = np.array(rot_num(*q_rad), dtype=float)
        roll, pitch, yaw = self.rotation_matrix_to_euler_zyx(R, transpose=False)

        err_roll = _wrap_angle(roll - rx_d_rad)
        err_pitch = _wrap_angle(pitch - ry_d_rad)
        err_yaw = _wrap_angle(yaw - rz_d_rad)

        return np.array([err_roll, err_pitch, err_yaw])

    def link_lengths(self) -> float:
        total_length = 0.0
        for a, alpha, d, theta in dh_table:
            total_length += abs(float(a)) + abs(float(d))
        return total_length

    def inverse_kinematics(self, x_target: float, y_target: float, z_target: float, 
                           rx_d: float, ry_d: float, rz_d: float, 
                           q_init: np.ndarray, max_reach: float,
                           max_iterations: int = 5000, tolerance: float = 1e-6) -> np.ndarray:
        self.validate_workspace(x_target, y_target, z_target)
        
        distance_from_origin = math.hypot(x_target, y_target, z_target)
        if distance_from_origin > max_reach:
            raise ValueError(f"Target outside robot reach")
        
        q_init = np.asarray(q_init, dtype=float)
        
        def combined_residual(q_all, pos_weight=1.0, ori_weight=10.0):
            pos_err = self.position_error(q_all, x_target, y_target, z_target)
            ori_err = self.orientation_error(q_all, rx_d, ry_d, rz_d)
            return np.concatenate([pos_err * pos_weight, ori_err * ori_weight])
        
        lower_bounds = np.array([limit[0] for limit in JOINT_LIMITS])
        upper_bounds = np.array([limit[1] for limit in JOINT_LIMITS])
        bounds = (lower_bounds, upper_bounds)
        
        result = least_squares(
            combined_residual, 
            q_init,
            bounds=bounds,
            method='trf',
            max_nfev=max_iterations,
            ftol=tolerance,
            xtol=tolerance,
            verbose=0
        )
        
        q_solution = result.x
        self.validate_joint_angles(q_solution)
        
        return q_solution

    def control_gripper(self, flag: int, speed: int = 70, gripper_type: int = 1, sleep_time: float = GRIPPER_SLEEP) -> bool:
        """
        Control gripper state.
        flag: 0 - open, 1 - close, 254 - release
        speed: 1 ~ 100
        gripper_type: 1 - Adaptive, 2 - Nimble, 3 - Parallel, 4 - Flexible
        """
        if self.mc is None:
            return False
        
        try:
            if flag not in [0, 1, 254]:
                logging.warning(f"Invalid gripper flag: {flag}")
                return False
            
            speed = max(1, min(100, int(speed)))
            self.mc.set_gripper_state(flag, speed, gripper_type)
            time.sleep(sleep_time)
            self.check_robot_errors()
            return True
        except Exception as e:
            logging.error(f"Failed to control gripper: {e}")
            self.check_robot_errors()
            return False

    def get_current_positions(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if self.mc is None:
            return None, None
        
        try:
            coords = self.mc.get_coords()
            angles = self.mc.get_angles()
            
            if coords is None or angles is None:
                return None, None
            
            return angles, coords
        except Exception as e:
            logging.error(f"Failed to read robot positions: {e}")
            return None, None

    def compare_coords(self, target: List[float], actual: List[float], 
                       pos_tol: float = COORD_TOLERANCE, 
                       ori_tol: float = ORIENT_TOLERANCE) -> Tuple[bool, dict]:
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

    def send_angles_with_correction(self, target_angles: List[float], target_coords: List[float],
                                    speed: int, timeout: float = 15) -> Tuple[bool, List[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                self.mc.sync_send_angles(target_angles, speed, timeout=timeout)
                
                current_angles, current_coords = self.get_current_positions()
                
                if current_coords is None:
                    continue
                
                coords_match, errors = self.compare_coords(target_coords, current_coords)
                
                if coords_match:
                    return True, current_coords
                
            except Exception as e:
                logging.warning(f"Movement failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.0)
        
        return False, target_coords

    def reset(self, sleep_time: float, speed: int = 30) -> None:
        if self.mc is None:
            return
        
        try:
            self.mc.power_off()
            time.sleep(sleep_time)
            self.mc.power_on()
            time.sleep(sleep_time)
            self.mc.send_angles(origin, speed)
            time.sleep(sleep_time)
        except Exception as e:
            logging.error(f"Reset failed: {e}")

    def load_targets_from_csv(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load target joint angles and coordinates from a CSV file.
        Replaces request_to_send with a more descriptive name.
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
            raise ValueError(f"CSV must have at least 13 columns")
        
        joint_angles = data[:, 1:7]
        positions = data[:, 7:10]
        orientations = data[:, 10:13]
        
        return joint_angles, positions, orientations

    def teach_positions(self, save_path: str, sleep_time: float = 2.0) -> Tuple[List[float], List[float]]:
        """
        Interactively teach START and END positions by manually moving the robot.
        Replaces record_manual_positions with a more descriptive name.
        """
        if self.mc is None:
            raise RuntimeError("Robot not initialized")
        
        try:
            print("Releasing servos for manual positioning...")
            self.mc.release_all_servos()
            
            input("Move robot to START position, then press Enter...")
            
            self.mc.power_on()
            time.sleep(sleep_time)
            
            _, start_coords = self.get_current_positions()
            if start_coords is None:
                raise RuntimeError("Failed to read START position")
            
            print("Releasing servos again for END position...")
            self.mc.release_all_servos()
            
            input("Move robot to END position, then press Enter...")
            
            self.mc.power_on()
            time.sleep(sleep_time)
            
            _, end_coords = self.get_current_positions()
            if end_coords is None:
                raise RuntimeError("Failed to read END position")
            
            with open(save_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Position", "X", "Y", "Z", "RX", "RY", "RZ"])
                writer.writerow(["START"] + start_coords)
                writer.writerow(["END"] + end_coords)
            
            return start_coords, end_coords
            
        except Exception as e:
            try:
                self.mc.power_on()
            except:
                pass
            raise e

    def load_taught_positions(self, load_path: str) -> Tuple[List[float], List[float]]:
        """
        Load previously taught START and END positions from CSV.
        Replaces load_manual_positions with a more descriptive name.
        """
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Positions file not found: {load_path}")
        
        try:
            with open(load_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                
                start_row = next(reader)
                end_row = next(reader)
                
                start_coords = [float(x) for x in start_row[1:7]]
                end_coords = [float(x) for x in end_row[1:7]]
                
                return start_coords, end_coords
                
        except Exception as e:
            raise ValueError(f"Failed to parse positions CSV: {e}")

    def log_performance_data(self, file_path: str, target: List[float], actual: List[float], 
                           attempts: int, success: bool, duration: float = 0.0):
        """
        Log detailed execution metrics for optimization analysis.
        Records target vs actual positions, errors, attempt counts, and timing.
        """
        file_exists = Path(file_path).exists()
        
        # Calculate errors
        errors = []
        if len(target) >= 6 and len(actual) >= 6:
            errors = [abs(t - a) for t, a in zip(target[:6], actual[:6])]
        else:
            errors = [0.0] * 6
            
        headers = [
            "timestamp", 
            "target_x", "target_y", "target_z", "target_rx", "target_ry", "target_rz",
            "actual_x", "actual_y", "actual_z", "actual_rx", "actual_ry", "actual_rz",
            "error_x", "error_y", "error_z", "error_rx", "error_ry", "error_rz",
            "attempts", "success", "duration"
        ]
        
        row_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_x": target[0], "target_y": target[1], "target_z": target[2],
            "target_rx": target[3], "target_ry": target[4], "target_rz": target[5],
            "actual_x": actual[0], "actual_y": actual[1], "actual_z": actual[2],
            "actual_rx": actual[3], "actual_ry": actual[4], "actual_rz": actual[5],
            "error_x": errors[0], "error_y": errors[1], "error_z": errors[2],
            "error_rx": errors[3], "error_ry": errors[4], "error_rz": errors[5],
            "attempts": attempts,
            "success": success,
            "duration": duration
        }
        
        try:
            with open(file_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
        except Exception as e:
            logging.error(f"Failed to log performance data: {e}")

    def test_gripper(self):
        if self.mc is None:
            return False
        
        try:
            self.control_gripper(0)
            time.sleep(2)
            self.control_gripper(1)
            time.sleep(2)
            self.control_gripper(0)
            time.sleep(1)
            return True
        except Exception as e:
            logging.error(f"Gripper test failed: {e}")
            return False
