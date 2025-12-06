"""
Ultra-Simplified MechArm Pick and Place - Straight Line Motion
===============================================================
Uses simple linear interpolation in joint space for smooth motion.
No RRT, no collision detection, no optimization - just smooth straight lines.

JSON Format (mission_simple.json):
{
  "start_angles": [0, 0, 0, 0, 0, 0],
  "safe_height_offset": 100.0,
  "waypoint_density": 20,
  "pegs": [
    {
      "peg_id": 0,
      "color": "RED",
      "pick_angles": [j1, j2, j3, j4, j5, j6],
      "pick_pose": [x, y, z, rx, ry, rz],
      "place_angles": [j1, j2, j3, j4, j5, j6],
      "place_pose": [x, y, z, rx, ry, rz]
    }
  ]
}
"""

import time
import json
import logging
import numpy as np
from scipy.optimize import least_squares
from sympy import symbols, cos, sin, pi, lambdify
from sympy.matrices import Matrix
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict

# Hardware Interface
try:
    from pymycobot.mycobot import MyCobot
    from pymycobot import PI_PORT, PI_BAUD
    PYMYCOBOT_AVAILABLE = True
except ImportError:
    PYMYCOBOT_AVAILABLE = False
    logging.warning("pymycobot not available - dry-run mode only")

# --- Configuration ---
JOINT_LIMITS = [
    (-165, 165), (-90, 90), (-180, 65),
    (-160, 160), (-115, 115), (-175, 175)
]

SPEED_PRECISION = 70
SPEED_NORMAL = 70
GRIPPER_OPEN = 80
GRIPPER_CLOSE = 20
GRIPPER_SLEEP = 2.0

# Kinematics Setup (DH Parameters)
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')
right = pi/2

dh_table = [
    [0.049, 0, 135.926, q1],
    [0, -right, 0, q2 - right],
    [99.973, 0, 0, q3],
    [10.012, -right, 107.011, q4],
    [0.003, right, 0.006, q5],
    [0.054, -right, 64.973, q6]
]

# --- Data Structures ---

@dataclass
class PegTask:
    """Pick and place task definition."""
    peg_id: int
    color: str
    pick_angles: List[float]
    pick_pose: List[float]
    place_angles: List[float]
    place_pose: List[float]
    
    def to_dict(self) -> Dict: 
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict): 
        return cls(**data)

@dataclass
class PlannedPath:
    """Computed trajectory."""
    path_id: str
    waypoints: List[List[float]]
    
    def to_dict(self) -> Dict:
        return {
            'path_id': self.path_id,
            'waypoints': [[float(x) for x in wp] for wp in self.waypoints]
        }
    
    @classmethod
    def from_dict(cls, data: Dict): 
        return cls(**data)

@dataclass
class MissionConfig:
    """Full mission specification."""
    start_angles: List[float]
    safe_height_offset: float
    waypoint_density: int  # Number of waypoints per path
    pegs: List[PegTask]
    planned_paths: Dict[str, PlannedPath]
    
    def to_dict(self) -> Dict:
        return {
            'start_angles': self.start_angles,
            'safe_height_offset': self.safe_height_offset,
            'waypoint_density': self.waypoint_density,
            'pegs': [p.to_dict() for p in self.pegs],
            'planned_paths': {k: v.to_dict() for k, v in self.planned_paths.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            data['start_angles'],
            data['safe_height_offset'],
            data.get('waypoint_density', 20),
            [PegTask.from_dict(p) for p in data['pegs']],
            {k: PlannedPath.from_dict(v) for k, v in data.get('planned_paths', {}).items()}
        )
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f: 
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Mission saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f: 
            return cls.from_dict(json.load(f))

# --- Kinematics ---

def get_transformation_matrix(a, alpha, d, theta):
    """Compute DH transformation matrix."""
    return Matrix([
        [cos(theta), -sin(theta), 0, a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
        [0, 0, 0, 1]
    ])

def overall_transformation(dh_table):
    """Compute total transformation."""
    T = Matrix(np.identity(4))
    for params in dh_table:
        T = T * get_transformation_matrix(*params)
    return T

T_symbolic = overall_transformation(dh_table)
fk_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, 3], modules='numpy')
rot_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, :3], modules='numpy')

class KinematicsEngine:
    """Forward and inverse kinematics."""
    
    @staticmethod
    def forward_kinematics(q_values: np.ndarray) -> np.ndarray:
        """Compute XYZ from joint angles."""
        q = np.radians(np.asarray(q_values, dtype=float))
        return np.asarray(fk_num(*q), dtype=float).ravel()
    
    @staticmethod
    def rotation_matrix_to_euler_zyx(R: np.ndarray, transpose: bool = False) -> tuple:
        """Convert rotation matrix to Euler angles."""
        if transpose: R = R.T
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
    
    def inverse_kinematics(self, target_pos: List[float], target_ori: List[float],
                          q_init: List[float], max_iterations: int = 5000) -> np.ndarray:
        """Solve IK using Levenberg-Marquardt."""
        x_t, y_t, z_t = target_pos
        rx_d, ry_d, rz_d = target_ori
        q_init = np.asarray(q_init, dtype=float)
        
        # Lock J4, J5, J6 to initial values (prevent wrist rotation)
        fixed_joints = q_init[3:6]
        
        def combined_residual(q_3d, pos_weight=1.0, ori_weight=10.0):
            # Reconstruct full 6D configuration
            q = np.concatenate([q_3d, fixed_joints])
            
            pos = self.forward_kinematics(q)
            pos_err = np.array([pos[0]-x_t, pos[1]-y_t, pos[2]-z_t])
            
            q_rad = np.radians(q)
            R = np.array(rot_num(*q_rad), dtype=float)
            r, p, y = self.rotation_matrix_to_euler_zyx(R)
            
            def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
            ori_err = np.array([
                wrap(r - np.radians(rx_d)),
                wrap(p - np.radians(ry_d)),
                wrap(y - np.radians(rz_d))
            ])
            return np.concatenate([pos_err * pos_weight, ori_err * ori_weight])
        
        bounds_3d = (np.array([l[0] for l in JOINT_LIMITS[:3]]), 
                     np.array([l[1] for l in JOINT_LIMITS[:3]]))
        
        res = least_squares(combined_residual, q_init[:3], bounds=bounds_3d, method='trf', 
                          max_nfev=max_iterations, ftol=1e-6, xtol=1e-6, verbose=0)
        
        q_sol = np.concatenate([res.x, fixed_joints])
        return q_sol

# --- Simple Path Planner ---

def interpolate_joint_path(start: List[float], end: List[float], min_density: int) -> List[List[float]]:
    """Create smooth linear interpolation with adaptive density (max 3 deg/step)."""
    start = np.array(start)
    end = np.array(end)
    
    # Adaptive density
    max_step = 3.0 
    max_diff = np.max(np.abs(end - start))
    needed = int(np.ceil(max_diff / max_step))
    num_waypoints = max(min_density, needed)
    
    waypoints = []
    for i in range(num_waypoints):
        alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 1.0
        waypoint = start + alpha * (end - start)
        waypoints.append(waypoint.tolist())
    
    return waypoints

# --- Robot Controller ---

class MechArmController:
    """Hardware interface."""
    
    def __init__(self, dry_run: bool = False):
        self.mc = None
        self.dry_run = dry_run
        self.init_robot()
    
    def init_robot(self) -> bool:
        if self.dry_run: return True
        if not PYMYCOBOT_AVAILABLE: return False
        try:
            self.mc = MyCobot(PI_PORT, PI_BAUD)
            self.mc.power_on()
            time.sleep(1.0)
            return True
        except Exception as e:
            logging.error(f"Init failed: {e}")
            return False
    
    def send_angles(self, angles: List[float], speed: int, timeout: float = 0.2) -> bool:
        if self.dry_run: 
            print(f"   [DRY RUN] Would send: {[f'{a:.1f}' for a in angles]}")
            return True
        try:
            self.mc.send_angles(angles, speed)
            time.sleep(timeout)
            return True
        except: return False
    
    def control_gripper(self, state: int, speed: int = 70, sleep_time: float = GRIPPER_SLEEP) -> bool:
        if self.dry_run:
            print(f"   [DRY RUN] Gripper set to: {state}")
            return True
        try:
            # Use set_gripper_value for 0-100 range control
            self.mc.set_gripper_value(state, speed,1)
            time.sleep(sleep_time)
            return True
        except Exception as e:
            logging.error(f"Gripper error: {e}")
            return False
    
    def power_on(self):
        if self.dry_run: return
        if self.mc: self.mc.power_on()

    def power_off(self):
        if self.dry_run: return
        if self.mc: self.mc.power_off()

    def release_all_servos(self):
        if self.dry_run:
            print("   [DRY RUN] Releasing servos")
            return
        if self.mc: self.mc.release_all_servos()

    def get_angles(self) -> Optional[List[float]]:
        if self.dry_run: return [0.0]*6
        if self.mc: return self.mc.get_angles()
        return None

    def get_coords(self) -> Optional[List[float]]:
        if self.dry_run: return [0.0]*6
        if self.mc: return self.mc.get_coords()
        return None

# --- Mission Planner ---

class SimplifiedMissionPlanner:
    """Plan and execute with straight-line interpolation."""
    
    def __init__(self, ctrl: MechArmController, kin: KinematicsEngine):
        self.ctrl = ctrl
        self.kin = kin
        self.config: Optional[MissionConfig] = None
    
    def load_mission(self, filepath: str) -> bool:
        """Load mission from JSON."""
        try:
            print(f"\n→ Loading mission from: {filepath}")
            
            if not Path(filepath).exists():
                logging.error(f"❌ File not found: {filepath}")
                return False
            
            self.config = MissionConfig.load(filepath)
            print(f"   ✓ Loaded {len(self.config.pegs)} peg(s)")
            print(f"   ✓ Waypoint density: {self.config.waypoint_density} points per path")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Load error: {e}")
            return False
    
    def plan_all_paths(self) -> bool:
        """Generate straight-line interpolated paths."""
        if not self.config:
            logging.error("No mission loaded")
            return False
        
        print("\n=== Planning Paths (Straight Line Interpolation) ===")
        
        self.config.planned_paths = {}
        curr_angles = self.config.start_angles
        curr_seed = np.array(curr_angles) # Always start IK from previous valid joint angles
        density = self.config.waypoint_density
        
        for i, peg in enumerate(self.config.pegs):
            print(f"\n{'='*50}")
            print(f"Planning Peg {i+1}/{len(self.config.pegs)}: {peg.color}")
            print(f"{'='*50}")
            
            try:
                # Extract positions
                pick_xyz = peg.pick_pose[:3]
                place_xyz = peg.place_pose[:3]
                
                # Compute approach positions
                pick_approach_xyz = [pick_xyz[0], pick_xyz[1], 
                                    pick_xyz[2] + self.config.safe_height_offset]
                place_approach_xyz = [place_xyz[0], place_xyz[1], 
                                     place_xyz[2] + self.config.safe_height_offset]
                
                # Compute approach angles
                print(f"→ Computing Pick Approach IK...")
                # Use curr_seed to ensure consistent configuration
                
                pick_approach_angles = self.kin.inverse_kinematics(
                    pick_approach_xyz, peg.pick_pose[3:], curr_seed, max_iterations=3000
                ).tolist()
                curr_seed = np.array(pick_approach_angles) # Update seed
                print(f"   ✓ J4={pick_approach_angles[3]:.1f}°, J5={pick_approach_angles[4]:.1f}°")
                
                print(f"→ Computing Place Approach IK...")
                # Use curr_seed
                place_approach_angles = self.kin.inverse_kinematics(
                    place_approach_xyz, peg.place_pose[3:], curr_seed, max_iterations=3000
                ).tolist()
                curr_seed = np.array(place_approach_angles) # Update seed
                print(f"   ✓ J4={place_approach_angles[3]:.1f}°, J5={place_approach_angles[4]:.1f}°")
                
                # Generate paths
                print(f"→ Generating paths (adaptive density)...")
                
                # 1. Travel to Pick Approach (Joint Interpolation)
                self.config.planned_paths[f"peg_{i}_to_pick_approach"] = PlannedPath(
                    f"peg_{i}_to_pick_approach",
                    interpolate_joint_path(curr_angles, pick_approach_angles, density)
                )
                
                # 2. Pick Descent (Cartesian Straight Line)
                self.config.planned_paths[f"peg_{i}_pick_descent"] = PlannedPath(
                    f"peg_{i}_pick_descent",
                    self._plan_cartesian(pick_approach_angles, peg.pick_pose, density, final_joint_target=peg.pick_angles)
                )
                
                # 3. Pick Ascent (Cartesian Straight Line)
                # Target is approach pose (xyz + pick orientation)
                approach_pose_pick = pick_approach_xyz + peg.pick_pose[3:]
                self.config.planned_paths[f"peg_{i}_pick_ascent"] = PlannedPath(
                    f"peg_{i}_pick_ascent",
                    self._plan_cartesian(peg.pick_angles, approach_pose_pick, density)
                )
                
                # 4. Travel to Place Approach (Joint Interpolation)
                self.config.planned_paths[f"peg_{i}_travel"] = PlannedPath(
                    f"peg_{i}_travel",
                    interpolate_joint_path(pick_approach_angles, place_approach_angles, density)
                )
                
                # 5. Place Descent (Cartesian Straight Line)
                self.config.planned_paths[f"peg_{i}_place_descent"] = PlannedPath(
                    f"peg_{i}_place_descent",
                    self._plan_cartesian(place_approach_angles, peg.place_pose, density, final_joint_target=peg.place_angles)
                )
                
                # 6. Place Ascent (Cartesian Straight Line)
                # Target is approach pose (xyz + place orientation)
                approach_pose_place = place_approach_xyz + peg.place_pose[3:]
                self.config.planned_paths[f"peg_{i}_place_ascent"] = PlannedPath(
                    f"peg_{i}_place_ascent",
                    self._plan_cartesian(peg.place_angles, approach_pose_place, density)
                )
                
                print(f"   ✓ Generated 6 paths × {density} waypoints = {6 * density} total points")
                
                curr_angles = place_approach_angles
                
            except Exception as e:
                logging.error(f"❌ Planning error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n{'='*50}")
        print(f"✓ All paths planned: {len(self.config.planned_paths)} total")
        print(f"{'='*50}")
        return True
    
    def execute_mission(self) -> Dict:
        """Execute all planned paths."""
        if not self.config or not self.config.planned_paths:
            logging.error("No planned paths")
            return {'placed': 0}
        
        print("\n=== Executing Mission ===")
        
        # Move to start
        print("\n→ Moving to start position...")
        if not self.ctrl.send_angles(self.config.start_angles, SPEED_NORMAL, timeout=2):
            logging.error("Failed to reach start")
            return {'placed': 0}
        
        # Open gripper
        self.ctrl.control_gripper(GRIPPER_OPEN)
        
        score = {'placed': 0, 'failed': 0}
        
        for i, peg in enumerate(self.config.pegs):
            print(f"\n{'='*60}")
            print(f"Peg {i+1}/{len(self.config.pegs)}: {peg.color}")
            print(f"{'='*60}")
            
            try:
                # 1. Move to pick approach
                print("  [1/6] → Pick Approach")
                self._execute_path(self.config.planned_paths[f"peg_{i}_to_pick_approach"])
                
                # 2. Descend
                print("  [2/6] ↓ Descending")
                self._execute_path(self.config.planned_paths[f"peg_{i}_pick_descent"])
                
                # FORCE EXACT POSITION: Ensure we are exactly at the recorded pick angles
                self.ctrl.send_angles(peg.pick_angles, SPEED_PRECISION)
                time.sleep(0.5)
                
                # 3. Grab
                print("  [GRAB] ✋ Closing gripper")
                self.ctrl.control_gripper(GRIPPER_CLOSE)
                
                # 4. Ascend
                print("  [3/6] ↑ Ascending")
                self._execute_path(self.config.planned_paths[f"peg_{i}_pick_ascent"])
                
                # 5. Travel
                print("  [4/6] → Traveling to place")
                self._execute_path(self.config.planned_paths[f"peg_{i}_travel"])
                
                # 6. Descend to place
                print("  [5/6] ↓ Descending to place")
                self._execute_path(self.config.planned_paths[f"peg_{i}_place_descent"])
                
                # FORCE EXACT POSITION: Ensure we are exactly at the recorded place angles
                self.ctrl.send_angles(peg.place_angles, SPEED_PRECISION)
                time.sleep(0.5)
                
                # 7. Release
                print("  [RELEASE] 🖐️  Opening gripper")
                self.ctrl.control_gripper(GRIPPER_OPEN)
                score['placed'] += 1
                
                # 8. Ascend
                print("  [6/6] ↑ Returning to safe height")
                self._execute_path(self.config.planned_paths[f"peg_{i}_place_ascent"])
                
                print(f"  ✓✓✓ {peg.color} PLACED ✓✓✓")
                
            except Exception as e:
                logging.error(f"❌ Error: {e}")
                score['failed'] += 1
                try:
                    self.ctrl.control_gripper(GRIPPER_OPEN)
                except:
                    pass
        
        return score
    
    def _execute_path(self, path: PlannedPath):
        """Execute path with smooth timing."""
        if self.ctrl.dry_run:
            print(f"      [DRY RUN] Executing {len(path.waypoints)} waypoints")
            return
            
        for i, wp in enumerate(path.waypoints):
            self.ctrl.send_angles(wp, SPEED_PRECISION)
    
    def _plan_cartesian(self, start_angles: List[float], end_pose: List[float], min_points: int, final_joint_target: Optional[List[float]] = None) -> List[List[float]]:
        """Generate waypoints for a straight Cartesian line with FK verification and adaptive steps."""
        waypoints = []
        current_q = np.array(start_angles)
        
        # Get start position from FK
        start_pos = self.kin.forward_kinematics(current_q)
        target_pos = np.array(end_pose[:3])
        target_ori = np.array(end_pose[3:]) # Keep orientation constant/target
        
        # Adaptive density: max 2mm per step
        dist = np.linalg.norm(target_pos - start_pos)
        needed = int(np.ceil(dist / 2.0))
        num_points = max(min_points, needed)
        
        for i in range(1, num_points + 1):
            alpha = i / num_points
            # Linear interpolation of position only
            interp_pos = start_pos + alpha * (target_pos - start_pos)
            
            # Determine seed:
            # If we are close to the end (last 20% of points) and have a final target,
            # blend the seed towards the final target to guide IK into the correct configuration.
            seed = current_q
            if final_joint_target is not None and i > num_points * 0.8:
                seed = np.array(final_joint_target)

            # Solve IK
            # Try to solve with current_q as seed first to encourage smoothness
            sol = self.kin.inverse_kinematics(
                interp_pos, target_ori, seed, max_iterations=500
            )
            
            # --- Verification ---
            # Check if the solution actually reaches the target point (FK check)
            actual_pos = self.kin.forward_kinematics(sol)
            pos_error = np.linalg.norm(actual_pos - interp_pos)
            
            # Check for large joint jumps (smoothness)
            joint_jump = np.linalg.norm(sol - current_q)
            
            # If jump is too large, try re-solving with stricter seed or different approach
            if joint_jump >= 0.5: # 0.5 rad ~ 28 deg
                # Retry IK with current_q explicitly as seed (already done, but maybe try more iterations)
                sol_retry = self.kin.inverse_kinematics(
                    interp_pos, target_ori, current_q, max_iterations=2000
                )
                if np.linalg.norm(sol_retry - current_q) < joint_jump:
                    sol = sol_retry
                    joint_jump = np.linalg.norm(sol - current_q)

            # Thresholds: 10mm position error, ~45 deg joint jump
            if pos_error < 10.0 and joint_jump < 0.8:
                waypoints.append(sol.tolist())
                current_q = sol
            else:
                # If point is bad, we skip it to remove "out of way" points
                print(f"      ⚠️ Pruned bad waypoint: Error={pos_error:.1f}mm, Jump={joint_jump:.2f}")
            
        return waypoints

# --- Mission Recorder ---

def record_mission_manual(ctrl: MechArmController):
    """
    Interactive wizard to record mission waypoints by manually moving the robot.
    """
    print("\n" + "="*60)
    print("  MANUAL MISSION RECORDING WIZARD")
    print("="*60)
    
    if ctrl.dry_run:
        print("⚠️  WARNING: Running in DRY RUN mode. Positions will be fake.")
    
    pegs = []
    
    try:
        num_pegs = int(input("\nHow many pegs to record? "))
    except ValueError:
        print("Invalid number, defaulting to 1")
        num_pegs = 1
        
    for i in range(num_pegs):
        print(f"\n--- Recording Peg {i+1} ---")
        color = input(f"Enter color for Peg {i+1} (e.g. RED, BLUE): ").upper()
        if not color: color = f"PEG_{i}"

        # Record PICK
        print(f"\n[PICK] Move robot to {color} PICK position.")
        print("Releasing servos in 2 seconds...")
        time.sleep(2)
        ctrl.release_all_servos()
        input("Move robot, then press ENTER to record...")
        ctrl.power_on()
        time.sleep(1.0) # Wait for lock
        
        pick_angles = ctrl.get_angles()
        pick_pose = ctrl.get_coords()
        print(f"✓ Recorded PICK: {pick_pose}")
        
        # Record PLACE
        print(f"\n[PLACE] Move robot to {color} PLACE position.")
        print("Releasing servos in 2 seconds...")
        time.sleep(2)
        ctrl.release_all_servos()
        input("Move robot, then press ENTER to record...")
        ctrl.power_on()
        time.sleep(1.0)
        
        place_angles = ctrl.get_angles()
        place_pose = ctrl.get_coords()
        print(f"✓ Recorded PLACE: {place_pose}")
        
        pegs.append({
            "peg_id": i,
            "color": color,
            "pick_angles": pick_angles,
            "pick_pose": pick_pose,
            "place_angles": place_angles,
            "place_pose": place_pose
        })
        
    # Get start angles (optional, use current)
    print("\n[START] Move robot to START/HOME position.")
    print("Releasing servos...")
    ctrl.release_all_servos()
    input("Move robot, then press ENTER...")
    ctrl.power_on()
    time.sleep(1.0)
    start_angles = ctrl.get_angles()
    
    mission_data = {
        "start_angles": start_angles,
        "safe_height_offset": 100.0,
        "waypoint_density": 20,
        "pegs": pegs
    }
    
    print("\n" + "="*60)
    print("  GENERATED MISSION JSON")
    print("="*60)
    print("Create mission_simple.json with:")
    json_str = json.dumps(mission_data, indent=2)
    print(json_str)
    print("="*60)
    
    save = input("\nSave to mission_simple.json? (y/n): ").lower()
    if save == 'y':
        filepath = Path(__file__).parent / "mission_simple.json"
        with open(filepath, "w") as f:
            f.write(json_str)
        print(f"✓ Saved to {filepath}")

# --- Main ---

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    dry = input("Dry Run? (y/n): ").lower().startswith('y')
    ctrl = MechArmController(dry)
    
    # Check for recording mode
    if input("Record new mission manually? (y/n): ").lower() == 'y':
        record_mission_manual(ctrl)
        if input("Run the recorded mission now? (y/n): ").lower() != 'y':
            return

    kin = KinematicsEngine()
    planner = SimplifiedMissionPlanner(ctrl, kin)
    
    mission_file = Path(__file__).parent / "mission_simple.json"
    
    # Load mission
    if not mission_file.exists():
        print(f"\n❌ Mission file not found: {mission_file}")
        print("\nCreate mission_simple.json with:")
        print(json.dumps({
            "start_angles": [0, 0, 0, 0, 0, 0],
            "safe_height_offset": 100.0,
            "waypoint_density": 20,
            "pegs": [
                {
                    "peg_id": 0,
                    "color": "RED",
                    "pick_angles": [0, 0, 0, 0, 90, 0],
                    "pick_pose": [150, 0, 100, 0, 0, 0],
                    "place_angles": [45, 0, 0, 0, 90, 0],
                    "place_pose": [100, 100, 50, 0, 0, 0]
                }
            ]
        }, indent=2))
        return
    
    if not planner.load_mission(mission_file):
        return
    
    # Plan
    if input("\nGenerate straight-line paths? (y/n): ").lower() == 'y':
        if planner.plan_all_paths():
            planner.config.save(mission_file)
            print(f"\n✓ Paths saved to {mission_file}")
        else:
            print("\n❌ Planning failed")
            return
    
    # Execute
    if input("\nExecute mission? (y/n): ").lower() == 'y':
        planner.load_mission(mission_file)  # Reload
        result = planner.execute_mission()
        print(f"\n{'='*50}")
        print(f"✓ Mission Complete!")
        print(f"  Placed: {result['placed']}/{len(planner.config.pegs)}")
        if result['failed'] > 0:
            print(f"  Failed: {result['failed']}")
        print(f"{'='*50}")
    
    # Record Mission
    if input("\nRecord new mission manually? (y/n): ").lower() == 'y':
        record_mission_manual(ctrl)

if __name__ == "__main__":
    main()