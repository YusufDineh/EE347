"""
FK-Only MechArm Pick and Place - Pure Joint Space Control
==========================================================
Uses only Forward Kinematics for verification and joint interpolation for motion.
No inverse kinematics - pure teach-and-repeat with recorded approach positions.

JSON Format (mission_simple.json):
{
  "start_angles": [0, 0, 0, 0, 0, 0],
  "waypoint_density": 20,
  "pegs": [
    {
      "peg_id": 0,
      "color": "RED",
      "pick_angles": [j1, j2, j3, j4, j5, j6],
      "pick_approach_angles": [j1, j2, j3, j4, j5, j6],
      "pick_pose": [x, y, z, rx, ry, rz],
      "place_angles": [j1, j2, j3, j4, j5, j6],
      "place_approach_angles": [j1, j2, j3, j4, j5, j6],
      "place_pose": [x, y, z, rx, ry, rz]
    }
  ]
}
"""

import time
import json
import logging
import numpy as np
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

# Kinematics Setup (DH Parameters) - For FK verification only
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
    """Pick and place task definition with approach angles."""
    peg_id: int
    color: str
    pick_angles: List[float]
    pick_approach_angles: List[float]
    pick_pose: List[float]
    place_angles: List[float]
    place_approach_angles: List[float]
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
    waypoint_density: int
    pegs: List[PegTask]
    planned_paths: Dict[str, PlannedPath]
    
    def to_dict(self) -> Dict:
        return {
            'start_angles': self.start_angles,
            'waypoint_density': self.waypoint_density,
            'pegs': [p.to_dict() for p in self.pegs],
            'planned_paths': {k: v.to_dict() for k, v in self.planned_paths.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            data['start_angles'],
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

# --- Forward Kinematics (Verification Only) ---

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

class KinematicsEngine:
    """Forward kinematics only - for verification."""
    
    @staticmethod
    def forward_kinematics(q_values: np.ndarray) -> np.ndarray:
        """Compute XYZ from joint angles."""
        q = np.radians(np.asarray(q_values, dtype=float))
        return np.asarray(fk_num(*q), dtype=float).ravel()

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
            self.mc.set_gripper_value(state, speed, 1)
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

# --- Mission Planner (FK Only) ---

class SimplifiedMissionPlanner:
    """Plan and execute with joint interpolation only."""
    
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
            import traceback
            traceback.print_exc()
            return False
    
    def plan_all_paths(self) -> bool:
        """Generate joint-interpolated paths (FK for verification only)."""
        if not self.config:
            logging.error("No mission loaded")
            return False
        
        print("\n=== Planning Paths (Joint Space Only - No IK) ===")
        
        self.config.planned_paths = {}
        curr_angles = self.config.start_angles
        density = self.config.waypoint_density
        
        for i, peg in enumerate(self.config.pegs):
            print(f"\n{'='*50}")
            print(f"Planning Peg {i+1}/{len(self.config.pegs)}: {peg.color}")
            print(f"{'='*50}")
            
            try:
                # Verify positions with FK
                pick_pos = self.kin.forward_kinematics(peg.pick_angles)
                pick_approach_pos = self.kin.forward_kinematics(peg.pick_approach_angles)
                place_pos = self.kin.forward_kinematics(peg.place_angles)
                place_approach_pos = self.kin.forward_kinematics(peg.place_approach_angles)
                
                print(f"→ FK Verification:")
                print(f"   Pick Approach: [{pick_approach_pos[0]:.1f}, {pick_approach_pos[1]:.1f}, {pick_approach_pos[2]:.1f}]")
                print(f"   Pick:          [{pick_pos[0]:.1f}, {pick_pos[1]:.1f}, {pick_pos[2]:.1f}]")
                print(f"   Place Approach:[{place_approach_pos[0]:.1f}, {place_approach_pos[1]:.1f}, {place_approach_pos[2]:.1f}]")
                print(f"   Place:         [{place_pos[0]:.1f}, {place_pos[1]:.1f}, {place_pos[2]:.1f}]")
                
                # Generate paths using only joint interpolation
                print(f"→ Generating joint-space paths...")
                
                # 1. Travel to Pick Approach
                self.config.planned_paths[f"peg_{i}_to_pick_approach"] = PlannedPath(
                    f"peg_{i}_to_pick_approach",
                    interpolate_joint_path(curr_angles, peg.pick_approach_angles, density)
                )
                
                # 2. Pick Descent
                self.config.planned_paths[f"peg_{i}_pick_descent"] = PlannedPath(
                    f"peg_{i}_pick_descent",
                    interpolate_joint_path(peg.pick_approach_angles, peg.pick_angles, density)
                )
                
                # 3. Pick Ascent
                self.config.planned_paths[f"peg_{i}_pick_ascent"] = PlannedPath(
                    f"peg_{i}_pick_ascent",
                    interpolate_joint_path(peg.pick_angles, peg.pick_approach_angles, density)
                )
                
                # 4. Travel to Place Approach
                self.config.planned_paths[f"peg_{i}_travel"] = PlannedPath(
                    f"peg_{i}_travel",
                    interpolate_joint_path(peg.pick_approach_angles, peg.place_approach_angles, density)
                )
                
                # 5. Place Descent
                self.config.planned_paths[f"peg_{i}_place_descent"] = PlannedPath(
                    f"peg_{i}_place_descent",
                    interpolate_joint_path(peg.place_approach_angles, peg.place_angles, density)
                )
                
                # 6. Place Ascent
                self.config.planned_paths[f"peg_{i}_place_ascent"] = PlannedPath(
                    f"peg_{i}_place_ascent",
                    interpolate_joint_path(peg.place_angles, peg.place_approach_angles, density)
                )
                
                print(f"   ✓ Generated 6 paths × {density} waypoints = {6 * density} total points")
                
                curr_angles = peg.place_approach_angles
                
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
                
                # FORCE EXACT POSITION
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
                
                # FORCE EXACT POSITION
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
            
        for wp in path.waypoints:
            self.ctrl.send_angles(wp, SPEED_PRECISION)

# --- Mission Recorder ---

def record_mission_manual(ctrl: MechArmController):
    """
    Interactive wizard to record mission waypoints with approach positions.
    """
    print("\n" + "="*60)
    print("  MANUAL MISSION RECORDING WIZARD (FK-Only)")
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
        print(f"\n{'='*50}")
        print(f"Recording Peg {i+1}/{num_pegs}")
        print(f"{'='*50}")
        color = input(f"Enter color for Peg {i+1} (e.g. RED, BLUE): ").upper()
        if not color: color = f"PEG_{i}"

        # Record PICK APPROACH
        print(f"\n[PICK APPROACH] Move robot ABOVE {color} pick position.")
        print("(This should be a safe height before descending to pick)")
        print("Releasing servos in 2 seconds...")
        time.sleep(2)
        ctrl.release_all_servos()
        input("Move robot, then press ENTER to record...")
        ctrl.power_on()
        time.sleep(1.0)
        
        pick_approach_angles = ctrl.get_angles()
        print(f"✓ Recorded PICK APPROACH: J={[f'{a:.1f}' for a in pick_approach_angles]}")
        
        # Record PICK
        print(f"\n[PICK] Move robot to exact {color} PICK position.")
        print("Releasing servos in 2 seconds...")
        time.sleep(2)
        ctrl.release_all_servos()
        input("Move robot, then press ENTER to record...")
        ctrl.power_on()
        time.sleep(1.0)
        
        pick_angles = ctrl.get_angles()
        pick_pose = ctrl.get_coords()
        print(f"✓ Recorded PICK: {pick_pose}")
        
        # Record PLACE APPROACH
        print(f"\n[PLACE APPROACH] Move robot ABOVE {color} place position.")
        print("(This should be a safe height before descending to place)")
        print("Releasing servos in 2 seconds...")
        time.sleep(2)
        ctrl.release_all_servos()
        input("Move robot, then press ENTER to record...")
        ctrl.power_on()
        time.sleep(1.0)
        
        place_approach_angles = ctrl.get_angles()
        print(f"✓ Recorded PLACE APPROACH: J={[f'{a:.1f}' for a in place_approach_angles]}")
        
        # Record PLACE
        print(f"\n[PLACE] Move robot to exact {color} PLACE position.")
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
            "pick_approach_angles": pick_approach_angles,
            "pick_pose": pick_pose,
            "place_angles": place_angles,
            "place_approach_angles": place_approach_angles,
            "place_pose": place_pose
        })
        
    # Get start angles
    print(f"\n{'='*50}")
    print("[START] Move robot to START/HOME position.")
    print(f"{'='*50}")
    print("Releasing servos in 2 seconds...")
    time.sleep(2)
    ctrl.release_all_servos()
    input("Move robot, then press ENTER...")
    ctrl.power_on()
    time.sleep(1.0)
    start_angles = ctrl.get_angles()
    print(f"✓ Recorded START: J={[f'{a:.1f}' for a in start_angles]}")
    
    mission_data = {
        "start_angles": start_angles,
        "waypoint_density": 20,
        "pegs": pegs
    }
    
    print("\n" + "="*60)
    print("  GENERATED MISSION JSON")
    print("="*60)
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
            "waypoint_density": 20,
            "pegs": [
                {
                    "peg_id": 0,
                    "color": "RED",
                    "pick_angles": [0, 0, 0, 0, 90, 0],
                    "pick_approach_angles": [0, -20, 0, 0, 90, 0],
                    "pick_pose": [150, 0, 100, 0, 0, 0],
                    "place_angles": [45, 0, 0, 0, 90, 0],
                    "place_approach_angles": [45, -20, 0, 0, 90, 0],
                    "place_pose": [100, 100, 50, 0, 0, 0]
                }
            ]
        }, indent=2))
        return
    
    if not planner.load_mission(mission_file):
        return
    
    # Plan
    if input("\nGenerate joint-space paths? (y/n): ").lower() == 'y':
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

if __name__ == "__main__":
    main()