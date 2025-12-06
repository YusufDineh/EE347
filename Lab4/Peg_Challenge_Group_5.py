#!/usr/bin/env python3
"""
MechArm unified mission runner for EE347 Lab 4
- Challenge 1: Precision peg pick-and-place (hover IK + joint-space interpolation)
- Challenge 2: Coordinated tower manipulation (Option C: pick->place sequential)
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
GRIPPER_SLEEP = 1.2

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
    start_angles: List[float]
    safe_height_offset: float
    waypoint_density: int
    pegs: List[PegTask]
    planned_paths: Dict[str, PlannedPath]
    # Backwards-compatible fields for Challenge 2 simplified JSON
    pick_pegs: Optional[List[Dict]] = None
    place_pegs: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        out = {
            'start_angles': self.start_angles,
            'safe_height_offset': self.safe_height_offset,
            'waypoint_density': self.waypoint_density,
            'pegs': [p.to_dict() for p in self.pegs],
            'planned_paths': {k: v.to_dict() for k, v in self.planned_paths.items()}
        }
        if self.pick_pegs is not None:
            out['pick_pegs'] = self.pick_pegs
        if self.place_pegs is not None:
            out['place_pegs'] = self.place_pegs
        return out
    
    @classmethod
    def from_dict(cls, data: Dict):
        pegs = [PegTask.from_dict(p) for p in data.get('pegs', [])]
        planned = {k: PlannedPath.from_dict(v) for k, v in data.get('planned_paths', {}).items()}
        return cls(
            data.get('start_angles', [0,0,0,0,0,0]),
            data.get('safe_height_offset', 100.0),
            data.get('waypoint_density', 20),
            pegs,
            planned,
            pick_pegs=data.get('pick_pegs'),
            place_pegs=data.get('place_pegs')
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
    return Matrix([
        [cos(theta), -sin(theta), 0, a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
        [0, 0, 0, 1]
    ])

def overall_transformation(dh_table):
    T = Matrix(np.identity(4))
    for params in dh_table:
        T = T * get_transformation_matrix(*params)
    return T

T_symbolic = overall_transformation(dh_table)
fk_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, 3], modules='numpy')
rot_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, :3], modules='numpy')

class KinematicsEngine:
    @staticmethod
    def forward_kinematics(q_values: np.ndarray) -> np.ndarray:
        q = np.radians(np.asarray(q_values, dtype=float))
        return np.asarray(fk_num(*q), dtype=float).ravel()
    
    @staticmethod
    def rotation_matrix_to_euler_zyx(R: np.ndarray, transpose: bool = False) -> tuple:
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
                          q_init: List[float], max_iterations: int = 3000) -> np.ndarray:
        x_t, y_t, z_t = target_pos
        rx_d, ry_d, rz_d = target_ori
        q_init = np.asarray(q_init, dtype=float)
        
        # Lock Joint 6 to the initial value (seed) to prevent rotation
        fixed_j6 = q_init[5]
        
        def combined_residual(q_5d, pos_weight=1.0, ori_weight=10.0):
            # Reconstruct full 6D configuration with fixed J6
            q = np.append(q_5d, fixed_j6)
            
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
        
        # Optimize only first 5 joints
        bounds_5d = (
            np.array([l[0] for l in JOINT_LIMITS[:5]]), 
            np.array([l[1] for l in JOINT_LIMITS[:5]])
        )
        
        res = least_squares(combined_residual, q_init[:5], bounds=bounds_5d, method='trf',
                            max_nfev=max_iterations, ftol=1e-6, xtol=1e-6, verbose=0)
        
        # Recombine solution with fixed J6
        q_sol = np.append(res.x, fixed_j6)
        return q_sol

# --- Path Utilities ---

def interpolate_joint_path(start: List[float], end: List[float], min_density: int) -> List[List[float]]:
    start = np.array(start)
    end = np.array(end)
    max_step = 3.0  # degrees per step max
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
    
    def send_angles(self, angles: List[float], speed: int, timeout: float = 0.02) -> bool:
        if self.dry_run:
            print(f"   [DRY RUN] send_angles: {[round(a,1) for a in angles]}")
            return True
        try:
            self.mc.send_angles(angles, speed)
            time.sleep(timeout)
            return True
        except Exception as e:
            logging.error(f"send_angles error: {e}")
            return False
    
    def control_gripper(self, state: int, speed: int = 70, sleep_time: float = GRIPPER_SLEEP) -> bool:
        if self.dry_run:
            print(f"   [DRY RUN] gripper -> {state}")
            return True
        try:
            self.mc.set_gripper_value(state, speed, 1)
            time.sleep(sleep_time)
            return True
        except Exception as e:
            logging.error(f"gripper error: {e}")
            return False
    
    def power_on(self):
        if self.dry_run: return
        if self.mc: self.mc.power_on()
    
    def power_off(self):
        if self.dry_run: return
        if self.mc: self.mc.power_off()
    
    def release_all_servos(self):
        if self.dry_run:
            print("   [DRY RUN] release_all_servos")
            return
        if self.mc: self.mc.release_all_servos()
    
    def get_angles(self) -> List[float]:
        if self.dry_run:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if self.mc: return self.mc.get_angles()
        return [0.0]*6
    
    def get_coords(self) -> List[float]:
        if self.dry_run:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if self.mc: return self.mc.get_coords()
        return [0.0]*6

# --- Mission Planner / Executor ---

class SimplifiedMissionPlanner:
    def __init__(self, ctrl: MechArmController, kin: KinematicsEngine):
        self.ctrl = ctrl
        self.kin = kin
        self.config: Optional[MissionConfig] = None
    
    def load_mission(self, filepath: str) -> bool:
        try:
            print(f"\n→ Loading mission: {filepath}")
            if not Path(filepath).exists():
                logging.error(f"Missing mission file: {filepath}")
                return False
            self.config = MissionConfig.load(filepath)
            print(f"   ✓ Loaded: {len(self.config.pegs)} peg(s) (pegs field)")
            if self.config.pick_pegs is not None:
                print(f"   ✓ pick_pegs found: {len(self.config.pick_pegs)}")
            if self.config.place_pegs is not None:
                print(f"   ✓ place_pegs found: {len(self.config.place_pegs)}")
            return True
        except Exception as e:
            logging.error(f"load_mission error: {e}")
            return False
    
    # helper: compute approach (hover) joint angles from a joint-angle pose
    def compute_safe_approach_from_angles(self, joint_angles: List[float]) -> List[float]:
        # Use FK to find xyz, lift by safe offset and solve IK for approach
        xyz = self.kin.forward_kinematics(joint_angles)
        target_xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2]) + self.config.safe_height_offset]
        # Use orientation from recorded pick pose if available: fallback keep zeros
        # We don't have orientation from pure angles in config.pick_pegs entries,
        # so use orientation of the recorded pick_pose if available in self.config.pegs.
        # Default to [0,0,0]
        ori = [0.0, 0.0, 0.0]
        # If this joint_angles matches a known peg in self.config.pegs, try to grab its pose orientation
        for p in self.config.pegs:
            if np.allclose(p.pick_angles, joint_angles, atol=1e-3):
                ori = p.pick_pose[3:]
                break
            if np.allclose(p.place_angles, joint_angles, atol=1e-3):
                ori = p.place_pose[3:]
                break
        q_init = np.array(joint_angles)
        sol = self.kin.inverse_kinematics(target_xyz, ori, q_init, max_iterations=2000)
        return sol.tolist()
    
    def compute_safe_approach_from_pose(self, pose: List[float], seed_angles: List[float]) -> List[float]:
        # pose: [x,y,z,rx,ry,rz]; lift z by safe_height_offset then IK
        target_xyz = [pose[0], pose[1], pose[2] + self.config.safe_height_offset]
        ori = pose[3:]
        sol = self.kin.inverse_kinematics(target_xyz, ori, seed_angles, max_iterations=2000)
        return sol.tolist()
    
    def plan_all_paths_challenge1(self) -> bool:
        """Plan the same joint-space sequence for pegs listed in self.config.pegs"""
        if not self.config:
            logging.error("No mission loaded")
            return False
        self.config.planned_paths = {}
        density = self.config.waypoint_density
        curr_angles = self.config.start_angles
        curr_seed = np.array(curr_angles)
        for i, peg in enumerate(self.config.pegs):
            try:
                pick_xyz = peg.pick_pose[:3]
                place_xyz = peg.place_pose[:3]
                # compute approach via IK once
                pick_approach_xyz = [pick_xyz[0], pick_xyz[1], pick_xyz[2] + self.config.safe_height_offset]
                place_approach_xyz = [place_xyz[0], place_xyz[1], place_xyz[2] + self.config.safe_height_offset]
                print(f"→ IK pick-approach (peg {i})")
                # Use pick_angles as seed to align J6 for straight descent
                pick_seed = np.array(peg.pick_angles)
                pick_approach_angles = self.kin.inverse_kinematics(pick_approach_xyz, peg.pick_pose[3:], pick_seed, max_iterations=3000).tolist()
                
                print(f"→ IK place-approach (peg {i})")
                # Use place_angles as seed to align J6 for straight descent
                place_seed = np.array(peg.place_angles)
                place_approach_angles = self.kin.inverse_kinematics(place_approach_xyz, peg.place_pose[3:], place_seed, max_iterations=3000).tolist()
                
                curr_seed = np.array(place_approach_angles)
                
                # joint-space segments
                self.config.planned_paths[f"peg_{i}_to_pick_approach"] = PlannedPath(
                    f"peg_{i}_to_pick_approach",
                    interpolate_joint_path(curr_angles, pick_approach_angles, density)
                )
                self.config.planned_paths[f"peg_{i}_pick_descent"] = PlannedPath(
                    f"peg_{i}_pick_descent",
                    interpolate_joint_path(pick_approach_angles, peg.pick_angles, density)
                )
                self.config.planned_paths[f"peg_{i}_pick_ascent"] = PlannedPath(
                    f"peg_{i}_pick_ascent",
                    interpolate_joint_path(peg.pick_angles, pick_approach_angles, density)
                )
                self.config.planned_paths[f"peg_{i}_travel"] = PlannedPath(
                    f"peg_{i}_travel",
                    interpolate_joint_path(pick_approach_angles, place_approach_angles, density)
                )
                self.config.planned_paths[f"peg_{i}_place_descent"] = PlannedPath(
                    f"peg_{i}_place_descent",
                    interpolate_joint_path(place_approach_angles, peg.place_angles, density)
                )
                self.config.planned_paths[f"peg_{i}_place_ascent"] = PlannedPath(
                    f"peg_{i}_place_ascent",
                    interpolate_joint_path(peg.place_angles, place_approach_angles, density)
                )
                curr_angles = place_approach_angles
            except Exception as e:
                logging.error(f"Planning error peg {i}: {e}")
                return False
        print("✓ Challenge 1 planning complete.")
        return True
    
    def plan_all_paths_challenge2(self) -> bool:
        """
        Use pick_pegs and place_pegs if present. For each pair (pick_i -> place_i),
        compute approach IK for pick and place and produce joint-space interpolations.
        If pick_pegs/place_pegs are not provided, fallback to self.config.pegs mapping by index.
        """
        if not self.config:
            logging.error("No mission loaded")
            return False
        self.config.planned_paths = {}
        density = self.config.waypoint_density
        curr_angles = self.config.start_angles
        curr_seed = np.array(curr_angles)
        
        # Build lists of pick and place entries
        if self.config.pick_pegs is not None and self.config.place_pegs is not None:
            picks = self.config.pick_pegs
            places = self.config.place_pegs
            # Expect picks and places aligned by index or peg_id
        else:
            # fallback to using self.config.pegs (which contain both pick/place entries)
            picks = []
            places = []
            for p in self.config.pegs:
                picks.append({'peg_id': p.peg_id, 'pick_angles': p.pick_angles, 'pick_pose': p.pick_pose})
                places.append({'peg_id': p.peg_id, 'place_angles': p.place_angles, 'place_pose': p.place_pose})
        
        # align pairs by peg_id where possible
        pairs = []
        for pk in picks:
            pid = pk.get('peg_id', None)
            # find matching place by peg_id
            match = None
            for pl in places:
                if pl.get('peg_id', None) == pid:
                    match = pl
                    break
            if match is None:
                # if not found, take same index if available
                idx = picks.index(pk)
                match = places[idx] if idx < len(places) else None
            pairs.append((pk, match))
        
        for i, (pk, pl) in enumerate(pairs):
            try:
                if pl is None:
                    logging.error(f"No place entry for pick {i} (peg_id={pk.get('peg_id')})")
                    return False
                # prefer pose if present, else use joint angles -> compute pose via FK
                pick_angles = pk.get('pick_angles', None)
                pick_pose = pk.get('pick_pose', None)
                place_angles = pl.get('target_angles', pl.get('place_angles', None))
                place_pose = pl.get('place_pose', None)
                
                # For consistency, we will plan approach angles using IK from poses if possible.
                # If only angles are available, we'll compute approach by raising FK z and IK.
                
                # Use pick_angles as seed if available to ensure J6 alignment
                pick_seed = np.array(pick_angles) if pick_angles is not None else curr_seed
                
                if pick_pose is not None:
                    pick_approach_angles = self.compute_safe_approach_from_pose(pick_pose, pick_seed)
                else:
                    pick_approach_angles = self.compute_safe_approach_from_angles(pick_angles)
                
                # Use place_angles as seed if available
                place_seed = np.array(place_angles) if place_angles is not None else np.array(pick_approach_angles)
                
                if place_pose is not None:
                    place_approach_angles = self.compute_safe_approach_from_pose(place_pose, place_seed)
                else:
                    place_approach_angles = self.compute_safe_approach_from_angles(place_angles)
                
                curr_seed = np.array(place_approach_angles)
                
                # create joint-space segments for pick->place sequentially
                base = i
                self.config.planned_paths[f"c2_{i}_to_pick_approach"] = PlannedPath(
                    f"c2_{i}_to_pick_approach",
                    interpolate_joint_path(curr_angles, pick_approach_angles, density)
                )
                self.config.planned_paths[f"c2_{i}_pick_descent"] = PlannedPath(
                    f"c2_{i}_pick_descent",
                    interpolate_joint_path(pick_approach_angles, pick_angles, density)
                )
                self.config.planned_paths[f"c2_{i}_pick_ascent"] = PlannedPath(
                    f"c2_{i}_pick_ascent",
                    interpolate_joint_path(pick_angles, pick_approach_angles, density)
                )
                self.config.planned_paths[f"c2_{i}_travel"] = PlannedPath(
                    f"c2_{i}_travel",
                    interpolate_joint_path(pick_approach_angles, place_approach_angles, density)
                )
                self.config.planned_paths[f"c2_{i}_place_descent"] = PlannedPath(
                    f"c2_{i}_place_descent",
                    interpolate_joint_path(place_approach_angles, place_angles, density)
                )
                self.config.planned_paths[f"c2_{i}_place_ascent"] = PlannedPath(
                    f"c2_{i}_place_ascent",
                    interpolate_joint_path(place_angles, place_approach_angles, density)
                )
                curr_angles = place_approach_angles
            except Exception as e:
                logging.error(f"Challenge2 planning error pair {i}: {e}")
                return False
        print("✓ Challenge 2 planning complete.")
        return True
    
    def _execute_path(self, path: PlannedPath):
        if self.ctrl.dry_run:
            print(f"   [DRY RUN] would execute {len(path.waypoints)} waypoints for {path.path_id}")
            return
        for wp in path.waypoints:
            self.ctrl.send_angles(wp, SPEED_PRECISION)
    
    def execute_challenge1(self) -> Dict:
        """Execute planned Challenge 1 paths from planned_paths keys."""
        if not self.config or not self.config.planned_paths:
            logging.error("No planned paths")
            return {'placed': 0, 'failed': 0}
        self.ctrl.send_angles(self.config.start_angles, SPEED_NORMAL, timeout=1.0)
        self.ctrl.control_gripper(GRIPPER_OPEN)
        score = {'placed': 0, 'failed': 0}
        for i, peg in enumerate(self.config.pegs):
            try:
                self._execute_path(self.config.planned_paths[f"peg_{i}_to_pick_approach"])
                self._execute_path(self.config.planned_paths[f"peg_{i}_pick_descent"])
                self.ctrl.send_angles(peg.pick_angles, SPEED_PRECISION)
                time.sleep(0.2)
                self.ctrl.control_gripper(GRIPPER_CLOSE)
                self._execute_path(self.config.planned_paths[f"peg_{i}_pick_ascent"])
                self._execute_path(self.config.planned_paths[f"peg_{i}_travel"])
                self._execute_path(self.config.planned_paths[f"peg_{i}_place_descent"])
                self.ctrl.send_angles(peg.place_angles, SPEED_PRECISION)
                time.sleep(0.2)
                self.ctrl.control_gripper(GRIPPER_OPEN)
                score['placed'] += 1
                self._execute_path(self.config.planned_paths[f"peg_{i}_place_ascent"])
                print(f"  ✓ Peg {i} placed")
            except Exception as e:
                logging.error(f"Execution error peg {i}: {e}")
                score['failed'] += 1
        return score
    
    def execute_challenge2(self) -> Dict:
        """
        Execute Challenge 2 sequential pick->place pairs (Option C).
        Follows the planned path keys c2_{i}_*
        """
        if not self.config or not self.config.planned_paths:
            logging.error("No planned paths")
            return {'placed': 0, 'failed': 0}
        self.ctrl.send_angles(self.config.start_angles, SPEED_NORMAL, timeout=1.0)
        self.ctrl.control_gripper(GRIPPER_OPEN)
        score = {'placed': 0, 'failed': 0}
        i = 0
        while True:
            keys = [
                f"c2_{i}_to_pick_approach",
                f"c2_{i}_pick_descent",
                f"c2_{i}_pick_ascent",
                f"c2_{i}_travel",
                f"c2_{i}_place_descent",
                f"c2_{i}_place_ascent"
            ]
            if keys[0] not in self.config.planned_paths:
                break
            try:
                self._execute_path(self.config.planned_paths[keys[0]])
                self._execute_path(self.config.planned_paths[keys[1]])
                # Force exact pick angles if present in JSON (safe)
                # Try to find corresponding pick_angles from config (pegs or pick_pegs)
                pick_angles = None
                # search in pegs
                for p in self.config.pegs:
                    if p.peg_id == i:
                        pick_angles = p.pick_angles
                        break
                # fallback to pick_pegs list if available
                if pick_angles is None and self.config.pick_pegs:
                    try:
                        pick_angles = self.config.pick_pegs[i].get('pick_angles')
                    except:
                        pick_angles = None
                if pick_angles:
                    self.ctrl.send_angles(pick_angles, SPEED_PRECISION)
                time.sleep(0.2)
                self.ctrl.control_gripper(GRIPPER_CLOSE)
                self._execute_path(self.config.planned_paths[keys[2]])
                self._execute_path(self.config.planned_paths[keys[3]])
                self._execute_path(self.config.planned_paths[keys[4]])
                # Force exact place angles if available
                place_angles = None
                for p in self.config.pegs:
                    if p.peg_id == i:
                        place_angles = p.place_angles
                        break
                if place_angles is None and self.config.place_pegs:
                    try:
                        place_angles = self.config.place_pegs[i].get('target_angles') or self.config.place_pegs[i].get('place_angles')
                    except:
                        place_angles = None
                if place_angles:
                    self.ctrl.send_angles(place_angles, SPEED_PRECISION)
                time.sleep(0.2)
                self.ctrl.control_gripper(GRIPPER_OPEN)
                score['placed'] += 1
                self._execute_path(self.config.planned_paths[keys[5]])
                print(f"  ✓ C2 pair {i} completed")
            except Exception as e:
                logging.error(f"Execution error C2 pair {i}: {e}")
                score['failed'] += 1
            i += 1
        return score

# --- Mission Recorder ---

def record_mission_manual(ctrl: MechArmController):
    print("\n" + "="*50)
    print("  MANUAL MISSION RECORDING WIZARD")
    print("="*50)
    pegs = []
    try:
        num_pegs = int(input("How many pegs to record? ") or "1")
    except ValueError:
        num_pegs = 1
    for i in range(num_pegs):
        color = input(f"Color for peg {i} (default PEG_{i}): ").upper() or f"PEG_{i}"
        print("\n-- RECORD PICK POSITION --")
        time.sleep(1.0)
        ctrl.release_all_servos()
        input("Move robot to pick pose, then press ENTER to record... ")
        ctrl.power_on()
        time.sleep(0.5)
        pick_angles = ctrl.get_angles()
        pick_pose = ctrl.get_coords()
        print(f"Recorded pick pose: {pick_pose}")
        print("\n-- RECORD PLACE POSITION --")
        time.sleep(1.0)
        ctrl.release_all_servos()
        input("Move robot to place pose, then press ENTER to record... ")
        ctrl.power_on()
        time.sleep(0.5)
        place_angles = ctrl.get_angles()
        place_pose = ctrl.get_coords()
        print(f"Recorded place pose: {place_pose}")
        pegs.append({
            "peg_id": i,
            "color": color,
            "pick_angles": pick_angles,
            "pick_pose": pick_pose,
            "place_angles": place_angles,
            "place_pose": place_pose
        })
    print("\n-- RECORD START / HOME --")
    time.sleep(1.0)
    ctrl.release_all_servos()
    input("Move robot to start/home pose, then press ENTER... ")
    ctrl.power_on()
    time.sleep(0.5)
    start_angles = ctrl.get_angles()
    mission_data = {
        "start_angles": start_angles,
        "safe_height_offset": float(input("Safe height offset (mm) [100]: ") or 100.0),
        "waypoint_density": int(input("Waypoint density (min points per segment) [20]: ") or 20),
        "pegs": pegs
    }
    print("\nGenerated mission JSON:")
    print(json.dumps(mission_data, indent=2))
    if input("Save to mission_simple.json? (y/n): ").lower().startswith('y'):
        filepath = Path(__file__).parent / "mission_simple.json"
        with open(filepath, 'w') as f:
            json.dump(mission_data, f, indent=2)
        print(f"Saved {filepath}")

# --- Main CLI ---

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    dry_input = input("Dry run mode? (y/n) [y]: ").lower()
    dry = dry_input.startswith('y') or dry_input == ''
    ctrl = MechArmController(dry_run=dry)
    kin = KinematicsEngine()
    planner = SimplifiedMissionPlanner(ctrl, kin)
    mission_file = Path(__file__).parent / "mission_simple.json"
    if not mission_file.exists():
        print(f"Mission file not found: {mission_file}")
        print("Create mission_simple.json via recorder or by hand.")
        if input("Record a mission now? (y/n): ").lower().startswith('y'):
            record_mission_manual(ctrl)
        else:
            print("Exiting.")
            return
    if not planner.load_mission(str(mission_file)):
        return
    # Menu
    choice = input("\nChoose action:\n  1) Plan Challenge 1 (peg pick/place)\n  2) Plan Challenge 2 (sequential pick->place)\n  3) Execute planned mission\n  4) Record mission manually\n  5) Exit\nSelect [1/2/3/4/5]: ").strip() or "5"
    if choice == "1":
        if planner.plan_all_paths_challenge1():
            planner.config.save(str(mission_file))
            print("Planned Challenge 1 and saved mission.")
    elif choice == "2":
        if planner.plan_all_paths_challenge2():
            planner.config.save(str(mission_file))
            print("Planned Challenge 2 and saved mission.")
    elif choice == "3":
        # Ask which challenge's plan to execute
        ch = input("Execute which challenge? (1 or 2) [1]: ").strip() or "1"
        if ch == "1":
            result = planner.execute_challenge1()
            print(f"Result: placed={result['placed']} failed={result['failed']}")
        else:
            result = planner.execute_challenge2()
            print(f"Result: placed={result['placed']} failed={result['failed']}")
    elif choice == "4":
        record_mission_manual(ctrl)
    else:
        print("Exit.")
    # power off & cleanup
    ctrl.power_off()

if __name__ == "__main__":
    main()
