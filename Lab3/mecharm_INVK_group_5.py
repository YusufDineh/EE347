import math
import numpy as np
import pandas as pd
from sympy.matrices import Matrix
from scipy.optimize import least_squares
from sympy  import symbols, cos, sin, atan2, pi, lambdify 
from pathlib import Path
from pathlib import Path


#Global variables
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6') # Define symbolic variables for joint angles
right = pi/2
output_dir = Path(".")  # Current directory
data = pd.read_csv(output_dir / "mecharm_control_group_5.csv")

#Joint Angles from Lab 2 task 3
known_joint_angles = data[['J1','J2','J3','J4','J5','J6']].to_numpy()
#End-Effector Positions from Lab 2 task 3
known_position = data[['X','Y','Z']].to_numpy()
#End-Effector Orientations from Lab 2 task 3
known_orientation = data[['RX','RY','RZ']].to_numpy()

dh_table = [
    [0.049,     0,          135.926,    q1],           # joint 1
    [0,     -right,     0,      q2 - right],   # joint 2
    [99.973,   0,          0,      q3],           # joint 3
    [10.012,    -right,     107.011,     q4],           # joint 4
    [0.003,     right,      0.006,      q5],           # joint 5
    [0.054,     -right,     64.973,   q6]              # joint 6
]

def get_transformation_matrix(a, alpha, d, theta):
    return Matrix([
        [cos(theta), -sin(theta), 0, a],
        [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
        [0, 0, 0, 1]
    ])
def overall_transformation(dh_table):
    T = Matrix(np.identity(4))
    for i in range(len(dh_table)):
        a, alpha, d, theta = dh_table[i]
        T_i = get_transformation_matrix(a, alpha, d, theta)
        T = T * T_i
    return T

T_symbolic = overall_transformation(dh_table)
fk_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, 3], modules='numpy')
rot_num = lambdify((q1, q2, q3, q4, q5, q6), T_symbolic[:3, :3], modules='numpy')

def fk(q_vals):
    q = np.asarray(q_vals, dtype=float)
    q = np.radians(q)
    xyz = np.asarray(fk_num(*q), dtype=float).ravel()
    return xyz

def rotation_matrix_to_euler_zyx(R, transpose=False):
    """
    Extract Z-Y-X Euler angles (yaw, pitch, roll) from a 3x3 rotation matrix.
    
    The matrix convention is:
    R = [[cos(γ)cos(β),  -cos(α)sin(γ) + cos(γ)sin(α)sin(β),   sin(γ)sin(α) + cos(γ)cos(α)sin(β)],
         [cos(β)sin(γ),   cos(γ)cos(α) + sin(γ)sin(α)sin(β),  -cos(γ)sin(α) + cos(α)sin(γ)sin(β)],
         [-sin(β),        cos(β)sin(α),                         cos(α)cos(β)]]
    
    Where: γ = yaw, β = pitch, α = roll
    
    Args:
        R: 3x3 rotation matrix
        transpose: If True, use R.T instead of R (default False)
        
    Returns:
        (roll, pitch, yaw) in radians
    """
    # Apply transpose if needed
    if transpose:
        R = R.T
    
    # Extract pitch from R[2,0] = -sin(β)
    sin_beta = -R[2, 0]
    sin_beta = np.clip(sin_beta, -1.0, 1.0)  # Clamp for numerical stability
    pitch = np.arcsin(sin_beta)
    
    cos_beta = np.cos(pitch)
    
    # Check for gimbal lock (cos(β) ≈ 0, i.e., pitch ≈ ±90°)
    if np.abs(cos_beta) < 1e-6:
        # Gimbal lock: set yaw = 0 and extract roll from remaining matrix elements
        yaw = 0.0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    else:
        # Normal case: extract roll and yaw from matrix elements
        # From R[2,1] = cos(β)sin(α)  and  R[2,2] = cos(α)cos(β)
        roll = np.arctan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
        
        # From R[0,0] = cos(γ)cos(β)  and  R[1,0] = cos(β)sin(γ)
        yaw = np.arctan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
    
    return roll, pitch, yaw

def position_error(q_position, x_target, y_target, z_target):
    """
    Compute position error between FK result and target.
    
    Args:
        q_position: Joint angles in degrees (6-element array)
        x_target, y_target, z_target: Target position in mm
        
    Returns:
        3-element error vector in mm
    """
    pos = fk(q_position)  # fk() handles degree→radian conversion internally
    return np.array([pos[0] - float(x_target),
                     pos[1] - float(y_target),
                     pos[2] - float(z_target)])

def orientation_error(q_orientation, rx_d, ry_d, rz_d):
    def _wrap_angle(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    q = np.asarray(q_orientation, dtype=float)
    q_rad = np.radians(q)
    rx_d_rad = np.radians(rx_d)
    ry_d_rad = np.radians(ry_d)
    rz_d_rad = np.radians(rz_d)

    R = np.array(rot_num(*q_rad), dtype=float)
    # Remove transpose=True - use the matrix directly
    roll, pitch, yaw = rotation_matrix_to_euler_zyx(R, transpose=False)

    err_roll  = _wrap_angle(roll  - rx_d_rad)
    err_pitch = _wrap_angle(pitch - ry_d_rad)
    err_yaw   = _wrap_angle(yaw   - rz_d_rad)

    return np.array([err_roll, err_pitch, err_yaw])

def link_lengths(dh_table):
    """Compute upper bound on robot reach from DH parameters."""
    link_lengths = 0.0
    for a, alpha, d, theta in dh_table:
        link_lengths += abs(float(a)) + abs(float(d))
    return link_lengths

def inverse_kinematics(x_target, y_target, z_target, rx_d, ry_d, rz_d, 
                       q_init, link_lengths, max_iterations=5000, tolerance=1e-6):
    """
    Unified inverse kinematics optimization for position AND orientation.
    """
    distance_from_origin = math.hypot(x_target, y_target, z_target)
    if distance_from_origin > link_lengths:
        raise ValueError("Target outside of range")
    
    q_init = np.asarray(q_init, dtype=float)
    
    # Unified residual function (position + orientation)
    def combined_residual(q_all,pos_weight = 1.0, ori_weight = 10.0):
        """
        Returns 6-element residual: [pos_x, pos_y, pos_z, ori_roll, ori_pitch, ori_yaw]
        """
        pos_err = position_error(q_all, x_target, y_target, z_target)
        ori_err = orientation_error(q_all, rx_d, ry_d, rz_d)
        
        
        return np.concatenate([pos_err * pos_weight, ori_err * ori_weight])
    
    # Use trust region reflective for better handling of non-linear problems
    result = least_squares(
        combined_residual, 
        q_init,
        method='trf',  # Trust Region Reflective - better than LM for this
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        verbose=0
    )
    
    if not result.success:
        print(f"IK optimization warning: {result.message}")
    
    q_solution = result.x
    
    # Verify solution quality
    final_pos_error = np.linalg.norm(position_error(q_solution, x_target, y_target, z_target))
    final_ori_error = np.linalg.norm(orientation_error(q_solution, rx_d, ry_d, rz_d))
    
    # More reasonable thresholds (in degrees for orientation)
    if final_pos_error > 10.0:  # 10mm position error
        print(f"Warning: Large position error: {final_pos_error:.2f} mm")
    
    if final_ori_error > np.radians(10):  # 10° orientation error
        print(f"Warning: Large orientation error: {np.degrees(final_ori_error):.2f}°")
    
    return q_solution


def angle_diff_deg(a_deg, b_deg):
    """Compute signed angle difference in degrees, wrapped to [-180, 180]."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return d

def test_single_pose(target_pos, target_ori, known_joints, q_init, 
                     max_iterations=5000, tolerance=1e-6, test_index=None):
    """
    Test IK for a single target pose and compute comprehensive error metrics.
    
    Args:
        target_pos: [x, y, z] target position in mm
        target_ori: [rx, ry, rz] target orientation in degrees
        known_joints: Known correct joint angles in degrees (ground truth)
        q_init: Initial guess for joint angles in degrees
        max_iterations: Max optimizer iterations
        tolerance: Convergence tolerance
        test_index: Optional index for this test case
        
    Returns:
        dict: Comprehensive test results including all error metrics
    """
    x, y, z = target_pos
    rx, ry, rz = target_ori
    
    # Use your link_lengths function to compute reach
    reach = link_lengths(dh_table)
    
    # Initialize result dictionary
    result = {
        'index': test_index,
        'target_x': x,
        'target_y': y,
        'target_z': z,
        'target_rx': rx,
        'target_ry': ry,
        'target_rz': rz,
    }
    
    # Store known (ground truth) joint angles
    for i, angle in enumerate(known_joints, 1):
        result[f'known_j{i}'] = angle
    
    # Store initial guess used
    for i, angle in enumerate(q_init, 1):
        result[f'init_j{i}'] = angle
    
    # Attempt inverse kinematics using YOUR function
    try:
        solved_joints = inverse_kinematics(x, y, z, rx, ry, rz, q_init, reach,
                                         max_iterations=max_iterations, tolerance=tolerance)
        solved_joints = np.asarray(solved_joints, dtype=float)
        
        # Store solved joint angles
        for i, angle in enumerate(solved_joints, 1):
            result[f'solved_j{i}'] = angle
        
        # Compute per-joint errors
        joint_errors_signed = np.array([angle_diff_deg(solved_joints[i], known_joints[i]) 
                                       for i in range(6)])
        joint_errors_abs = np.abs(joint_errors_signed)
        
        for i, (signed, absolute) in enumerate(zip(joint_errors_signed, joint_errors_abs), 1):
            result[f'j{i}_error_signed'] = signed
            result[f'j{i}_error_abs'] = absolute
        
        # Joint space error metrics
        result['joint_error_mean'] = np.mean(joint_errors_abs)
        result['joint_error_max'] = np.max(joint_errors_abs)
        result['joint_error_rms'] = np.sqrt(np.mean(joint_errors_abs**2))
        result['joint_error_total'] = np.sum(joint_errors_abs)
        
        # CRITICAL: Use solved joints with YOUR FK function to see actual end-effector error
        actual_pos = fk(solved_joints)
        
        # Task space position errors
        pos_error_vec = actual_pos - target_pos
        result['fk_pos_error_x'] = pos_error_vec[0]
        result['fk_pos_error_y'] = pos_error_vec[1]
        result['fk_pos_error_z'] = pos_error_vec[2]
        result['fk_pos_error_norm'] = np.linalg.norm(pos_error_vec)
        
        # Also compute position error if we used known joints (should be ~0)
        known_pos = fk(known_joints)
        known_pos_error = np.linalg.norm(known_pos - target_pos)
        result['known_fk_pos_error'] = known_pos_error
        
        # Success indicators
        result['converged'] = True
        result['status'] = 'success'
        
        # Quality assessment based on TASK SPACE error (what matters!)
        if result['fk_pos_error_norm'] < 1.0:
            result['quality'] = 'excellent'
        elif result['fk_pos_error_norm'] < 5.0:
            result['quality'] = 'good'
        elif result['fk_pos_error_norm'] < 10.0:
            result['quality'] = 'acceptable'
        elif result['fk_pos_error_norm'] < 50.0:
            result['quality'] = 'poor'
        else:
            result['quality'] = 'failed'
            
    except Exception as e:
        # IK failed
        result['converged'] = False
        result['status'] = f'failed: {str(e)}'
        result['quality'] = 'failed'
        
        # Fill with NaN for missing values
        for i in range(1, 7):
            result[f'solved_j{i}'] = np.nan
            result[f'j{i}_error_signed'] = np.nan
            result[f'j{i}_error_abs'] = np.nan
        
        for key in ['joint_error_mean', 'joint_error_max', 'joint_error_rms', 'joint_error_total',
                   'fk_pos_error_x', 'fk_pos_error_y', 'fk_pos_error_z', 'fk_pos_error_norm',
                   'known_fk_pos_error']:
            result[key] = np.nan
    
    return result

def test_multiple_poses_honest(targets_pos, targets_ori, known_joints_list,
                               initial_guess_strategy='zero',
                               max_iterations=5000, tolerance=1e-6,
                               verbose=True):
    """
    Test IK for multiple poses with HONEST initial guesses that don't cheat.
    
    Args:
        targets_pos: Array of target positions [[x, y, z], ...]
        targets_ori: Array of target orientations [[rx, ry, rz], ...]
        known_joints_list: Array of known joint angles for each pose
        initial_guess_strategy: How to generate initial guesses:
            - 'zero': All joints at [0, 0, 0, 0, 0, 0]
            - 'random': Random angles within reasonable bounds
            - 'home': Common home position [0, -90, 90, 0, 0, 0]
            - 'constant': Use [10, 10, 10, 10, 10, 10] for all
        max_iterations: Max optimizer iterations
        tolerance: Convergence tolerance
        verbose: Print progress information
        
    Returns:
        pandas.DataFrame: Complete test results
    """
    results = []
    
    for i, (pos, ori, known) in enumerate(zip(targets_pos, targets_ori, known_joints_list)):
        if verbose and i % 10 == 0:
            print(f"Testing pose {i+1}/{len(targets_pos)}...")
        
        # Generate initial guess based on strategy
        if initial_guess_strategy == 'zero':
            q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif initial_guess_strategy == 'random':
            # Random angles between -90 and 90 degrees
            q_init = np.random.uniform(-90, 90, 6)
        elif initial_guess_strategy == 'home':
            q_init = np.array([0.0, -90.0, 90.0, 0.0, 0.0, 0.0])
        elif initial_guess_strategy == 'constant':
            q_init = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        else:
            raise ValueError(f"Unknown strategy: {initial_guess_strategy}")
        
        result = test_single_pose(
            pos, ori, known, q_init,
            max_iterations=max_iterations,
            tolerance=tolerance,
            test_index=i
        )
        
        results.append(result)
    
    return pd.DataFrame(results)

def test_multiple_poses_with_previous(targets_pos, targets_ori, known_joints_list,
                                     q_init=None, max_iterations=5000, tolerance=1e-6,
                                     verbose=True):
    """
    Test IK using previous solution as initial guess (EASY MODE - may hide problems).
    
    This is what your original code was doing - it makes IK look better than it is!
    """
    if q_init is None:
        q_init = known_joints_list[0]  # Start with first known joints
    
    results = []
    
    for i, (pos, ori, known) in enumerate(zip(targets_pos, targets_ori, known_joints_list)):
        if verbose and i % 10 == 0:
            print(f"Testing pose {i+1}/{len(targets_pos)}...")
        
        result = test_single_pose(
            pos, ori, known, q_init,
            max_iterations=max_iterations,
            tolerance=tolerance,
            test_index=i
        )
        
        results.append(result)
        
        # Use previous solution as next initial guess (THIS MAKES IT EASY!)
        if result['converged']:
            q_init = np.array([result[f'solved_j{j}'] for j in range(1, 7)])
    
    return pd.DataFrame(results)

def print_summary(results_df, strategy_name="Unknown"):
    """Print a clean, actionable summary of test results."""
    if len(results_df) == 0:
        print("No test results available.")
        return
    
    print("\n" + "="*80)
    print(f"INVERSE KINEMATICS TEST SUMMARY - Strategy: {strategy_name}")
    print("="*80)
    
    # Overall statistics
    total_tests = len(results_df)
    converged = results_df['converged'].sum()
    failed = total_tests - converged
    
    print(f"\n📊 Overall Performance:")
    print(f"   Total tests:      {total_tests}")
    print(f"   Converged:        {converged} ({100*converged/total_tests:.1f}%)")
    print(f"   Failed:           {failed} ({100*failed/total_tests:.1f}%)")
    
    if converged > 0:
        converged_df = results_df[results_df['converged']]
        
        # Quality distribution
        print(f"\n⭐ Quality Distribution (converged cases):")
        quality_counts = converged_df['quality'].value_counts()
        for quality in ['excellent', 'good', 'acceptable', 'poor', 'failed']:
            count = quality_counts.get(quality, 0)
            if count > 0:
                print(f"   {quality.capitalize():12s}: {count:4d} ({100*count/converged:.1f}%)")
        
        # Task space errors (what really matters!)
        print(f"\n🎯 Task Space Errors (using IK solution with FK):")
        print(f"   Position Error (mm):")
        print(f"      Mean:  {converged_df['fk_pos_error_norm'].mean():8.3f}")
        print(f"      Median:{converged_df['fk_pos_error_norm'].median():8.3f}")
        print(f"      Max:   {converged_df['fk_pos_error_norm'].max():8.3f}")
        print(f"      Min:   {converged_df['fk_pos_error_norm'].min():8.3f}")
        print(f"      Std:   {converged_df['fk_pos_error_norm'].std():8.3f}")
        
        # Per-coordinate errors
        print(f"\n   Position Error by Coordinate (mm):")
        for coord in ['x', 'y', 'z']:
            col = f'fk_pos_error_{coord}'
            mean_err = converged_df[col].abs().mean()
            max_err = converged_df[col].abs().max()
            print(f"      {coord.upper()}: mean={mean_err:7.3f}, max={max_err:7.3f}")
        
        # Joint space errors
        print(f"\n🔧 Joint Space Errors (degrees):")
        print(f"   Mean per-joint error: {converged_df['joint_error_mean'].mean():.3f}°")
        print(f"   Max per-joint error:  {converged_df['joint_error_max'].mean():.3f}°")
        print(f"   RMS per-joint error:  {converged_df['joint_error_rms'].mean():.3f}°")
        
        # Individual joint analysis
        print(f"\n   Error by Joint (degrees, absolute):")
        for i in range(1, 7):
            col = f'j{i}_error_abs'
            mean_err = converged_df[col].mean()
            max_err = converged_df[col].max()
            print(f"      J{i}: mean={mean_err:6.2f}°, max={max_err:6.2f}°")
        
        # Ground truth FK validation
        print(f"\n✅ Ground Truth Validation:")
        mean_known_error = converged_df['known_fk_pos_error'].mean()
        print(f"   FK with known joints error: {mean_known_error:.6f} mm")
        if mean_known_error < 1.0:
            print(f"   → Forward kinematics is working correctly! ✓")
        else:
            print(f"   → WARNING: FK errors suggest DH parameter issues!")
        
        # Key insight
        print(f"\n💡 Key Insight:")
        mean_task_error = converged_df['fk_pos_error_norm'].mean()
        mean_joint_error = converged_df['joint_error_mean'].mean()
        
        if mean_task_error < 5.0:
            print(f"   ✓ Task space errors are acceptable ({mean_task_error:.2f} mm)")
            print(f"     Joint errors ({mean_joint_error:.2f}°) map to good end-effector accuracy")
        elif mean_task_error < 10.0:
            print(f"   ⚠ Task space errors are marginal ({mean_task_error:.2f} mm)")
            print(f"     Consider improving optimization or joint error tolerance")
        else:
            print(f"   ✗ Task space errors are too large ({mean_task_error:.2f} mm)")
            print(f"     Joint errors ({mean_joint_error:.2f}°) lead to poor end-effector accuracy")
            print(f"     Action: Review IK optimization strategy")
    
    print("\n" + "="*80)

def get_worst_cases(results_df, n=5, metric='fk_pos_error_norm'):
    """Get the worst performing test cases for debugging."""
    converged_df = results_df[results_df['converged']].copy()
    if len(converged_df) == 0:
        print("No converged cases to analyze.")
        return pd.DataFrame()
    
    worst = converged_df.nlargest(min(n, len(converged_df)), metric)
    
    print(f"\n🔍 Top {min(n, len(converged_df))} Worst Cases (by {metric}):")
    print("-" * 80)
    
    for idx, row in worst.iterrows():
        print(f"\nCase {row['index']}:")
        print(f"  Target: ({row['target_x']:.1f}, {row['target_y']:.1f}, {row['target_z']:.1f}) mm")
        print(f"  FK Position Error: {row['fk_pos_error_norm']:.3f} mm")
        print(f"  Joint Errors (deg): ", end="")
        for i in range(1, 7):
            print(f"J{i}={row[f'j{i}_error_abs']:.1f}° ", end="")
        print()
    
    return worst

def save_results(results_df, filepath):
    """
    Save detailed results to CSV.
    
    Args:
        results_df: DataFrame to save
        filepath: Path to save CSV file
    """
    filepath = Path(filepath)
    results_df.to_csv(filepath, index=False)
    print(f"\n💾 Results saved to: {filepath}")
    print(f"   Total rows: {len(results_df)}")
    print(f"   Total columns: {len(results_df.columns)}")

def test_multiple_poses(targets_pos, targets_ori, known_joints_list,
                       q_init=None, max_iterations=5000, tolerance=1e-6,
                       use_previous_solution=True, verbose=True):
    """
    Test IK for multiple poses - wrapper function for compatibility.
    
    Args:
        targets_pos: Array of target positions [[x, y, z], ...]
        targets_ori: Array of target orientations [[rx, ry, rz], ...]
        known_joints_list: Array of known joint angles for each pose
        q_init: Initial joint guess (if None, uses first known joints)
        max_iterations: Max optimizer iterations
        tolerance: Convergence tolerance
        use_previous_solution: Use previous solution as next initial guess
        verbose: Print progress information
        
    Returns:
        pandas.DataFrame: Complete test results
    """
    if use_previous_solution:
        return test_multiple_poses_with_previous(
            targets_pos, targets_ori, known_joints_list,
            q_init=q_init, max_iterations=max_iterations, 
            tolerance=tolerance, verbose=verbose
        )
    else:
        # Use honest testing with zero initial guess
        return test_multiple_poses_honest(
            targets_pos, targets_ori, known_joints_list,
            initial_guess_strategy='zero',
            max_iterations=max_iterations,
            tolerance=tolerance, verbose=verbose
        )

def run_comprehensive_ik_testing():
    """
    Run HONEST IK Tests
    Test your IK with different initial guess strategies to see how robust it really is.
    """
    print("="*80)
    print("COMPREHENSIVE IK TESTING - MULTIPLE STRATEGIES")
    print("="*80)

    # Test 1: Zero initial guess (hardest test)
    print("\n\n🔥 TEST 1: Zero Initial Guess (All joints at 0°)")
    print("This is the HARDEST test - your IK must find the solution from scratch!")
    results_zero = test_multiple_poses_honest(
        known_position,
        known_orientation,
        known_joint_angles,
        initial_guess_strategy='zero',
        max_iterations=5000,
        tolerance=1e-6,
        verbose=True
    )
    print_summary(results_zero, "Zero Initial Guess")
    worst_zero = get_worst_cases(results_zero, n=3)

    # Test 2: Constant initial guess
    print("\n\n🔸 TEST 2: Constant Initial Guess (All joints at 10°)")
    print("Tests if your IK can handle a consistent but non-zero starting point.")
    results_constant = test_multiple_poses_honest(
        known_position,
        known_orientation,
        known_joint_angles,
        initial_guess_strategy='constant',
        max_iterations=5000,
        tolerance=1e-6,
        verbose=True
    )
    print_summary(results_constant, "Constant Initial Guess")
    worst_constant = get_worst_cases(results_constant, n=3)

    # Test 3: Using previous solution (EASY MODE - what your code was doing)
    print("\n\n✨ TEST 3: Using Previous Solution (EASY MODE)")
    print("This makes IK look better because each guess is already close to the answer!")
    print("This is what your original code was doing - it hides problems!")
    results_previous = test_multiple_poses_with_previous(
        known_position,
        known_orientation,
        known_joint_angles,
        q_init=None,  # Will use first known joints
        max_iterations=5000,
        tolerance=1e-6,
        verbose=True
    )
    print_summary(results_previous, "Previous Solution (Easy Mode)")
    worst_previous = get_worst_cases(results_previous, n=3)


    results_zero.to_csv(output_dir / "ik_results_zero_init.csv", index=False)
    results_constant.to_csv(output_dir / "ik_results_constant_init.csv", index=False)
    results_previous.to_csv(output_dir / "ik_results_previous_init.csv", index=False)

    print("\n\n💾 Results saved to:")
    print(f"   {output_dir / 'ik_results_zero_init.csv'}")
    print(f"   {output_dir / 'ik_results_constant_init.csv'}")
    print(f"   {output_dir / 'ik_results_previous_init.csv'}")

    # Compare strategies
    print("\n\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    strategies = [
        ("Zero Init (Hard)", results_zero),
        ("Constant Init", results_constant),
        ("Previous Solution (Easy)", results_previous)
    ]

    print(f"\n{'Strategy':<30} {'Success Rate':<15} {'Mean Error (mm)':<20} {'Quality':<15}")
    print("-"*80)

    for name, df in strategies:
        converged = df['converged'].sum()
        total = len(df)
        success_rate = f"{converged}/{total} ({100*converged/total:.1f}%)"
        
        if converged > 0:
            converged_df = df[df['converged']]
            mean_error = f"{converged_df['fk_pos_error_norm'].mean():.3f}"
            
            # Count excellent cases
            excellent = (converged_df['quality'] == 'excellent').sum()
            quality = f"{excellent}/{converged} excellent"
        else:
            mean_error = "N/A"
            quality = "N/A"
        
        print(f"{name:<30} {success_rate:<15} {mean_error:<20} {quality:<15}")

    print("\n💡 If 'Previous Solution' works well but 'Zero Init' fails badly,")
    print("   your IK algorithm is NOT robust - it only works when given good guesses!")
    
    return results_zero, results_constant, results_previous

def main():
    """
    Main function that runs when the script is executed directly.
    Performs comprehensive IK testing using your own functions.
    """
    print("="*80)
    print("COMPREHENSIVE IK TESTING - USING YOUR FUNCTIONS")
    print("="*80)
    print("\nThis test suite uses:")
    print("- Your fk() function for forward kinematics")
    print("- Your inverse_kinematics() function for IK solving")
    print("- Your link_lengths() function for reach calculation")
    print("\nGenerates:")
    print("- Detailed console output with statistics")
    print("- CSV file with all results")
    print("- Worst case analysis for debugging")
    
    # Run tests on all poses from your CSV data using your own functions
    print("\n" + "="*60)
    print("Starting comprehensive IK testing...")
    print("This will test each pose and compute actual end-effector errors.")
    print("="*60)

    results_df = test_multiple_poses(
        targets_pos=known_position,
        targets_ori=known_orientation,
        known_joints_list=known_joint_angles,
        q_init=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Small non-zero initial guess
        max_iterations=5000,
        tolerance=1e-6,
        use_previous_solution=True,  # Use previous solution as next initial guess
        verbose=True
    )

    # Print comprehensive summary
    print_summary(results_df, "Using Your Functions")

    # Show worst 5 cases for debugging
    worst_cases = get_worst_cases(results_df, n=5)

    # Save results to CSV
    output_path = Path(".") / "ik_results_your_functions.csv"
    save_results(results_df, output_path)

    # Display first few rows of results
    print("\n📋 Sample Results (first 5 rows):")
    print(results_df.head().to_string())
    
    # Show basic statistics
    if results_df['converged'].sum() > 0:
        converged_results = results_df[results_df['converged']]
        
        print(f"\n" + "="*60)
        print("QUICK SUMMARY:")
        print("="*60)
        print(f"  Success Rate: {results_df['converged'].sum()}/{len(results_df)} ({100*results_df['converged'].sum()/len(results_df):.1f}%)")
        print(f"  Avg Task Space Error: {converged_results['fk_pos_error_norm'].mean():.3f} mm")
        print(f"  Avg Joint Error: {converged_results['joint_error_mean'].mean():.3f}°")
        print(f"\n  Quality Breakdown:")
        quality_counts = converged_results['quality'].value_counts()
        for quality, count in quality_counts.items():
            print(f"    {quality.capitalize()}: {count} cases")
    
    print(f"\n✅ Testing complete! Results saved to: {output_path}")
    
    return results_df

# Run the main function when script is executed directly
if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("\nThis could be due to:")
        print("- Missing CSV file (mecharm_control_group_5.csv)")
        print("- Issues with your inverse kinematics function")
        print("- Missing dependencies (numpy, pandas, scipy, sympy)")
        raise




