import numpy as np
from sympy  import symbols, cos, sin, atan2, pi, Matrix, lambdify 
from scipy.optimize import least_squares

# Reuse your FK function from Challenge 1
# forward_kinematics_func = lambdify(q_sym, symbolic_forward_kinematics(q_sym), 'numpy')

# Converts the transformation matrix into a pose vector: [X, Y, Z, roll, pitch, yaw]
def transf_to_pose(t_matrix):
    # Position (fill in indices based on your matrix structure)
    X, Y, Z = t_matrix[?,?], t_matrix[?,?], t_matrix[?,?]


   # Breakdown the rotation matrix into axes
   roll = np.arctan2(t_matrix[?,?], t_matrix[?,?])
   pitch = np.arctan2(-t_matrix[?,?], np.sqrt((?? ** 2) + (?? ** 2)))
   yaw = np.arctan2(??,??)

   return X, Y, Z, roll, pitch, yaw

# Checks the current joint angles against the target pose and returns their delta
def target_pose_error(joint_angles, *args):
   target_pose = args[0]
   current_fk = forward_kinematics_func(*joint_angles)

   # Convert translation matrix to pose vector
   current_pose = transf_to_pose(current_fk)

   error = ?? - ??
   return error

# Looks for a solution to inverse kinematics for a target pose and returns a transformation matrix
def ik(target_pose, init_pose, max_iter=1000, tolerance=1e-5, bounds=joint_limits):

   result = least_squares(error(s)..., some pose, args=(some other pose, ...), method='trf', max_nfev=max_iter, ftol=tolerance, bounds=bounds)

   if result.success:
       # print(f"Inverse kinematics converged after {result.nfev} function evaluations.")
       return result.x
   else:
       print("Inverse kinematics did not converge.")
       return None
       