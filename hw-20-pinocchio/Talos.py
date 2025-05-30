import numpy as np

class Talos:
    def __init__(self, robot, model, data, left_foot_id, right_foot_id, 
                 left_ank, right_ank, com_traj, footsteps, robot_base_height, base_foot_height, dt=0.05):
        self.robot = robot
        self.model = model
        self.data = data
        self.left_foot_id = left_foot_id
        self.right_foot_id = right_foot_id
        self.left_ank = left_ank
        self.right_ank = right_ank
        self.com_traj = com_traj
        self.footsteps = footsteps
        self.robot_base_height = robot_base_height
        self.base_foot_height = base_foot_height
        self.dt = dt

    def inverse_kinematics(self, frame_id, target_pose, q_init, max_iter=100):
        import pinocchio as pin
        oMtarget = pin.SE3(target_pose)
        q = q_init.copy()
        model = self.model
        data = self.data

        if "left" in model.frames[frame_id].name:
            knee_joint_names = ["leg_left_3_joint"]
            leg_joint_names = ["leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", 
                            "leg_left_4_joint", "leg_left_5_joint", "leg_left_6_joint"]
        else:
            knee_joint_names = ["leg_right_3_joint"]
            leg_joint_names = ["leg_right_1_joint", "leg_right_2_joint", "leg_right_3_joint",
                            "leg_right_4_joint", "leg_right_5_joint", "leg_right_6_joint"]
        
        for i in range(max_iter):
            pin.framesForwardKinematics(model, data, q)
            dM = oMtarget.actInv(data.oMf[frame_id])
            err = pin.log(dM).vector
            
            if np.linalg.norm(err) < 1e-6:  # Tighter threshold
                break
                
            J = pin.computeFrameJacobian(model, data, q, frame_id)
            
            W = np.eye(model.nv) * 0.1
            for joint_name in leg_joint_names:
                try:
                    joint_id = model.getJointId(joint_name)
                    joint_idx = model.joints[joint_id].idx_v
                    if joint_idx < model.nv:
                        W[joint_idx, joint_idx] = 1.0
                except:
                    continue
            
            for joint_name in knee_joint_names:
                try:
                    joint_id = model.getJointId(joint_name)
                    joint_idx = model.joints[joint_id].idx_v
                    if joint_idx < model.nv:
                        W[joint_idx, joint_idx] = 2.0
                except:
                    continue
            
            JW = J @ W
            v = -W @ J.T @ np.linalg.solve(JW @ J.T + 1e-4*np.eye(6), err)
            q = pin.integrate(model, q, v*0.1)
        
        return q

    def get_corrected_leg_position(self, ankle_func, t, is_left_leg=True, ground_height=None):
        phase_type = self.footsteps.get_phase_type(t)
        ankle_pos = ankle_func(t)
        if ground_height is None:
            ground_height = self.base_foot_height
        if (is_left_leg and phase_type == 'left') or (not is_left_leg and phase_type == 'right'):
            ankle_pos[2] = ground_height
        elif (is_left_leg and phase_type != 'left') or (not is_left_leg and phase_type != 'right'):
            ankle_pos[2] = ground_height
        return ankle_pos

    def animate(self, t_total):
        import pinocchio as pin
        import time
        N = int(t_total / self.dt)
        prev_com_z = self.robot_base_height

        for i in range(N):
            t = i * self.dt
            com_pos = self.com_traj(t)
            left_ankle_pos = self.get_corrected_leg_position(self.left_ank, t, is_left_leg=True, ground_height=self.base_foot_height)
            right_ankle_pos = self.get_corrected_leg_position(self.right_ank, t, is_left_leg=False, ground_height=self.base_foot_height)
            q = self.robot.q0.copy()
            phase_type = self.footsteps.get_phase_type(t)
            if phase_type == 'left':
                target_com_z = self.robot_base_height - (self.base_foot_height - right_ankle_pos[2]) * 0.3
                q[2] = 0.9 * prev_com_z + 0.1 * target_com_z 
            elif phase_type == 'right':
                target_com_z = self.robot_base_height - (self.base_foot_height - left_ankle_pos[2]) * 0.3
                q[2] = 0.9 * prev_com_z + 0.1 * target_com_z
            else:
                q[2] = self.robot_base_height
            prev_com_z = q[2]
            q[0] = com_pos[0]
            q[1] = com_pos[1]
            left_target = pin.SE3(np.eye(3), np.array([left_ankle_pos[0], left_ankle_pos[1], left_ankle_pos[2]]))
            right_target = pin.SE3(np.eye(3), np.array([right_ankle_pos[0], right_ankle_pos[1], right_ankle_pos[2]]))
            q = self.inverse_kinematics(self.left_foot_id, left_target, q)
            q = self.inverse_kinematics(self.right_foot_id, right_target, q)
            self.robot.display(q)
            time.sleep(self.dt)
            if i % 20 == 0:
                print(
                    f"Time: {t:.2f}s / {t_total:.2f}s | "
                    f"Phase: {phase_type:>5} | "
                    f"CoM: [{com_pos[0]:.3f}, {com_pos[1]:.3f}, {com_pos[2]:.3f}] | "
                    f"Base height: {q[2]:.3f} | "
                    f"Left foot Z: {left_ankle_pos[2]:.3f} | Right foot Z: {right_ankle_pos[2]:.3f}"
                )
