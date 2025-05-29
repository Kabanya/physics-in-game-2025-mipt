# # starting in the middle of the ankles of the two first steps,
# # finishing in the middle of the two ankles of the two last steps,
# # constant under the foot support during single support phases.

# class ZmpRef(object):
#     def __init__(self, footsteps):
#         self.footsteps = footsteps
 
#     def __call__(self, t):
#         return np.array(self.footsteps[0])

# # starting in the middle of the ankles of the two first steps,
# # finishing in the middle of the two ankles of the two last steps,
# # constant under the foot support during single support phases.

import numpy as np
import matplotlib.pyplot as plt
from FootSteps import FootSteps

class ZmpClass(object):
    def __init__(self, footsteps: FootSteps):
        self.footsteps = footsteps
 
    def __call__(self, t):
        foot = self.footsteps.get_phase_type(t)
        left_pos = self.footsteps.get_left_position(t)
        right_pos = self.footsteps.get_right_position(t)
        
        if foot == 'left':
            return np.array(right_pos)
        elif foot == 'right':
            return np.array(left_pos)
        else: # 'none' phase, double support
            return np.array([(left_pos[0] + right_pos[0]) / 2, 
                           (left_pos[1] + right_pos[1]) / 2])

def plot_zmp(footsteps: FootSteps):
    test = ZmpClass(footsteps)
    t_max = footsteps.timetime[-1]
    dt = 0.1
    N = int(t_max / dt + 1)
    zmp_traj = np.zeros((N, 2))
    left_traj = np.zeros((N, 2))
    right_traj = np.zeros((N, 2))
    
    for k in range(N):
        t = k * dt
        zmp_traj[k, :] = test(t)
        left_traj[k, :] = footsteps.get_left_position(t)
        right_traj[k, :] = footsteps.get_right_position(t)
    
    plt.figure(figsize=(10, 8))
    plt.plot(zmp_traj[:, 0], zmp_traj[:, 1], 'ro-', label='ZMP trajectory', markersize=3)
    plt.plot(left_traj[:, 0], left_traj[:, 1], 'bx-', label='Left foot', markersize=4)
    plt.plot(right_traj[:, 0], right_traj[:, 1], 'gx-', label='Right foot', markersize=4)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.title('ZMP and Foot Trajectories')
    plt.show()