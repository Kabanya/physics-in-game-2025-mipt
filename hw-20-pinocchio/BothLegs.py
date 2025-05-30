from FootSteps import *
import numpy as np
import matplotlib.pyplot as plt

d_foot_ankle= 0.08
h_lift=0.1
h_base=0.862+d_foot_ankle

class LeftLeg(object):

    def __init__(self, footsteps: FootSteps) -> None:
        self.footsteps=footsteps

    def __call__(self, t):        
        left=np.array(self.footsteps.get_left_position(t))
        left_next=np.array(self.footsteps.get_left_next_position(t))        
        t_start=self.footsteps.get_phase_start(t)
        t_total=self.footsteps.get_phase_duration(t)
        k=(t-t_start)/t_total
        
        phase_type = self.footsteps.get_phase_type(t)
        
        if phase_type == 'left':
            z_foot=h_lift*np.sin(k*np.pi)+d_foot_ankle
            act_left = left + (left_next - left) * k
        else:
            z_foot = d_foot_ankle
            act_left = left
            
        return np.array([act_left[0], act_left[1], z_foot])

class RightLeg(object):

    def __init__(self, footsteps: FootSteps) -> None:        
        self.footsteps=footsteps

    def __call__(self, t):        
        right=np.array(self.footsteps.get_right_position(t))
        right_next=np.array(self.footsteps.get_right_next_position(t))
        t_start=self.footsteps.get_phase_start(t)
        t_total=self.footsteps.get_phase_duration(t)
        k=(t-t_start)/t_total
        
        phase_type = self.footsteps.get_phase_type(t)
        
        if phase_type == 'right':
            z_foot=h_lift*np.sin(k*np.pi)+d_foot_ankle
            act_right = right + (right_next - right) * k
        else:
            z_foot = d_foot_ankle
            act_right = right
            
        return np.array([act_right[0], act_right[1], z_foot])

def plot_legs(footsteps:FootSteps):
    tmp_left=LeftLeg(footsteps)
    tmp_right=RightLeg(footsteps)

    t_max=footsteps.timetime[-1]
    dt=0.01
    N=int(t_max/dt+1)
    zmp_traj=np.zeros((N,2))
    ankle_left_traj=np.zeros((N,3))
    ankle_right_traj=np.zeros((N,3))
    for k in range(N):
        t_q=dt*k
        ankle_left_traj[k,:]=tmp_left(t_q)
        ankle_right_traj[k,:]=tmp_right(t_q)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(ankle_left_traj[:,0],ankle_left_traj[:,1],ankle_left_traj[:,2], label='Left leg')
    ax.plot(ankle_right_traj[:,0],ankle_right_traj[:,1],ankle_right_traj[:,2], label='Right leg')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()