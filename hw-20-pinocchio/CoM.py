## Center of Mass

import numpy as np
from Factor import *
from Zmp import *
import matplotlib.pyplot as plt

MASS = 10.0

class CoMClass(object):
    def __init__(self, zmp_traj: ZmpClass,
                 com_z_nominal: float = 0.9,  
                 solver='LCQP',
                 flag_plot=False):
        self.zmp_traj = zmp_traj
        self.dt=0.005 #sampling time
        self.t_end= self.zmp_traj.footsteps.timetime[-1]
        self.com_z=com_z_nominal
        self.g=9.8        
        self.x_opt=[]
        self.y_opt=[]
        self.N=int(self.t_end/self.dt)
        self.L=self.com_z/self.g
        self.solver=solver
        self.flag_plot=flag_plot        

        if solver == "LCQP":
            self.x_traj = self.solve_CoM_traj(0) #solve x trajectory
            self.y_traj = self.solve_CoM_traj(1) #y
        else:
            pass

    def solve_CoM_traj(self, dir: int):
        '''
        dir: 0 for x direction, 1 for y direction
        Solves CoM trajectory ensuring ZMP = CoM projection (zero moment).
        '''
        N = self.N
        nx = 3 
        L = self.L  # com_z / g
        f = FactorGraph(nx, N)
        
        M1=np.array([[1, self.dt, 0], [0, 1, self.dt], [0, 0, 1]]) 
        M2=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for k in range(N-1):
            f.add_factor_constraint([Factor(k, M1), Factor(k+1, -M2)], zero(3))
        
        Mc = np.array([[1, 0, -L]])  
        Mf=eye(1)
        for k in range(N):
            traj_x = self.zmp_traj(self.dt * k)[dir]
            f.add_factor_constraint([Factor(k, Mc)], Mf * traj_x)  
        
        opt= f.solve()
        if dir == 0:
            self.x_opt=np.copy(opt)
        else:
            self.y_opt=np.copy(opt)
        
        return np.reshape(opt[0::3], len(opt[0::3]))
 
    def __call__(self, t):     
        idx = int(t / self.dt)
        idx = min(idx, len(self.x_traj) - 1) # check bounds
        idx = max(idx, 0)
        return np.array([self.x_traj[idx], self.y_traj[idx], self.com_z]) 
    
    def compute_moment(self, t):
            idx = int(t / self.dt)
            idx = min(idx, self.N - 1)
            idx = max(idx, 0)

            com_pos_x = self.x_traj[idx]
            com_pos_y = self.y_traj[idx]
            
            ddc_x = self.x_opt[idx * 3 + 2]
            ddc_y = self.y_opt[idx * 3 + 2]
            
            zmp_ref = self.zmp_traj(t) # This is [zmp_ref_x, zmp_ref_y]
            moment_y_component = MASS * self.g * (com_pos_x - zmp_ref[0] - self.L * ddc_x)            
            moment_x_component = -MASS * self.g * (com_pos_y - zmp_ref[1] - self.L * ddc_y)
            
            return np.linalg.norm([moment_x_component, moment_y_component])

    def plot_ctrl_error_x(self):
        if self.solver != "LCQP":
            return
        #verify solution of LCQP along x direction
        ref_x= np.zeros((self.N,1))
        act_u= self.x_opt[2::3]
        act_c= self.x_opt[0::3]
        act_dc=self.x_opt[1::3]
        ddc= (act_dc[1:-1]-act_dc[0:-2])/self.dt
        act_x=act_c-self.L*act_u
        for k in range(self.N):
            ref_x[k]=self.zmp_traj(self.dt*k)[0]
        plt.subplot(1,2,1)
        plt.plot(ref_x)
        plt.plot(act_x,'x')
        plt.subplot(1,2,2)
        plt.plot(act_u)
        plt.plot(ddc,'x')
        
        plt.show()
        
    def plot_ctrl_error_y(self):
        if self.solver != "LCQP":
            return
        ref_y= np.zeros((self.N,1))
        act_u= self.y_opt[2::3]
        act_c= self.y_opt[0::3]
        act_dc=self.y_opt[1::3]
        ddc= (act_dc[1:-1]-act_dc[0:-2])/self.dt
        act_y=act_c-self.L*act_u
        for k in range(self.N):
            ref_y[k]=self.zmp_traj(self.dt*k)[1]
        plt.subplot(1,2,1)
        plt.plot(ref_y)
        plt.plot(act_y,'x')
        plt.subplot(1,2,2)
        plt.plot(act_u)
        plt.plot(ddc,'x')
        
        plt.show()

    def plot_com_traj(self):
        plt.plot(self.x_traj, self.y_traj)
        plt.show()