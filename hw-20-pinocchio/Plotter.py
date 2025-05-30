from Zmp import plot_zmp
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    @staticmethod
    def plot_all(com_traj, left_ank, right_ank, t_total, footsteps):

        t_plot = np.linspace(0, t_total, 100)
        com_traj_plot = np.array([com_traj(t) for t in t_plot])
        left_foot_traj = np.array([left_ank(t) for t in t_plot])
        right_foot_traj = np.array([right_ank(t) for t in t_plot])

        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        plt.plot(t_plot, com_traj_plot[:, 0], label='CoM X')
        plt.plot(t_plot, left_foot_traj[:, 0], '--', label='Left foot X')
        plt.plot(t_plot, right_foot_traj[:, 0], '--', label='Right foot X')
        plt.title('CoM X trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('X position (m)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(t_plot, com_traj_plot[:, 1], label='CoM Y')
        plt.plot(t_plot, left_foot_traj[:, 1], '--', label='Left foot Y')
        plt.plot(t_plot, right_foot_traj[:, 1], '--', label='Right foot Y')
        plt.title('CoM Y trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Y position (m)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(com_traj_plot[:, 0], com_traj_plot[:, 1], label='CoM XY')
        plt.plot(left_foot_traj[:, 0], left_foot_traj[:, 1], '--', label='Left foot XY')
        plt.plot(right_foot_traj[:, 0], right_foot_traj[:, 1], '--', label='Right foot XY')
        plt.title('CoM XY trajectory')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        plot_zmp(footsteps)
        print("Plots generated successfully!")
        
    def plot_moment(com_traj, t_total):
        t_plot = np.linspace(0, t_total, 100)
        moments = np.array([com_traj.compute_moment(t) for t in t_plot])
        plt.figure()
        plt.plot(t_plot, moments, label='Moment about ZMP')
        plt.title('Moment of Forces')
        plt.xlabel('Time (s)')
        plt.ylabel('Moment (N·m)')
        plt.legend()
        plt.grid(True)
        plt.show()