import numpy as np
import matplotlib.pyplot as plt

def plot_com_trajectory(coms, com_errors, desired_com, dtime):
    time_steps = np.arange(len(coms)) * dtime
    coms = np.array(coms)
    com_errors = np.array(com_errors)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(time_steps, coms[:, 2], label='Actual Z')
    plt.plot(time_steps, desired_com[:, 2], label='Desired Z', linestyle='--')
    plt.title('Center of Mass Z Position: Fall and Recovery')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(time_steps[:-1], com_errors[:, 2], label='Z error')
    plt.title('Z Error During Fall and Recovery')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Error (m)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(time_steps, coms[:, 2] - desired_com[:, 2], label='Z deviation')
    plt.title('Deviation from Desired Z Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Deviation (m)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
