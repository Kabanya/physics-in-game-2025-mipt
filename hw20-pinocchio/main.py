import time
import numpy as np
import pinocchio as pin
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer

from BothLegs import LeftLeg, RightLeg
from Plotter import Plotter
from Talos import Talos
from FootSteps import *
from Factor import *
from CoM import *
from Zmp import *

ankle_to_sole_height= 0.08 
foot_lift_height=0.1
robot_base_height=0.862+ankle_to_sole_height
com_height=0.9

footsteps = FootSteps([.0, -.1], [.0, .1])
footsteps.add_phase(.3, 'none')
footsteps.add_phase(.7, 'left', [.1, .1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'right', [.2, -.1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'left', [.3, .1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'right', [.4, -.1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'left', [.5, .1])
footsteps.add_phase(.1, 'none')
footsteps.add_phase(.7, 'right', [.5, -.1])
footsteps.add_phase(.5, 'none')

print("Started computing ...")
zmptraj = ZmpClass(footsteps)
com_traj = CoMClass(zmptraj)
print("CoM trajectory computed")

left_ank = LeftLeg(footsteps)
right_ank = RightLeg(footsteps)

robot = example_robot_data.load("talos")
model = robot.model
data = robot.data

robot.setVisualizer(MeshcatVisualizer())
robot.initViewer(open=True)
robot.loadViewerModel("pinocchio")

robot.display(robot.q0)

left_foot_id = model.getFrameId("leg_left_sole_fix_joint")
right_foot_id = model.getFrameId("leg_right_sole_fix_joint")

print(f"Left foot frame ID: {left_foot_id}")
print(f"Right foot frame ID: {right_foot_id}")

pin.framesForwardKinematics(model, data, robot.q0)
initial_left_height = data.oMf[left_foot_id].translation[2]
initial_right_height = data.oMf[right_foot_id].translation[2]
base_foot_height = (initial_left_height + initial_right_height) / 2.0

print(f"Initial left foot height: {initial_left_height:.3f}")
print(f"Initial right foot height: {initial_right_height:.3f}")
print(f"Using base foot height: {base_foot_height:.3f}")

t_total = footsteps.timetime[-1]
dt = 0.05
N = int(t_total / dt)

print(f"Starting animation for {t_total:.2f} seconds...")

prev_com_z = robot_base_height

animator = Talos(
    robot, model, data, left_foot_id, right_foot_id,
    left_ank, right_ank, com_traj, footsteps,
    robot_base_height, base_foot_height, dt
)
animator.animate(t_total)

print("Animation completed")

Plotter.plot_all(com_traj, left_ank, right_ank, t_total, footsteps)

print("Plots generated")