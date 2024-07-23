import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Astar_algorithm import a_star_algorithm
import kinematics_n_dynamics as knd
import control_simulation as cs


class Link:
    def __init__(self, link_length, pos1, pos2, color):
        self.link_len = link_length
        self.link_color = color
        self.joint1 = pos1  # expected [x, y, z]
        self.joint2 = pos2
        # self.pause = 0.05
        # self.fps = 30

    def draw(self, link_number, plot_ax):

        link_num = f'line{link_number},'
        (link_num) = plot_ax.plot([self.joint1[0], self.joint2[0]],
                                  [self.joint1[1], self.joint2[1]],
                                  [self.joint1[2], self.joint2[2]],
                                  color=self.link_color,
                                  linewidth=2,)


# System parameters
links = [1.1, 1.6, 0.8]  # Link lengths
masses = [1, 1.4, 0.9]  # Link masses
inertias = [0.1, 0.2, 0.1]  # Link inertias
initial_pos = (2, 1, 0)  # initial position of the end-effector
target = (-1, 0, 1)  # set a target
x_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)
y_lim = np.linspace(0, 4, 32, endpoint=False, retstep=True)
z_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)
cylinder_center = (0, 0, 0)  # Center of the cylinder (x, y, z)
cylinder_radius = 0.5  # Radius of the cylinder in the xz-plane
cylinder_height = 4  # Height of the cylinder along the y-axis
path = a_star_algorithm(initial_pos, target, x_lim, y_lim, z_lim, x_lim[-1], y_lim[-1], z_lim[-1],
                        cylinder_center, cylinder_radius, cylinder_height)

ic_angles = knd.inverse_kinematics(path[0], links)
d_angles = knd.inverse_kinematics(path[-1], links)

# Simulation parameters
t_span = (0, 10)  # 10 seconds simulation
y0 = [ic_angles[1][0], ic_angles[1][1], ic_angles[1][2], 0, 0, 0]  # Initial conditions: [theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot]
theta_desired = [d_angles[1][0], d_angles[1][1], d_angles[1][2]]  # Desired joint angles

# Control gains
Kp = np.diag([150, 200, 170])
Kd = np.diag([150, 150, 170])


# Run simulation
t = np.linspace(t_span[0], t_span[1], 1000, endpoint=False)
sol = odeint(cs.simulate_system, y0, t, args=(links, masses, inertias, Kp, Kd, theta_desired))

j1_coord = []
j2_coord = []
j3_coord = []

for j in range(len(sol[:, 0])):
    theta_sol = sol[j, 0], sol[j, 1], sol[j, 2]
    coord = knd.forward_kinematics(theta_sol, links)
    j1_coord.append(coord[0])
    j2_coord.append(coord[1])
    j3_coord.append(coord[2])

if j1_coord and j2_coord and j3_coord is not None:
    j1_coord = np.squeeze(j1_coord)
    j2_coord = np.squeeze(j2_coord)
    j3_coord = np.squeeze(j3_coord)

    for tm, pos1, pos2, pos3 in zip(t, j1_coord, j2_coord, j3_coord):

        fig = plt.figure(figsize=plt.figaspect(0.6))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_title('Arm simulation')

        plt.plot(initial_pos[0], initial_pos[1], initial_pos[2], 'ok', linewidth=4)  # plot the target
        plt.plot(target[0], target[1], target[2], 'xm', linewidth=4)  # plot the target

        # plot the arm
        robot_base = Link(links[0], [0, 0, 0], pos1, 'blue')
        robot_base.draw(1, ax)

        # link1 = Link(links[1], position1[0], position1[1], 'red')
        # link1.draw(2, ax)
        #
        # link2 = Link(links[1], position1[1], position1[2], 'red')
        # link2.draw(3, ax)

        link3 = Link(links[1], pos1, pos2, 'orange')
        link3.draw(4, ax)

        link4 = Link(links[1], pos2, pos3, 'green')
        link4.draw(5, ax)

        # set environment limit
        ax.set_xlim([-4, 4])
        ax.set_ylim([0, 4])
        ax.set_zlim([-4, 4])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        fig.suptitle(f'Arm simulation \n Time = {tm:.2f}', y=0.15)  # Adjust the y value to position the title

        ax.view_init(azim=90, elev=-57, roll=0)
        plt.savefig(f'images/time_seq/Time_{tm}.png')
        plt.close()

else:
    print('target is out of reach')
