import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kinematics import inverse_kinematics as ik
from kinematics import forward_kinematics as fk
from Astar_algorithm import a_star_algorithm

export_to_excel = True


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


x_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)
y_lim = np.linspace(0, 4, 32, endpoint=False, retstep=True)
z_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)

# define the links lengths
links = [1.1, 1.6, 1.8]
initial_pos = (2, 2, 0)  # initial position of the end-effector
target = (-2.5, 0, -2)  # set a target
cylinder_center = (0, 0, 0)  # Center of the cylinder (x, y, z)
cylinder_radius = 0.5  # Radius of the cylinder in the xz-plane
cylinder_height = 4  # Height of the cylinder along the y-axis
current_pos = initial_pos
counter = 0
prev = 0
data = np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0, 0, 0])
# get an optimal path to target
path = a_star_algorithm(initial_pos, target, x_lim, y_lim, z_lim, x_lim[-1], y_lim[-1], z_lim[-1],
                        cylinder_center, cylinder_radius, cylinder_height)
if path is not None:
    for pos in path:

        fig = plt.figure(figsize=plt.figaspect(0.6))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_title('Arm simulation')

        plt.plot(initial_pos[0], initial_pos[1], initial_pos[2], 'ok', linewidth=4)  # plot the target
        plt.plot(target[0], target[1], target[2], 'xm', linewidth=4)  # plot the target

        # get desired angles with inverse kinematics
        angles1, angles2 = ik(pos, links)

        # fix theta 1 with 360 degrees rotation to keep angular transition smooth
        if 350 <= abs(angles2[0] - prev) <= 365:
            angles2[0] = 360 + angles2[0]

        prev = angles2[0]

        # control the arm with desired angles
        position1 = fk(angles1, links)
        position2 = fk(angles2, links)
        print(f'the angles are: {angles2[0]:.1f}, {angles2[1]:.1f}, {angles2[2]:.1f}')
        # Collect data for record
        data = np.vstack((data,
                         np.array([position2[2][0], position2[2][1], position2[2][2],
                                   angles2[0], angles2[1], angles2[2]])))

        # plot the arm
        robot_base = Link(links[0], [0, 0, 0], position1[0], 'blue')
        robot_base.draw(1, ax)

        # link1 = Link(links[1], position1[0], position1[1], 'red')
        # link1.draw(2, ax)
        #
        # link2 = Link(links[1], position1[1], position1[2], 'red')
        # link2.draw(3, ax)

        link3 = Link(links[1], position2[0], position2[1], 'orange')
        link3.draw(4, ax)

        link4 = Link(links[1], position2[1], position2[2], 'green')
        link4.draw(5, ax)

        # set environment limit
        ax.set_xlim([-4, 4])
        ax.set_ylim([0, 4])
        ax.set_zlim([-4, 4])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        counter += 1

        fig.suptitle(f'Arm simulation \n Steps = {counter}', y=0.15)  # Adjust the y value to position the title

        ax.view_init(azim=90, elev=-57, roll=0)
        plt.savefig(f'images/Steps_{counter}.png')
        plt.close()
    df = pd.DataFrame(data, columns=['X', 'Y', 'Z', 'Theta_1', 'Theta_2', 'Theta_3'])

    if export_to_excel:
        with pd.ExcelWriter("Simulation.xlsx") as writer:
            df.to_excel(writer)

else:
    print('target is out of reach')
