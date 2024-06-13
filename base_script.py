import numpy as np
import matplotlib.pyplot as plt


class Link:
    def __init__(self, link_length, initial_rotational_angle,
                 initial_angle, first_joint_position):
        self.link_len = link_length
        self.link_color = 'red'
        self.theta = initial_rotational_angle
        self.alpha = initial_angle
        self.joint1 = first_joint_position  # expected [x, y, z]
        self.joint1_color = 'bo'
        link_vector = [first_joint_position[0], first_joint_position[1], first_joint_position[2] + self.link_len]
        z_rotation = [[np.cos(self.theta), -np.sin(self.theta), 0],
                      [np.sin(self.theta), np.cos(self.theta), 0],
                      [0, 0, 1]]
        body_rotation = [[np.cos(self.alpha), 0, np.sin(self.alpha)],
                         [0, 1, 0],
                         [-np.sin(self.alpha), 0, np.cos(self.alpha)]]
        self.joint2 = np.matmul(np.matmul(link_vector, body_rotation), z_rotation)
        self.joint2_color = 'bo'
        # self.pause = 0.05
        # self.fps = 30

    def draw(self, link_number, plot_ax):

        link_num = f'line{link_number},'
        (link_num) = plot_ax.plot([self.joint1[0], self.joint2[0]],
                                  [self.joint1[1], self.joint2[1]],
                                  [self.joint1[2], self.joint2[2]],
                                  color=self.link_color,
                                  linewidth=2,)
        # point = plot_ax.plot(self.joint2[0], self.joint2[1], self.joint2[2], self.joint2_color)


class Joint(Link):
    def __init__(self, link_length, initial_rotational_angle,
                 initial_angle, first_joint_position):
        # inherit variables from a specified link
        super().__init__(self, link_length, initial_rotational_angle, initial_angle,
                         first_joint_position)
        self.joint_position = first_joint_position  # [x, y, z]
        self.joint_color = 'co'  # 'bo'

    def draw_joint(self, plot_ax):
        point = plot_ax.plot(self.joint_position[0], self.joint_position[1], self.joint_position[2], self.joint_color)

    def rotate(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta

# def attach_motor(self, resulution: float, input_voltage: float, ):
#     self.deg_pulse = resulution


fig = plt.figure(figsize=plt.figaspect(0.6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Arm simulation')

robot_base = Link(2, 0, 0, [0, 0, 0])
robot_base.link_color = 'blue'
robot_base.draw(1, ax)
link1 = Link(2, 0, np.pi/5, robot_base.joint2)
link1.draw(2, ax)
# link2 = Link(2, 0, np.pi/10, link1.joint2)
# link2.joint2_color = 'mo'
# link2.draw(3, ax)


# joint1 = Joint(link1.link_len, 0, 0, link1.joint2)

ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([0, 5])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.view_init(azim=60, elev=30)
plt.show()
