import matplotlib.pyplot as plt
from kinematics import inverse_kinematics as ik
from kinematics import forward_kinematics as fk


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


fig = plt.figure(figsize=plt.figaspect(0.6))
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Arm simulation')

target = [-2, 1.5, 2]  # set a target
plt.plot(target[0], target[1], target[2], 'xm', linewidth=4) # plot the target
# define the links lengths
links = [1.1, 1.6, 1.8]
# get desired angles with inverse kinematics
angles1, angles2 = ik(target, links)
# control the arm with desired angles
position1 = fk(angles1, links)
position2 = fk(angles2, links)

# plot the arm
robot_base = Link(links[0], [0, 0, 0], position1[0], 'blue')
robot_base.draw(1, ax)

link1 = Link(links[1], position1[0], position1[1], 'red')
link1.draw(2, ax)

link2 = Link(links[1], position1[1], position1[2], 'red')
link2.draw(3, ax)

link3 = Link(links[1], position2[0], position2[1], 'green')
link3.draw(4, ax)

link4 = Link(links[1], position2[1], position2[2], 'green')
link4.draw(5, ax)

# set environment limit
ax.set_xlim([-4, 4])
ax.set_ylim([0, 3])
ax.set_zlim([-4, 4])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

time = 0.2
fig.suptitle(f'Arm simulation \n Time = {time:.2f}', y=0.15)  # Adjust the y value to position the title

ax.view_init(azim=90, elev=-57, roll=0)
plt.show()
