class Joint:
    def __init__(self, initial_rotational_angle: float, initial_angle: float,
                 first_joint_position: list[float]):
        self.rot_pos = initial_rotational_angle
        self.angle_pos = initial_angle
        self.pos = first_joint_position

    def roll_rotate(self, angle):
        self.rot_pos += angle

    def yaw_rotate(self, angle):
        self.angle_pos += angle

    def update_position(self, pos):  # update the joint position
        self.pos = pos

    def __repr__(self):  # generate a report of the current state
        return self.pos, self.rot_pos, self.angle_pos


class Link(Joint):
    def __init__(self, link_length: float, initial_rotational_angle: float,
                 initial_angle: float, first_joint_position: list[float], attach_to_joint=None):
        # inherit Joint class variables submission
        super().__init__(initial_rotational_angle, initial_angle, first_joint_position)
        self.link_len = link_length
        if attach_to_joint is None:
            self.joint = []
            print('No Joint attached to Link')
        else:
            self.joint = attach_to_joint

    def update(self, position, rotation, angle):
        self.pos = position
        self.rot_pos = rotation
        self.angle_pos = angle

    def __repr__(self):
        return self.link_len, self.pos, self.rot_pos, self.angle_pos


base_joint = Joint(20, 45, [0, 0, 0])  # using joint
state = base_joint.__repr__()  # report the current joint state
print("joint state: ")
print(state)


base_link = Link(10, base_joint.rot_pos, base_joint.angle_pos, base_joint.pos, base_joint)  # using a link
link_state = base_link.__repr__()  # report the current link state
print("link state: ")
print(link_state)

# rotate the arm
# base_joint.rot_pos = 30


base_joint.roll_rotate(10)  # move 'x' angle in the roll angle state
base_joint.yaw_rotate(15)  # move 'x' angle in the yaw angle state

# base_link.rot_pos = base_joint.rot_pos
# base_link.angle_pos = base_joint.angle_pos

print("new joint state: ")
print(base_joint.__repr__())

update_state = base_joint.__repr__()  # update the link position with the current joint state

base_link.update(update_state[0], update_state[1], update_state[2])
print("new link state: ")
print(base_link.__repr__())
