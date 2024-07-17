from kinematics import inverse_kinematics as ik
from kinematics import forward_kinematics as fk


pos = [0, 1.01, 0.96]
links = [1.1, 1.6, 1.8]
angles = [-25, 34.28, -20.45]

position = fk(angles, links)
angles = ik(position[-1], links)
print(position)
print(angles)
