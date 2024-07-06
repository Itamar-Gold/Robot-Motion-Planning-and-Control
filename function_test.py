from kinematics import inverse_kinematics as ik
from kinematics import forward_kinematics as fk


pos = [0, 1.01, 0.96]
links = [1.1, 1.6, 1.8]
angles = [0, 34.28, 116.02]

# test
# ----- angles ------------- position --------
# [0, 34.04, -66.65] <-> [0, 0.99, 3.76]
# [0, 34.28, 116.02] <-> [0, 1.58, 0.22]
# [0, 75.68, 145.77] <-> [0, 1.01, 0.96]

pos = fk(angles, links)
angles = ik(pos, links)
print(pos)
