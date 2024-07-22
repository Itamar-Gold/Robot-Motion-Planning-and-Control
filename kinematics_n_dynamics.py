import numpy as np


def forward_kinematics(thetas: [float], links: [float]) -> [float]:
    """
    :param thetas:
    :param links:
    :return:
    """

    # Kinematics Matrices.

    r01 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, links[0]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r12 = np.array([
        [np.cos(thetas[0]), 0, np.sin(thetas[0]), 0],
        [0, 1, 0, 0],
        [-np.sin(thetas[0]), 0, np.cos(thetas[0]), 0],
        [0, 0, 0, 1]
    ])

    r23 = np.array([
        [np.cos(thetas[1]), -np.sin(thetas[1]), 0, links[1]*np.cos(thetas[1])],
        [np.sin(thetas[1]), np.cos(thetas[1]), 0, links[1]*np.sin(thetas[1])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r34 = np.array([
        [np.cos(thetas[2]), -np.sin(thetas[2]), 0, links[2]*np.cos(thetas[2])],
        [np.sin(thetas[2]), np.cos(thetas[2]), 0, links[2]*np.sin(thetas[2])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Calculated Matrixes

    r02 = np.matmul(r01, r12)
    r03 = np.matmul(r02, r23)
    r04 = np.matmul(r03, r34)

    # Position calculation
    p_1 = np.squeeze(r02[0:3, 3:])
    p_2 = np.squeeze(r03[0:3, 3:])
    p_3 = np.squeeze(r04[0:3, 3:])

    return p_1, p_2, p_3


def inverse_kinematics(position: [float], links: [float]) -> [float]:
    """
    this function was designed based on the exercise in page 5 lecture 5 of kinematics and dynamics of robotics arm

    :param position: the [x, y, z] position of the end-effector
    :param links: the [L1, L2, L3] lengths of the links [links[0], links[1], links[2]]
    :return: the [q1, q2, q3] angles of the arm
    """
    pos = position

    x_til = np.sqrt(pos[2] ** 2 + pos[0] ** 2)
    y_til = pos[1] - links[0]

    c = x_til ** 2 + y_til ** 2
    d = (c - (links[1] ** 2) - (links[2] ** 2)) / (2 * links[1] * links[2])

    q3_1 = np.arctan2(np.sqrt(1 - d ** 2), d)
    q3_2 = np.arctan2(-np.sqrt(1 - d ** 2), d)

    q1 = np.arctan2(pos[0] / x_til, pos[2] / x_til) - np.pi / 2

    alpha_1 = np.arctan2(links[2] * np.sin(q3_1) / c, (links[1] + (links[2] * np.cos(q3_1))) / c)
    alpha_2 = np.arctan2(links[2] * np.sin(q3_2) / c, (links[1] + (links[2] * np.cos(q3_2))) / c)

    q2_1 = np.arctan2(y_til / np.sqrt(c), x_til / np.sqrt(c)) - alpha_1
    q2_2 = np.arctan2(y_til / np.sqrt(c), x_til / np.sqrt(c)) - alpha_2

    angles_1 = [q1, q2_1, q3_1]
    angles_2 = [q1, q2_2, q3_2]

    return angles_1, angles_2


def mass_matrix(thetas, links, masses, inertias):
    theta1, theta2, theta3 = thetas
    l1, l2, l3 = links
    m1, m2, m3 = masses
    i1, i2, i3 = inertias

    cos2 = np.cos(theta2)
    cos23 = np.cos(theta2 + theta3)

    M11 = i1 + m3 * (l2 * cos2 + l3 / 2 * cos23)**2
    M22 = i2 + m2 * l2**2 / 4 + m3 * (2 * l2**2 + l3**2 / 2 + 2 * cos23 * l2 * l3)
    M23 = M32 = m3 / 2 * (l3**2 + l2 * l3 * cos2)
    M33 = i3 + m3 * l3**2 / 4

    M = np.array([
        [M11, 0, 0],
        [0, M22, M23],
        [0, M32, M33]
    ])

    return M


def coriolis_forces(thetas, thetas_dot, links, masses):
    theta1, theta2, theta3 = thetas
    theta1_dot, theta2_dot, theta3_dot = thetas_dot
    L1, L2, L3 = links
    m1, m2, m3 = masses

    cos2 = np.cos(theta2)
    cos23 = np.cos(theta2 + theta3)
    sin2 = np.sin(theta2)
    sin23 = np.sin(theta2 + theta3)

    C1 = 2 * m3 * (L2**2 * cos2 * sin2 + L3 / 2 * cos23 * L2 * sin2) * theta1_dot * theta2_dot
    C1 += 2 * m3 * (L2 * L3 / 2 * cos2 + L3**2 / 4 * cos23 * sin23) * theta1_dot * (theta2_dot + theta3_dot)

    C2 = m3 / 2 * L2 * L3 * (2 * theta2_dot * theta3_dot - theta3_dot**2 * sin23)

    C = np.array([C1, C2, 0])

    return C


def gravitational_forces(thetas, links, masses, g=9.81):
    theta1, theta2, theta3 = thetas
    L1, L2, L3 = links
    m1, m2, m3 = masses

    cos2 = np.cos(theta2)
    cos23 = np.cos(theta2 + theta3)

    G1 = 0
    G2 = cos2 * (L2 / 2 * m2 * g + m3 * g * L2) + m3 * g * L3 / 2 * cos23
    G3 = m3 * g * L3 / 2 * cos23

    G = np.array([G1, G2, G3])

    return G


def dynamics(thetas, thetas_dot, thetas_ddot, links, masses, inertias):
    M = mass_matrix(thetas, links, masses, inertias)
    C = coriolis_forces(thetas, thetas_dot, links, masses)
    G = gravitational_forces(thetas, links, masses)

    tau = np.dot(M, thetas_ddot) + C + G
    return tau
