import numpy as np
import sympy as sp


def forward_kinematics(thetas: [float], links: [float], thetad: [float], thetadd: [float]) -> [float]:
    """
    :param thetas:
    :param links:
    :return:
    """

    # Converting deg to rad
    thetas = np.deg2rad(thetas)

    # Kinematics Matrixes.

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

    # Matrixes for Mass Center calculation

    r12mass = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[1]), - np.sin(thetas[1]), 0],
        [0, np.sin(thetas[1]), np.cos(thetas[1]), 0.5 * links[0]],
        [0, 0, 0, 1]
    ])

    r23mass = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[2]), -np.sin(thetas[2]), 0.5 * links[1]],
        [0, np.sin(thetas[2]), np.cos(thetas[2]), 0],
        [0, 0, 0, 1]
    ])

    r34mass = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.5 * links[2]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Calculated Matrixes

    r02 = np.matmul(r01, r12)
    r03 = np.matmul(r02, r23)
    r04 = np.matmul(r03, r34)

    # Position calculation
    p = r04[0:3, 3:]

    # Jacobian Matrix
    # Prismatic Vector
    v1 = np.array([0, 0, 1])

    # Row 1-3, Column 1
    j1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    j1 = np.matmul(v1, j1)
    j1 = np.array([
        [j1[0]],
        [j1[1]],
        [j1[2]]
    ])

    # Row 1-3, Column 2
    j2a = r01[0:3, 0:3]
    j2b2 = r01[0:3, 3:]
    j2a = np.matmul(v1, j2a)

    j2b1 = r03[0:3, 3:]

    j2b = j2b1 - j2b2

    j2 = np.array([
        (j2a[1] * j2b[2]) - (j2a[2] * j2b[1]),
        (j2a[2] * j2b[0]) - (j2a[0] * j2b[2]),
        (j2a[0] * j2b[1]) - (j2a[1] * j2b[0])
    ])

    # Row 1-3 Column 3
    j3a = r02[0:3, 0:3]
    j3b2 = r02[0:3, 3:]
    j3a = np.dot(v1, j3a)

    j3b1 = r03[0:3, 3:]

    j3b = j3b1 - j3b2

    j3 = np.array([
        (j3a[1] * j3b[2]) - (j3a[2] * j3b[1]),
        (j3a[2] * j3b[0]) - (j3a[0] * j3b[2]),
        (j3a[0] * j3b[1]) - (j3a[1] * j3b[0])
    ])

    # Rotation / Orientation Vector

    # Row 4-6 Column 1
    j4 = np.array([
        [0],
        [0],
        [0]
    ])

    # Row 4-6, Column 2
    j5 = np.array([
        [j2a[0]],
        [j2a[1]],
        [j2a[2]]
    ])

    # Row 4-6, Column 3
    j6 = np.array([
        [j3a[0]],
        [j3a[1]],
        [j3a[2]]
    ])

    # Creating the Jacobian Matrix
    jm1 = np.concatenate((j1, j2, j3), 1)
    jm2 = np.concatenate((j4, j5, j6), 1)
    j = np.concatenate((jm1, jm2), 0)
    jp = j[0:3, 0:3]

    pdot = np.matmul(jp, thetad)
    pdubeldot = pdot + np.matmul(jp, thetadd)

    # Debugging
    print(f'pos: ', p)
    print(f'pos speed: ', pdot)
    print(f'pos exaleration: ', pdubeldot)

    return p, pdot, pdubeldot


def inverse_kinematics(position: [float], links: [float]) -> [float]:
    """
    this function was designed based on the excersize in page 5 lecture 5 of kinematics and dynamics of robotics arm

    :param position: the [x, y, z] position of the end-effector
    :param links: the [L1, L2, L3] lengths of the links [links[0], links[1], links[2]]
    :return: the [q1, q2, q3] angles of the arm
    """
    x_y = np.power(position[0], 2) + np.power(position[1], 2)
    x_y_norm = np.sqrt(np.power(position[0], 2) + np.power(position[1], 2))

    d = (x_y - np.power(links[1], 2) - np.power(links[2], 2)) / (2 * links[1] * links[2])

    q1 = np.arctan2(position[1] / x_y_norm, position[0] / x_y_norm)
    q3 = np.arctan2((1 * np.sqrt(1 - np.power(d, 2))), d)
    alpha = np.arctan2((links[2] * np.sin(q3) / x_y), (links[1] + links[2] * np.cos(q3)) / x_y)
    q2 = q1 - alpha
    angles = [np.rad2deg(q1), np.rad2deg(q2), np.rad2deg(q3)]

    return angles
