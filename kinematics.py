import numpy as np
import sympy as sp


def forward_kinematics(thetas: [float], links: [float]) -> [float]:
    """
    :param thetas:
    :param links:
    :return:
    """

    # Kinematics Matrixes.

    r01 = np.array([
        [np.cos(thetas[0]), - np.sin(thetas[0]), 0, 0],
        [np.sin(thetas[0]), np.cos(thetas[0]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r12 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[1]), - np.sin(thetas[1]), 0],
        [0, np.sin(thetas[1]), np.cos(thetas[1]), links[0]],
        [0, 0, 0, 1]
    ])

    r23 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[2]), -np.sin(thetas[2]), links[1]],
        [0, np.sin(thetas[2]), np.cos(thetas[2]), 0],
        [0, 0, 0, 1]
    ])

    r34 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, links[2]],
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

    r02 = np.dot(r01, r12)
    r03 = np.dot(r02, r23)
    r04 = np.dot(r03, r34)

    ## Jacobian Matrix

    # Prismatic Vector
    v1 = np.array([0, 0, 1])
    print(f'v1: ', v1)

    # Row 1-3, Column 1
    j1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    j1 = np.dot(v1, j1)

    # Row 1-3, Column 2
    j2a = r01[0:3, 3:]
    print(f'j2a: ', j2a)
    j2b2 = j2a
    j2a = np.dot(v1, j2a)
    print(f'j2a: ', j2a)

    j2b1 = r03[0:3, 3:]

    j2b = j2b1 - j2b2
    print(f'j2b: ', j2b)

    j2 = np.array([
        [(j2a[1] * j2b[2]) - (j2a[2] * j2b[1])],
        [(j2a[2] * j2b[0]) - (j2a[0] * j2b[2])],
        [(j2a[0] * j2b[1]) - (j2a[1] * j2b[0])]
    ])

    # Row 1-3 Column 3
    j3a = r02[0:3, 3:]
    j3b2 = j3a
    j3a = np.dot(v1, j3a)

    j3b1 = r03[0:3, 3:]

    j3b = j3b1 - j3b2

    j3 = np.array([
        [(j3a[1, 0] * j3b[2, 0]) - (j3a[2, 0] * j3b[1, 0])],
        [(j3a[2, 0] * j3b[0, 0]) - (j3a[0, 0] * j3b[2, 0])],
        [(j3a[0, 0] * j3b[1, 0]) - (j3a[1, 0] * j3b[0, 0])]
    ])

    # Rotation / Orientation Vector

    # Row 4-6 Column 1
    j4 = np.array([
        [0],
        [0],
        [0]
    ])

    # Row 4-6, Column 2
    j5 = j2a

    # Row 4-6, Column 3
    j6 = j3a

    print(f'j1: ', j1)
    print(f'j2: ', j2)
    print(f'j3: ', j3)

    # Creating the Jacobian Matrix
    jm1 = np.concatenate((j1, j2, j3), 1)
    jm2 = np.concatenate((j4, j5, j6), 1)
    # debugging line 138
    print(jm1.shape, jm2.shape)
    j = np.concatenate((jm1, jm2), 0)

    # Diff Eq.
    xp, yp, zp = sp.symbols('x* y* z*')
    wx, wy, wz = sp.symbols('wx wy wz')
    the1p, the2p, the3p = sp.symbols('theta1* theta2* theta3*')

    q = np.array([[the1p], [the2p], [the3p]])
    e = np.dot(j, q)

    xp = e[0, 0]
    yp = e[1, 0]
    zp = e[2, 0]
    wx = e[3, 0]
    wy = e[4, 0]
    wz = e[5, 0]

    return [j, e, xp, yp, zp, wx, wy, wz]

    ## WE NEED TO ADD A FEATCHER TO DESIDE THE DIRECTION OF THE TOOL TO THE END POINT


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
