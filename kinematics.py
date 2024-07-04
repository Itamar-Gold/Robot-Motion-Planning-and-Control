import numpy as np
import sympy as sp
import math


def forward_kinematics(thetas: [float], links: [float]) -> [float]:
    """
    :param thetas:
    :param links:
    :return:
    """

    # Kinematics Matrixes.

    r01 = np.array([
        [np.cos(thetas[1]), - np.sin(thetas[1]), 0, 0],
        [np.sin(thetas[1]), np.cos(thetas[1]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r12 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[2]), - np.sin(thetas[2]), 0],
        [0, np.sin(thetas[2]), np.cos(thetas[2]), links[1]],
        [0, 0, 0, 1]
    ])

    r23 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[3]), -np.sin(thetas[3]), links[2]],
        [0, np.sin(thetas[3]), np.cos(thetas[3]), 0],
        [0, 0, 0, 1]
    ])

    r34 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, links[3]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Matrixes for Mass Center calculation

    r12mass = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[2]), - np.sin(thetas[2]), 0],
        [0, np.sin(thetas[2]), np.cos(thetas[2]), 0.5 * links[1]],
        [0, 0, 0, 1]
    ])

    r23mass = np.array([
        [1, 0, 0, 0],
        [0, np.cos(thetas[3]), -np.sin(thetas[3]), 0.5 * links[2]],
        [0, np.sin(thetas[3]), np.cos(thetas[3]), 0],
        [0, 0, 0, 1]
    ])

    r34mass = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.5 * links[3]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Calculated Matrixes

    r02 = np.dot(r01, r12)
    r03 = np.dot(r02, r23)
    r04 = np.dot(r03, r34)

    ## Jacobian Matrix

    # Prismatic Vector
    v1 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]
                   ])

    # Row 1-3, Column 1
    j1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    j1 = np.dot(j1, v1)

    # Row 1-3, Column 2
    j2a = r01[0:3, 3:]
    j2b2 = j2a
    j2a = np.dot(j2a, v1)

    j2b1 = r03[0:3, 3:]

    j2b = j2b1 - j2b2

    j2 = np.array([
        [(j2a[1, 0] * j2b[2, 0]) - (j2a[2, 0] * j2b[1, 0])],
        [(j2a[2, 0] * j2b[0, 0]) - (j2a[0, 0] * j2b[2, 0])],
        [(j2a[0, 0] * j2b[1, 0]) - (j2a[1, 0] * j2b[0, 0])]
    ])

    # Row 1-3 Column 3
    j3a = r02[0:3, 3:]
    j3b2 = j3a
    j3a = np.dot(j3a, v1)

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

    # Creating the Jacobian Matrix
    jm1 = np.concatenate((j1, j2, j3), 1)
    jm2 = np.concatenate((j4, j5, j6), 1)
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
