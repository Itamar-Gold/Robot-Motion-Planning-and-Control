import numpy as np
import sympy as sp

def forward_kinematics(thetas: [float], links: [float], mass: [float], thetad: [float], thetadd: [float]) -> [float]:
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

    # Calculated Matrixes

    r02 = np.matmul(r01, r12)
    r03 = np.matmul(r02, r23)
    r04 = np.matmul(r03, r34)

    # Position calculation
    p = r04[0:3, 3:]

    # Jacobian Matrix
    r01diff = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r12diff = np.array([
        [-np.sin(thetas[0]), 0, np.cos(thetas[0]), 0],
        [0, 1, 0, 0],
        [-np.cos(thetas[0]), 0, -np.sin(thetas[0]), 0],
        [0, 0, 0, 1]
    ])

    r23diff = np.array([
        [-np.sin(thetas[1]), -np.cos(thetas[1]), 0, links[1] * -np.sin(thetas[1])],
        [np.cos(thetas[1]), -np.sin(thetas[1]), 0, links[1] * np.cos(thetas[1])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r23diff3 = np.array([
        [np.cos(thetas[1]), -np.sin(thetas[1]), 0, 0],
        [np.sin(thetas[1]), np.cos(thetas[1]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r34diff = np.array([
        [-np.sin(thetas[2]), -np.cos(thetas[2]), 0, links[2] * -np.sin(thetas[2])],
        [np.cos(thetas[2]), -np.sin(thetas[2]), 0, links[2] * np.cos(thetas[2])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r02diff1 = np.matmul(r01, r12diff)
    r03diff1 = np.matmul(r02diff1, r23)
    r04diff1 = np.matmul(r03diff1, r34)

    r02diff2 = np.matmul(r01diff, r12)
    r03diff2 = np.matmul(r02diff2, r23diff)
    r04diff2 = np.matmul(r03diff2, r34)

    r03diff3 = np.matmul(r02diff2, r23diff3)
    r04diff3 = np.matmul(r03diff3, r34diff)

    jp = np.array([
        [r04diff1[0, 3], r04diff2[0, 3], r04diff3[0, 3]],
        [r04diff1[1, 3], r04diff2[1, 3], r04diff3[1, 3]],
        [r04diff1[2, 3], r04diff2[2, 3], r04diff3[2, 3]]
    ])

    print(f'jp: ', jp)
    print(f'r04diff3: ', r04diff3)
    print(f'r34diff3: ', r34diff)
    print(f'r03: ', r03)

    # Center of mass calculations
    r01mass = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.5 * links[0]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r23mass = np.array([
        [np.cos(thetas[1]), -np.sin(thetas[1]), 0, 0.5 * links[1] * np.cos(thetas[1])],
        [np.sin(thetas[1]), np.cos(thetas[1]), 0, 0.5 * links[1] * np.sin(thetas[1])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r34mass = np.array([
        [np.cos(thetas[2]), -np.sin(thetas[2]), 0, 0.5 * links[2] * np.cos(thetas[2])],
        [np.sin(thetas[2]), np.cos(thetas[2]), 0, 0.5 * links[2] * np.sin(thetas[2])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Jv calculations
    r23massdiff = np.array([
        [-np.sin(thetas[1]), -np.cos(thetas[1]), 0, 0.5 * links[1] * -np.sin(thetas[1])],
        [np.cos(thetas[1]), -np.sin(thetas[1]), 0, 0.5 * links[1] * np.cos(thetas[1])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    r34massdiff = np.array([
        [-np.sin(thetas[2]), -np.cos(thetas[2]), 0, links[2] * -np.sin(thetas[2])],
        [np.cos(thetas[2]), -np.sin(thetas[2]), 0, links[2] * np.cos(thetas[2])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Jv1
    r02massdiff1 = np.matmul(r01mass, r12diff)

    jv1 = np.array([
        [r02massdiff1[0, 3], 0, 0],
        [r02massdiff1[1, 3], 0, 0],
        [r02massdiff1[2, 3], 0, 0]
    ])

    jv1trans = jv1.transpose()

    # JV2
    r02diff1 = np.matmul(r01, r12diff)
    r03diff1 = np.matmul(r02diff1, r23mass)

    r02diff2 = np.matmul(r01diff, r12)
    r03diff2 = np.matmul(r02diff2, r23massdiff)

    jv2 = np.array([
        [r03diff1[0, 3], r03diff2[0, 3], 0],
        [r03diff1[1, 3], r03diff2[1, 3], 0],
        [r03diff1[2, 3], r03diff2[2, 3], 0]
    ])

    jv2trans = jv2.transpose()

    # JV3
    r02diff1 = np.matmul(r01, r12diff)
    r03diff1 = np.matmul(r02diff1, r23)
    r04diff1 = np.matmul(r03diff1, r34mass)

    r02diff2 = np.matmul(r01diff, r12)
    r03diff2 = np.matmul(r02diff2, r23diff)
    r04diff2 = np.matmul(r03diff2, r34mass)

    r03diff3 = np.matmul(r02diff2, r23diff3)
    r04diff3 = np.matmul(r03diff3, r34massdiff)

    jv3 = np.array([
        [r04diff1[0, 3], r04diff2[0, 3], r04diff3[0, 3]],
        [r04diff1[1, 3], r04diff2[1, 3], r04diff3[1, 3]],
        [r04diff1[2, 3], r04diff2[2, 3], r04diff3[2, 3]]
    ])

    jv3trans = jv3.transpose()

    # JW
    jw1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])

    jw1trans = jw1.transpose()

    jw2 = np.array([
        [0, np.sin(thetas[0]), 0],
        [1, 0, 0],
        [0, np.cos(thetas[0]), 0]
    ])

    jw2trans = jw2.transpose()

    jw3 = np.array([
        [0, np.sin(thetas[0]), np.sin(thetas[0])],
        [1, 0, 0],
        [0, np.cos(thetas[0]), np.cos(thetas[0])]
    ])

    jw3trans = jw3.transpose()

    # Inertia model

    i = np.array([
        [(mass[0]*links[0]**2)/12, 0, 0],
        [0, (mass[1]*links[1]**2)/12, 0],
        [0, 0, (mass[2]*links[2]**2)/12]
    ])

    m1 = mass[0]*jv1trans*jv1 + mass[1]*jv2trans*jv2 + mass[2]*jv2trans*jv2
    m2 = jw1trans*i*jw1 + jw2trans*i*jw2 + jw3trans*i*jw3
    m = m1 + m2

    g1 = - jv1trans*9.81*mass[0]
    g2 = - jv2trans*9.81*mass[1]
    g3 = - jv3trans*9.81*mass[2]
    g = g1 + g2 + g3

    print(f'p: ', p)
    print(f'm: ', m)
    print(f'g: ', g)


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
