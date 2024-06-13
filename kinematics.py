import numpy as np


def forward_kinematics(thetas: [float], links) -> [float]:
    """

    :param thetas:
    :param links:
    :return:
    """
    state = []

    r1 = [[np.cos(thetas[1]), - np.sin(thetas[1]), 0],
          [np.sin(thetas[1]), np.cos(thetas[1]), 0],
          [0, 0, 1]]

    r2 = [[np.cos(thetas[2]), - np.sin(thetas[2]), 0],
          [np.sin(thetas[2]), np.cos(thetas[2]), 0],
          [0, 0, 1]]

    for i in thetas:
        state[i] = np.matmul(r1, r2)

def inverse_kinematics(thetas: [float]) -> None:
