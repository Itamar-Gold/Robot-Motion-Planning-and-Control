import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Astar_algorithm import a_star_algorithm
import kinematics_n_dynamics as knd

def pd_control(theta_desired, theta_current, theta_dot_current, Kp, Kd, links, masses):
    # Calculate the error
    error = theta_desired - theta_current

    # Calculate the derivative of the error (assuming desired velocity is 0)
    error_dot = -theta_dot_current

    # Calculate the gravity compensation term
    G = knd.gravitational_forces(theta_current, links, masses)

    # Calculate the control input
    tau = np.dot(Kp, error) + np.dot(Kd, error_dot) + G

    return tau


def simulate_system(y, t, links, masses, inertias, Kp, Kd, theta_desired):
    # Unpack the state variables
    theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot = y

    theta = np.array([theta1, theta2, theta3])
    theta_dot = np.array([theta1_dot, theta2_dot, theta3_dot])

    # Calculate control input
    tau = pd_control(theta_desired, theta, theta_dot, Kp, Kd, links, masses)

    # Calculate acceleration using inverse dynamics
    M = knd.mass_matrix(theta, links, masses, inertias)
    C = knd.coriolis_forces(theta, theta_dot, links, masses)
    G = knd.gravitational_forces(theta, links, masses)

    theta_ddot = np.linalg.solve(M, tau - C - G)

    return [theta1_dot, theta2_dot, theta3_dot, theta_ddot[0], theta_ddot[1], theta_ddot[2]]


# System parameters
links = [1.1, 1.6, 0.8]  # Link lengths
masses = [1, 1.4, 0.9]  # Link masses
inertias = [0.1, 0.2, 0.1]  # Link inertias
initial_pos = (2, 1, 0)  # initial position of the end-effector
target = (-1.5, 0, -1)  # set a target
x_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)
y_lim = np.linspace(0, 4, 32, endpoint=False, retstep=True)
z_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)
cylinder_center = (0, 0, 0)  # Center of the cylinder (x, y, z)
cylinder_radius = 0.5  # Radius of the cylinder in the xz-plane
cylinder_height = 4  # Height of the cylinder along the y-axis
path = a_star_algorithm(initial_pos, target, x_lim, y_lim, z_lim, x_lim[-1], y_lim[-1], z_lim[-1],
                        cylinder_center, cylinder_radius, cylinder_height)

ic_angles = knd.inverse_kinematics(path[0], links)
d_angles = knd.inverse_kinematics(path[-1], links)

# Simulation parameters
t_span = (0, 10)  # 10 seconds simulation
y0 = [ic_angles[1][0], ic_angles[1][1], ic_angles[1][2], 0, 0, 0]  # Initial conditions: [theta1, theta2, theta3, theta1_dot, theta2_dot, theta3_dot]
theta_desired = [d_angles[1][0], d_angles[1][1], d_angles[1][2]]  # Desired joint angles

# Control gains
Kp = np.diag([150, 200, 170])
Kd = np.diag([150, 150, 170])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Run simulation
t = np.linspace(t_span[0], t_span[1], 1000, endpoint=False)
sol = odeint(simulate_system, y0, t, args=(links, masses, inertias, Kp, Kd, theta_desired))


# Plot joint angles in the first subplot
for i in range(3):
    ax1.plot(t, sol[:, i], label=f'Joint {i+1}')
ax1.plot(t, [theta_desired[0]]*len(t), 'k--', label='Desired 1')
ax1.plot(t, [theta_desired[1]]*len(t), 'k--', label='Desired 2')
ax1.plot(t, [theta_desired[2]]*len(t), 'k--', label='Desired 3')
ax1.set_ylabel('Joint Angle (rad)')
ax1.legend()
ax1.grid(True)
ax1.set_title('Joint Angles')

# Plot joint angle speeds in the second subplot
for i in range(3):
    ax2.plot(t, sol[:, i+3], label=f'Joint {i+1}')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Joint Angle Speed (rad/sec)')
ax2.legend()
ax2.grid(True)
ax2.set_title('Joint Angle Speeds')

# Adjust layout to prevent overlap
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.3)  # Adjust vertical space between subplots

# Show the plot
plt.show()
plt.savefig(f'images/control_res.png')
x_y_z = []

for j in range(len(sol[:, 0])):
    theta_sol = sol[j, 0], sol[j, 1], sol[j, 2]
    coord = knd.forward_kinematics(theta_sol, links)
    x_y_z.append(coord[2])
print(x_y_z)
