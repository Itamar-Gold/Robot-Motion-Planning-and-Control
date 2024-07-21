from queue import PriorityQueue
import itertools


# # Example usage
# import numpy as np
# x_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)
# y_lim = np.linspace(0, 4, 32, endpoint=False, retstep=True)
# z_lim = np.linspace(-4, 4, 64, endpoint=False, retstep=True)
#
# print(f'x step = {x_lim[-1]}, y step = {y_lim[-1]}, z step = {z_lim[-1]}')
# start = (2, 2, 0)
# end = (-2, 0, 2)
# cylinder_center = (0, 0, 0)  # Center of the cylinder (x, y, z)
# cylinder_radius = 0.25  # Radius of the cylinder in the xz-plane
# cylinder_height = 4  # Height of the cylinder along the y-axis
#
# path = a_star_algorithm(start, end, x_lim, y_lim, z_lim, x_lim[-1], y_lim[-1], z_lim[-1],
#                         cylinder_center, cylinder_radius, cylinder_height)
# if path:
#     print("Path found:", path)
#     steps = list(enumerate(path))
#     print(f'Number of path steps = ', steps[-1][0])
# else:
#     print("No path found")


# Heuristic function for A* (Manhattan distance)
def heuristic(a, b) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


# Check if an  end-effector point is in collision with cylindrical region of the base link
def is_in_forbidden_region(point, cylinder_center, cylinder_radius, cylinder_height) -> bool:
    x, y, z = point
    cx, cy, cz = cylinder_center

    # Check if the point is within the cylinder's height
    if abs(y - cy) > cylinder_height:
        return False

    # Check if the point is within the cylinder's radius (in xz-plane)
    if (x - cx) ** 2 + (z - cz) ** 2 <= cylinder_radius ** 2:
        return True

    return False


# A* Pathfinding algorithm
def a_star_algorithm(start, end, x_lim, y_lim, z_lim, x_step_size, y_step_size, z_step_size,
                     cylinder_center, cylinder_radius, cylinder_height):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    i = 0

    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == end:
            return reconstruct_path(came_from, end)
        # List all combinations for steps
        # Assume that steps in every direction is equal (x = y = z = number)
        possible_values = [0, x_step_size, -x_step_size, y_step_size, -y_step_size, z_step_size, -z_step_size]
        # create a list of all possible steps
        directions = list(itertools.product(possible_values, repeat=3))

        # print(f'Number of combinations: ', len(directions))

        # Remove (0, 0, 0) from the list
        directions = [d for d in directions if d != (0, 0, 0)]

        for direction in directions:
            # Count computational steps
            i += 1
            neighbor = (current[0] + direction[0], current[1] + direction[1], current[2] + direction[2])

            # Check if target reached
            if (abs(neighbor[0]) < abs(x_lim[0].size) and
                    abs(neighbor[1]) < abs(y_lim[0].size) and
                    abs(neighbor[2]) < abs(z_lim[0].size) and
                    not is_in_forbidden_region(neighbor, cylinder_center, cylinder_radius, cylinder_height)):

                temp_g_score = g_score.get(current, float('inf')) + 1

                if temp_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, end)

                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

            if i == 100000000:  # If number of computational steps exceed stop the search
                return None

    return None


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]
