import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, rotate
import numpy as np
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_safety_z(state, p_matrix):
    state = np.array(state)
    state_transpose = np.transpose(state, (0, 2, 1))  # Adjust shape for matrix multiplication
    safe_z = state_transpose @ p_matrix @ state
    return safe_z

def cal_lamda(tem, safe_z):
    """
    Calculate the lamda value based on the temperature, safety_evaluation
    :param tem: Temperature parameter
    :param state: State vector
    :return: Calculated lamda value
    """
    lamda_value = (np.exp(safe_z * tem) - 1)/ (np.exp(tem) - 1)
    return lamda_value

t_list = [2.85, 3.4, 3.84, 4.56, 5.108, 9.565, 25]
x_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

x_min = -0.9
x_max = 0.9
y_min = -0.85
y_max = 0.85

rect = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

P_matrix = np.array([[4.6074554, 1.49740096, 5.80266046, 0.99189224],
                     [1.49740096, 0.81703147, 2.61779592, 0.51179642],
                     [5.80266046, 2.61779592, 11.29182733, 1.87117709],
                     [0.99189224, 0.51179642, 1.87117709, 0.37041435]])

cP = P_matrix

tP = np.zeros((2, 2))

tP[0][0] = cP[0][0]
tP[1][1] = cP[2][2]
tP[0][1] = cP[0][2]
tP[1][0] = cP[0][2]

for t, x_goal in zip(t_list, x_list):

    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    # Create a grid of points
    X, Y = np.meshgrid(x_range, y_range)

    all_points = np.vstack([X.ravel(), Y.ravel()]).T
    all_points = np.insert(all_points, 1, 0, axis=-1)  # Add a column for the second dimension (x_dot)
    all_points = np.insert(all_points, 3, 0, axis=-1)  # Add a column for the last dimension (theta_dot)

    all_points = all_points[:,:, np.newaxis] # n_pts x 4 x 1

    all_safety_values = get_safety_z(state=all_points, p_matrix=P_matrix)

    safe_points_id = np.where(all_safety_values <= 1.0)[0]
    pts_in_envelope = all_points[safe_points_id].squeeze()
    values_in_envelope = all_safety_values[safe_points_id].squeeze()

    pts_lamda = cal_lamda(tem=t, safe_z=values_in_envelope)

    wp, vp = np.linalg.eig(tP)

    angle = math.atan2(vp[0][1], vp[0][0])
    angle = -1 * math.degrees(angle)

    # Create an ellipse:
    ellipse = Point(0, 0).buffer(1)
    ellipse = scale(ellipse, math.sqrt(1/wp[0]), math.sqrt(1/wp[1]))
    ellipse = rotate(ellipse, angle, origin=(0, 0))

    intersection = rect.intersection(ellipse)

    fig, ax = plt.subplots(figsize=(7, 6))

    xs = pts_in_envelope[:, 0]
    ys = pts_in_envelope[:, 2]
    sc = plt.scatter(xs, ys, c=pts_lamda, cmap='plasma', s=20, edgecolors='none')
    cbar=plt.colorbar(sc)
    cbar.set_label(r'$\lambda$', fontsize=20, weight='bold')
    rect_patch = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor='black', facecolor='none', lw=1.5)
    ax.add_patch(rect_patch)

    ellipse_patch = patches.Ellipse((0, 0), math.sqrt(1/wp[0]) * 2, math.sqrt(1/wp[1]) * 2, angle=angle, edgecolor='black', facecolor='none', lw=1.5)
    ax.add_patch(ellipse_patch)

    ax.scatter(x_goal, 0, color='green', s=150, marker='o', label="Control Goal")
    # plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.18), fontsize=20, frameon=False)
    # Adjust plot limits and display the result
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.95, 0.95)
    plt.xlabel('$X$', fontsize=20, weight='bold')
    plt.ylabel('$\\theta$', fontsize=20, weight='bold')
    plt.subplots_adjust(bottom=0.14)
    plt.title(f'$\lambda$ ($p = {t})$', fontsize=18, weight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'lamda_heatmap_cartpole_{x_goal}.pdf', bbox_inches='tight', dpi=300)
