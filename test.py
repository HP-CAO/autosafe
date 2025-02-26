import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Example task data: list of tuples (x, y, x1, y1, success_flag)
tasks = [
    (0.0, 0.0, 0.5, 1.2, True),  # Success example
    (-0.5, 0.8, 0.2, -1.0, False),  # Failure example
    # Add more tasks as needed
]

plt.figure(figsize=(6, 6))
for (x, y, x1, y1, success) in tasks:
    dx = x1 - x
    dy = y1 - y
    color = 'blue' if success else 'red'

    # Draw arrow from start (x, y) to goal (x1, y1)
    plt.arrow(x, y, dx, dy, color=color, width=0.005, length_includes_head=True)

    # Mark the starting point with a circle marker
    plt.scatter(x, y, marker='o', color='black', s=50)

    # Mark the ending point with a triangle marker
    plt.scatter(x1, y1, marker='^', color='black', s=50)

# Set x-axis limit manually
plt.xlim(-1, 1)
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)

# Create custom legend handles:
# For arrow colors (success and failure)
success_patch = Patch(color='blue', label='Success')
failure_patch = Patch(color='red', label='Failure')
# For markers (start and goal)
start_marker = Line2D([0], [0], marker='o', color='w', label='Start',
                      markerfacecolor='black', markersize=8)
goal_marker = Line2D([0], [0], marker='^', color='w', label='Goal',
                     markerfacecolor='black', markersize=8)

# Combine all legend handles
plt.legend(handles=[success_patch, failure_patch, start_marker, goal_marker])

plt.title('RL Task Evaluations: Arrows with Start and Goal Markers')
plt.show()
