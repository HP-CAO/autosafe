# Geometry illustration for the relation: (1-λ)(a_θ - a_safe) = (a_rl - a_safe)
# You can change a_safe, a_theta, and lam to explore other configurations.

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters (edit these) ---
a_safe = np.array([0.5, 0.5])
a_theta = np.array([4.0, 2.5])
lam = 0.15  # λ(s) in [0,1]

# --- Derived quantities ---
a_rl = a_safe + (1 - lam) * (a_theta - a_safe)
v_theta = a_theta - a_safe
v_rl = a_rl - a_safe

# Sanity: ratio of magnitudes should be 1/(1-λ)
ratio = np.linalg.norm(v_theta) / max(np.linalg.norm(v_rl), 1e-12)
expected_ratio = 1.0 / max(1 - lam, 1e-12)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6.0, 4.0))

# Draw arrows from a_safe
ax.quiver(
    a_safe[0], a_safe[1], v_theta[0], v_theta[1],
    angles='xy', scale_units='xy', scale=1, color='grey', label=r'$\mathbf{a}_\theta - \mathbf{a}_{\mathrm{safe}}$'
)
ax.quiver(
    a_safe[0], a_safe[1], v_rl[0], v_rl[1],
    angles='xy', scale_units='xy', scale=1, color='blue', label=r'$\mathbf{a}_{\mathrm{rl}} - \mathbf{a}_{\mathrm{safe}}$'
)
ax.hlines(3.2, 0.0, 4.5, color='red', linewidth=1.0, linestyle='--', label='$a_{1_{max}}$')
ax.vlines(4.2, 0, 3.8, color='pink', linewidth=1.0, linestyle='--', label='$a_{2_{max}}$')

# Points
ax.scatter([a_safe[0], a_theta[0], a_rl[0]], [a_safe[1], a_theta[1], a_rl[1]], s=50)
ax.annotate(r'$\mathbf{a}_{\mathrm{safe}}$', xy=a_safe, xytext=(5, 5), textcoords='offset points')
ax.annotate(r'$\mathbf{a}_\theta$', xy=a_theta, xytext=(5, 5), textcoords='offset points')
ax.annotate(r'$\mathbf{a}_{\mathrm{rl}}$', xy=a_rl, xytext=(5, 5), textcoords='offset points')

# Colinearity guide (dashed line through a_safe -> a_theta)
t = np.linspace(-0.2, 1.2, 50)
line = a_safe[:, None] + v_theta[:, None] * t
ax.plot(line[0], line[1], linestyle='--', linewidth=1)

# Text box with the key identity and measured ratios
text = (
    r'$(1-\lambda)\,(\mathbf{a}_\theta-\mathbf{a}_{\mathrm{safe}})'
    r'\;=\;\mathbf{a}_{\mathrm{rl}}-\mathbf{a}_{\mathrm{safe}}$' + '\n'
    rf'$\lambda={lam:.2f}$'
)
ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top')

# Axes settings
all_pts = np.vstack([a_safe, a_theta, a_rl])
mins = all_pts.min(axis=0) - 0.5
maxs = all_pts.max(axis=0) + 0.5
ax.set_xlim(0, 4.5)
ax.set_ylim(0, 3.8)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle=':')
ax.set_xlabel('$a_1$')
ax.set_ylabel('$a_2$')
ax.set_title('Geometric relation between action vectors')
ax.legend(loc='lower right')
plt.savefig('action_vector_relation.pdf', bbox_inches='tight', dpi=300)
# plt.show()
