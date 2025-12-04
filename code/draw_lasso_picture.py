import numpy as np
import matplotlib.pyplot as plt

# High-resolution figure
fig, ax = plt.subplots(figsize=(3, 6), dpi=300)
slateblue = (45/255, 62/255, 80/255)

# ---- L1 ball (LASSO constraint) ----
t = 1.5  # radius of the L1 ball
diamond = np.array([
    [0,  t],
    [t,  0],
    [0, -t],
    [-t, 0]
])
ax.fill(diamond[:, 0], diamond[:, 1],
        color=slateblue, alpha=0.95, zorder=1)

ax.scatter(0, t, color='C1', s=15, zorder=3)

# ---- Quadratic loss contours (OLS objective) ----
center = np.array([1.4, 2.35])  # location of unconstrained estimator \hat{δ}
Sigma = np.array([[3.0, 1.6],
                  [1.6, 1.0]])
Sigma_inv = np.linalg.inv(Sigma)

d1 = np.linspace(-1.5, 4, 400)
d2 = np.linspace(-2.0, 4, 400)
D1, D2 = np.meshgrid(d1, d2)

Z = np.stack([D1 - center[0], D2 - center[1]], axis=-1)
quad = np.einsum('...i,ij,...j->...', Z, Sigma_inv, Z)

levels = np.linspace(0.1, 2, 10)
ax.contour(D1, D2, quad, levels=levels,
           colors='red', linewidths=1)

# ---- Unconstrained estimator point ----
ax.scatter(center[0], center[1], color='black', s=15, zorder=3)
#ax.text(center[0] + 0.1, center[1] + 0.05, r'$\hat{\delta}$', fontsize=14)

# ---- Axes and labels ----
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Remove box and ticks
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Axis labels near arrow tips
ax.set_xlim(-2.0, 4.0)
ax.set_ylim(-2.0, 4.0)
ax.set_aspect('equal', 'box')

ax.text(ax.get_xlim()[1], -0.15, r'$\delta_1$',
        ha='right', va='top', fontsize=14)
ax.text(-0.15, ax.get_ylim()[1], r'$\delta_2$',
        ha='right', va='top', fontsize=14)

fig.tight_layout()

# Save VERY high-res version for slides
plt.savefig("lasso_geometry_delta.png", dpi=600, bbox_inches="tight")
plt.show()