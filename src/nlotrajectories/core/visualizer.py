import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(X_opt, geometry, title="Trajectory", goal=None, obstacle_draw_fn=None, cfg=None):
    fig, ax = plt.subplots()
    N = X_opt.shape[1]

    for k in range(N):
        pose_k = X_opt[:, k]
        geometry.draw(ax, pose_k)

    if goal is not None:
        ax.plot(goal[0], goal[1], "ro", label="Goal")

    if obstacle_draw_fn:
        obstacle_draw_fn(ax, cfg)

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)
    plt.legend()
    plt.savefig('foo2.png', bbox_inches='tight')
