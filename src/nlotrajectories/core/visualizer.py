from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plot_trajectory(X_opt, geometry, obstacles, title="Trajectory", goal=None):
    path = Path("result")
    path.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots()
    N = X_opt.shape[1]

    for k in range(N):
        pose_k = X_opt[:, k]
        geometry.draw(ax, pose_k)

    obstacles.draw(ax, alpha=0.7, color="r")

    if goal is not None:
        ax.plot(goal[0], goal[1], "ro", label="Goal")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)
    plt.legend()
    plt.savefig(path / f"{title}.png", bbox_inches="tight")
    plt.close()


def plot_levels(func, x_range=(-1, 2), y_range=(-1, 2), n_samples=500, title="sdf"):
    path = Path("result")
    path.mkdir(parents=True, exist_ok=True)
    x = np.linspace(x_range[0], x_range[1], n_samples)
    y = np.linspace(y_range[0], y_range[1], n_samples)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # Plot contour lines at levels 0, 1, 2, ...
    levels = [0.05, 0.1, 0.2, 0.5, 1]
    plt.contour(X, Y, Z, levels=[0], colors="red")
    contours = plt.contour(X, Y, Z, levels=levels, colors="black")
    plt.clabel(contours, inline=True, fontsize=8, fmt="%1.2f")
    plt.colorbar(label="f(x, y)")
    plt.title(f"Levels for {title}")
    plt.axis("equal")
    plt.savefig(path / f"{title}_levels.png", bbox_inches="tight")
    plt.close()

def plot_control(U_opt, dt: float, title="Control Inputs"):    
    path = Path("result")
    path.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots()
    N = U_opt.shape[1]

    time = np.arange(N) * dt
    for i in range(U_opt.shape[0]):
        ax.plot(time, U_opt[i, :], label=f"Control {i + 1}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title(title)
    ax.grid(True)
    plt.legend()
    plt.savefig(path / f"{title}.png", bbox_inches="tight")
    plt.close()  


