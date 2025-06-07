from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(X_opt, geometry, obstacles, X_init, title="Trajectory", goal=None):
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
    if X_init is not None:
        plt.plot(X_init[:, 0], X_init[:, 1], "r--", linewidth=1, label="Original Path")
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


def plot_control(U_opt, dt: float, title: str = "Control Inputs"):
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


def animation_plot(X_opt, U_opt, geometry, obstacles, title: str = "Trajectory Animation", goal=None):
    dir = Path("result")
    dir.mkdir(parents=True, exist_ok=True)
    # Ensure shapes match by slicing the longer array
    min_length = min(U_opt.shape[1], X_opt.shape[1])
    U_opt = U_opt[:, :min_length]
    X_opt = X_opt[:, :min_length]

    x_vals = X_opt[0, :]
    y_vals = X_opt[1, :]

    if X_opt.shape[0] == 2:  # dot first order
        vx_vals = U_opt[0, :]
        vy_vals = U_opt[1, :]
    elif X_opt.shape[0] == 3:  # unicycle first order
        vx_vals = U_opt[0, :] * np.cos(X_opt[2, :])
        vy_vals = U_opt[0, :] * np.sin(X_opt[2, :])
    elif X_opt.shape[0] == 4:  # dot second order
        vx_vals = X_opt[2, :]
        vy_vals = X_opt[3, :]
    elif X_opt.shape[0] == 5:  # unicycle second order
        vx_vals = X_opt[3, :] * np.cos(X_opt[2, :])
        vy_vals = X_opt[3, :] * np.sin(X_opt[2, :])
    else:
        raise ValueError("Unsupported state dimension for animation.")
    # ---------- Create Figure and Axes ----------
    fig, ax = plt.subplots()
    ax.set_xlim(-0.1, 1.4)
    ax.set_ylim(-0.1, 1.4)
    ax.set_aspect("equal")
    ax.set_title("2D Trajectory with Obstacles and Velocity Vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    obstacles.draw(ax, alpha=0.7, color="r")

    # Draw start and goal points
    if goal is not None:
        ax.plot(goal[0], goal[1], "go", label="Goal")
    ax.plot(x_vals[0], y_vals[0], "ro", label="Start")

    # Animation Elements
    (point,) = ax.plot([], [], "bo", label="Current Position")
    (path,) = ax.plot([], [], "b--", alpha=0.5)
    velocity_quiver = ax.quiver([], [], [], [], color="green", scale=1, label="Velocity")
    speed_text = ax.text(
        0.02,
        1.05,
        "",
        transform=ax.transAxes,
        fontsize=12,
        color="purple",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # ---------- Initialize Animation Elements ----------
    def init():
        point.set_data([], [])
        path.set_data([], [])
        velocity_quiver.set_UVC([], [])
        speed_text.set_text("")
        return point, path, velocity_quiver, speed_text

    # ---------- Update Function per Frame ----------
    def update(frame):

        if frame < len(x_vals):
            # pose = X_opt[:, frame]  # Current pose
            # geometry.draw(ax, pose)  # Draw geometry at the current pose
            point.set_data([x_vals[frame]], [y_vals[frame]])  # Update point position
            path.set_data(x_vals[: frame + 1], y_vals[: frame + 1])  # Update path trace

            vx = vx_vals[frame]
            vy = vy_vals[frame]
            velocity_quiver.set_offsets([[x_vals[frame], y_vals[frame]]])  # Set the position of the velocity vector
            velocity_quiver.set_UVC([vx], [vy])
            speed = np.sqrt(vx**2 + vy**2)  # Compute speed
            speed_text.set_text(f"Speed: {speed:.2f} m/s")  # Update speed text

        return point, path, velocity_quiver, speed_text

    # ---------- Generate and Save Animation ----------
    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True, interval=100)

    ani.save(dir / f"{title}.gif", writer="pillow", fps=10)
    plt.show()
    plt.close()
