from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from nlotrajectories.core.geometry import DotGeometry

def plot_sampling(obstacles, xs=None, ys=None, title="Sampling", goal=None):
    path = Path("result")
    path.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots()

    # Plot obstacles
    obstacles.draw(ax, alpha=0.7, color="r")

    # Plot sampling points if provided
    if xs is not None and ys is not None:
        ax.scatter(xs, ys, s=0.1, c='b', alpha=0.1, label="Samples")

    ax.set_aspect("equal")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(path / f"{title}.png", bbox_inches="tight")
    plt.close()


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
    elif X_opt.shape[0] == 4:  # dot second order or ackermann first order
        if isinstance(geometry, DotGeometry):
            vx_vals = X_opt[2, :]
            vy_vals = X_opt[3, :]
        else:  # ackermann first order
            vx_vals = U_opt[0, :] * np.cos(X_opt[2, :])
            vy_vals = U_opt[0, :] * np.sin(X_opt[2, :])
    elif X_opt.shape[0] == 5:  # unicycle second order
        vx_vals = X_opt[3, :] * np.cos(X_opt[2, :])
        vy_vals = X_opt[3, :] * np.sin(X_opt[2, :])
    elif X_opt.shape[0] == 7:  # ackermann second order
        vx_vals = X_opt[4, :] * np.cos(X_opt[2, :])
        vy_vals = X_opt[4, :] * np.sin(X_opt[2, :])
    else:
        raise ValueError("Unsupported state dimension for animation.")
    # ---------- Create Figure and Axes ----------
    fig, ax = plt.subplots()
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 1.4)
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

    # decide if we need a polygon patch
    use_poly = geometry is not None and not isinstance(geometry, DotGeometry)
    if use_poly:
        # initial pose for polygon
        theta0 = X_opt[2, 0] if X_opt.shape[0] >= 3 else 0.0
        pose0 = np.array([x_vals[0], y_vals[0], theta0])
        verts0 = geometry.transform(pose0)
        robot_patch = Polygon(verts0, closed=True, edgecolor="blue", facecolor="none", lw=2, label="Robot")
        ax.add_patch(robot_patch)
    else:
        robot_patch = None

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
        # Initialize robot polygon to starting pose
        if use_poly:
            robot_patch.set_xy(verts0)
            return point, path, velocity_quiver, speed_text, robot_patch
        else:
            return point, path, velocity_quiver, speed_text

    # ---------- Update Function per Frame ----------
    def update(frame):
        # Current position and orientation
        xi, yi = x_vals[frame], y_vals[frame]
        thetai = X_opt[2, frame] if X_opt.shape[0] >= 3 else 0.0
        pose_i = np.array([xi, yi, thetai])

        # Update point and path line
        point.set_data([xi], [yi])
        path.set_data(x_vals[: frame + 1], y_vals[: frame + 1])

        # polygon if needed
        if use_poly:
            verts = geometry.transform(pose_i)
            robot_patch.set_xy(verts)

        # Update velocity arrow
        vx, vy = vx_vals[frame], vy_vals[frame]
        velocity_quiver.set_offsets([[xi, yi]])
        velocity_quiver.set_UVC([vx], [vy])
        speed = np.hypot(vx, vy)
        speed_text.set_text(f"Speed: {speed:.2f} m/s")

        if use_poly:
            return point, path, velocity_quiver, speed_text, robot_patch
        else:
            return point, path, velocity_quiver, speed_text

    # ---------- Generate and Save Animation ----------
    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True, interval=100)

    ani.save(dir / f"{title}.gif", writer="pillow", fps=10)
    plt.show()
    plt.close()


def plot_initialization(initializer, X_init: np.ndarray, obstacles=None, title="RRT Initialization"):
    """
    Plot the RRT tree saved in the initializer alongside the initial trajectory.

    Args:
        initializer: RRTInitializer instance containing a `_last_tree` attribute with nodes having `.pos` and `.parent`.
        X_init: np.ndarray of shape (N, >=2), initial trajectory; the first two dimensions are used for XY plotting.
        obstacles: Optional obstacle object providing a `draw(ax, alpha, color)` method to render obstacles.
        title: Title for the plot.
    """
    # Retrieve the stored RRT tree from the initializer
    tree = getattr(initializer, "_last_tree", None)
    if tree is None:
        raise ValueError("No RRT tree stored in initializer. Ensure get_initial_guess() has been called.")

    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw obstacles if provided
    if obstacles is not None:
        obstacles.draw(ax, alpha=0.7, color="r")

    # Plot all edges of the RRT tree
    for node in tree:
        if node.parent is not None:
            p1 = node.parent.pos
            p2 = node.pos
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-k", linewidth=0.5)

    # Plot the initial trajectory in red
    xy = X_init[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], "-r", linewidth=2, label="Initial Path")

    # Mark start and goal points
    ax.scatter(xy[0, 0], xy[0, 1], c="green", s=50, label="Start")
    ax.scatter(xy[-1, 0], xy[-1, 1], c="blue", s=50, label="Goal")

    # Finalize plot
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")
    plt.show()
