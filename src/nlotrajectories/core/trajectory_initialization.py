import math
import random
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.interpolate import interp1d, splev, splprep, CubicSpline

from nlotrajectories.core.geometry import DotGeometry, IRobotGeometry, RectangleGeometry


class Initializer(str, Enum):
    LINEAR = "linear"
    RRT = "rrt"
    DEFAULT = "default"


class TrajectoryInitializer(ABC):
    """
    Abstract base class for trajectory initialization methods.
    """

    def __init__(self, x0: np.ndarray, x_goal: np.ndarray):
        self.x0 = x0
        self.x_goal = x_goal

    @abstractmethod
    def get_initial_guess(self) -> np.ndarray:
        """
        Compute an initial state trajectory guess between x0 and x_goal.

        Args:
            x0 (np.ndarray): Initial state (should contain numeric values).
            x_goal (np.ndarray): Goal state (should contain numeric values).

        Returns:
            np.ndarray: An array of shape (N+1, state_dim) representing the initial guess.
        """
        pass


class LinearInitializer(TrajectoryInitializer):
    """
    Simple straight-line (linear interpolation) initializer in state space.
    """

    def __init__(self, x0: np.ndarray, x_goal: np.ndarray, N: int):
        super().__init__(x0, x_goal)
        """
        Args:
            N (int): Number of control intervals; initial trajectory will have N+1 points.
        """
        self.N = N

    def get_initial_guess(self) -> np.ndarray:
        return np.linspace(self.x0, self.x_goal, self.N + 1)


class RRTInitializer(TrajectoryInitializer):
    """
    RRT-based initializer that plans a collision-free path in (x,y) and lifts it to full state.
    """

    class _Node:
        def __init__(self, pos, parent):
            self.pos = pos
            self.parent = parent

    def __init__(
        self,
        N: int,
        x0: np.ndarray,
        x_goal: np.ndarray,
        dt: float,
        sdf_func: callable,
        geometry: IRobotGeometry,
        bounds: np.ndarray,
        step_size: float = 0.05,
        max_iter: int = 1000,
        margin: float = 0.01,
        goal_sample_rate: float = 0.05,
    ):
        """
        Args:
            N:            Number of points in the returned trajectory.
            x0:           Initial full state (array of length ≥2; uses only x0[:2]).
            x_goal:       Goal full state (uses only x_goal[:2]).
            dt:           Time-step for computing v, omega (not used here; velocities set to zero).
            sdf_func:     Callable sdf_func(x,y)->float; must return signed distance to obstacles.
            geometry:     IRobotGeometry instance with .body_points (robot footprint vertices).
            bounds:       np.ndarray [[xmin,ymin],[xmax,ymax]] sampling domain.
            step_size:    RRT extension step size.
            max_iter:     Maximum RRT iterations.
            margin:       Extra safety distance.
            goal_sample_rate: Probability of sampling goal to bias RRT.
        """
        super().__init__(x0, x_goal)
        self.N = N
        self.dt = dt
        self.sdf_func = sdf_func
        self.bounds = np.array(bounds)
        self.geometry = geometry
        self.step_size = step_size
        self.max_iter = max_iter
        self.margin = margin
        self.goal_sample_rate = goal_sample_rate
        self._last_tree = None

        # Precompute inflation radius = min distance from footprint points to center + margin
        if isinstance(self.geometry, RectangleGeometry):
            radii = [np.linalg.norm(np.min(pt)) for pt in geometry.body_points]
            self.inflation = max(radii) + margin
        else:
            self.inflation = 0

    def _collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Sample along p1->p2 in steps of ~step_size,
        check sdf(pt) >= inflation to ensure the whole footprint is collision-free.
        """
        if self.sdf_func is None:
            return True
        dist = np.linalg.norm(p2 - p1)
        n = max(1, int(np.ceil(dist / self.step_size)))
        for i in range(n + 1):
            pt = p1 + (p2 - p1) * (i / n)
            if self.sdf_func(pt[0], pt[1]) < self.inflation:
                return False
        return True

    def _shortcut_path(self, path: np.ndarray) -> np.ndarray:
        def is_collision_free(p1, p2):
            dist = np.linalg.norm(p2 - p1)
            n = max(1, int(np.ceil(dist / self.step_size)))
            for i in range(n + 1):
                pt = p1 + (p2 - p1) * (i / n)
                if self.sdf_func(pt[0], pt[1]) < self.inflation:
                    return False
            return True

        new_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if is_collision_free(path[i], path[j]):
                    break
                j -= 1
            new_path.append(path[j])
            i = j
        return np.array(new_path)

    def _bspline_curve(self, points: np.ndarray, num_points: int) -> np.ndarray:
        if len(points) <= 2:
            return points
        s = np.linspace(0, 1, len(points))
        cs_x = CubicSpline(s, points[:, 0])
        cs_y = CubicSpline(s, points[:, 1])
        s_new = np.linspace(0, 1, num_points)
        return np.vstack((cs_x(s_new), cs_y(s_new))).T

    def _build_rrt_path(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """
        1) Run RRT in XY plane from start[:2] to end[:2].
        2) Extract path, resample to exactly self.N points.
        3) Lift to full-state (x,y,theta,v,omega).
        Returns:
            state_traj: np.ndarray of shape (self.N, 5).
        """
        # --- 1. Build RRT tree ---
        tree = [self._Node(start[:2], None)]
        final = None

        for i in range(self.max_iter):
            # sample
            if random.random() < self.goal_sample_rate:
                ref = end[:2].copy()
            else:
                ref = np.array(
                    [
                        random.uniform(self.bounds[0, 0], self.bounds[1, 0]),
                        random.uniform(self.bounds[0, 1], self.bounds[1, 1]),
                    ]
                )
            # nearest
            dists = [np.linalg.norm(ref - node.pos) for node in tree]
            nearest_node = tree[int(np.argmin(dists))]
            # steer
            direction = ref - nearest_node.pos
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            new_pos = nearest_node.pos + (direction / norm) * self.step_size
            # collision check (point inflation)
            if self._collision_free(nearest_node.pos, new_pos):
                new_node = self._Node(new_pos, nearest_node)
                tree.append(new_node)
                if np.linalg.norm(new_pos - end[:2]) < self.step_size:
                    final = self._Node(end[:2], new_node)
                    tree.append(final)
                    break
        else:
            raise RuntimeError("RRT failed to find a path within max_iter.")
        self._last_tree = tree
        # --- 2. Extract raw path and resample ---
        path = []
        node = final
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path = np.array(path[::-1])  # shape (M,2)

        # --- 2b. Optional path shortcut ---
        path = self._shortcut_path(path)
        # --- 2c. Smooth path with B-spline ---
        xy = self._bspline_curve(path, self.N)

        # --- 3. Lift to full-state ---
        state_traj = np.zeros((self.N, self.x0.shape[0]))
        state_traj[:, 0:2] = xy

        return state_traj

    def get_initial_guess(self) -> np.ndarray:
        return self._build_rrt_path(self.x0, self.x_goal)


class DefualtInitializer(TrajectoryInitializer):
    """
    Null initializer: returns None to skip any custom initialization.
    """

    def __init__(self):
        pass

    def get_initial_guess(self) -> None:
        return None


INITIALIZER_CLASS_MAP = {
    Initializer.LINEAR: LinearInitializer,
    Initializer.RRT: RRTInitializer,
    Initializer.DEFAULT: DefualtInitializer,
}
