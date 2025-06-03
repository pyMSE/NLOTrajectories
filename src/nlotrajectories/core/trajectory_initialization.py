from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Initializer(str, Enum):
    LINEAR = "linear"
    RRT = "rrt"


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

    class Node:
        def __init__(self, pos, parent=None):
            self.pos = np.array(pos)
            self.parent = parent

    def __init__(
        self,
        N: int,
        x0: np.ndarray,
        x_goal: np.ndarray,
        dt: float,
        sdf_func: callable,
        bounds: list,
        step_size: float = 0.05,
        max_iter: int = 1000,
        min_sdf: float = 0.01,
    ):
        """
        Args:
            N (int): Number of samples along the path (trajectories will have N points).
            dt (float): Time step used for velocities/omegas.
            sdf_func (callable): Signed-distance function sdf_func(x, y) -> float.
            bounds (list): [[xmin, ymin], [xmax, ymax]] sampling bounds for RRT.
            step_size (float, optional): Extension step size. Defaults to 0.05.
            max_iter (int, optional): Maximum RRT iterations. Defaults to 1000.
            min_sdf (float, optional): Minimum safe distance to obstacles. Defaults to 0.01.
        """
        super().__init__(x0, x_goal)
        self.N = N
        self.dt = dt
        self.sdf_func = sdf_func
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.min_sdf = min_sdf

    def _collision(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Check collision by sampling along the segment p1->p2.
        """
        n_samples = 10
        for i in range(n_samples + 1):
            interp = p1 + (p2 - p1) * (i / n_samples)
            if self.sdf_func(interp[0], interp[1]) < self.min_sdf:
                return True
        return False

    def _nearest(self, tree: list, rnd: np.ndarray) -> "RRTInitializer.Node":
        """
        Find the nearest node in the tree to a random sample.
        """
        return min(tree, key=lambda node: np.linalg.norm(node.pos - rnd))

    def _build_rrt_path(self, start, end) -> np.ndarray:
        """
        Run basic RRT to find a collision-free path from start->goal in XY plane,
        then resample it to exactly self.N points and lift to full-state (x,y,theta,v,omega).

        Returns:
            state_traj (np.ndarray): Array of shape (self.N, 5).
        """
        tree = [self.Node(start)]
        final = None

        for _ in range(self.max_iter):
            rnd = np.random.uniform(self.bounds[0], self.bounds[1])
            nearest_node = self._nearest(tree, rnd)
            direction = rnd - nearest_node.pos
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction = direction / norm
            new_pos = nearest_node.pos + self.step_size * direction

            if not self._collision(nearest_node.pos, new_pos):
                new_node = self.Node(new_pos, nearest_node)
                tree.append(new_node)
                if np.linalg.norm(new_pos - end) < self.step_size:
                    final = self.Node(end, new_node)
                    tree.append(final)
                    break
        else:
            raise RuntimeError("RRT failed to find a path within max_iter.")

        # Trace back from final to start
        path = []
        node = final
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path = path[::-1]  # reverse

        path = np.array(path)  # shape (M, 2)
        # Compute segment lengths
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_length = np.sum(segment_lengths)
        cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
        # Resample to exactly N points along the curve
        target_lengths = np.linspace(0, total_length, self.N)

        resampled_xy = []
        j = 0
        for t in target_lengths:
            while j < len(cumulative_lengths) - 2 and cumulative_lengths[j + 1] < t:
                j += 1
            if cumulative_lengths[j + 1] - cumulative_lengths[j] == 0:
                point = path[j].copy()
            else:
                ratio = (t - cumulative_lengths[j]) / (cumulative_lengths[j + 1] - cumulative_lengths[j])
                point = path[j] + ratio * (path[j + 1] - path[j])
            resampled_xy.append(point)
        resampled_xy = np.array(resampled_xy)  # shape (N, 2)

        # Build full state trajectory: [x, y, theta, v, omega]
        delta_xy = np.diff(resampled_xy, axis=0)
        theta = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
        theta = np.append(theta, theta[-1])  # repeat last
        theta = np.unwrap(theta)

        distances = np.linalg.norm(delta_xy, axis=1)
        v = distances / self.dt
        v = np.append(v, v[-1])

        omega = np.diff(theta) / self.dt
        omega = np.append(omega, omega[-1])

        state_traj = np.column_stack((resampled_xy[:, 0], resampled_xy[:, 1], theta, v, omega))  # shape (N, 5)
        return state_traj

    def get_initial_guess(self) -> np.ndarray:
        # Extract (x, y) positions from full state
        start_xy = self.x0[:2]
        goal_xy = self.x_goal[:2]

        state_traj = self._build_rrt_path(start_xy, goal_xy)  # shape (N, 5)
        return state_traj


INITIALIZER_CLASS_MAP = {
    Initializer.LINEAR: LinearInitializer,
    Initializer.RRT: RRTInitializer,
}
