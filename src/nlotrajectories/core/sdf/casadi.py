from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from matplotlib.patches import Circle, Rectangle

from nlotrajectories.core.utils import soft_min


class IObstacle(ABC):
    @abstractmethod
    def sdf(self, x: float, y: float) -> float:
        pass

    @abstractmethod
    def approximated_sdf(self, x: float, y: float) -> float:
        pass

    @abstractmethod
    def draw(self, ax, **kwargs) -> None:
        pass


class CircleObstacle(IObstacle):
    def __init__(self, center: tuple[float, float], radius: float, margin: float = 0.0):
        self.center = np.array(center)
        self.radius = radius
        self.margin = margin

    def sdf(self, x: ca.MX, y: ca.MX) -> float:
        dx = x - self.center[0]  # shape (500, 500)
        dy = y - self.center[1]  # shape (500, 500)

        dist = np.sqrt(dx**2 + dy**2)  # shape (500, 500)
        return dist - (self.radius + self.margin)

    def approximated_sdf(self, x: ca.MX, y: ca.MX) -> float:
        return self.sdf(x, y)

    def draw(self, ax, **kwargs) -> None:
        circle = Circle(self.center, self.radius, **kwargs)
        ax.add_patch(circle)


class SquareObstacle(IObstacle):
    def __init__(self, center: tuple[float, float], size: float, margin: float = 0.0):
        self.center = np.array(center)
        self.size = size
        self.margin = margin

    def sdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cx, cy = self.center
        half = self.size / 2 + self.margin

        dx = np.abs(x - cx) - half
        dy = np.abs(y - cy) - half

        dx_clamped = np.maximum(dx, 0)
        dy_clamped = np.maximum(dy, 0)

        outside_dist = np.sqrt(dx_clamped**2 + dy_clamped**2)
        inside_dist = np.minimum(np.maximum(dx, dy), 0)

        return outside_dist + inside_dist

    def approximated_sdf(self, x, y):
        is_numpy = isinstance(x, np.ndarray)

        # Extract center
        cx, cy = self.center
        half = self.size / 2 + self.margin

        # Build coordinate delta
        dx = x - cx
        dy = y - cy

        # Soft absolute
        def soft_abs(val):
            if is_numpy:
                return np.sqrt(val**2 + 1e-6)
            else:
                return ca.sqrt(val**2 + 1e-6)

        dx = soft_abs(dx)
        dy = soft_abs(dy)

        # d = soft_abs(p) - half
        d_x = dx - half
        d_y = dy - half

        # soft_max for [d_x, 0] and [d_y, 0]
        def soft_max(a, b):
            if is_numpy:
                return 0.5 * (a + b + np.sqrt((a - b) ** 2 + 1e-6))
            else:
                return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + 1e-6))

        def soft_min(a, b):
            if is_numpy:
                return 0.5 * (a + b - np.sqrt((a - b) ** 2 + 1e-6))
            else:
                return 0.5 * (a + b - ca.sqrt((a - b) ** 2 + 1e-6))

        d_x_out = soft_max(d_x, 0)
        d_y_out = soft_max(d_y, 0)

        if is_numpy:
            outside = np.sqrt(d_x_out**2 + d_y_out**2)
        else:
            outside = ca.sqrt(d_x_out**2 + d_y_out**2)

        inner_max = soft_max(d_x, d_y)
        inside = soft_min(inner_max, 0)

        return outside + inside

    def draw(self, ax, **kwargs) -> None:
        half = self.size / 2
        lower_left = self.center - half
        rect = Rectangle(lower_left, 2 * half, 2 * half, **kwargs)
        ax.add_patch(rect)


class MultiObstacle(IObstacle):
    def __init__(self, obstacles: list[IObstacle]):
        self.obstacles = obstacles

    def sdf(self, x: ca.MX, y: ca.MX) -> float:
        sdf_values = [obs.sdf(x, y) for obs in self.obstacles]
        return np.min(np.stack(sdf_values, axis=0), axis=0)

    def approximated_sdf(self, x: ca.MX, y: ca.MX) -> float:
        return soft_min([obs.approximated_sdf(x, y) for obs in self.obstacles])

    def draw(self, ax, **kwargs) -> None:
        for obs in self.obstacles:
            obs.draw(ax, **kwargs)
