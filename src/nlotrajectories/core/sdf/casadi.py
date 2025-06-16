from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon as MplPolygon
from shapely.geometry import Polygon, Point

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


class PolygonObstacle(IObstacle):
    def __init__(self, points: list[tuple[float, float]], margin: float = 0.0):
        self.points = points
        self.polygon = Polygon(points)
        self.margin = margin
        self.edges = list(zip(points, points[1:] + [points[0]]))
        self.centroid = np.mean(self.points, axis=0)

    def sdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if isinstance(x, ca.MX) or isinstance(x, ca.SX):
            raise TypeError("PolygonObstacle SDF only supports NumPy arrays, not CasADi expressions.")

        flat_x = x.ravel()
        flat_y = y.ravel()

        points = [Point(xi, yi) for xi, yi in zip(flat_x, flat_y)]

        distances = np.array([p.distance(self.polygon.boundary) for p in points])
        signs = np.array([-1 if self.polygon.contains(p) else 1 for p in points])

        sdf_flat = distances * signs - self.margin
        return sdf_flat.reshape(x.shape)

    def approximated_sdf(self, x, y):
        is_numpy = isinstance(x, np.ndarray)

        def norm(val):
            return np.sqrt(val) if is_numpy else ca.sqrt(val)

        def maximum(a, b):
            return np.maximum(a, b) if is_numpy else ca.fmax(a, b)

        def minimum(a, b):
            return np.minimum(a, b) if is_numpy else ca.fmin(a, b)

        dists = []
        for (x0, y0), (x1, y1) in self.edges:
            dx = x1 - x0
            dy = y1 - y0
            seg_len_sq = dx**2 + dy**2 + 1e-6

            px = x - x0
            py = y - y0
            t_raw = (px * dx + py * dy) / seg_len_sq
            t = minimum(1.0, maximum(0.0, t_raw))

            proj_x = x0 + t * dx
            proj_y = y0 + t * dy

            dist = norm((x - proj_x) ** 2 + (y - proj_y) ** 2)
            dists.append(dist)

        min_dist = soft_min(dists)

        # Use precomputed centroid
        cx, cy = self.centroid
        direction = (x - cx) * (y - cy)
        sign = np.tanh(100 * direction) if is_numpy else ca.tanh(100 * direction)

        return sign * min_dist - self.margin

    def draw(self, ax, **kwargs) -> None:
        polygon_patch = MplPolygon(self.points, closed=True, **kwargs)
        ax.add_patch(polygon_patch)


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
