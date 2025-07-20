from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon

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


class EllipticRingObstacle(PolygonObstacle):
    def __init__(
        self,
        center: tuple[float, float],
        semi_axes: tuple[float, float],
        width: float,
        angle: float = np.pi,
        num_arc_points: int = 15,
        margin: float = 0.0,
        rotation: float = 0.0,
    ):
        """
        An elliptic half-ring obstacle: the region between outer and inner ellipses,
        only for the “upper” part (from angle 0 to `angle`).

        Args:
            center: (x, y) coordinates of the ellipse center.
            semi_axes: Tuple (a, b) where 'a' is the semi-major axis, 'b' is the semi-minor axis of the outer ellipse.
            width: Radial width of the ring (subtracted from both semi-major and semi-minor axes for the inner ellipse).
            angle: Angular span of the half-ring in radians (default is np.pi for a half-ellipse).
            num_arc_points: Number of points to sample along each arc.
            margin: Safety margin to apply.
            rotation: Rotation of the entire shape (in radians).
        """
        self.center = np.array(center)
        self.outer_a, self.outer_b = semi_axes
        self.inner_a = self.outer_a - width
        self.inner_b = self.outer_b - width
        self.margin = margin
        self.angle = angle
        self.rotation = rotation

        if self.inner_a <= 0 or self.inner_b <= 0:
            raise ValueError("Width too large for given semi-axes.")

        cx, cy = self.center

        # Generate the outer and inner arcs (before rotation)
        t = np.linspace(0.0, self.angle, num_arc_points)

        outer = [(self.outer_a * np.cos(ti), self.outer_b * np.sin(ti)) for ti in t]
        inner = [(self.inner_a * np.cos(ti), self.inner_b * np.sin(ti)) for ti in t[::-1]]

        # Combine and translate the arcs
        points = outer + inner

        # Apply rotation and translate to center
        rotated_translated_points = [
            (
                cx + (x * np.cos(self.rotation) - y * np.sin(self.rotation)),
                cy + (x * np.sin(self.rotation) + y * np.cos(self.rotation)),
            )
            for x, y in points
        ]

        super().__init__(points=rotated_translated_points, margin=margin)


class TrapezoidObstacle(PolygonObstacle):
    """
    Axis-aligned trapezoid with horizontal bases.

    Parameters
    ----------
    center : (float, float)
    top_width : float            # width of the top base  (y = +h/2)
    bottom_width : float         # width of the bottom base (y = −h/2)
    height : float               # total height
    margin : float, optional     # safety margin
    """

    def __init__(
        self,
        points: list[tuple[float, float]],
        margin: float = 0.0,
        k: float = 200.0,
    ):
        if len(points) != 4:
            raise ValueError("TrapezoidObstacle requires exactly 4 vertices.")
        super().__init__(points=points, margin=margin)
        self.k = k
        # Precompute edge normals and vertices for half-space distance
        self._edges = []
        V = np.array(points, dtype=float)
        for i in range(4):
            a = V[i]
            b = V[(i + 1) % 4]
            e = b - a
            # inward normal = (e.y, -e.x) / ||e||
            n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-8)
            self._edges.append((a, n))

    # ------------------------------------------------------------------
    # Soft helpers (vectorised & CasADi-friendly)
    # ------------------------------------------------------------------
    @staticmethod
    def _soft_abs(val, is_numpy):
        return ca.sqrt(val**2 + 1e-8) if not is_numpy else np.sqrt(val**2 + 1e-8)

    @staticmethod
    def _soft_max(a, b, is_numpy):
        return 0.5 * (a + b + TrapezoidObstacle._soft_abs(a - b, is_numpy))

    @staticmethod
    def _soft_min(a, b, is_numpy):
        return 0.5 * (a + b - TrapezoidObstacle._soft_abs(a - b, is_numpy))

    @classmethod
    def _soft_max_many(cls, vals, is_numpy):
        out = vals[0]
        for v in vals[1:]:
            out = cls._soft_max(out, v, is_numpy)
        return out

    @classmethod
    def _soft_min_many(cls, vals, is_numpy):
        out = vals[0]
        for v in vals[1:]:
            out = cls._soft_min(out, v, is_numpy)
        return out

    # ------------------------------------------------------------------
    # Approximated SDF
    # ------------------------------------------------------------------
    def approximated_sdf(self, x, y):
        is_numpy = isinstance(x, np.ndarray)

        # ------------------------------------------------------------------
        # 1) signed distance to each supporting line (half-plane)
        # ------------------------------------------------------------------
        half_plane_ds = []
        for (x0, y0), (x1, y1) in self.edges:
            # edge vector and outward normal (clockwise vertex order)
            ex, ey = x1 - x0, y1 - y0
            nx, ny = ey, -ex
            if is_numpy:
                norm_len = np.sqrt(nx**2 + ny**2 + 1e-6)
            else:
                norm_len = ca.sqrt(nx**2 + ny**2 + 1e-6)
            nx, ny = nx / norm_len, ny / norm_len

            # signed distance of (x,y) to the infinite line, minus margin
            d = (nx * (x - x0) + ny * (y - y0)) - self.margin
            half_plane_ds.append(d)

        # ------------------------------------------------------------------
        # 2) inside term   (negative in the interior, 0 outside)
        # ------------------------------------------------------------------
        inner_max = self._soft_max_many(half_plane_ds, is_numpy)
        inside = self._soft_min(inner_max, 0, is_numpy)

        # ------------------------------------------------------------------
        # 3) Euclidean distance to the polygon boundary
        #     – project on each segment, keep the minimum
        # ------------------------------------------------------------------
        seg_dists = []
        for (x0, y0), (x1, y1) in self.edges:
            ex, ey = x1 - x0, y1 - y0
            seg_len_sq = ex**2 + ey**2 + 1e-6

            # parametric projection of p on the supporting line
            t_raw = ((x - x0) * ex + (y - y0) * ey) / seg_len_sq

            # smooth clamp t ∈ [0,1]
            t0 = self._soft_max(0, t_raw, is_numpy)
            t = self._soft_min(1, t0, is_numpy)

            proj_x = x0 + t * ex
            proj_y = y0 + t * ey

            # distance to the segment
            diff_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
            if is_numpy:
                dist = np.sqrt(diff_sq + 1e-6)
            else:
                dist = ca.sqrt(diff_sq + 1e-6)
            seg_dists.append(dist)

        outside = self._soft_min_many(seg_dists, is_numpy)

        # ------------------------------------------------------------------
        return outside + inside - self.margin


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


class ConvexEllipticRing(MultiObstacle):
    """
    Convex decomposition of an elliptic half-ring by tiling the region
    between outer and inner arcs into convex quadrilaterals.
    """

    def __init__(
        self,
        center: tuple[float, float],
        semi_axes: tuple[float, float],
        width: float,
        angle: float = np.pi,
        num_arc_points: int = 15,
        margin: float = 0.0,
        rotation: float = 0.0,
    ):
        cx, cy = center
        outer_a, outer_b = semi_axes
        inner_a = outer_a - width
        inner_b = outer_b - width
        if inner_a <= 0 or inner_b <= 0:
            raise ValueError("Width too large for given semi-axes.")

        # 1) sample both arcs at the same t values
        t = np.linspace(0.0, angle, num_arc_points)
        outer_raw = [(outer_a * np.cos(ti), outer_b * np.sin(ti)) for ti in t]
        inner_raw = [(inner_a * np.cos(ti), inner_b * np.sin(ti)) for ti in t]

        # 2) rotate & translate into world frame
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)

        def transform(pt):
            x, y = pt
            xr = x * cos_r - y * sin_r + cx
            yr = x * sin_r + y * cos_r + cy
            return (xr, yr)

        outer_pts = [transform(p) for p in outer_raw]
        inner_pts = [transform(p) for p in inner_raw]

        # 3) build quads between corresponding samples
        convex_pieces: list[IObstacle] = []
        for i in range(len(t) - 1):
            quad = [
                outer_pts[i],
                outer_pts[i + 1],
                inner_pts[i + 1],
                inner_pts[i],
            ]
            convex_pieces.append(TrapezoidObstacle(quad, margin=margin))

        # 4) hand off to MultiObstacle
        super().__init__(convex_pieces)


class ConvexSObstacle(MultiObstacle):
    """
    Convex decomposition of an elliptic half-ring by tiling the region
    between outer and inner arcs into convex quadrilaterals.
    """

    def __init__(
        self,
        center: tuple[float, float],
        semi_axes: tuple[float, float],
        width: float,
        angle: float = np.pi,
        num_arc_points: int = 15,
        margin: float = 0.0,
        rotation: float = 0.0,
    ):
        cx, cy = center
        outer_a, outer_b = semi_axes
        inner_a = outer_a - width
        inner_b = outer_b - width
        if inner_a <= 0 or inner_b <= 0:
            raise ValueError("Width too large for given semi-axes.")

        # 1) sample both arcs at the same t values
        t = np.linspace(0.0, angle, num_arc_points)
        outer_raw = [(outer_a * np.cos(ti), outer_b * np.sin(ti)) for ti in t]
        inner_raw = [(inner_a * np.cos(ti), inner_b * np.sin(ti)) for ti in t]

        # 2) rotate & translate into world frame
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)

        def transform(pt):
            x, y = pt
            xr = x * cos_r - y * sin_r + cx
            yr = x * sin_r + y * cos_r + cy
            return (xr, yr)

        outer_pts = [transform(p) for p in outer_raw]
        inner_pts = [transform(p) for p in inner_raw]

        # 3) build quads between corresponding samples
        convex_pieces: list[IObstacle] = []
        for i in range(len(t) - 1):
            quad = [
                outer_pts[i],
                outer_pts[i + 1],
                inner_pts[i + 1],
                inner_pts[i],
            ]
            convex_pieces.append(TrapezoidObstacle(quad, margin=margin))

        # 4) second half
        cx += 0.45
        t = np.linspace(0.0, -angle, num_arc_points)
        outer_raw = [(outer_a * np.cos(ti), outer_b * np.sin(ti)) for ti in t]
        inner_raw = [(inner_a * np.cos(ti), inner_b * np.sin(ti)) for ti in t]

        cos_r, sin_r = np.cos(rotation), np.sin(rotation)

        def transform(pt):
            x, y = pt
            xr = x * cos_r - y * sin_r + cx
            yr = x * sin_r + y * cos_r + cy
            return (xr, yr)

        outer_pts = [transform(p) for p in outer_raw]
        inner_pts = [transform(p) for p in inner_raw]
        for i in range(len(t) - 1):
            quad = [
                outer_pts[i],
                outer_pts[i + 1],
                inner_pts[i + 1],
                inner_pts[i],
            ][::-1]
            convex_pieces.append(TrapezoidObstacle(quad, margin=margin))

        # 4) hand off to MultiObstacle
        super().__init__(convex_pieces)
