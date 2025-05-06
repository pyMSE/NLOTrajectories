from abc import ABC, abstractmethod
from enum import Enum

import casadi as ca


class GoalMode(str, Enum):
    CENTER = "center"
    ANY_POINT = "any_point"


class Shape(str, Enum):
    DOT = "dot"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"


class IRobotGeometry(ABC):
    @abstractmethod
    def transform(self, pose: ca.MX) -> list[tuple[ca.MX, ca.MX]]:
        """
        Returns a list of (x, y) points in world coordinates based on the current robot pose.

        Parameters:
            pose (ca.MX): A symbolic vector [x, y, theta] representing the robot pose

        Returns:
            list[tuple[ca.MX, ca.MX]]: List of symbolic (x, y) points defining the robot shape
        """
        pass

    @abstractmethod
    def sdf_constraints(self, opti: ca.Opti, pose: ca.MX, sdf_func: callable, margin: float) -> None:
        """
        Adds symbolic obstacle avoidance constraints to the optimization problem.

        Parameters:
            opti (ca.Opti): The CasADi optimization object
            pose (ca.MX): The symbolic robot pose [x, y, theta]
            sdf_func (Callable): A symbolic function sdf(x, y) returning signed distance
            margin (float): Minimum required clearance from obstacles
        """
        pass

    @abstractmethod
    def goal_cost(self, pose: ca.MX, goal_vec: ca.MX) -> ca.MX:
        """
        Computes the symbolic cost for reaching the goal (can use center, any point, or full shape).

        Parameters:
            pose (ca.MX): The symbolic robot pose [x, y, theta]
            goal_vec (ca.MX): Symbolic vector of goal position [x, y]

        Returns:
            ca.MX: A symbolic expression representing the goal-reaching cost
        """
        pass

    def draw(self, ax, pose: ca.MX, **kwargs):
        """Matplotlib drawing using pose (x, y, theta if available)"""
        ax.plot(pose[0], pose[1], "bo")


class DotGeometry(IRobotGeometry):
    def transform(self, pose: ca.MX) -> list[tuple[ca.MX, ca.MX]]:
        return [(pose[0], pose[1])]

    def sdf_constraints(self, opti: ca.Opti, pose: ca.MX, sdf_func: callable, margin: float) -> None:
        x, y = pose[0], pose[1]
        opti.subject_to(sdf_func(x, y) >= margin)

    def goal_cost(self, pose: ca.MX, goal_vec: ca.MX) -> ca.MX:
        return ca.sumsqr(pose[0:2] - goal_vec)


class PolygonGeometry(IRobotGeometry):
    def __init__(self, body_points: list[tuple[float, float]], goal_mode: GoalMode = GoalMode.CENTER):
        self.body_points = body_points
        self.goal_mode = goal_mode

    def transform(self, pose: ca.MX) -> list[tuple[ca.MX, ca.MX]]:
        x, y, theta = pose[0], pose[1], pose[2]
        return [
            (
                x + ca.cos(theta) * px - ca.sin(theta) * py,
                y + ca.sin(theta) * px + ca.cos(theta) * py
            )
            for px, py in self.body_points
        ]

    def sdf_constraints(self, opti: ca.Opti, pose: ca.MX, sdf_func: callable, margin: float) -> None:
        for px, py in self.transform(pose):
            opti.subject_to(sdf_func(px, py) >= margin)

    def goal_cost(self, pose: ca.MX, goal_vec: ca.MX) -> ca.MX:
        shape_pts = self.transform(pose)
        if self.goal_mode == GoalMode.CENTER:
            return ca.sumsqr(pose[0:2] - goal_vec)
        return ca.mmin([ca.sumsqr(ca.vertcat(px, py) - goal_vec) for px, py in shape_pts])


class RectangleGeometry(PolygonGeometry):
    def __init__(self, length: float, width: float, goal_mode: GoalMode = GoalMode.CENTER):
        l = length / 2
        w = width / 2
        body_points = [(-l, -w), (-l, w), (l, w), (l, -w)]
        super().__init__(body_points, goal_mode)


class TriangleGeometry(PolygonGeometry):
    def __init__(self, length: float, width: float, goal_mode: GoalMode = GoalMode.CENTER):
        tip = (length / 2, 0.0)
        base_left = (-length / 2, -width / 2)
        base_right = (-length / 2, width / 2)
        body_points = [tip, base_right, base_left]
        super().__init__(body_points, goal_mode)
