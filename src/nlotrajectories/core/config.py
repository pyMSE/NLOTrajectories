from pydantic import BaseModel

from nlotrajectories.core.dynamics import (
    Dynamics,
    PointMass1stOrder,
    PointMass2ndOrder,
    Unicycle,
)
from nlotrajectories.core.geometry import (
    DotGeometry,
    GoalMode,
    RectangleGeometry,
    Shape,
    TriangleGeometry,
)


class BodyConfig(BaseModel):
    shape: Shape
    dynamic: Dynamics
    goal_mode: GoalMode = GoalMode.CENTER
    length: float | None = None
    width: float | None = None

    def create_geometry(self):
        if self.shape == Shape.DOT:
            return DotGeometry()
        if self.shape == Shape.RECTANGLE:
            return RectangleGeometry(self.length, self.width, self.goal_mode)
        return TriangleGeometry(self.length, self.width, self.goal_mode)

    def create_dynamics(self):
        if self.dynamic == Dynamics.POINT_1ST:
            return PointMass1stOrder()
        if self.dynamic == Dynamics.POINT_2ND:
            return PointMass2ndOrder()
        return Unicycle()


class Config(BaseModel):
    body: BodyConfig
