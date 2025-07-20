from typing import List, Literal, Union
import numpy as np
from pydantic import BaseModel, Field, RootModel

from nlotrajectories.core.dynamics import DYNAMICS_CLASS_MAP, Dynamics, IRobotDynamics
from nlotrajectories.core.geometry import (
    DotGeometry,
    GoalMode,
    RectangleGeometry,
    Shape,
    TriangleGeometry,
)
from nlotrajectories.core.sdf.casadi import (
    CircleObstacle,
    IObstacle,
    MultiObstacle,
    PolygonObstacle,
    SquareObstacle,
    EllipticRingObstacle,
    TrapezoidObstacle,
    ConvexSObstacle
)


class BodyConfig(BaseModel):
    shape: Shape
    dynamic: Dynamics
    goal_mode: GoalMode = GoalMode.CENTER
    length: float | None = None
    width: float | None = None
    wheelbase: float | None = None
    start_state: list[float]
    goal_state: list[float]
    control_bounds: list[tuple[float, float]]

    def create_geometry(self):
        if self.shape == Shape.DOT:
            return DotGeometry()
        if self.shape == Shape.RECTANGLE:
            return RectangleGeometry(self.length, self.width, self.goal_mode)
        return TriangleGeometry(self.length, self.width, self.goal_mode)

    def create_dynamics(self) -> IRobotDynamics:
        cls = DYNAMICS_CLASS_MAP.get(self.dynamic)
        if self.dynamic == Dynamics.ACKERMANN or self.dynamic == Dynamics.ACKERMANN_2ND:
            return cls(self.wheelbase)
        if cls is None:
            raise ValueError(f"Unknown dynamics type: {self.dynamic}")
        return cls()


class CircleObstacleConfig(BaseModel):
    type: Literal["circle"]
    center: tuple[float, float]
    radius: float
    margin: float = 0.0

    def to_obstacle(self) -> IObstacle:
        return CircleObstacle(center=self.center, radius=self.radius, margin=self.margin)


class SquareObstacleConfig(BaseModel):
    type: Literal["square"]
    center: tuple[float, float]
    size: float
    margin: float = 0.0

    def to_obstacle(self) -> IObstacle:
        return SquareObstacle(center=self.center, size=self.size, margin=self.margin)


class PolygonObstacleConfig(BaseModel):
    type: Literal["polygon"]
    points: list[tuple[float, float]]
    margin: float = 0.0

    def to_obstacle(self) -> IObstacle:
        return PolygonObstacle(points=self.points, margin=self.margin)


class EllipticRingObstacleConfig(BaseModel):
    type: Literal["elliptical_ring"]
    center: tuple[float, float]
    semi_axes: tuple[float, float]
    width: float
    angle: float = np.pi
    margin: float = 0.0
    rotation: float = 0.0
    num_arc_points: int = 15

    def to_obstacle(self) -> IObstacle:
        return EllipticRingObstacle(
            center=self.center,
            semi_axes=self.semi_axes,
            width=self.width,
            angle=self.angle,
            margin=self.margin,
            rotation=self.rotation,
            num_arc_points=self.num_arc_points
        )


class DiscreteSObstacleConfig(BaseModel):
    type: Literal["discr_s"]
    center: tuple[float, float]
    semi_axes: tuple[float, float]
    width: float
    angle: float = np.pi
    margin: float = 0.0
    rotation: float = 0.0
    num_arc_points: int = 30

    def to_obstacle(self) -> IObstacle:
        return ConvexSObstacle(
            center=self.center,
            semi_axes=self.semi_axes,
            width=self.width,
            angle=self.angle,
            margin=self.margin,
            rotation=self.rotation,
            num_arc_points=self.num_arc_points
        )

class TrapezoidObstacleConfig(BaseModel):
    type: Literal["trapezoid"]
    points: list[tuple[float, float]]
    margin: float = 0.0

    def to_obstacle(self) -> IObstacle:
        return TrapezoidObstacle(points=self.points, margin=self.margin)


ObstacleConfigs = list[CircleObstacleConfig | SquareObstacleConfig | PolygonObstacleConfig | EllipticRingObstacleConfig | TrapezoidObstacleConfig | DiscreteSObstacleConfig]


class ObstacleConfig(RootModel[ObstacleConfigs]):
    model_config = {"arbitrary_types_allowed": True, "smart_union": True, "discriminator": "type"}

    def to_obstacles(self) -> MultiObstacle:
        return MultiObstacle([ob.to_obstacle() for ob in self.root])


class DefaultInitializerConfig(BaseModel):
    mode: Literal["default"] = Field("default", description="Casadi default init")


class LinearInitializerConfig(BaseModel):
    mode: Literal["linear"] = Field("linear", description="Linear interpolation init")


class RRTInitializerConfig(BaseModel):
    mode: Literal["rrt"] = Field("rrt", description="RRT init")
    rrt_bounds: List[List[float]] = Field(
        ..., description="[[xmin, ymin], [xmax, ymax]]，for RRT random sample boundary"
    )
    step_size: float = Field(0.05, ge=1e-6, description="RRT step size")
    max_iter: int = Field(1000, ge=1, description="RRT maximum iteration")
    margin: float = Field(0.01, ge=0.0, description="minimum sign distance")

    class Config:
        smart_union = True
        discriminator = "mode"


class InitializerConfig(
    RootModel[list[Union[DefaultInitializerConfig, LinearInitializerConfig, RRTInitializerConfig]]]
):
    model_config = {
        "arbitrary_types_allowed": True,
        "smart_union": True,
        "discriminator": "mode",
    }

    @property
    def choice(self) -> LinearInitializerConfig | RRTInitializerConfig:
        return self.root[0]


class SolverConfig(BaseModel):
    N: int = Field(default=20, ge=1, description="Number of steps")
    dt: float = 0.1
    use_slack: bool = False
    slack_penalty: float | None = Field(default=1000, ge=1)
    use_smooth: bool = Field(False, description="Enable smoothing penalty on control rate du")
    smooth_weight: float = Field(10.0, ge=0, description="Weight of the du smoothing penalty in the cost")
    mode: Literal["casadi", "l4casadi"]
    initializer: InitializerConfig = Field(
        default_factory=lambda: InitializerConfig(root=[{"mode": "linear"}]), description="Casadi Initializer"
    )
    enforce_heading: bool = True
    type: Literal["ipopt","sqpmethod"]


class ModelConfig(BaseModel):
    type: Literal["mlp", "fourier", "siren"] = "mlp"
    hidden_dim: int = Field(default=64, ge=1)
    num_hidden_layers: int = Field(default=3, ge=1)
    activation_function: str = "ReLU"
    omega_0: float = 30.0  # Only used for SIREN
    n_samples: int = 200_000
    boundary_fraction: float = 0.3
    surface_loss_weight: float = Field(1.0, ge=0.0, description="Weight of the surface loss")
    eikonal_loss_weight: float = Field(1.0, ge=0.0, description="Weight of the eikonal loss")


class Config(BaseModel):
    body: BodyConfig
    obstacles: ObstacleConfig
    solver: SolverConfig
    model: ModelConfig

    def get_obstacles(self) -> MultiObstacle:
        return self.obstacles.to_obstacles()
