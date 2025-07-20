from abc import ABC, abstractmethod
from enum import Enum

import casadi as ca


class Dynamics(str, Enum):
    POINT_1ST = "point_1st"
    POINT_2ND = "point_2nd"
    UNICYCLE = "unicycle"
    UNICYCLE_2ND = "unicycle_2nd"
    ACKERMANN = "ackermann"
    ACKERMANN_2ND = "ackermann_2nd"


class IRobotDynamics(ABC):
    @abstractmethod
    def state_dim(self) -> int:
        """Returns the dimensionality of the state vector."""
        pass

    @abstractmethod
    def control_dim(self) -> int:
        """Returns the dimensionality of the control vector."""
        pass

    @abstractmethod
    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Returns symbolic expression for f(x, u) = dx/dt"""
        pass


class PointMass1stOrder(IRobotDynamics):
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    def control_dim(self) -> int:
        return 2  # vx, vy

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        return ca.vertcat(u[0], u[1], 0, 0)  # dx/dt = u


class PointMass2ndOrder(IRobotDynamics):
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    def control_dim(self) -> int:
        return 2  # ax, ay

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        vx = x[2]
        vy = x[3]
        ax = u[0]
        ay = u[1]
        return ca.vertcat(vx, vy, ax, ay)


class Unicycle(IRobotDynamics):
    def state_dim(self) -> int:
        return 3  # x, y, theta

    def control_dim(self) -> int:
        return 2  # v, omega

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        theta = x[2]
        v = u[0]
        omega = u[1]
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega
        return ca.vertcat(dx, dy, dtheta)


class Unicycle2ndOrder(IRobotDynamics):
    def state_dim(self) -> int:
        return 5  # x, y, theta, v, omega

    def control_dim(self) -> int:
        return 2  # a (linear acceleration), alpha (angular acceleration)

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        theta = x[2]
        v = x[3]
        omega = x[4]
        a = u[0]
        alpha = u[1]

        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega
        dv = a
        domega = alpha

        return ca.vertcat(dx, dy, dtheta, dv, domega)


class Ackermann(IRobotDynamics):
    def __init__(self, wheelbase: float = 0.1):
        self.wheelbase = wheelbase  # Wheelbase of the vehicle, default is 1.0

    def state_dim(self) -> int:
        return 4  # x, y, theta: global vehicle heading, psi: steering angle

    def control_dim(self) -> int:
        return 2  # v, psi_dot: speed steering angle

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        theta = x[2]
        psi = x[3]
        v = u[0]
        psi_dot = u[1]
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v * ca.tan(psi)) / self.wheelbase  # wheel base is assumed to be 1
        dpsi = psi_dot
        return ca.vertcat(dx, dy, dtheta, dpsi)


class Ackermann2ndOrder(IRobotDynamics):
    def __init__(self, wheelbase: float = 1.0):
        self.wheelbase = wheelbase

    def state_dim(self) -> int:
        return 7  # x, y, theta, psi, v, omega, psi_dot

    def control_dim(self) -> int:
        return 2  # a (linear acceleration), alpha (steering acceleration)

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        theta = x[2]
        psi = x[3]
        v = x[4]
        # omega = x[5]
        psi_dot = x[6]
        a = u[0]
        alpha = u[1]

        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = (v * ca.tan(psi)) / self.wheelbase
        dpsi = psi_dot
        domega = 1 / self.wheelbase * (dpsi / (1 + psi**2) * v + ca.tan(psi) * a)  # assuming wheel base is 1
        dv = a
        ddpsi = alpha

        return ca.vertcat(dx, dy, dtheta, dpsi, domega, dv, ddpsi)


DYNAMICS_CLASS_MAP = {
    Dynamics.POINT_1ST: PointMass1stOrder,
    Dynamics.POINT_2ND: PointMass2ndOrder,
    Dynamics.UNICYCLE: Unicycle,
    Dynamics.UNICYCLE_2ND: Unicycle2ndOrder,
    Dynamics.ACKERMANN: Ackermann,
    Dynamics.ACKERMANN_2ND: Ackermann2ndOrder,
}
