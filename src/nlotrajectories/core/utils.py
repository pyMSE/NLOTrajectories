import casadi as ca
import numpy as np


def soft_abs(x: ca.MX, epsilon: float = 1e-3):
    return ca.sqrt(x**2 + epsilon)


def soft_max(args: list[ca.MX], alpha: float = 10.0) -> ca.MX:
    """
    Smooth approximation of max(args) using log-sum-exp trick.
    Larger alpha = sharper approximation.
    """
    args = ca.vertcat(*args)
    return 1.0 / alpha * ca.log(ca.sum1(ca.exp(alpha * args)))


def soft_min(args, alpha: float = 10.0):
    """
    Smooth elementwise approximation of min(args) using log-sum-exp trick.
    Supports both NumPy arrays and CasADi MX arrays.
    args: list of arrays of the same shape
    Returns: array of the same shape
    """
    if isinstance(args[0], np.ndarray):
        stacked = np.stack(args, axis=0)  # shape (N_obstacles, H, W)
        return -1.0 / alpha * np.log(np.sum(np.exp(-alpha * stacked), axis=0))
    elif isinstance(args[0], ca.MX):
        stacked = ca.hcat([ca.reshape(a, -1, 1) for a in args])  # (H*W, N_obs)
        result = -1.0 / alpha * ca.log(ca.sum2(ca.exp(-alpha * stacked)))
        return ca.reshape(result, args[0].shape[0], args[0].shape[1])
    else:
        raise TypeError("soft_min only supports lists of np.ndarray or casadi.MX")
