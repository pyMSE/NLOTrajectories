import numpy as np


def mse(target: np.array, prediction: np.array):
    # compute the MSE metric between the target and prediction
    """
    Compute the Mean Squared Error (MSE) metric.
    Args:
        target (np.ndarray): Ground truth SDF grid.
        prediction (np.ndarray): Predicted SDF grid.
    Returns:
        float: MSE score between the predicted and ground truth SDFs.
    """
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction must have the same shape.")
    mse_value = np.mean((target - prediction) ** 2)
    return mse_value


def iou(sdf_target: np.ndarray, sdf_pred: np.ndarray, threshold: float = 0.0):
    """
    Compute the intersection over union (IoU) metric.

    Args:
        target (np.ndarray): Ground truth SDF grid.
        prediction (np.ndarray): Predicted SDF grid.
        threshold (float): Threshold to determine occupancy from SDF (default is 0.0).

    Returns:
        float: IoU score between the predicted and ground truth occupancies.
    """
    target_mask = sdf_target < threshold
    prediction_mask = sdf_pred < threshold

    intersection = np.logical_and(target_mask, prediction_mask).sum()
    union = np.logical_or(target_mask, prediction_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou


def hausdorf(target, prediction):
    return 0.0


def chamfer(sdf_target: np.array, sdf_pred: np.array, X: np.array, Y: np.array, eps: float = 1e-2):
    # compute the chamfer distance between sdf_target and sdf_pred
    """
    Compute the Chamfer distance between two SDF grids.
    Args:
        sdf_target (np.ndarray): Ground truth SDF grid.
        sdf_pred (np.ndarray): Predicted SDF grid.
        X (np.ndarray): X coordinates of the grid.
        Y (np.ndarray): Y coordinates of the grid.
        eps (float): value to approximate the contour (default is 1e-2).
    Returns:
        float: Chamfer distance between the predicted and ground truth SDFs.
    """
    if sdf_target.shape != sdf_pred.shape:
        raise ValueError("Target and prediction must have the same shape.")

    # Get coordinates on the surface of each SDF
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)  # shape: (n_samples², 2)
    sdf_pred_flat = sdf_pred.flatten()
    sdf_target_flat = sdf_target.flatten()
    pred_points = coords[np.abs(sdf_pred_flat) < eps]  # points where the predicted SDF is close to zero
    target_points = coords[np.abs(sdf_target_flat) < eps]  # points where the target SDF is close to zero

    if pred_points.shape[0] == 0 or target_points.shape[0] == 0:
        return float("inf")  # No surface points to compare

    # Compute the chamfer distance between pred_points and target_points
    dists = np.linalg.norm(pred_points[:, None] - target_points[None, :], axis=-1)
    chamfer_distance = (np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))) / 2
    return chamfer_distance
