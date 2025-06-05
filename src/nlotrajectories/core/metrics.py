from typing import Union
import numpy as np
import torch


def mse(sdf_target: Union[np.ndarray, torch.tensor], sdf_pred: Union[np.ndarray, torch.tensor]) -> float:
    """
    Compute the Mean Squared Error (MSE) metric.
    Args:
        target (np.ndarray): Ground truth SDF grid.
        prediction (np.ndarray): Predicted SDF grid.
    Returns:
        float: MSE score between the predicted and ground truth SDFs.
    """
    if sdf_target.shape != sdf_pred.shape:
        raise ValueError("Target and prediction must have the same shape.")
    return np.mean((sdf_target - sdf_pred) ** 2)


def iou(sdf_target: Union[np.ndarray, torch.tensor], sdf_pred: Union[np.ndarray, torch.tensor], threshold: float = 0.0) -> float:
    """
    Compute the intersection over union (IoU) metric.

    Args:
        sdf_target (np.ndarray): Ground truth SDF grid.
        sdf_pred (np.ndarray): Predicted SDF grid.
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

    return intersection / union


def chamfer(
    sdf_target: Union[np.ndarray, torch.tensor], sdf_pred: Union[np.ndarray, torch.tensor], X: Union[np.ndarray, torch.tensor], Y: Union[np.ndarray, torch.tensor], eps: float = 1e-2
) -> float | None:
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
        return None  # No surface points to compare

    # Compute the chamfer distance between pred_points and target_points
    dists = np.linalg.norm(pred_points[:, None] - target_points[None, :], axis=-1)
    chamfer_distance = (np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))) / 2
    return chamfer_distance


def surface_loss(sdf_target: Union[np.ndarray, torch.tensor], sdf_pred: Union[np.ndarray, torch.tensor], eps: float = 1e-2) -> float | None:
    """
    Compute the surface loss of the approximated sdf
    Args:
        sdf_target (np.ndarray): Ground truth SDF grid.
        sdf_pred (np.ndarray): Predicted SDF grid.
        X (np.ndarray): X coordinates of the grid.
        Y (np.ndarray): Y coordinates of the grid.
        eps (float): value to approximate the contour (default is 1e-2).
    Returns:
        float: Surface loss between the predicted and ground truth SDFs.
    """
    is_numpy = isinstance(sdf_pred, np.ndarray)
    if is_numpy:
        sdf_target_flat = sdf_target.flatten()
        sdf_pred_flat = sdf_pred.flatten()
        surface_mask = np.abs(sdf_target_flat) < eps
        pred_values_surface = sdf_pred_flat[surface_mask]
        if pred_values_surface.size > 0:
            return np.mean(pred_values_surface**2)
        else:
            return None
    else:
        sdf_target_flat = sdf_target.view(-1)
        sdf_pred_flat = sdf_pred.view(-1)
        surface_mask = sdf_target_flat.abs() < eps
        if surface_mask.any():
            pred_values_surface = sdf_pred_flat[surface_mask]
            return (pred_values_surface**2).mean()
        else:
            return None


def hausdorff(
    sdf_target: Union[np.ndarray, torch.tensor], sdf_pred: Union[np.ndarray, torch.tensor], X: Union[np.ndarray, torch.tensor], Y: Union[np.ndarray, torch.tensor], eps: float = 1e-2
) -> float | None:
    """
    Compute the Hausdorff distance between two SDF grids.

    Args:
        sdf_target (np.ndarray): Ground truth SDF grid.
        sdf_pred (np.ndarray): Predicted SDF grid.
        X (np.ndarray): X coordinates of the grid.
        Y (np.ndarray): Y coordinates of the grid.
        eps (float): Threshold to extract surface points (default is 1e-2).

    Returns:
        float: Hausdorff distance between the predicted and ground truth SDFs.
    """
    if sdf_target.shape != sdf_pred.shape:
        raise ValueError("Target and prediction must have the same shape.")

    # Flatten coordinates and SDFs
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)
    sdf_pred_flat = sdf_pred.flatten()
    sdf_target_flat = sdf_target.flatten()

    # Extract surface points
    pred_points = coords[np.abs(sdf_pred_flat) < eps]
    target_points = coords[np.abs(sdf_target_flat) < eps]

    if pred_points.shape[0] == 0 or target_points.shape[0] == 0:
        return None  # Cannot compute distance without surface points

    # Compute pairwise distances
    dists = np.linalg.norm(pred_points[:, None] - target_points[None, :], axis=-1)

    # Hausdorff distance: max(min(d(x, Y))) and max(min(d(y, X)))
    hd_pred_to_target = np.max(np.min(dists, axis=1))
    hd_target_to_pred = np.max(np.min(dists, axis=0))

    hausdorff_distance = max(hd_pred_to_target, hd_target_to_pred)
    return hausdorff_distance
