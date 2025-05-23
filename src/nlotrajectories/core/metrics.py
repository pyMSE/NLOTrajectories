import numpy as np


def mse(target, prediction):
    return 0.0


def iou(target: np.ndarray, prediction: np.ndarray, threshold: float = 0.0):
    """
    Compute the intersection over union (IoU) metric.

    Args:
        target (np.ndarray): Ground truth SDF grid.
        prediction (np.ndarray): Predicted SDF grid.
        threshold (float): Threshold to determine occupancy from SDF (default is 0.0).

    Returns:
        float: IoU score between the predicted and ground truth occupancies.
    """
    target_mask = target < threshold
    prediction_mask = prediction < threshold

    intersection = np.logical_and(target_mask, prediction_mask).sum()
    union = np.logical_or(target_mask, prediction_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou


def hausdorf(target, prediction):
    return 0.0


def chamfer(target, prediction):
    return 0.0