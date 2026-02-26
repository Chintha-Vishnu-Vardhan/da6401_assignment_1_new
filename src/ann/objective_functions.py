"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

from typing import Tuple

import numpy as np

from .activations import softmax


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error loss.

    Args:
        y_true: One-hot or continuous targets, shape (batch_size, num_classes)
        y_pred: Predictions (typically logits or activations from last layer),
                shape (batch_size, num_classes)

    Returns:
        Scalar MSE loss.
    """
    diff = y_pred - y_true
    # 1/2 factor is conventional and simplifies derivative
    return 0.5 * np.mean(np.sum(diff ** 2, axis=1))


def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss w.r.t. predictions.

    Note: This returns gradient per sample; averaging over the batch is handled
    inside the layer backward pass.
    """
    return y_pred - y_true


def cross_entropy_loss(
    y_true: np.ndarray,
    logits: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss for multi-class classification with softmax outputs.

    Args:
        y_true: One-hot encoded true labels, shape (batch_size, num_classes)
        logits: Raw output logits from the network,
                shape (batch_size, num_classes)
        eps: Small constant for numerical stability.

    Returns:
        (loss, probabilities):
            loss: scalar cross-entropy loss
            probabilities: softmax(logits), shape (batch_size, num_classes)
    """
    probs = softmax(logits)
    log_probs = np.log(probs + eps)
    loss = -np.mean(np.sum(y_true * log_probs, axis=1))
    return loss, probs


def cross_entropy_grad(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Gradient of cross-entropy loss w.r.t. logits when using softmax.

    Args:
        y_true: One-hot encoded true labels, shape (batch_size, num_classes)
        probs: Softmax probabilities, shape (batch_size, num_classes)

    Returns:
        Gradient w.r.t. logits, per-sample (batch averaging is done in layers).
    """
    return probs - y_true
