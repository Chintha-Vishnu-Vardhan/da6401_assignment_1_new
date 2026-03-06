"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

from typing import Tuple

import numpy as np

from .activations import softmax


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Match reference exact math
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    batch_size = y_true.shape[0]
    # Apply batch size division and factor of 2 here
    return 2.0 * (y_pred - y_true) / batch_size


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
    batch_size = y_true.shape[0]
    return probs - y_true / batch_size
