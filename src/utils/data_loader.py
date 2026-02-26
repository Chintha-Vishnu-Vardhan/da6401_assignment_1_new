"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def _one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    y = y.astype(int).ravel()
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


def load_dataset(
    dataset: str,
    validation_split: float = 0.1,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load and preprocess MNIST or Fashion-MNIST using scikit-learn's fetch_openml.

    Args:
        dataset: 'mnist' or 'fashion' / 'fashion_mnist'
        validation_split: Fraction of training data used for validation.

    Returns:
        X_train: Training images, shape (N_train, 784)
        y_train_onehot: Training labels one-hot, shape (N_train, 10)
        X_val: Validation images, shape (N_val, 784)
        y_val_onehot: Validation labels one-hot, shape (N_val, 10)
        X_test: Test images, shape (N_test, 784)
        y_test_onehot: Test labels one-hot, shape (N_test, 10)
        y_test_labels: Test labels as integers, shape (N_test,)
    """
    name = dataset.lower()
    if name == "mnist":
        mnist_data = fetch_openml("mnist_784", version=1, as_frame=False)
    elif name in ("fashion", "fashion_mnist", "fashion-mnist"):
        mnist_data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    X = mnist_data.data.astype("float32") / 255.0
    y = mnist_data.target.astype(int)

    # MNIST has 70k samples; we mimic keras split: 60k train, 10k test
    X_train_full, X_test = X[:60000], X[60000:]
    y_train_full, y_test = y[:60000], y[60000:]

    # Train/validation split from training portion
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_split,
        random_state=42,
        stratify=y_train_full,
    )

    y_train_onehot = _one_hot_encode(y_train_labels, num_classes=10)
    y_val_onehot = _one_hot_encode(y_val_labels, num_classes=10)
    y_test_onehot = _one_hot_encode(y_test, num_classes=10)

    return (
        X_train,
        y_train_onehot,
        X_val,
        y_val_onehot,
        X_test,
        y_test_onehot,
        y_test,
    )


"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
