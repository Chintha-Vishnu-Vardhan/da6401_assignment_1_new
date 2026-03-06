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
    Load and preprocess MNIST or Fashion-MNIST.

    Uses parser='liac-arff' so that pandas is not required.
    """
    name = dataset.lower()
    if name == "mnist":
        mnist_data = fetch_openml(
            "mnist_784", version=1, as_frame=False, parser="liac-arff"
        )
    elif name in ("fashion", "fashion_mnist", "fashion-mnist"):
        mnist_data = fetch_openml(
            "Fashion-MNIST", version=1, as_frame=False, parser="liac-arff"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    X = mnist_data.data.astype("float32") / 255.0
    y = mnist_data.target.astype(int)

    # Standard 60k/10k split
    X_train_full, X_test = X[:60000], X[60000:]
    y_train_full, y_test = y[:60000], y[60000:]

    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_split,
        random_state=42,
        stratify=y_train_full,
    )

    y_train_onehot = _one_hot_encode(y_train_labels, num_classes=10)
    y_val_onehot   = _one_hot_encode(y_val_labels,   num_classes=10)
    y_test_onehot  = _one_hot_encode(y_test,          num_classes=10)

    return (
        X_train,
        y_train_onehot,
        X_val,
        y_val_onehot,
        X_test,
        y_test_onehot,
        y_test,
    )