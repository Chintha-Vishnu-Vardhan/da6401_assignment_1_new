"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets via keras.datasets (fast, no pandas required).
"""

from typing import Tuple

import numpy as np
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

    Uses keras.datasets for fast, cached, pandas-free loading.

    Returns:
        X_train, y_train_onehot, X_val, y_val_onehot,
        X_test, y_test_onehot, y_test_labels
    """
    name = dataset.lower()

    # --- Load via keras (cached locally, no internet after first use) ---
    try:
        if name == "mnist":
            from keras.datasets import mnist
            (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
        elif name in ("fashion", "fashion_mnist", "fashion-mnist"):
            from keras.datasets import fashion_mnist
            (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    except ImportError:
        # Fallback: scikit-learn fetch_openml with liac-arff (no pandas needed)
        from sklearn.datasets import fetch_openml
        if name == "mnist":
            data = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        elif name in ("fashion", "fashion_mnist", "fashion-mnist"):
            data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        X = data.data.astype("float32") / 255.0
        y = data.target.astype(int)
        X_train_full = X[:60000].reshape(-1, 28, 28)
        X_test = X[60000:].reshape(-1, 28, 28)
        y_train_full, y_test = y[:60000], y[60000:]

    # --- Normalise to [0, 1] and flatten to (N, 784) ---
    X_train_full = X_train_full.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train_full = X_train_full.reshape(len(X_train_full), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # --- Train / validation split ---
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_split,
        random_state=42,
        stratify=y_train_full,
    )

    return (
        X_train,
        _one_hot_encode(y_train_labels, 10),
        X_val,
        _one_hot_encode(y_val_labels, 10),
        X_test,
        _one_hot_encode(y_test, 10),
        y_test.astype(int),
    )