"""
Gradient checking utility for the NumPy MLP.

This script:
- Builds a tiny network with your existing classes.
- Runs a forward + backward pass to get analytic gradients.
- Computes numerical gradients for a subset of parameters via finite differences.
- Reports the max absolute difference per checked parameter.

Run (from repo root):
    python -m src.check_gradients
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from ann.neural_network import NeuralNetwork


@dataclass
class GradCheckResult:
    param_name: str
    max_abs_diff: float
    mean_abs_diff: float


def _build_toy_network() -> Tuple[NeuralNetwork, np.ndarray, np.ndarray]:
    """
    Build a very small network and toy data for gradient checking.
    """

    class DummyArgs:
        # match your CLI attributes
        activation = "relu"
        loss = "cross_entropy"
        optimizer = "sgd"
        learning_rate = 1e-3
        weight_decay = 0.0
        weight_init = "xavier"
        hidden_size = [5]  # single hidden layer with 5 units

    rng = np.random.default_rng(42)
    input_dim = 4
    num_classes = 3

    model = NeuralNetwork(DummyArgs(), input_dim=input_dim, num_classes=num_classes, rng=rng)

    # Tiny batch
    X = rng.normal(size=(2, input_dim))
    y_labels = np.array([0, 2], dtype=int)
    y = np.zeros((y_labels.size, num_classes), dtype=float)
    y[np.arange(y_labels.size), y_labels] = 1.0

    return model, X, y


def _forward_loss(model: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> float:
    """
    Convenience wrapper: forward + loss scalar.
    """
    model.forward(X)
    loss, _ = model.compute_loss_and_output(y)
    return float(loss)


def numerical_grad_check(
    eps: float = 1e-5,
    max_params_per_layer: int = 5,
) -> List[GradCheckResult]:
    """
    Run numerical gradient checking on a tiny network.

    Args:
        eps: Finite-difference step size.
        max_params_per_layer: Number of weight entries to probe per layer.

    Returns:
        List of GradCheckResult with error statistics.
    """
    model, X, y = _build_toy_network()

    # Analytic gradients
    model.forward(X)
    _, y_out = model.compute_loss_and_output(y)
    model.backward(y, y_out)

    results: List[GradCheckResult] = []

    for layer_idx, layer in enumerate(model.layers):
        if layer.grad_W is None or layer.grad_b is None:
            raise RuntimeError("Backward must populate grad_W and grad_b.")

        # Flatten weight indices and sample a few positions to probe
        W = layer.W
        grad_W = layer.grad_W

        flat_indices = np.arange(W.size)
        rng = np.random.default_rng(0)
        rng.shuffle(flat_indices)
        flat_indices = flat_indices[: max_params_per_layer]

        diffs: List[float] = []

        for flat_idx in flat_indices:
            i, j = divmod(int(flat_idx), W.shape[1])

            original = W[i, j]

            # f(theta + eps)
            W[i, j] = original + eps
            loss_plus = _forward_loss(model, X, y)

            # f(theta - eps)
            W[i, j] = original - eps
            loss_minus = _forward_loss(model, X, y)

            # restore
            W[i, j] = original

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            ana_grad = grad_W[i, j]
            diffs.append(abs(num_grad - ana_grad))

        if diffs:
            diffs_arr = np.asarray(diffs, dtype=float)
            results.append(
                GradCheckResult(
                    param_name=f"layer_{layer_idx}_W",
                    max_abs_diff=float(diffs_arr.max()),
                    mean_abs_diff=float(diffs_arr.mean()),
                )
            )

        # Biases: check all, they are few
        b = layer.b
        grad_b = layer.grad_b
        b_diffs: List[float] = []

        for k in range(b.size):
            original_b = b[k]

            b[k] = original_b + eps
            loss_plus = _forward_loss(model, X, y)

            b[k] = original_b - eps
            loss_minus = _forward_loss(model, X, y)

            b[k] = original_b

            num_grad_b = (loss_plus - loss_minus) / (2.0 * eps)
            ana_grad_b = grad_b[k]
            b_diffs.append(abs(num_grad_b - ana_grad_b))

        if b_diffs:
            b_arr = np.asarray(b_diffs, dtype=float)
            results.append(
                GradCheckResult(
                    param_name=f"layer_{layer_idx}_b",
                    max_abs_diff=float(b_arr.max()),
                    mean_abs_diff=float(b_arr.mean()),
                )
            )

    return results


def main() -> None:
    results = numerical_grad_check()

    print("Gradient check results (finite differences vs analytic):")
    for r in results:
        print(
            f"{r.param_name:15s} | max_abs_diff = {r.max_abs_diff:.3e} "
            f"| mean_abs_diff = {r.mean_abs_diff:.3e}"
        )

    worst = max(results, key=lambda r: r.max_abs_diff)
    print("\nWorst discrepancy:")
    print(
        f"{worst.param_name}: max_abs_diff = {worst.max_abs_diff:.3e}, "
        f"mean_abs_diff = {worst.mean_abs_diff:.3e}"
    )
    print("\nIf max_abs_diff is comfortably below 1e-7, you are within the assignment tolerance.")


if __name__ == "__main__":
    main()

