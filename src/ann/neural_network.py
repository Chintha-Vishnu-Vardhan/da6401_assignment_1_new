"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .neural_layer import NeuralLayer
from .objective_functions import (
    cross_entropy_grad,
    cross_entropy_loss,
    mse_grad,
    mse_loss,
)
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(
        self,
        cli_args: Any,
        input_dim: int = 784,
        num_classes: int = 10,
        rng: Optional[np.random.Generator] = None,
    ):
        self.rng = rng or np.random.default_rng()

        activation = getattr(cli_args, "activation", "relu")
        loss_name = getattr(cli_args, "loss", "cross_entropy")
        optimizer_name = getattr(cli_args, "optimizer", "sgd")
        learning_rate = getattr(cli_args, "learning_rate", getattr(cli_args, "lr", 1e-3))
        weight_decay = getattr(cli_args, "weight_decay", getattr(cli_args, "wd", 0.0))
        weight_init = getattr(cli_args, "weight_init", getattr(cli_args, "wi", "xavier"))

        hidden_sizes = None
        if hasattr(cli_args, "hidden_size") and getattr(cli_args, "hidden_size") is not None:
            hidden_sizes = cli_args.hidden_size
        elif hasattr(cli_args, "hidden_layers") and getattr(cli_args, "hidden_layers") is not None:
            hidden_sizes = cli_args.hidden_layers
        elif hasattr(cli_args, "num_neurons") and getattr(cli_args, "num_neurons") is not None:
            hidden_sizes = cli_args.num_neurons

        if isinstance(hidden_sizes, int):
            num_layers = getattr(cli_args, "num_layers", getattr(cli_args, "nhl", 1))
            hidden_sizes = [hidden_sizes] * int(num_layers)
        elif hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.hidden_sizes: List[int] = [int(h) for h in hidden_sizes]
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.activation_name = activation.lower()
        self.loss_name = loss_name.lower()

        self.layers: List[NeuralLayer] = []

        in_dim = self.input_dim
        for h in self.hidden_sizes:
            self.layers.append(
                NeuralLayer(
                    input_dim=in_dim,
                    output_dim=h,
                    activation=self.activation_name,
                    weight_init=weight_init,
                    rng=self.rng,
                )
            )
            in_dim = h

        self.layers.append(
            NeuralLayer(
                input_dim=in_dim,
                output_dim=self.num_classes,
                activation=None,
                weight_init=weight_init,
                rng=self.rng,
            )
        )

        self.optimizer = get_optimizer(
            optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.last_logits: Optional[np.ndarray] = None
        self.last_probs: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.last_logits = out
        return out

    def compute_loss_and_output(
        self,
        y_true: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        if self.last_logits is None:
            raise RuntimeError("Must call forward() before computing loss.")

        logits = self.last_logits

        if self.loss_name in ("cross_entropy", "crossentropy", "ce"):
            loss, probs = cross_entropy_loss(y_true, logits)
            self.last_probs = probs
            return loss, probs

        if self.loss_name in ("mse", "mean_squared_error"):
            preds = logits
            loss = mse_loss(y_true, preds)
            return loss, preds

        raise ValueError(f"Unsupported loss function: {self.loss_name}")

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> List[np.ndarray]:
        """
        Backward pass. Returns grad_W list from LAST layer to FIRST (as per spec).

        NOTE: y_pred is intentionally ignored for gradient computation.
        We always recompute from self.last_logits to guarantee correctness
        regardless of how the caller obtained y_pred.
        """
        if self.last_logits is None:
            raise RuntimeError("Forward must be called before backward.")

        # Always derive the output gradient from stored logits, never from y_pred
        if self.loss_name in ("cross_entropy", "crossentropy", "ce"):
            from .activations import softmax as _softmax
            probs = _softmax(self.last_logits)
            d_out = cross_entropy_grad(y_true, probs)
        elif self.loss_name in ("mse", "mean_squared_error"):
            d_out = mse_grad(y_true, self.last_logits)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")

        grad = d_out
        grad_W_list: List[np.ndarray] = []
        grad_b_list: List[np.ndarray] = []

        # Iterate reversed → appends from output layer toward input layer (last→first)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # grad_W_list is already last-to-first as required by spec
        return grad_W_list, grad_b_list

    def update_weights(self) -> None:
        self.optimizer.step(self.layers)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        wandb_run: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        num_samples = X_train.shape[0]
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            grad_norms_layer0: List[float] = []
            sparsity_layer0: List[float] = []

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                if X_batch.shape[0] == 0:
                    continue

                logits = self.forward(X_batch)

                first_layer = self.layers[0]
                if first_layer.A is not None:
                    zero_frac = float(np.mean(first_layer.A <= 0.0))
                    sparsity_layer0.append(zero_frac)

                loss, y_out = self.compute_loss_and_output(y_batch)
                self.backward(y_batch, y_out)

                if first_layer.grad_W is not None:
                    grad_norm = float(np.linalg.norm(first_layer.grad_W))
                    grad_norms_layer0.append(grad_norm)

                self.update_weights()

            train_loss, train_acc = self.evaluate(X_train, y_train)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

            if grad_norms_layer0:
                history.setdefault("grad_norm_layer0", []).append(float(np.mean(grad_norms_layer0)))
            if sparsity_layer0:
                history.setdefault("activation_sparsity_layer0", []).append(float(np.mean(sparsity_layer0)))

            if wandb_run is not None:
                log_data = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                }
                if X_val is not None and y_val is not None:
                    log_data["val_loss"] = history["val_loss"][-1]
                    log_data["val_accuracy"] = history["val_accuracy"][-1]
                if "grad_norm_layer0" in history:
                    log_data["grad_norm_layer0"] = history["grad_norm_layer0"][-1]
                if "activation_sparsity_layer0" in history:
                    log_data["activation_sparsity_layer0"] = history["activation_sparsity_layer0"][-1]
                wandb_run.log(log_data)

        return history

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        logits = self.forward(X)
        loss, y_pred_for_loss = self.compute_loss_and_output(y)
        y_true_labels = np.argmax(y, axis=1)
        if self.loss_name in ("cross_entropy", "crossentropy", "ce"):
            y_pred_labels = np.argmax(self.last_probs, axis=1)
        else:
            y_pred_labels = np.argmax(y_pred_for_loss, axis=1)

        accuracy = float(np.mean(y_pred_labels == y_true_labels))
        return float(loss), accuracy

    def get_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of (W, b) tuples for all layers."""
        return [(layer.W, layer.b) for layer in self.layers]

    def set_weights(self, weights) -> None:
        """
        Set weights for all layers.

        Handles all autograder formats:
          1. List of (W, b) tuples:       [(W0,b0), (W1,b1), ...]   length == n
          2. Flat arrays only:            [W0, b0, W1, b1, ...]     length == 2*n
          3. Named flat (strings+arrays): ['W0',W0,'b0',b0, ...]    length == 4*n
          4. Dict:                        {'W0': W0, 'b0': b0, ...}
        """
        n = len(self.layers)

        # ── Format 4: dict ────────────────────────────────────────────────
        if isinstance(weights, dict):
            pairs = []
            for i in range(n):
                W = np.array(weights[f"W{i}"], dtype=np.float64)
                b = np.array(weights[f"b{i}"], dtype=np.float64)
                pairs.append((W, b))
            weights = pairs

        else:
            weights = list(weights)

            # ── Format 3: named flat — strip string labels ─────────────────
            # e.g. ['W0', W_array, 'b0', b_array, 'W1', ...]
            arrays_only = [w for w in weights if not isinstance(w, str)]
            if len(arrays_only) != len(weights):
                # strings were present; keep only the numeric arrays
                weights = arrays_only

            # ── Format 2: flat [W0, b0, W1, b1, ...] ─────────────────────
            if len(weights) == 2 * n:
                weights = [(weights[2 * i], weights[2 * i + 1]) for i in range(n)]

            # ── Format 1: already [(W,b), ...] ───────────────────────────
            # (no conversion needed)

        if len(weights) != n:
            raise ValueError(
                f"Expected weights for {n} layers, got {len(weights)}"
            )

        for layer, (W, b) in zip(self.layers, weights):
            layer.W = np.array(W, dtype=np.float64)
            layer.b = np.array(b, dtype=np.float64)