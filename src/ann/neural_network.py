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
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
            input_dim: Input feature dimension (default 784 for MNIST)
            num_classes: Number of output classes (default 10)
        """
        self.rng = rng or np.random.default_rng()

        # Defensive parsing of CLI arguments (support multiple possible names)
        activation = getattr(cli_args, "activation", "relu")
        loss_name = getattr(cli_args, "loss", "cross_entropy")
        optimizer_name = getattr(cli_args, "optimizer", "sgd")
        learning_rate = getattr(cli_args, "learning_rate", getattr(cli_args, "lr", 1e-3))
        weight_decay = getattr(cli_args, "weight_decay", getattr(cli_args, "wd", 0.0))
        weight_init = getattr(cli_args, "weight_init", getattr(cli_args, "wi", "xavier"))

        # Hidden layer configuration
        hidden_sizes = None
        if hasattr(cli_args, "hidden_size") and getattr(cli_args, "hidden_size") is not None:
            hidden_sizes = cli_args.hidden_size
        elif hasattr(cli_args, "hidden_layers") and getattr(cli_args, "hidden_layers") is not None:
            hidden_sizes = cli_args.hidden_layers
        elif hasattr(cli_args, "num_neurons") and getattr(cli_args, "num_neurons") is not None:
            hidden_sizes = cli_args.num_neurons

        if isinstance(hidden_sizes, int):
            # If a single value is given, replicate it across layers (if num_layers is provided)
            num_layers = getattr(cli_args, "num_layers", getattr(cli_args, "nhl", 1))
            hidden_sizes = [hidden_sizes] * int(num_layers)
        elif hidden_sizes is None:
            # Sensible default if nothing is provided
            hidden_sizes = [128, 128]

        self.hidden_sizes: List[int] = [int(h) for h in hidden_sizes]
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.activation_name = activation.lower()
        self.loss_name = loss_name.lower()

        # Construct layers (hidden + output)
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

        # Output layer: linear (no activation), logits of size num_classes
        self.layers.append(
            NeuralLayer(
                input_dim=in_dim,
                output_dim=self.num_classes,
                activation=None,
                weight_init=weight_init,
                rng=self.rng,
            )
        )

        # Optimizer
        self.optimizer = get_optimizer(
            optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Internal caches
        self.last_logits: Optional[np.ndarray] = None
        self.last_probs: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers.

        Args:
            X: Input data of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        self.last_logits = out
        return out

    def compute_loss_and_output(
        self,
        y_true: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute loss and the prediction used for backprop.

        For cross-entropy, applies softmax internally and returns probabilities.
        For MSE, uses raw logits as predictions.
        """
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

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation to compute gradients.

        Args:
            y_true: True labels (one-hot), shape (batch_size, num_classes)
            y_pred: Predicted outputs used for loss (either logits or probabilities),
                    shape (batch_size, num_classes)

        Returns:
            (grad_W_list, grad_b_list): lists of gradients for each layer.
        """
        if self.loss_name in ("cross_entropy", "crossentropy", "ce"):
            d_out = cross_entropy_grad(y_true, y_pred)
        elif self.loss_name in ("mse", "mean_squared_error"):
            d_out = mse_grad(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")

        grad = d_out
        grad_W_list: List[np.ndarray] = []
        grad_b_list: List[np.ndarray] = []

        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_W_list.insert(0, layer.grad_W)
            grad_b_list.insert(0, layer.grad_b)

        return grad_W_list, grad_b_list

    def update_weights(self) -> None:
        """
        Update weights using the optimizer.
        """
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
        """
        Train the network for specified epochs.

        Args:
            X_train: Training inputs, shape (N, input_dim)
            y_train: Training targets (one-hot), shape (N, num_classes)
            epochs: Number of epochs
            batch_size: Mini-batch size
            X_val: Optional validation inputs
            y_val: Optional validation targets (one-hot)
            wandb_run: Optional wandb run object for logging

        Returns:
            History dictionary with lists of losses and accuracies.
        """
        num_samples = X_train.shape[0]
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            # Per-epoch diagnostic accumulators
            grad_norms_layer0: List[float] = []
            sparsity_layer0: List[float] = []

            # Shuffle indices
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch training
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                if X_batch.shape[0] == 0:
                    continue

                logits = self.forward(X_batch)

                # Track activation sparsity for the first hidden layer (for W&B analyses)
                first_layer = self.layers[0]
                if first_layer.A is not None:
                    # Fraction of activations that are exactly zero or negative
                    zero_frac = float(np.mean(first_layer.A <= 0.0))
                    sparsity_layer0.append(zero_frac)

                loss, y_out = self.compute_loss_and_output(y_batch)
                self.backward(y_batch, y_out)

                # Track gradient norm for the first hidden layer (for vanishing gradient analysis)
                if first_layer.grad_W is not None:
                    grad_norm = float(np.linalg.norm(first_layer.grad_W))
                    grad_norms_layer0.append(grad_norm)

                self.update_weights()

            # End of epoch: evaluate on training (and validation if provided)
            train_loss, train_acc = self.evaluate(X_train, y_train)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

            # Aggregate diagnostics for the epoch
            if grad_norms_layer0:
                avg_grad_norm0 = float(np.mean(grad_norms_layer0))
                history.setdefault("grad_norm_layer0", []).append(avg_grad_norm0)
            if sparsity_layer0:
                avg_sparsity0 = float(np.mean(sparsity_layer0))
                history.setdefault("activation_sparsity_layer0", []).append(avg_sparsity0)

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
        """
        Evaluate the network on given data.

        Args:
            X: Inputs, shape (N, input_dim)
            y: Targets (one-hot), shape (N, num_classes)

        Returns:
            (loss, accuracy)
        """
        logits = self.forward(X)
        loss, y_pred_for_loss = self.compute_loss_and_output(y)
        y_true_labels = np.argmax(y, axis=1)
        if self.loss_name in ("cross_entropy", "crossentropy", "ce"):
            y_pred_labels = np.argmax(self.last_probs, axis=1)
        else:
            y_pred_labels = np.argmax(y_pred_for_loss, axis=1)

        accuracy = float(np.mean(y_pred_labels == y_true_labels))
        return float(loss), accuracy

