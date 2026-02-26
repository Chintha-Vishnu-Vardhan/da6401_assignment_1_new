"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

from typing import Optional

import numpy as np

from .activations import ACTIVATIONS, DERIVATIVES


class NeuralLayer:
    """
    Fully-connected (dense) neural network layer.

    Attributes exposed for autograder:
        W: Weight matrix of shape (input_dim, output_dim)
        b: Bias vector of shape (output_dim,)
        grad_W: Gradient of loss w.r.t. W, same shape as W
        grad_b: Gradient of loss w.r.t. b, same shape as b
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[str] = None,
        weight_init: str = "xavier",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation_name = activation
        self.weight_init = weight_init.lower() if weight_init is not None else "xavier"
        self.rng = rng or np.random.default_rng()

        self.W, self.b = self._init_parameters()

        # Placeholders for cache and gradients
        self.X: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None

        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None

    def _init_parameters(self):
        """Initialize weights and biases."""
        if self.weight_init == "zeros":
            W = np.zeros((self.input_dim, self.output_dim), dtype=np.float64)
        elif self.weight_init == "random":
            # Small random values
            limit = 0.01
            W = self.rng.uniform(
                low=-limit,
                high=limit,
                size=(self.input_dim, self.output_dim),
            )
        else:  # "xavier"
            # Xavier/Glorot uniform initialization
            limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
            W = self.rng.uniform(
                low=-limit,
                high=limit,
                size=(self.input_dim, self.output_dim),
            )

        b = np.zeros(self.output_dim, dtype=np.float64)
        return W, b

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
            X: Input of shape (batch_size, input_dim)

        Returns:
            Output of shape (batch_size, output_dim)
        """
        self.X = X
        self.Z = X @ self.W + self.b  # (batch_size, output_dim)

        if self.activation_name is None:
            self.A = self.Z
        else:
            activation_fn = ACTIVATIONS[self.activation_name]
            self.A = activation_fn(self.Z)

        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass for the layer.

        Args:
            dA: Gradient of loss w.r.t. layer output A,
                shape (batch_size, output_dim)

        Returns:
            dX: Gradient of loss w.r.t. layer input X,
                shape (batch_size, input_dim)

        Side effects:
            Updates self.grad_W and self.grad_b for optimizer access.
        """
        if self.X is None or self.Z is None:
            raise RuntimeError("Forward must be called before backward.")

        if self.activation_name is None:
            dZ = dA
        else:
            derivative_fn = DERIVATIVES[self.activation_name]
            dZ = dA * derivative_fn(self.Z)

        batch_size = self.X.shape[0]

        # Gradients averaged over the mini-batch
        self.grad_W = (self.X.T @ dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0) / batch_size

        dX = dZ @ self.W.T
        return dX
