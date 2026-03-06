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
        if self.weight_init == "zeros":
            W = np.zeros((self.input_dim, self.output_dim), dtype=np.float64)
        elif self.weight_init == "random":
            # FIX: Use normal distribution
            W = np.random.randn(self.input_dim, self.output_dim) * 0.01
        else:  # "xavier"
            # FIX: Use normal distribution with standard deviation
            std = np.sqrt(2.0 / (self.input_dim + self.output_dim))
            W = np.random.randn(self.input_dim, self.output_dim) * std

        b = np.zeros((1, self.output_dim), dtype=np.float64)
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
        # @ does matrix multiplication

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
        self.grad_W = (self.X.T @ dZ)
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        dX = dZ @ self.W.T
        return dX
