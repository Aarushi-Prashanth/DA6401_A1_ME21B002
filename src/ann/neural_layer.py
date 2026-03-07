"""
neural_layer.py - Dense layer implementation using only NumPy.
Exposes self.grad_W and self.grad_b after every backward() call.
"""

import numpy as np


class DenseLayer:
    """
    Fully connected (dense) layer.
    W shape: (input_size, output_size)
    b shape: (1, output_size)
    After backward(), self.grad_W and self.grad_b are populated.
    """

    def __init__(self, input_size: int, output_size: int, weight_init: str = "xavier"):
        self.input_size = input_size
        self.output_size = output_size
        self.weight_init = weight_init

        self.W, self.b = self._initialize_weights(weight_init)

        # Gradients - exposed for autograder
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self._input_cache = None

    def _initialize_weights(self, method: str):
        if method == "xavier":
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            W = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif method == "random":
            W = np.random.randn(self.input_size, self.output_size) * 0.01
        elif method == "zeros":
            W = np.zeros((self.input_size, self.output_size))
        else:
            raise ValueError(f"Unknown weight_init: {method}")
        b = np.zeros((1, self.output_size))
        return W, b

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._input_cache = X
        return X @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output: gradient w.r.t. layer output, shape (batch_size, output_size)
        NOTE: NO division by batch_size here - that is handled by the loss function.
        """
        # Gradient w.r.t. weights: average over batch
        self.grad_W = self._input_cache.T @ grad_output

        # Gradient w.r.t. biases: sum over batch
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # Gradient w.r.t. input
        grad_input = grad_output @ self.W.T
        return grad_input
