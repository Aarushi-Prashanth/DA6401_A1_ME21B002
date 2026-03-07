
import numpy as np

class Sigmoid:
    """Sigmoid activation"""
    def __init__(self):
        self._output_cache = None  #Stores forward output for backward

    def forward(self, Z: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow in exp
        Z_clipped = np.clip(Z, -500, 500)
        out = 1.0 / (1.0 + np.exp(-Z_clipped))
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Gradient: σ(z) * (1 - σ(z)) * grad_output."""
        s = self._output_cache
        return grad_output * s * (1.0 - s)


class Tanh:
    """Tanh activation: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))."""

    def __init__(self):
        self._output_cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        out = np.tanh(Z)
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Gradient: (1 - tanh^2(z)) * grad_output."""
        return grad_output * (1.0 - self._output_cache ** 2)


class ReLU:
    """ReLU activation: max(0, z)."""

    def __init__(self):
        self._input_cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self._input_cache = Z
        return np.maximum(0.0, Z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Gradient: grad_output where Z > 0, else 0."""
        return grad_output * (self._input_cache > 0).astype(float)


class Softmax:
    """
    Softmax activation for output layer.
    Uses numerically stable version: subtract max before exp.
    """

    def __init__(self):
        self._output_cache = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Args:
            Z: shape (batch_size, num_classes)
        Returns:
            Probabilities summing to 1 per sample.
        """
        #Shift by max for numerical stability
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        out = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        When Softmax is combined with CrossEntropy, this backward is NOT used directly.
        The combined gradient is computed in the loss function.
        This is provided for completeness.
        """
        # full Jacobian-vector product (not used in practice with CE loss)
        s = self._output_cache
        return grad_output * s * (1.0 - s)


def get_activation(name: str):
    """Factory function to get activation by name."""
    activations = {
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
        "softmax": Softmax,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]()
