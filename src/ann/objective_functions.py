"""
objective_functions.py - Loss functions using NumPy.
"""

import numpy as np


class CrossEntropyLoss:
    """
    Softmax + Cross-Entropy combined for numerical stability.
    Gradient w.r.t. logits = (probs - y) / batch_size
    """

    def __init__(self):
        self._probs_cache = None
        self._labels_cache = None

    def forward(self, logits: np.ndarray, y_onehot: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        self._probs_cache = probs
        self._labels_cache = y_onehot
        eps = 1e-12
        log_probs = np.log(np.clip(probs, eps, 1.0))
        loss = -np.mean(np.sum(y_onehot * log_probs, axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Returns gradient of shape (batch_size, num_classes).
        Divides by batch_size so layer.backward() just does matrix multiply.
        """
        batch_size = self._labels_cache.shape[0]
        return (self._probs_cache - self._labels_cache) / batch_size

    def get_probs(self) -> np.ndarray:
        return self._probs_cache


class MSELoss:
    """Mean Squared Error loss with Softmax output."""

    def __init__(self):
        self._probs_cache = None
        self._labels_cache = None

    def _softmax(self, logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, logits: np.ndarray, y_onehot: np.ndarray) -> float:
        probs = self._softmax(logits)
        self._probs_cache = probs
        self._labels_cache = y_onehot
        diff = probs - y_onehot
        loss = np.mean(np.sum(diff ** 2, axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        batch_size = self._labels_cache.shape[0]
        p = self._probs_cache
        diff = p - self._labels_cache
        dp = diff * p
        sum_dp = np.sum(dp, axis=1, keepdims=True)
        return 2.0 * (dp - p * sum_dp) / batch_size

    def get_probs(self) -> np.ndarray:
        return self._probs_cache


def get_loss(name: str):
    losses = {"cross_entropy": CrossEntropyLoss, "mse": MSELoss}
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'")
    return losses[name]()
