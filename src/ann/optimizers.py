"""

All optimizers support L2 weight decay (applied to weights only, not biases).

Optimizers implemented:
    - SGD (vanilla stochastic gradient descent)
    - Momentum (SGD with momentum)
    - NAG (Nesterov Accelerated Gradient)
    - RMSProp
    - Adam
    - Nadam (Nesterov Adam)
"""

import numpy as np


class SGD:
    """
    Vanilla Stochastic Gradient Descent.
    Update rule: W = W - lr * grad_W  (+ L2 decay)
    """

    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layer):
        """Apply SGD update to a DenseLayer."""
        # L2 regularization gradient contribution
        grad_W = layer.grad_W + self.weight_decay * layer.W
        layer.W -= self.lr * grad_W
        layer.b -= self.lr * layer.grad_b


class Momentum:
    """
    SGD with Momentum.
    v = beta * v - lr * grad
    W = W + v
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self._velocity = {}  # dict keyed by layer id

    def update(self, layer):
        lid = id(layer)
        if lid not in self._velocity:
            self._velocity[lid] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
            }

        v = self._velocity[lid]
        grad_W = layer.grad_W + self.weight_decay * layer.W

        v["vW"] = self.beta * v["vW"] - self.lr * grad_W
        v["vb"] = self.beta * v["vb"] - self.lr * layer.grad_b

        layer.W += v["vW"]
        layer.b += v["vb"]


class NAG:
    """
    Nesterov Accelerated Gradient.
    
    Implementation using the "look-ahead" trick:
    1. Compute look-ahead weights: W_look = W + beta * v
    2. Compute gradient at W_look (caller must handle this; here we apply standard NAG update)
    
    Practical implementation (Sutskever form):
    v_new = beta * v - lr * grad(W + beta*v)
    W = W + v_new
    
    We use the simplified reformulation where:
    v = beta*v - lr*grad
    W = W - beta*v_prev + (1+beta)*v
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self._velocity = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self._velocity:
            self._velocity[lid] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
            }

        v = self._velocity[lid]
        grad_W = layer.grad_W + self.weight_decay * layer.W

        # Standard Nesterov update
        vW_prev = v["vW"].copy()
        vb_prev = v["vb"].copy()

        v["vW"] = self.beta * v["vW"] - self.lr * grad_W
        v["vb"] = self.beta * v["vb"] - self.lr * layer.grad_b

        # Nesterov "look-ahead" correction
        layer.W += -self.beta * vW_prev + (1 + self.beta) * v["vW"]
        layer.b += -self.beta * vb_prev + (1 + self.beta) * v["vb"]


class RMSProp:
    """
    RMSProp optimizer.
    v = rho * v + (1 - rho) * grad^2
    W = W - lr / sqrt(v + eps) * grad
    """

    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self._cache = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self._cache:
            self._cache[lid] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
            }

        c = self._cache[lid]
        grad_W = layer.grad_W + self.weight_decay * layer.W

        c["vW"] = self.rho * c["vW"] + (1 - self.rho) * grad_W ** 2
        c["vb"] = self.rho * c["vb"] + (1 - self.rho) * layer.grad_b ** 2

        layer.W -= self.lr * grad_W / (np.sqrt(c["vW"]) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(c["vb"]) + self.eps)


class Adam:
    """
    Adam optimizer.
    m = beta1 * m + (1 - beta1) * grad        (1st moment: mean)
    v = beta2 * v + (1 - beta2) * grad^2      (2nd moment: variance)
    m_hat = m / (1 - beta1^t)                  (bias correction)
    v_hat = v / (1 - beta2^t)
    W = W - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._state = {}
        self._t = 0  # global step counter

    def update(self, layer):
        # Increment global step once per update call group
        # (each layer call is one sub-step; use per-layer step counter)
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "mW": np.zeros_like(layer.W),
                "vW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b),
                "vb": np.zeros_like(layer.b),
                "t": 0,
            }

        s = self._state[lid]
        s["t"] += 1
        t = s["t"]

        grad_W = layer.grad_W + self.weight_decay * layer.W

        # Update moments
        s["mW"] = self.beta1 * s["mW"] + (1 - self.beta1) * grad_W
        s["vW"] = self.beta2 * s["vW"] + (1 - self.beta2) * grad_W ** 2
        s["mb"] = self.beta1 * s["mb"] + (1 - self.beta1) * layer.grad_b
        s["vb"] = self.beta2 * s["vb"] + (1 - self.beta2) * layer.grad_b ** 2

        # Bias correction
        mW_hat = s["mW"] / (1 - self.beta1 ** t)
        vW_hat = s["vW"] / (1 - self.beta2 ** t)
        mb_hat = s["mb"] / (1 - self.beta1 ** t)
        vb_hat = s["vb"] / (1 - self.beta2 ** t)

        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class Nadam:
    """
    Nadam: Nesterov-accelerated Adam.
    Combines Nesterov momentum with Adam's adaptive learning rates.
    
    Key difference from Adam: uses look-ahead gradient correction:
    m_hat_nesterov = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1^t)
    W = W - lr * m_hat_nesterov / (sqrt(v_hat) + eps)
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._state = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self._state:
            self._state[lid] = {
                "mW": np.zeros_like(layer.W),
                "vW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b),
                "vb": np.zeros_like(layer.b),
                "t": 0,
            }

        s = self._state[lid]
        s["t"] += 1
        t = s["t"]

        grad_W = layer.grad_W + self.weight_decay * layer.W

        # Update biased first and second moment estimates
        s["mW"] = self.beta1 * s["mW"] + (1 - self.beta1) * grad_W
        s["vW"] = self.beta2 * s["vW"] + (1 - self.beta2) * grad_W ** 2
        s["mb"] = self.beta1 * s["mb"] + (1 - self.beta1) * layer.grad_b
        s["vb"] = self.beta2 * s["vb"] + (1 - self.beta2) * layer.grad_b ** 2

        # Bias-corrected second moment
        vW_hat = s["vW"] / (1 - self.beta2 ** t)
        vb_hat = s["vb"] / (1 - self.beta2 ** t)

        # Nesterov-corrected first moment (look-ahead)
        mW_nesterov = (self.beta1 * s["mW"] / (1 - self.beta1 ** (t + 1))
                       + (1 - self.beta1) * grad_W / (1 - self.beta1 ** t))
        mb_nesterov = (self.beta1 * s["mb"] / (1 - self.beta1 ** (t + 1))
                       + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** t))

        layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.eps)


def get_optimizer(name: str, **kwargs):
    """
    Factory function to get optimizer by name.
    
    Args:
        name: One of 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'.
        **kwargs: Optimizer hyperparameters (learning_rate, weight_decay, etc.)
    Returns:
        Optimizer instance.
    """
    optimizers = {
        "sgd": SGD,
        "momentum": Momentum,
        "nag": NAG,
        "rmsprop": RMSProp,
        "adam": Adam,
        "nadam": Nadam,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(optimizers.keys())}")
    return optimizers[name](**kwargs)
