"""
data_utils.py - Data loading, preprocessing, and utility functions.
Supports MNIST and Fashion-MNIST via keras.datasets.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(name: str):
    """
    Load and preprocess a dataset.

    Args:
        name: 'mnist' or 'fashion_mnist'.
    Returns:
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names)
        - X arrays: float32 in [0, 1], shape (N, 784)
        - y arrays: int32, shape (N,), values in [0, 9]
    """
    name = name.lower().replace("-", "_")

    if name == "mnist":
        from keras.datasets import mnist
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
        class_names = [str(i) for i in range(10)]
    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'.")

    # Flatten and normalize to [0, 1]
    X_train_full = X_train_full.reshape(X_train_full.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

    y_train_full = y_train_full.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Split training set into train (90%) and validation (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    print(f"[Data] Dataset: {name}")
    print(f"[Data] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def to_onehot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.

    Args:
        y: Integer array of shape (N,).
        num_classes: Number of classes.
    Returns:
        One-hot matrix of shape (N, num_classes).
    """
    N = len(y)
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    onehot[np.arange(N), y] = 1.0
    return onehot


def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    Generator that yields (X_batch, y_batch) tuples.

    Args:
        X: Input array of shape (N, D).
        y: Label array of shape (N,).
        batch_size: Batch size.
        shuffle: Whether to shuffle data before batching.
    Yields:
        (X_batch, y_batch) where X_batch has shape (batch_size, D).
    """
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def compute_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_pred: Predicted class indices, shape (N,).
        y_true: True class indices, shape (N,).
    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(np.mean(y_pred == y_true))
