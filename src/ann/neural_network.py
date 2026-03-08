"""

Architecture:
    Input -> [Dense -> Activation] * num_hidden_layers -> Dense -> (loss handles softmax)

The network supports:
    - Configurable depth and width
    - Multiple activation functions
    - Gradient logging per layer (grad norms, dead neuron analysis)
    - Weight serialization / deserialization
"""

import numpy as np
from ann.neural_layer import DenseLayer
from ann.activations import get_activation


class MLP:
    """
    Configurable Multi-Layer Perceptron.

    Args:
        input_size: Dimensionality of input (e.g., 784 for MNIST).
        hidden_sizes: List of hidden layer sizes (e.g., [128, 128, 128]).
        output_size: Number of output classes (e.g., 10).
        activation: Activation name for hidden layers ('relu', 'sigmoid', 'tanh').
        weight_init: Weight initialization method ('xavier' or 'random').
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        activation: str = "relu",
        weight_init: str = "xavier",
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init

        # Build layers
        self.dense_layers = []   # DenseLayer objects
        self.activations = []    # Activation objects (one per hidden layer)

        # Input -> first hidden layer
        layer_sizes = [input_size] + hidden_sizes

        for i in range(len(hidden_sizes)):
            dense = DenseLayer(layer_sizes[i], layer_sizes[i + 1], weight_init)
            act = get_activation(activation)
            self.dense_layers.append(dense)
            self.activations.append(act)

        # Last hidden -> output layer (no activation here; softmax inside loss)
        output_layer = DenseLayer(hidden_sizes[-1], output_size, weight_init)
        self.dense_layers.append(output_layer)

        # Storage for activation outputs (useful for dead neuron analysis)
        self._activation_outputs = []


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute forward pass.

        Args:
            X: Input array of shape (batch_size, input_size).
        Returns:
            Logits of shape (batch_size, output_size).
        """
        self._activation_outputs = []
        out = X

        # Hidden layers: Dense -> Activation
        for dense, act in zip(self.dense_layers[:-1], self.activations):
            out = dense.forward(out)   # linear transformation
            out = act.forward(out)     # non-linearity
            self._activation_outputs.append(out)

        # Output layer: Dense only (softmax applied in loss)
        logits = self.dense_layers[-1].forward(out)
        return logits

    def backward(self, grad_loss: np.ndarray) -> None:
        """
        Backpropagate the loss gradient through all layers.

        Args:
            grad_loss: Gradient of loss w.r.t. logits, shape (batch_size, output_size).
                       Provided by the loss function's .backward() method.
        
        After calling backward(), each layer in self.dense_layers will have:
            - layer.grad_W  (gradient w.r.t. weights)
            - layer.grad_b  (gradient w.r.t. biases)
        """
        grad = grad_loss

        # Backward through output layer (dense only, no activation)
        grad = self.dense_layers[-1].backward(grad)

        # Backward through hidden layers in reverse order
        for dense, act in zip(
            reversed(self.dense_layers[:-1]),
            reversed(self.activations)
        ):
            grad = act.backward(grad)   # through activation
            grad = dense.backward(grad) # through linear layer

    def update_params(self, optimizer) -> None:
        """
        Apply optimizer update to all DenseLayer parameters.

        Args:
            optimizer: An optimizer instance (SGD, Adam, etc.)
        """
        for layer in self.dense_layers:
            optimizer.update(layer)



    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predicted class indices for input X.

        Args:
            X: Shape (batch_size, input_size).
        Returns:
            Array of shape (batch_size,) with predicted class indices.
        """
        logits = self.forward(X)
        return np.argmax(logits, axis=1)


    def get_gradient_norms(self) -> list:
        """
        Returns list of gradient norms (Frobenius) for each dense layer's weights.
        Useful for vanishing/exploding gradient analysis.
        """
        return [np.linalg.norm(layer.grad_W) for layer in self.dense_layers]

    def get_dead_neuron_fraction(self, layer_idx: int) -> float:
        """
        Fraction of neurons in hidden layer `layer_idx` that output 0 for all samples
        in the last forward pass. Relevant for ReLU dead neuron analysis.

        Args:
            layer_idx: Index into self._activation_outputs (0 = first hidden layer).
        Returns:
            Float in [0, 1] representing fraction of dead neurons.
        """
        if not self._activation_outputs:
            return 0.0
        acts = self._activation_outputs[layer_idx]  # (batch_size, neurons)
        dead = np.all(acts == 0, axis=0)            # True where all batch outputs are 0
        return float(dead.mean())

    def get_per_neuron_gradients(self, layer_idx: int, neuron_indices: list) -> dict:
        """
        Returns per-neuron gradient norms for weight symmetry analysis.

        Args:
            layer_idx: Index of the dense layer.
            neuron_indices: List of neuron (output dimension) indices to inspect.
        Returns:
            Dict mapping neuron index -> gradient norm for that neuron's incoming weights.
        """
        layer = self.dense_layers[layer_idx]
        return {i: float(np.linalg.norm(layer.grad_W[:, i])) for i in neuron_indices}


    def save_weights(self, filepath: str) -> None:
        """
        Save all layer weights and biases to a single .npy file.

        Saves a list of dicts: [{"W": array, "b": array}, ...]
        """
        weights = [{"W": layer.W, "b": layer.b} for layer in self.dense_layers]
        np.save(filepath, weights, allow_pickle=True)
        print(f"[MLP] Weights saved to {filepath}")

    def load_weights(self, filepath: str) -> None:
        """
        Load weights from a .npy file and assign to layers.

        Args:
            filepath: Path to the saved .npy file.
        """
        weights = np.load(filepath, allow_pickle=True)
        if len(weights) != len(self.dense_layers):
            raise ValueError(
                f"Loaded weights have {len(weights)} layers, "
                f"but model has {len(self.dense_layers)} layers."
            )
        for layer, w in zip(self.dense_layers, weights):
            layer.W = w["W"].copy()
            layer.b = w["b"].copy()
        print(f"[MLP] Weights loaded from {filepath}")

    def __repr__(self):
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        desc = " -> ".join(str(s) for s in sizes)
        return (f"MLP({desc}, activation={self.activation_name}, "
                f"init={self.weight_init})")
