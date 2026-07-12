"""
Neural Network from Scratch — NumPy implementation.

A simple multi-layer perceptron (MLP) with manual backpropagation.
Supports binary cross-entropy (binary classification) and categorical
cross-entropy (multi-class classification).

This module is extracted from the Jupyter notebook so that:
- Notebooks can import it cleanly
- Automated tests can exercise the implementation
- The core logic is portable outside Jupyter

Usage
-----
    from project11_5_neural_networks.neural_network import NeuralNetwork

    net = NeuralNetwork(input_dim=2, hidden_dims=[32, 16],
                        output_dim=1, task='binary')
    net.train(X_train, y_train, epochs=200,
              learning_rate=0.5, batch_size=None,
              validate_every=10)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Activation helpers (also used by the notebook)
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid with numerical clipping."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    """Derivative: σ(x) * (1 - σ(x))."""
    return sigmoid_output * (1.0 - sigmoid_output)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract max per sample)."""
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Neural Network class
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """Simple 2–3 layer MLP implemented from scratch in NumPy.

    Supports:
    - Configurable hidden layer sizes
    - ReLU hidden activations, sigmoid (binary) or softmax (multiclass) output
    - Binary cross-entropy (binary) or categorical cross-entropy (multiclass)
    - Full-batch or mini-batch gradient descent
    - Gradient clipping for stability

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        Hidden layer sizes, e.g. ``[64, 32]``.
    output_dim : int
        Number of output units (1 for binary, >2 for multiclass).
    task : str
        ``'binary'`` or ``'multiclass'``.
    use_gradient_clipping : bool
        If ``True``, clip gradients to ``[−1, 1]`` before update.

    Examples
    --------
    >>> net = NeuralNetwork(input_dim=2, hidden_dims=[32, 16],
    ...                     output_dim=1, task='binary')
    >>> net.train(X, y, epochs=50, learning_rate=0.5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        task: str = "binary",
        use_gradient_clipping: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task = task
        self.use_gradient_clipping = use_gradient_clipping

        # Xavier/Glorot weight init (better than 0.01 * randn)
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            w = np.random.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

        self.num_layers = len(self.weights)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass, caching activations and pre-activations for backprop.

        Parameters
        ----------
        x : (batch_size, input_dim)

        Returns
        -------
        output : (batch_size, output_dim) probabilities
        """
        self._activations: list[np.ndarray] = [x]
        self._zs: list[np.ndarray] = []

        a = x
        for layer_idx in range(self.num_layers):
            z = a @ self.weights[layer_idx] + self.biases[layer_idx]
            self._zs.append(z)

            if layer_idx < self.num_layers - 1:
                a = relu(z)
            else:
                a = sigmoid(z) if self.task == "binary" else softmax(z)

            self._activations.append(a)

        return a

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute loss without updating weights (pure forward pass)."""
        output = self.forward(x)

        if self.task == "binary":
            y_flat = y.ravel() if y.ndim == 2 else y
            eps = 1e-15
            output_clipped = np.clip(output.ravel(), eps, 1.0 - eps)
            return float(
                -np.mean(y_flat * np.log(output_clipped)
                         + (1.0 - y_flat) * np.log(1.0 - output_clipped))
            )
        else:
            y_onehot = self._onehot(y)
            eps = 1e-15
            output_clipped = np.clip(output, eps, 1.0)
            return float(
                -np.mean(np.sum(y_onehot * np.log(output_clipped), axis=1))
            )

    # ------------------------------------------------------------------
    # Backward pass (with optional clipping)
    # ------------------------------------------------------------------

    def backward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
    ) -> None:
        """Backpropagation. Compute gradients and update weights in-place.

        Parameters
        ----------
        x : (batch_size, input_dim)
        y : (batch_size,) or (batch_size, output_dim) labels
        learning_rate : float — step size
        """
        batch_size = x.shape[0]

        # Ensure y is shaped correctly
        if self.task == "binary":
            y_flat = y.ravel() if y.ndim == 2 else y
            delta = (self._activations[-1].ravel() - y_flat).reshape(-1, 1)
        else:
            delta = self._activations[-1] - self._onehot(y)

        for layer_idx in range(self.num_layers - 1, -1, -1):
            dW = self._activations[layer_idx].T @ delta / batch_size
            dB = np.sum(delta, axis=0, keepdims=True) / batch_size

            # Gradient clipping for stability
            if self.use_gradient_clipping:
                max_norm = 5.0
                dW = np.clip(dW, -max_norm, max_norm)
                dB = np.clip(dB, -max_norm, max_norm)

            self.weights[layer_idx] -= learning_rate * dW
            self.biases[layer_idx] -= learning_rate * dB

            if layer_idx > 0:
                delta = delta @ self.weights[layer_idx].T
                delta *= relu_derivative(self._zs[layer_idx - 1])

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        output = self.forward(x)
        if self.task == "binary":
            return (output.ravel() > 0.5).astype(int)
        return np.argmax(output, axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correct predictions."""
        preds = self.predict(x)
        y_t = y.argmax(axis=1) if y.ndim == 2 else y
        return float(np.mean(preds == y_t))

    # ------------------------------------------------------------------
    # One-hot encoding helper (multiclass only)
    # ------------------------------------------------------------------

    def _onehot(self, y: np.ndarray) -> np.ndarray:
        """Convert integer labels to one-hot vectors."""
        n = y.shape[0] if y.ndim > 0 else 1
        if y.ndim == 0:
            one_hot = np.zeros(self.output_dim)
            one_hot[int(y)] = 1.0
            return one_hot.reshape(1, -1)
        one_hot = np.zeros((n, self.output_dim))
        one_hot[np.arange(n), y.astype(int)] = 1.0
        return one_hot

    # ------------------------------------------------------------------
    # High-level training API
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        learning_rate: float = 0.5,
        batch_size: Optional[int] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        log_every: int = 10,
    ) -> dict:
        """Train the network with optional validation tracking.

        Parameters
        ----------
        X, y : training data
        epochs : number of passes over the training set
        learning_rate : step size per update
        batch_size : ``None`` for full-batch, or int for mini-batch
        X_val, y_val : optional validation set
        log_every : print training status every N epochs

        Returns
        -------
        dict with keys ``'train_losses'``, ``'val_losses'`` (if provided).
        """
        train_losses: list[float] = []
        val_losses: list[float] = []

        n_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            if batch_size is None or batch_size >= n_samples:
                # Full-batch
                loss = self.compute_loss(X, y)
                self.backward(X, y, learning_rate=learning_rate)
                train_losses.append(loss)
            else:
                # Mini-batch
                epoch_losses = []
                indices = np.random.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_X = X[indices[start:end]]
                    batch_y = y[indices[start:end]]
                    loss = self.compute_loss(batch_X, batch_y)
                    self.backward(batch_X, batch_y, learning_rate=learning_rate)
                    epoch_losses.append(loss)
                train_losses.append(float(np.mean(epoch_losses)))

            # Validation
            if X_val is not None and y_val is not None:
                val_losses.append(self.compute_loss(X_val, y_val))

            if log_every and epoch % log_every == 0:
                t_acc = self.accuracy(X, y)
                v_msg = ""
                if X_val is not None:
                    v_acc = self.accuracy(X_val, y_val)
                    v_msg = f" (val_acc={v_acc:.3f})"
                print(
                    f"Epoch {epoch:4d}: train_loss={train_losses[-1]:.4f}"
                    f" (acc={t_acc:.3f}{v_msg})"
                )

        result: dict = {"train_losses": train_losses}
        if X_val is not None:
            result["val_losses"] = val_losses
        return result
