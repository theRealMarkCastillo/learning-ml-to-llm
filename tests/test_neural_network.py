"""Tests for the NeuralNetwork class."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports
_repo_root = None
for candidate in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
    if (candidate / "requirements.txt").exists():
        _repo_root = candidate
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        break

# Add project dir for the extracted module
_proj_dir = _repo_root / "projects" / "phase1_classical_ml" / "project11_5_neural_networks"
_proj_str = str(_proj_dir)
if _proj_str not in sys.path:
    sys.path.insert(0, _proj_str)

from neural_network import (
    NeuralNetwork,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
)
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


class TestActivationHelpers:
    """Unit tests for activation functions and their derivatives."""

    def test_relu(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(relu(x), expected)

    def test_relu_derivative(self):
        x = np.array([-1.0, 0.0, 1.0])
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu_derivative(x), expected)

    def test_sigmoid_bounds(self):
        for val in [-100.0, -10.0, 0.0, 10.0, 20.0]:
            out = sigmoid(np.array([float(val)]))[0]
            assert 0.0 < out < 1.0

    def test_sigmoid_derivative_identity(self):
        """σ'(x) = σ(x)·(1−σ(x))."""
        x = np.array([-2.0, 0.0, 2.0])
        sig = sigmoid(x)
        expected = sig * (1.0 - sig)
        np.testing.assert_allclose(sigmoid_derivative(sig), expected, rtol=1e-10)

    def test_softmax_normalises(self):
        x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 10.0]])
        out = softmax(x)
        np.testing.assert_allclose(out.sum(axis=1), np.array([1.0, 1.0]), atol=1e-6)

    def test_softmax_shift_invariance(self):
        """softmax(x + c) = softmax(x)."""
        x = np.array([[0.0, 1.0, 2.0]])
        shifted = x + 100.0
        np.testing.assert_allclose(softmax(x), softmax(shifted), atol=1e-6)


# ---------------------------------------------------------------------------
# NeuralNetwork — forward
# ---------------------------------------------------------------------------


class TestNeuralNetworkForward:
    """Test the forward pass produces expected shapes."""

    def test_binary_forward(self):
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=1, task="binary")
        x = np.array([[0.0, 1.0]])
        out = net.forward(x)
        assert out.shape == (1, 1)
        assert 0.0 <= out[0, 0] <= 1.0

    def test_multiclass_forward(self):
        net = NeuralNetwork(input_dim=3, hidden_dims=[8], output_dim=3, task="multiclass")
        x = np.random.randn(5, 3)
        out = net.forward(x)
        assert out.shape == (5, 3)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(5), atol=1e-6)

    def test_caching(self):
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=1, task="binary")
        net.forward(np.array([[0.0, 1.0]]))
        assert hasattr(net, "_activations")
        assert hasattr(net, "_zs")
        assert len(net._activations) == net.num_layers + 1
        assert len(net._zs) == net.num_layers

    def test_multiclass_onehot(self):
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=3, task="multiclass")
        y = np.array([0, 2, 1])
        oh = net._onehot(np.array(y))
        assert oh.shape == (3, 3)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(oh, expected)

    def test_single_label_onehot(self):
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=3, task="multiclass")
        oh = net._onehot(np.array(2))
        assert oh.shape == (1, 3)
        np.testing.assert_array_equal(oh, np.array([[0.0, 0.0, 1.0]]))


# ---------------------------------------------------------------------------
# NeuralNetwork — backward
# ---------------------------------------------------------------------------


class TestNeuralNetworkBackward:
    """Test that backward correctly updates weights and gradients."""

    def test_backward_changes_weights(self):
        np.random.seed(0)
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=1, task="binary")
        net.forward(np.array([[0.0, 1.0]]))  # set activations first
        initial_weights = [w.copy() for w in net.weights]
        net.backward(np.array([[0.0, 1.0]]), np.array([1.0]), learning_rate=0.5)
        for old_w, new_w in zip(initial_weights, net.weights):
            assert not np.allclose(old_w, new_w)

    def test_bias_backward_changes_biases(self):
        np.random.seed(0)
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=1, task="binary")
        net.forward(np.array([[0.0, 1.0]]))  # set activations first
        initial_biases = [b.copy() for b in net.biases]
        net.backward(np.array([[0.0, 1.0]]), np.array([1.0]), learning_rate=0.5)
        for old_b, new_b in zip(initial_biases, net.biases):
            assert not np.allclose(old_b, new_b)

    def test_gradient_clipping(self):
        np.random.seed(42)
        net = NeuralNetwork(
            input_dim=2, hidden_dims=[4], output_dim=1, task="binary",
            use_gradient_clipping=True,
        )
        large_x = np.full((1, 2), 100.0)
        net.forward(large_x)
        net.backward(large_x, np.array([1.0]), learning_rate=1.0)
        for w in net.weights:
            assert np.all(np.abs(w) <= 10.0)


# ---------------------------------------------------------------------------
# NeuralNetwork — prediction / accuracy
# ---------------------------------------------------------------------------


class TestNeuralNetworkPrediction:
    """Test prediction and accuracy."""

    def test_binary_predict(self):
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=1, task="binary")
        net.forward(np.array([[0.0, 0.0]]))
        preds = net.predict(np.array([[0.0, 0.0]]))
        assert preds.dtype in (np.int32, np.int64, int)

    def test_multiclass_predict(self):
        net = NeuralNetwork(input_dim=3, hidden_dims=[8], output_dim=3, task="multiclass")
        net.forward(np.random.randn(5, 3))
        preds = net.predict(np.random.randn(5, 3))
        assert preds.shape == (5,)
        assert all(0 <= p < 3 for p in preds)

    def test_accuracy_returns_float(self):
        net = NeuralNetwork(input_dim=2, hidden_dims=[4], output_dim=1, task="binary")
        acc = net.accuracy(np.array([[0.0, 0.0]]), np.array([0.0]))
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# NeuralNetwork — high-level train API
# ---------------------------------------------------------------------------


class TestNeuralNetworkTrain:
    """Test the high-level train() method."""

    def test_train_reduces_loss(self):
        np.random.seed(42)
        net = NeuralNetwork(input_dim=2, hidden_dims=[16], output_dim=1, task="binary")
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(float)

        result = net.train(
            X, y, epochs=50, learning_rate=0.5, log_every=50,
        )
        losses = result["train_losses"]
        assert losses[-1] < losses[0]

    def test_train_with_validation(self):
        np.random.seed(42)
        net = NeuralNetwork(input_dim=2, hidden_dims=[8], output_dim=1, task="binary")
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(float)
        X_val = X[:20]
        y_val = y[:20]

        result = net.train(
            X[20:], y[20:], epochs=30, learning_rate=0.3,
            X_val=X_val, y_val=y_val, log_every=30,
        )
        assert "val_losses" in result
        assert len(result["val_losses"]) == 30

    def test_train_with_minibatch(self):
        np.random.seed(42)
        net = NeuralNetwork(input_dim=2, hidden_dims=[8], output_dim=1, task="binary")
        X = np.random.randn(64, 2)
        y = (X[:, 0] > 0).astype(float)

        result = net.train(
            X, y, epochs=10, learning_rate=0.5, batch_size=16, log_every=10,
        )
        assert len(result["train_losses"]) == 10

    def test_multiclass_train_reduces_loss(self):
        np.random.seed(42)
        net = NeuralNetwork(input_dim=3, hidden_dims=[16, 8], output_dim=3, task="multiclass")
        X = np.random.randn(100, 3)
        y = np.array([0, 1, 2] * 33 + [0])

        result = net.train(
            X, y, epochs=50, learning_rate=0.3, log_every=50,
        )
        assert result["train_losses"][-1] < result["train_losses"][0]

    def test_returns_final_accuracy(self):
        np.random.seed(42)
        net = NeuralNetwork(input_dim=2, hidden_dims=[16], output_dim=1, task="binary")
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(float)

        net.train(X, y, epochs=50, learning_rate=0.5, log_every=50)
        acc = net.accuracy(X, y)
        assert acc > 0.5, f"Accuracy {acc} should exceed random baseline"
