"""Tests for the visualization module."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI)
import matplotlib.pyplot as plt

from utils.visualization import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_curve,
    plot_parameter_trajectory,
    plot_regression_line,
)


class MockModel:
    """Minimal mock classifier for decision boundary plotting."""

    def __init__(self, weight: float, bias: float):
        self.weight = weight
        self.bias = bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] * self.weight + self.bias > 0).astype(float)


# ---------------------------------------------------------------------------
# plot_loss_curve
# ---------------------------------------------------------------------------


class TestPlotLossCurve:
    """Test loss curve plotting."""

    def setup_method(self):
        plt.close("all")  # Clean slate before each test

    def test_basic_plot(self):
        plot_loss_curve([0.5, 0.4, 0.3, 0.2, 0.1])
        assert plt.gcf() is not None

    def test_log_scale(self):
        plot_loss_curve([0.01, 0.001, 0.0001], log_scale=True)

    def test_custom_labels(self):
        plot_loss_curve(
            [1.0, 0.5], title="Custom", xlabel="x", ylabel="y"
        )


# ---------------------------------------------------------------------------
# plot_regression_line
# ---------------------------------------------------------------------------


class TestPlotRegressionLine:
    """Test regression line plotting."""

    def setup_method(self):
        plt.close("all")

    def test_1d_input(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.5, 1.5, 2.5])
        preds = np.array([0.5, 1.5, 2.5])
        plot_regression_line(X, y, preds)

    def test_1d_also_works(self):
        X = np.random.randn(20, 1).ravel()  # 1D
        y = np.random.randn(20)
        preds = np.random.randn(20)
        plot_regression_line(X, y, preds)


# ---------------------------------------------------------------------------
# plot_decision_boundary
# ---------------------------------------------------------------------------


class TestPlotDecisionBoundary:
    """Test decision boundary plotting."""

    def setup_method(self):
        plt.close("all")

    def test_2d_features(self):
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(float)
        model = MockModel(weight=1.0, bias=0.0)
        plot_decision_boundary(X, y, model)

    def test_non_2d_raises(self):
        X = np.random.randn(50, 3)
        y = np.zeros(50)
        model = MockModel(1.0, 0.0)
        with pytest.raises(ValueError, match="2D features"):
            plot_decision_boundary(X, y, model)


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------


class TestPlotConfusionMatrix:
    """Test confusion matrix plotting."""

    def setup_method(self):
        plt.close("all")

    def test_basic_plot(self):
        cm = np.array([[10, 2], [3, 15]])
        plot_confusion_matrix(cm, classes=["A", "B"])

    def test_normalize(self):
        cm = np.array([[10, 5], [2, 13]])
        plot_confusion_matrix(cm, normalize=True)


# ---------------------------------------------------------------------------
# plot_parameter_trajectory
# ---------------------------------------------------------------------------


class TestPlotParameterTrajectory:
    """Test parameter trajectory plotting."""

    def setup_method(self):
        plt.close("all")

    def test_single_param(self):
        history = np.random.randn(20, 1)
        plot_parameter_trajectory(history)

    def test_multiple_params(self):
        history = np.random.randn(20, 3)
        plot_parameter_trajectory(history, ["w1", "w2", "b"])
