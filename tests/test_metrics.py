"""Tests for the evaluation metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import (
    accuracy,
    binary_cross_entropy,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision,
    recall,
    r_squared,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    """Test regression (MSE, RMSE, MAE, R²) metrics."""

    def test_mse_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y_true, y_pred) == 0.0

    def test_mse_simple(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 0.0])
        expected = ((0 - 1) ** 2 + (1 - 0) ** 2) / 2  # = 1.0
        assert pytest.approx(mean_squared_error(y_true, y_pred)) == expected

    def test_rmse_is_sqrt_of_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert pytest.approx(root_mean_squared_error(y_true, y_pred)) == pytest.approx(
            np.sqrt(mean_squared_error(y_true, y_pred))
        )

    def test_mae_perfect(self):
        assert mean_absolute_error(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

    def test_r_squared_perfect(self):
        assert r_squared(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0

    def test_r_squared_worst(self):
        """Predicting the mean gives R² = 0."""
        y = np.array([1.0, 2.0, 3.0])
        preds = np.array([2.0, 2.0, 2.0])
        assert pytest.approx(r_squared(y, preds)) == 0.0


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    """Test accuracy, precision, recall, F1."""

    def test_accuracy_perfect(self):
        assert accuracy(np.array([0, 1, 0]), np.array([0, 1, 0])) == 1.0

    def test_accuracy_worst(self):
        assert accuracy(np.array([0, 0]), np.array([1, 1])) == 0.0

    def test_accuracy_partial(self):
        result = accuracy(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        assert pytest.approx(result) == 0.75  # 3/4 correct

    def test_precision_zero_predictions(self):
        """Precision should be 0 when no positive predictions."""
        result = precision(np.array([1, 1]), np.array([0, 0]), pos_label=1)
        assert result == 0.0

    def test_recall_zero_actual(self):
        """Recall should be 0 when no actual positives."""
        result = recall(np.array([0, 0]), np.array([1, 1]), pos_label=1)
        assert result == 0.0

    def test_f1_both_zero(self):
        """F1 should be 0 when both precision and recall are 0."""
        result = f1_score(np.array([0, 0]), np.array([0, 0]), pos_label=1)
        assert result == 0.0

    def test_f1_perfect(self):
        result = f1_score(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), pos_label=1)
        assert result == 1.0

    def test_f1_harmonic_mean(self):
        """F1 is the harmonic mean of precision and recall."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1])
        p = precision(y_true, y_pred, pos_label=1)
        r = recall(y_true, y_pred, pos_label=1)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert pytest.approx(f1_score(y_true, y_pred, pos_label=1)) == expected


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    """Test confusion matrix computation."""

    def test_binary_confusion_matrix(self):
        cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]))
        assert cm.shape == (2, 2)
        np.testing.assert_array_equal(cm[0], [1, 1])
        np.testing.assert_array_equal(cm[1], [1, 1])

    def test_multiclass_confusion_matrix(self):
        cm = confusion_matrix(
            np.array([0, 1, 2]), np.array([0, 1, 2]), n_classes=3
        )
        assert cm.shape == (3, 3)
        np.testing.assert_array_equal(cm.diagonal(), np.array([1, 1, 1]))

    def test_auto_detect_n_classes(self):
        cm = confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2]))
        assert cm.shape == (3, 3)


# ---------------------------------------------------------------------------
# Binary cross-entropy
# ---------------------------------------------------------------------------


class TestBinaryCrossEntropy:
    """Test BCE loss computation."""

    def test_bce_perfect_predictions(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.001, 0.999])
        loss = binary_cross_entropy(y_true, y_pred)
        assert loss > 0
        assert loss < 0.01  # Very close to 0

    def test_bce_worst_predictions(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.999, 0.001])
        loss = binary_cross_entropy(y_true, y_pred)
        assert loss > 1.0  # Large loss for inverted predictions

    def test_bce_clipping_avoids_nan(self):
        y_pred_extreme = np.array([-10.0, 10.0])  # Outside [0,1]
        y_true = np.array([0.0, 1.0])
        loss = binary_cross_entropy(y_true, y_pred_extreme)
        assert not np.isnan(loss)
        assert not np.isinf(loss)
