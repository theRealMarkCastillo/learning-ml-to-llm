"""Tests for data generators."""

from __future__ import annotations

import numpy as np
import pytest

from utils.data_generators import (
    generate_binary_classification_data,
    generate_linear_data,
    generate_multiclass_data,
    generate_noisy_sine_wave,
    generate_polynomial_data,
)


# ---------------------------------------------------------------------------
# generate_linear_data
# ---------------------------------------------------------------------------


class TestGenerateLinearData:
    """Test synthetic linear regression data generation."""

    def test_default_shape(self):
        X, y = generate_linear_data(n_samples=100)
        assert X.shape == (100, 1)
        assert y.shape == (100, 1)

    def test_custom_samples(self):
        X, y = generate_linear_data(n_samples=500)
        assert X.shape[0] == 500

    def test_random_state_deterministic(self):
        X1, y1 = generate_linear_data(n_samples=50, random_state=123)
        X2, y2 = generate_linear_data(n_samples=50, random_state=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_y_follows_linear_pattern(self):
        """Mean of y should be close to slope·mean(X) + intercept."""
        X, y = generate_linear_data(n_samples=2000, slope=5.0, intercept=10.0, noise_std=0.01)
        predicted_y = 5.0 * X[:, 0] + 10.0
        np.testing.assert_allclose(
            y[:, 0].mean(), predicted_y.mean(), rtol=0.1
        )


# ---------------------------------------------------------------------------
# generate_polynomial_data
# ---------------------------------------------------------------------------


class TestGeneratePolynomialData:
    """Test polynomial regression data generation."""

    def test_default_shape(self):
        X, y = generate_polynomial_data(n_samples=100)
        assert X.shape == (100, 1)
        assert y.shape == (100, 1)

    def test_polynomial_degrees(self):
        """Output should involve multiple degrees up to degree+1."""
        _, y = generate_polynomial_data(n_samples=100, degree=5, noise_std=0.0)
        # With no noise, polynomial coefficients determine output
        assert len(set(np.round(y[:, 0], 2))) > 5  # Many unique values

    def test_deterministic_with_random_state(self):
        X1, y1 = generate_polynomial_data(n_samples=50, random_state=42)
        X2, y2 = generate_polynomial_data(n_samples=50, random_state=42)
        np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# generate_binary_classification_data
# ---------------------------------------------------------------------------


class TestGenerateBinaryData:
    """Test binary classification data generation."""

    def test_default_shape(self):
        X, y = generate_binary_classification_data(n_samples=100)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_binary_labels(self):
        _, y = generate_binary_classification_data(n_samples=100)
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_class_balance(self):
        _, y = generate_binary_classification_data(n_samples=200)
        assert abs(np.mean(y) - 0.5) < 0.05

    def test_deterministic(self):
        X1, y1 = generate_binary_classification_data(n_samples=50, random_state=77)
        X2, y2 = generate_binary_classification_data(n_samples=50, random_state=77)
        np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# generate_multiclass_data
# ---------------------------------------------------------------------------


class TestGenerateMulticlassData:
    """Test multiclass classification data generation."""

    def test_default_shape(self):
        X, y = generate_multiclass_data(n_samples=150, n_classes=3)
        assert X.shape == (150, 2)
        assert y.shape == (150,)

    def test_correct_number_of_classes(self):
        _, y = generate_multiclass_data(n_samples=300, n_classes=5)
        assert len(np.unique(y)) == 5

    def test_deterministic(self):
        X1, y1 = generate_multiclass_data(n_samples=60, n_classes=3, random_state=99)
        X2, y2 = generate_multiclass_data(n_samples=60, n_classes=3, random_state=99)
        np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# generate_noisy_sine_wave
# ---------------------------------------------------------------------------


class TestGenerateSineWave:
    """Test noisy sine wave generation."""

    def test_default_shape(self):
        X, y = generate_noisy_sine_wave(n_samples=100)
        assert X.shape == (100, 1)
        assert y.shape == (100, 1)

    def test_linearly_spaced_x(self):
        X, _ = generate_noisy_sine_wave(n_samples=50)
        np.testing.assert_array_equal(
            X[:, 0], np.linspace(0, 2 * np.pi, 50)
        )

    def test_oscillates(self):
        _, y = generate_noisy_sine_wave(n_samples=100, noise_std=0.0)
        assert y.min() < 0
        assert y.max() > 0  # Sine oscillates between -1 and 1
