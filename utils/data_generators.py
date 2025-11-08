"""
Data generation utilities for ML projects
Synthetic data generators for different learning scenarios
"""

import numpy as np
from typing import Tuple, Optional


def generate_linear_data(n_samples: int = 100,
                        slope: float = 3.0,
                        intercept: float = 4.0,
                        noise_std: float = 1.0,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for linear regression
    
    Args:
        n_samples: Number of data points
        slope: True slope of the line
        intercept: True intercept of the line
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility
    
    Returns:
        X: Features (n_samples, 1)
        y: Target values (n_samples, 1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate X uniformly
    X = 2 * np.random.rand(n_samples, 1)
    
    # Generate y with noise
    y = intercept + slope * X + noise_std * np.random.randn(n_samples, 1)
    
    return X, y


def generate_polynomial_data(n_samples: int = 100,
                            degree: int = 3,
                            noise_std: float = 0.1,
                            x_range: Tuple[float, float] = (-1, 1),
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for polynomial regression
    
    Args:
        n_samples: Number of data points
        degree: Degree of polynomial
        noise_std: Standard deviation of noise
        x_range: Range of X values
        random_state: Random seed
    
    Returns:
        X: Features (n_samples, 1)
        y: Target values (n_samples, 1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    
    # Generate polynomial
    y = np.zeros((n_samples, 1))
    for i in range(degree + 1):
        coef = np.random.randn() * 0.5
        y += coef * (X ** i)
    
    # Add noise
    y += noise_std * np.random.randn(n_samples, 1)
    
    return X, y


def generate_binary_classification_data(n_samples: int = 100,
                                       n_features: int = 2,
                                       n_clusters_per_class: int = 1,
                                       class_sep: float = 1.0,
                                       random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic binary classification data
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_clusters_per_class: Number of clusters per class
        class_sep: Separation between classes
        random_state: Random seed
    
    Returns:
        X: Features (n_samples, n_features)
        y: Binary labels (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples_per_class = n_samples // 2
    
    # Generate class 0
    X_class0 = np.random.randn(samples_per_class, n_features)
    y_class0 = np.zeros(samples_per_class)
    
    # Generate class 1 (shifted by class_sep)
    X_class1 = np.random.randn(samples_per_class, n_features) + class_sep
    y_class1 = np.ones(samples_per_class)
    
    # Combine
    X = np.vstack([X_class0, X_class1])
    y = np.concatenate([y_class0, y_class1])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def generate_multiclass_data(n_samples: int = 150,
                            n_features: int = 2,
                            n_classes: int = 3,
                            class_sep: float = 1.5,
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic multi-class classification data
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        class_sep: Separation between classes
        random_state: Random seed
    
    Returns:
        X: Features (n_samples, n_features)
        y: Class labels (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    
    X_list = []
    y_list = []
    
    for class_idx in range(n_classes):
        # Generate cluster center
        center = np.random.randn(n_features) * class_sep
        
        # Generate samples around center
        X_class = np.random.randn(samples_per_class, n_features) + center
        y_class = np.full(samples_per_class, class_idx)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    # Combine
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def generate_noisy_sine_wave(n_samples: int = 100,
                             noise_std: float = 0.1,
                             x_range: Tuple[float, float] = (0, 2 * np.pi),
                             random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy sine wave data (useful for regularization experiments)
    
    Args:
        n_samples: Number of data points
        noise_std: Standard deviation of noise
        x_range: Range of X values
        random_state: Random seed
    
    Returns:
        X: Features (n_samples, 1)
        y: Target values (n_samples, 1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    y = np.sin(X) + noise_std * np.random.randn(n_samples, 1)
    
    return X, y
