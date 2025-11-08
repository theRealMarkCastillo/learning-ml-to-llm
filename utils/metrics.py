"""
Evaluation metrics for ML projects
Custom implementations to understand metrics deeply
"""

import numpy as np
from typing import Optional


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R-squared value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy value
    """
    return np.mean(y_true == y_pred)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Calculate binary cross-entropy loss
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small constant to avoid log(0)
    
    Returns:
        Binary cross-entropy value
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes (auto-detected if None)
    
    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    if n_classes is None:
        n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    
    return cm


def precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Calculate precision for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label
    
    Returns:
        Precision value
    """
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    predicted_positives = np.sum(y_pred == pos_label)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives


def recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Calculate recall for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label
    
    Returns:
        Recall value
    """
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    actual_positives = np.sum(y_true == pos_label)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Calculate F1 score for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label
    
    Returns:
        F1 score value
    """
    prec = precision(y_true, y_pred, pos_label)
    rec = recall(y_true, y_pred, pos_label)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)
