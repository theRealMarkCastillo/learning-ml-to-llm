"""
Visualization utilities for ML projects
Common plotting functions to reuse across projects
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_loss_curve(losses: List[float], 
                    title: str = "Training Loss Over Iterations",
                    xlabel: str = "Iteration",
                    ylabel: str = "Loss",
                    figsize: Tuple[int, int] = (10, 6),
                    log_scale: bool = False) -> None:
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=figsize)
    plt.plot(losses, linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_regression_line(X: np.ndarray, 
                         y: np.ndarray, 
                         predictions: np.ndarray,
                         title: str = "Linear Regression Fit",
                         figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot data points and regression line
    
    Args:
        X: Input features (1D or 2D array)
        y: True values
        predictions: Predicted values
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Handle both 1D and 2D arrays
    X_plot = X.flatten() if X.ndim > 1 else X
    
    # Scatter plot of actual data
    plt.scatter(X_plot, y, alpha=0.6, s=50, label='Actual Data', color='blue')
    
    # Sort for line plot
    sorted_idx = np.argsort(X_plot)
    plt.plot(X_plot[sorted_idx], predictions[sorted_idx], 
             color='red', linewidth=2, label='Predicted Line')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(X: np.ndarray,
                          y: np.ndarray,
                          model,
                          title: str = "Decision Boundary",
                          figsize: Tuple[int, int] = (10, 8),
                          resolution: int = 100) -> None:
    """
    Plot decision boundary for 2D classification
    
    Args:
        X: Input features (Nx2 array)
        y: Labels
        model: Model with predict method
        title: Plot title
        figsize: Figure size
        resolution: Resolution of decision boundary mesh
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plot only works with 2D features")
    
    plt.figure(figsize=figsize)
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', s=50, alpha=0.8)
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                         classes: Optional[List[str]] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6),
                         normalize: bool = False) -> None:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class labels
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True, cbar_kws={'label': 'Count'})
    
    if classes is not None:
        plt.xticks(np.arange(len(classes)) + 0.5, classes)
        plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_learning_rate_comparison(learning_rates: List[float],
                                  losses_dict: dict,
                                  figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Compare loss curves for different learning rates
    
    Args:
        learning_rates: List of learning rates
        losses_dict: Dictionary mapping learning rate to list of losses
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for lr in learning_rates:
        if lr in losses_dict:
            plt.plot(losses_dict[lr], label=f'LR = {lr}', linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Curves for Different Learning Rates', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def plot_parameter_trajectory(param_history: np.ndarray,
                              param_names: List[str] = None,
                              figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot how parameters change during training
    
    Args:
        param_history: Array of shape (n_iterations, n_params)
        param_names: List of parameter names
        figsize: Figure size
    """
    n_params = param_history.shape[1]
    
    if param_names is None:
        param_names = [f'Param {i}' for i in range(n_params)]
    
    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    
    if n_params == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(param_history[:, i], linewidth=2)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(param_names[i], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
