"""
Minimal numpy-based LinearRegression + StandardScaler fallback.

When scikit-learn is not available in the environment (e.g. minimal
Docker base image or restricted network) this module provides two
drop-in replacements that implement the same math using only numpy.

Classes:
    SimpleScaler: StandardScaler equivalent (z-score normalisation).
    SimpleLinearRegression: Ordinary Least Squares via np.linalg.lstsq.

Both classes expose the same ``fit`` / ``predict`` API as their
scikit-learn counterparts so the rest of the codebase can treat them
interchangeably via duck-typing.
"""

from __future__ import annotations

import numpy as np


class SimpleScaler:
    """Z-score normalisation — (X - μ) / σ.

    Mirrors sklearn.preprocessing.StandardScaler with ``with_mean=True``
    and ``with_std=True``.  Handles the zero-stdedge case (constant
    features) by replacing std with 1.0 to avoid division-by-zero.

    Attributes:
        mean_: Per-feature mean array, shape (n_features,).
        std_: Per-feature std array, shape (n_features,).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SimpleScaler":
        """Compute mean and std from ``X`` and return self.

        Args:
            X: Training features, shape (n_samples, n_features).

        Returns:
            self, for method-chaining compatibility with sklearn.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Prevent division-by-zero for constant columns.
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale ``X`` using previously learned mean and std.

        Args:
            X: Features to scale, shape (n_samples, n_features).

        Returns:
            Scaled array of the same shape.
        """
        return (X - self.mean_) / self.std_


class SimpleLinearRegression:
    """Ordinary Least Squares linear regression via np.linalg.lstsq.

    Matches sklearn.linear_model.LinearRegression with
    ``fit_intercept=True``.  The intercept is handled by prepending a
    column of ones to the design matrix before calling ``lstsq``.

    Attributes:
        coef_: Learned coefficients, shape (n_features,).
        intercept_: Learned bias term (float).
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLinearRegression":
        """Fit the linear model to ``X`` and ``y``.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).

        Returns:
            self, for method-chaining compatibility with sklearn.
        """
        # Augment with a bias column so lstsq learns intercept too.
        X_bias = np.column_stack([np.ones(len(X)), X])
        theta, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for ``X``.

        Args:
            X: Features, shape (n_samples, n_features) or (n_features,)
                (single sample).

        Returns:
            Predicted values as a 1-D array.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_ + self.intercept_
