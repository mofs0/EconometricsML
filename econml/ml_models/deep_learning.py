"""
Lightweight deep learning style models.

These implementations use stable, widely available dependencies (NumPy and
scikit-learn) so the core package stays easy to install.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def _as_2d(x):
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class DeepMLPRegressor:
    """A deeper MLP regressor wrapper."""

    def __init__(self, hidden_layers=(128, 64, 32), activation="relu", learning_rate=0.001):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            learning_rate_init=learning_rate,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(_as_2d(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        return self.model.predict(_as_2d(X))


class Conv1DRegressor:
    """Sequence regressor using random 1D convolutional features."""

    def __init__(self, n_kernels=8, kernel_size=3, alpha=1.0, random_state=42):
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.random_state = random_state
        self.kernels_ = None
        self.model = Ridge(alpha=alpha)

    def _conv_features(self, X):
        X = _as_2d(X)
        if X.shape[1] < self.kernel_size:
            raise ValueError("Input sequence length must be >= kernel_size")
        feats = []
        for row in X:
            row_feats = []
            for kernel in self.kernels_:
                values = [np.dot(row[i : i + self.kernel_size], kernel) for i in range(len(row) - self.kernel_size + 1)]
                row_feats.extend([np.mean(values), np.max(values), np.std(values)])
            feats.append(row_feats)
        return np.asarray(feats)

    def fit(self, X, y):
        X = _as_2d(X)
        rng = np.random.default_rng(self.random_state)
        self.kernels_ = rng.normal(size=(self.n_kernels, self.kernel_size))
        feats = self._conv_features(X)
        self.model.fit(feats, np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        feats = self._conv_features(X)
        return self.model.predict(feats)


class RNNRegressor:
    """Simple recurrent feature regressor."""

    def __init__(self, hidden_size=16, alpha=1.0, random_state=42):
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.random_state = random_state
        self.Wx_ = None
        self.Wh_ = None
        self.b_ = None
        self.model = Ridge(alpha=alpha)

    def _rnn_features(self, X):
        X = _as_2d(X)
        features = []
        for row in X:
            h = np.zeros(self.hidden_size)
            for value in row:
                h = np.tanh(value * self.Wx_ + h @ self.Wh_ + self.b_)
            features.append(np.concatenate([h, [np.mean(row), np.std(row)]]))
        return np.asarray(features)

    def fit(self, X, y):
        X = _as_2d(X)
        rng = np.random.default_rng(self.random_state)
        self.Wx_ = rng.normal(scale=0.5, size=self.hidden_size)
        self.Wh_ = rng.normal(scale=0.5, size=(self.hidden_size, self.hidden_size))
        self.b_ = rng.normal(scale=0.1, size=self.hidden_size)
        feats = self._rnn_features(X)
        self.model.fit(feats, np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        feats = self._rnn_features(X)
        return self.model.predict(feats)


class TransformerRegressor:
    """Attention-style sequence regressor."""

    def __init__(self, d_model=16, alpha=1.0, random_state=42):
        self.d_model = d_model
        self.alpha = alpha
        self.random_state = random_state
        self.query_ = None
        self.proj_ = None
        self.model = Ridge(alpha=alpha)

    def _attention_features(self, X):
        X = _as_2d(X)
        logits = X @ self.query_
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights = weights / weights.sum(axis=1, keepdims=True)
        pooled = np.sum(X * weights, axis=1, keepdims=True)
        return np.column_stack([pooled, X.mean(axis=1), X.std(axis=1)])

    def fit(self, X, y):
        X = _as_2d(X)
        rng = np.random.default_rng(self.random_state)
        self.query_ = rng.normal(size=(X.shape[1], 1))
        self.proj_ = rng.normal(size=(X.shape[1], self.d_model))
        feats = self._attention_features(X)
        self.model.fit(feats, np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        feats = self._attention_features(X)
        return self.model.predict(feats)


class AutoencoderModel:
    """PCA-based autoencoder style model."""

    def __init__(self, latent_dim=3):
        self.latent_dim = latent_dim
        self.encoder_ = PCA(n_components=latent_dim)
        self.fitted_ = False

    def fit(self, X):
        self.encoder_.fit(_as_2d(X))
        self.fitted_ = True
        return self

    def encode(self, X):
        if not self.fitted_:
            raise ValueError("Model must be fit before encoding")
        return self.encoder_.transform(_as_2d(X))

    def reconstruct(self, X):
        z = self.encode(X)
        return self.encoder_.inverse_transform(z)

    def predict(self, X):
        return self.reconstruct(X)


__all__ = [
    "DeepMLPRegressor",
    "Conv1DRegressor",
    "RNNRegressor",
    "TransformerRegressor",
    "AutoencoderModel",
]
