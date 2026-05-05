"""
Neural Network Models for Economic Applications

Deep learning architectures for complex pattern recognition in economic data,
time series forecasting, and nonlinear relationship modeling.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


class NeuralNetworkRegressor:
    """
    Multi-layer Perceptron for economic regression.
    
    Parameters
    ----------
    hidden_layers : tuple, default=(100, 50)
        Number of neurons in each hidden layer
    activation : str, default='relu'
        Activation function: 'relu', 'tanh', 'logistic'
    learning_rate : float, default=0.001
        Learning rate for optimization
    """
    
    def __init__(self, hidden_layers=(100, 50), activation='relu', learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, epochs=200, batch_size=32, early_stopping=True, validation_split=0.2):
        """
        Fit the neural network.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values
        epochs : int, default=200
            Number of epochs
        batch_size : int, default=32
            Batch size for training
        early_stopping : bool, default=True
            Whether to use early stopping
        validation_split : float, default=0.2
            Fraction of data for validation
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            learning_rate_init=self.learning_rate,
            max_iter=epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            validation_fraction=validation_split,
            random_state=42,
            n_iter_no_change=20,
        )
        
        self.model.fit(X_scaled, y)
        
        return self
        
    def predict(self, X):
        """
        Generate predictions.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fit before prediction")
        
        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def score(self, X, y):
        """
        Calculate R² score.
        
        Parameters
        ----------
        X : array-like
            Test features
        y : array-like
            Test targets
            
        Returns
        -------
        score : float
            R² score
        """
        X = np.asarray(X)
        y = np.asarray(y)
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
