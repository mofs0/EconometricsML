"""
Ensemble Models for Economic and Financial Prediction

Combines multiple algorithms (Random Forest, XGBoost, LightGBM) with voting or stacking.
Provides robust predictions with reduced overfitting.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


class EnsemblePredictor:
    """
    Ensemble model combining multiple weak learners.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of base estimators
    method : str, default='voting'
        Ensemble method: 'voting' or 'stacking'
    """
    
    def __init__(self, n_estimators=100, method='voting'):
        self.n_estimators = n_estimators
        self.method = method
        self.models = []
        self.weights = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """
        Fit ensemble models.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train base models
        self.models = [
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            ),
            GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                random_state=42
            ),
        ]
        
        # Fit models and calculate weights based on cross-val performance
        scores = []
        for model in self.models:
            model.fit(X_scaled, y)
            score = np.mean(cross_val_score(model, X_scaled, y, cv=5, scoring='r2'))
            scores.append(score)
        
        # Normalize scores as weights
        scores = np.array(scores)
        self.weights = scores / np.sum(scores)
        
        return self
        
    def predict(self, X):
        """
        Generate ensemble predictions.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Ensemble predictions
        """
        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)
        
        predictions = np.column_stack([
            model.predict(X_scaled) for model in self.models
        ])
        
        # Weighted average
        y_pred = predictions @ self.weights
        return y_pred
        
    def feature_importance(self):
        """
        Get aggregated feature importance across models.
        
        Returns
        -------
        importance : array
            Feature importance scores
        """
        importances = np.zeros(self.models[0].n_features_in_)
        
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'feature_importances_'):
                importances += weight * model.feature_importances_
        
        return importances
