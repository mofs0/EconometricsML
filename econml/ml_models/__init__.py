"""
Machine Learning Models Module

Contains ML models for economics and finance:
- Tree-based models (Random Forest, XGBoost, LightGBM)
- Neural networks
- Ensemble methods
- Feature engineering utilities
"""

from .ensemble import EnsemblePredictor
from .neural_network import NeuralNetworkRegressor
from .advanced import (
    DecisionTreeClassifierModel,
    DecisionTreeRegressorModel,
    GradientBoostingClassifierModel,
    GradientBoostingRegressorModel,
    KMeansClusterer,
    KNNClassifierModel,
    KNNRegressorModel,
    LassoRegressor,
    LogisticClassifier,
    PCAProjector,
    RandomForestClassifierModel,
    RandomForestRegressorModel,
    RidgeRegressor,
    SVRModel,
    XGBoostRegressorModel,
)
from .deep_learning import (
    AutoencoderModel,
    Conv1DRegressor,
    DeepMLPRegressor,
    RNNRegressor,
    TransformerRegressor,
)

__all__ = [
    "EnsemblePredictor",
    "NeuralNetworkRegressor",
    "RidgeRegressor",
    "LassoRegressor",
    "RandomForestRegressorModel",
    "GradientBoostingRegressorModel",
    "DecisionTreeRegressorModel",
    "SVRModel",
    "LogisticClassifier",
    "RandomForestClassifierModel",
    "GradientBoostingClassifierModel",
    "DecisionTreeClassifierModel",
    "KMeansClusterer",
    "PCAProjector",
    "XGBoostRegressorModel",
    "KNNRegressorModel",
    "KNNClassifierModel",
    "DeepMLPRegressor",
    "Conv1DRegressor",
    "RNNRegressor",
    "TransformerRegressor",
    "AutoencoderModel",
]
