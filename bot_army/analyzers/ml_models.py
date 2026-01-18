"""Machine Learning models for prediction and classification."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path

logger = logging.getLogger("analyzer.ml")

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class Prediction:
    """Model prediction result."""
    market_id: str
    model_name: str
    prediction: float
    confidence: float
    features_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "model_name": self.model_name,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "features_used": self.features_used,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class MLEngine:
    """
    Main ML engine managing multiple models.
    Handles training, prediction, and model ensemble.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        
        # Performance tracking
        self.predictions_log: List[Prediction] = []
        self.model_performance: Dict[str, Dict] = {}
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize ML models."""
        if SKLEARN_AVAILABLE:
            # Price direction classifier
            self.models["direction_rf"] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            # Price regressor
            self.models["price_gbr"] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            # Scalers
            self.scalers["features"] = StandardScaler()
        
        if TORCH_AVAILABLE:
            # Deep learning models
            self.models["lstm"] = PriceLSTM()
            self.models["attention"] = AttentionModel()
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "direction_rf",
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """Train a model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features
        if "features" in self.scalers:
            X_scaled = self.scalers["features"].fit_transform(X)
        else:
            X_scaled = X
        
        # Train
        model = self.models[model_name]
        
        if hasattr(model, "fit"):
            model.fit(X_scaled, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)
            
            metrics = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "n_samples": len(y),
                "n_features": X.shape[1]
            }
            
            self.model_performance[model_name] = metrics
            logger.info(f"Trained {model_name}: CV={metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}")
            
            return metrics
        
        return {}
    
    def predict(
        self,
        X: np.ndarray,
        market_id: str,
        model_name: str = "direction_rf"
    ) -> Prediction:
        """Make prediction with a model."""
        if model_name not in self.models:
            # Return neutral prediction
            return Prediction(
                market_id=market_id,
                model_name=model_name,
                prediction=0.5,
                confidence=0.0
            )
        
        model = self.models[model_name]
        
        # Scale features
        if "features" in self.scalers and hasattr(self.scalers["features"], "transform"):
            try:
                X_scaled = self.scalers["features"].transform(X.reshape(1, -1))
            except:
                X_scaled = X.reshape(1, -1)
        else:
            X_scaled = X.reshape(1, -1)
        
        # Predict
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[0]
                prediction = proba[1] if len(proba) > 1 else proba[0]
                confidence = max(proba) - 0.5
            elif hasattr(model, "predict"):
                prediction = model.predict(X_scaled)[0]
                confidence = 0.5
            else:
                prediction = 0.5
                confidence = 0.0
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            prediction = 0.5
            confidence = 0.0
        
        result = Prediction(
            market_id=market_id,
            model_name=model_name,
            prediction=float(prediction),
            confidence=float(confidence),
            features_used=self.feature_names
        )
        
        self.predictions_log.append(result)
        return result
    
    def ensemble_predict(
        self,
        X: np.ndarray,
        market_id: str,
        weights: Dict[str, float] = None
    ) -> Prediction:
        """Ensemble prediction from multiple models."""
        weights = weights or {name: 1.0 for name in self.models}
        
        predictions = []
        total_weight = 0
        
        for model_name, weight in weights.items():
            if model_name in self.models:
                pred = self.predict(X, market_id, model_name)
                predictions.append((pred.prediction, pred.confidence, weight))
                total_weight += weight
        
        if not predictions:
            return Prediction(
                market_id=market_id,
                model_name="ensemble",
                prediction=0.5,
                confidence=0.0
            )
        
        # Weighted average
        weighted_pred = sum(p * w for p, c, w in predictions) / total_weight
        avg_confidence = sum(c * w for p, c, w in predictions) / total_weight
        
        return Prediction(
            market_id=market_id,
            model_name="ensemble",
            prediction=weighted_pred,
            confidence=avg_confidence,
            metadata={"models_used": list(weights.keys())}
        )
    
    def save_model(self, model_name: str):
        """Save model to disk."""
        if model_name not in self.models:
            return
        
        model_path = self.model_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.models[model_name], f)
        
        # Save scaler
        if "features" in self.scalers:
            scaler_path = self.model_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scalers["features"], f)
        
        logger.info(f"Saved model: {model_name}")
    
    def load_model(self, model_name: str):
        """Load model from disk."""
        model_path = self.model_dir / f"{model_name}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.models[model_name] = pickle.load(f)
            logger.info(f"Loaded model: {model_name}")
        
        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scalers["features"] = pickle.load(f)
    
    def update_performance(
        self,
        prediction_id: str,
        actual_outcome: float
    ):
        """Update model performance with actual outcome."""
        for pred in self.predictions_log:
            if pred.timestamp.isoformat() == prediction_id:
                error = abs(pred.prediction - actual_outcome)
                pred.metadata["actual"] = actual_outcome
                pred.metadata["error"] = error
                
                # Update model performance
                model_name = pred.model_name
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = {"errors": []}
                
                self.model_performance[model_name].setdefault("errors", []).append(error)
                break
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics for all models."""
        stats = {}
        for name, perf in self.model_performance.items():
            errors = perf.get("errors", [])
            stats[name] = {
                "cv_mean": perf.get("cv_mean", 0),
                "cv_std": perf.get("cv_std", 0),
                "predictions": len(errors),
                "mae": np.mean(errors) if errors else 0,
                "mse": np.mean([e**2 for e in errors]) if errors else 0
            }
        return stats


class PricePredictionModel:
    """Specialized model for price prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self._trained = False
        
        if SKLEARN_AVAILABLE:
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            self.scaler = StandardScaler()
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train price prediction model."""
        if not SKLEARN_AVAILABLE or self.model is None:
            return {}
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._trained = True
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        return {
            "mse": np.mean((self.model.predict(X_scaled) - y) ** 2),
            "r2": self.model.score(X_scaled, y),
            "top_features": np.argsort(importances)[-5:].tolist()
        }
    
    def predict(self, X: np.ndarray) -> Tuple[float, float]:
        """Predict price with confidence."""
        if not self._trained or self.model is None:
            return 0.5, 0.0
        
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        prediction = self.model.predict(X_scaled)[0]
        
        # Estimate confidence from prediction variance
        confidence = min(0.9, abs(prediction - 0.5) * 2)
        
        return float(prediction), float(confidence)


class SignalClassifier:
    """Classifier for trading signals (BUY/SELL/HOLD)."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self._trained = False
        self.classes = ["SELL", "HOLD", "BUY"]
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42
            )
            self.scaler = StandardScaler()
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Train signal classifier."""
        if not SKLEARN_AVAILABLE or self.model is None:
            return {}
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._trained = True
        
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        return {
            "accuracy": self.model.score(X_scaled, y),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
    
    def predict(self, X: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Predict signal with probabilities."""
        if not self._trained or self.model is None:
            return "HOLD", {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
        
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        proba_dict = {
            cls: float(prob) 
            for cls, prob in zip(self.model.classes_, probabilities)
        }
        
        return self.classes[int(prediction) + 1] if isinstance(prediction, (int, float)) else str(prediction), proba_dict


# PyTorch Models (if available)
if TORCH_AVAILABLE:
    class PriceLSTM(nn.Module):
        """LSTM model for price sequence prediction."""
        
        def __init__(
            self,
            input_size: int = 50,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    
    class AttentionModel(nn.Module):
        """Attention-based model for market analysis."""
        
        def __init__(
            self,
            input_size: int = 50,
            d_model: int = 64,
            n_heads: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.input_proj = nn.Linear(input_size, d_model)
            self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.norm = nn.LayerNorm(d_model)
            
            self.fc = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.input_proj(x)
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            return self.fc(x.mean(dim=1))

else:
    # Dummy classes if PyTorch not available
    class PriceLSTM:
        def __init__(self, *args, **kwargs):
            pass
    
    class AttentionModel:
        def __init__(self, *args, **kwargs):
            pass
