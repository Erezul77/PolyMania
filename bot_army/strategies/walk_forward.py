"""
Walk-Forward Optimization & Model Retraining
=============================================

Industry-standard approach used by top quant funds:
- Train on rolling window of data
- Validate on out-of-sample period
- Automatic model retraining on schedule
- Performance degradation detection
- Hyperparameter optimization
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import pickle
from pathlib import Path
from enum import Enum

logger = logging.getLogger("strategy.walk_forward")

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class RetrainTrigger(Enum):
    """Reasons for model retraining."""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"
    COLD_START = "cold_start"


@dataclass
class ModelMetrics:
    """Performance metrics for a model."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "timestamp": self.timestamp.isoformat()
        }
    
    def degradation_score(self, baseline: "ModelMetrics") -> float:
        """Calculate performance degradation vs baseline."""
        if baseline.sharpe == 0:
            return 0
        
        sharpe_drop = (baseline.sharpe - self.sharpe) / max(abs(baseline.sharpe), 0.1)
        wr_drop = (baseline.win_rate - self.win_rate) / max(baseline.win_rate, 0.1)
        dd_increase = (self.max_drawdown - baseline.max_drawdown) / max(baseline.max_drawdown, 0.01)
        
        # Weighted degradation score (higher = worse)
        return sharpe_drop * 0.4 + wr_drop * 0.3 + dd_increase * 0.3


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    train_window_days: int = 30  # Training data window
    test_window_days: int = 7   # Out-of-sample test window
    retrain_interval_hours: int = 24  # How often to retrain
    min_train_samples: int = 100  # Minimum samples to train
    degradation_threshold: float = 0.3  # Trigger retrain if degradation > threshold
    n_splits: int = 5  # Cross-validation splits
    save_models: bool = True
    model_path: str = "/app/models/walk_forward"


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Manager.
    
    Implements rolling-window training with out-of-sample validation,
    commonly used by quantitative hedge funds.
    """
    
    def __init__(
        self,
        config: WalkForwardConfig = None,
        model_factory: Callable = None
    ):
        self.config = config or WalkForwardConfig()
        self.model_factory = model_factory
        
        # Training data buffer
        self._train_buffer: List[Tuple[np.ndarray, float]] = []  # (features, outcome)
        self._max_buffer_size = 10000
        
        # Model state
        self._current_model = None
        self._model_trained_at: datetime = None
        self._baseline_metrics: ModelMetrics = None
        self._current_metrics: ModelMetrics = None
        
        # Performance tracking
        self._recent_predictions: List[Tuple[float, float]] = []  # (predicted, actual)
        self._recent_pnl: List[float] = []
        
        # History
        self._retrain_history: List[Dict] = []
        self._metrics_history: List[ModelMetrics] = []
        
        # Model persistence
        self._model_path = Path(self.config.model_path)
        self._model_path.mkdir(parents=True, exist_ok=True)
        
        self._load_state()
        
        logger.info(f"WalkForward initialized: train={self.config.train_window_days}d, "
                   f"test={self.config.test_window_days}d, retrain={self.config.retrain_interval_hours}h")
    
    def add_sample(
        self,
        features: np.ndarray,
        outcome: float,  # 1 for profitable, 0 for loss
        pnl: float = 0,
        timestamp: datetime = None
    ):
        """Add a training sample."""
        self._train_buffer.append((features, outcome))
        self._train_buffer = self._train_buffer[-self._max_buffer_size:]
        
        self._recent_pnl.append(pnl)
        self._recent_pnl = self._recent_pnl[-500:]
        
        # Check if retrain needed
        if self._should_retrain():
            self.retrain()
    
    def add_prediction(self, predicted: float, actual: float):
        """Record a prediction and actual outcome for tracking."""
        self._recent_predictions.append((predicted, actual))
        self._recent_predictions = self._recent_predictions[-500:]
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        """Make prediction with current model."""
        if self._current_model is None:
            return None
        
        try:
            features = np.nan_to_num(features, nan=0, posinf=1, neginf=-1)
            
            if hasattr(self._current_model, "predict_proba"):
                proba = self._current_model.predict_proba(features.reshape(1, -1))[0]
                return proba[1] if len(proba) > 1 else proba[0]
            else:
                return self._current_model.predict(features.reshape(1, -1))[0]
        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            return None
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        # Cold start
        if self._current_model is None:
            if len(self._train_buffer) >= self.config.min_train_samples:
                logger.info("Cold start retrain triggered")
                return True
            return False
        
        # Scheduled retrain
        if self._model_trained_at:
            hours_since_train = (datetime.utcnow() - self._model_trained_at).total_seconds() / 3600
            if hours_since_train >= self.config.retrain_interval_hours:
                logger.info(f"Scheduled retrain: {hours_since_train:.1f}h since last train")
                return True
        
        # Performance degradation
        current = self._calculate_current_metrics()
        if self._baseline_metrics and current:
            degradation = current.degradation_score(self._baseline_metrics)
            if degradation > self.config.degradation_threshold:
                logger.info(f"Performance degradation retrain: score={degradation:.3f}")
                return True
        
        return False
    
    def retrain(self, trigger: RetrainTrigger = RetrainTrigger.SCHEDULED) -> bool:
        """Retrain the model using walk-forward methodology."""
        if len(self._train_buffer) < self.config.min_train_samples:
            logger.warning(f"Not enough samples for retrain: {len(self._train_buffer)}")
            return False
        
        logger.info(f"Starting walk-forward retrain ({trigger.value})...")
        
        # Prepare data
        X = np.array([s[0] for s in self._train_buffer])
        y = np.array([s[1] for s in self._train_buffer])
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=1, neginf=-1)
        
        # Walk-forward validation
        best_model = None
        best_score = -np.inf
        cv_scores = []
        
        if SKLEARN_AVAILABLE and len(y) >= self.config.n_splits * 10:
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Create and train model
                model = self._create_model()
                if model is None:
                    continue
                
                try:
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    cv_scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
                except Exception as e:
                    logger.error(f"Fold {fold} training error: {e}")
        else:
            # Simple train on all data if not enough for CV
            model = self._create_model()
            if model:
                try:
                    model.fit(X, y)
                    best_model = model
                    best_score = 0.5
                except Exception as e:
                    logger.error(f"Training error: {e}")
        
        if best_model is None:
            logger.error("Retraining failed - no valid model")
            return False
        
        # Update model
        self._current_model = best_model
        self._model_trained_at = datetime.utcnow()
        
        # Update baseline metrics
        self._baseline_metrics = self._calculate_current_metrics()
        self._current_metrics = self._baseline_metrics
        
        # Record history
        self._retrain_history.append({
            "timestamp": self._model_trained_at.isoformat(),
            "trigger": trigger.value,
            "samples": len(self._train_buffer),
            "cv_scores": cv_scores,
            "best_score": best_score
        })
        self._retrain_history = self._retrain_history[-100:]  # Keep last 100
        
        if self._baseline_metrics:
            self._metrics_history.append(self._baseline_metrics)
            self._metrics_history = self._metrics_history[-100:]
        
        # Save
        self._save_state()
        
        logger.info(f"Retrain complete: score={best_score:.3f}, cv_mean={np.mean(cv_scores) if cv_scores else 0:.3f}")
        return True
    
    def _create_model(self):
        """Create a new model instance."""
        if self.model_factory:
            return self.model_factory()
        
        # Default: Gradient Boosting Classifier
        if SKLEARN_AVAILABLE:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
        
        return None
    
    def _calculate_current_metrics(self) -> Optional[ModelMetrics]:
        """Calculate current performance metrics."""
        if len(self._recent_predictions) < 20:
            return None
        
        predictions = np.array([p[0] for p in self._recent_predictions[-100:]])
        actuals = np.array([p[1] for p in self._recent_predictions[-100:]])
        
        # Binarize predictions
        pred_binary = (predictions > 0.5).astype(int)
        actual_binary = actuals.astype(int)
        
        # Calculate metrics
        metrics = ModelMetrics()
        
        try:
            if SKLEARN_AVAILABLE:
                metrics.accuracy = accuracy_score(actual_binary, pred_binary)
                metrics.precision = precision_score(actual_binary, pred_binary, zero_division=0)
                metrics.f1 = f1_score(actual_binary, pred_binary, zero_division=0)
            
            # Trading metrics
            metrics.win_rate = np.mean(actual_binary)
            metrics.total_trades = len(actuals)
            
            # PnL-based metrics
            if len(self._recent_pnl) >= 20:
                pnl_array = np.array(self._recent_pnl[-100:])
                
                # Sharpe
                if np.std(pnl_array) > 0:
                    metrics.sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252)
                
                # Profit factor
                wins = pnl_array[pnl_array > 0]
                losses = pnl_array[pnl_array < 0]
                if len(losses) > 0 and np.sum(np.abs(losses)) > 0:
                    metrics.profit_factor = np.sum(wins) / np.sum(np.abs(losses))
                
                # Max drawdown
                cumulative = np.cumsum(pnl_array)
                peak = np.maximum.accumulate(cumulative)
                drawdown = (peak - cumulative) / np.maximum(peak, 1e-8)
                metrics.max_drawdown = np.max(drawdown)
            
            metrics.timestamp = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
        
        return metrics
    
    def _save_state(self):
        """Save model and state to disk."""
        if not self.config.save_models:
            return
        
        try:
            state = {
                "model": self._current_model,
                "trained_at": self._model_trained_at,
                "baseline_metrics": self._baseline_metrics,
                "retrain_history": self._retrain_history[-20:],
                "train_buffer_size": len(self._train_buffer)
            }
            
            with open(self._model_path / "model_state.pkl", "wb") as f:
                pickle.dump(state, f)
            
            logger.info("Walk-forward state saved")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load model and state from disk."""
        state_path = self._model_path / "model_state.pkl"
        if not state_path.exists():
            return
        
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            
            self._current_model = state.get("model")
            self._model_trained_at = state.get("trained_at")
            self._baseline_metrics = state.get("baseline_metrics")
            self._retrain_history = state.get("retrain_history", [])
            
            logger.info(f"Walk-forward state loaded: trained_at={self._model_trained_at}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def force_retrain(self):
        """Force immediate retrain."""
        return self.retrain(trigger=RetrainTrigger.MANUAL)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "model_ready": self._current_model is not None,
            "trained_at": self._model_trained_at.isoformat() if self._model_trained_at else None,
            "buffer_size": len(self._train_buffer),
            "recent_predictions": len(self._recent_predictions),
            "retrain_count": len(self._retrain_history),
            "current_metrics": self._current_metrics.to_dict() if self._current_metrics else None,
            "baseline_metrics": self._baseline_metrics.to_dict() if self._baseline_metrics else None,
            "last_retrain": self._retrain_history[-1] if self._retrain_history else None
        }
