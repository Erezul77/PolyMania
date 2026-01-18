"""
Meta-Learner: ML model that learns which strategy to trust when.
=================================================================

Inspired by Renaissance Technologies / Two Sigma approach:
- Learns from historical strategy performance
- Considers market regime, features, and context
- Dynamically adjusts strategy weights
- Uses gradient boosting for fast online learning
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import pickle
from pathlib import Path

logger = logging.getLogger("strategy.meta_learner")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available for meta-learner")


@dataclass
class StrategyContext:
    """Context for a strategy signal."""
    strategy_name: str
    signal_type: str  # BUY, SELL
    confidence: float
    market_id: str
    features: Dict[str, float]
    regime: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaFeatures:
    """Features for the meta-learner."""
    # Market features
    regime: str
    volatility: float
    trend_strength: float
    rsi: float
    
    # Strategy agreement
    num_buy_signals: int
    num_sell_signals: int
    avg_confidence: float
    max_confidence: float
    strategy_agreement: float  # -1 to 1
    
    # Time features
    hour: int
    day_of_week: int
    
    # Recent performance (rolling)
    strategy_recent_wins: Dict[str, float]
    strategy_recent_pnl: Dict[str, float]
    
    def to_vector(self, strategy_names: List[str]) -> np.ndarray:
        """Convert to feature vector."""
        base_features = [
            self._regime_to_num(self.regime),
            self.volatility,
            self.trend_strength,
            self.rsi / 100,
            self.num_buy_signals,
            self.num_sell_signals,
            self.avg_confidence,
            self.max_confidence,
            self.strategy_agreement,
            self.hour / 24,
            self.day_of_week / 7,
        ]
        
        # Add per-strategy recent performance
        for name in strategy_names:
            base_features.append(self.strategy_recent_wins.get(name, 0.5))
            base_features.append(self.strategy_recent_pnl.get(name, 0))
        
        return np.array(base_features, dtype=np.float32)
    
    def _regime_to_num(self, regime: str) -> float:
        """Convert regime to numeric."""
        regime_map = {
            "trending_up": 1.0,
            "trending_down": -1.0,
            "mean_reverting": 0.0,
            "high_volatility": 0.5,
            "low_volatility": -0.5,
            "ranging": 0.0,
            "unknown": 0.0,
        }
        return regime_map.get(regime.lower(), 0.0)


class MetaLearner:
    """
    Meta-learner that predicts which strategy will perform best.
    
    Uses historical data to learn:
    - Which strategy works in which regime
    - How to weight strategies based on recent performance
    - Context-dependent strategy selection
    """
    
    def __init__(
        self,
        strategy_names: List[str],
        model_path: str = "/app/models/meta_learner.pkl",
        min_samples: int = 50,
        retrain_interval: int = 100,
        lookback_trades: int = 50
    ):
        self.strategy_names = sorted(strategy_names)
        self.model_path = Path(model_path)
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval
        self.lookback_trades = lookback_trades
        
        # Models (one per strategy - predicts if strategy will be profitable)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Training data
        self.training_data: List[Tuple[np.ndarray, str, float]] = []  # (features, strategy, outcome)
        self._samples_since_train = 0
        
        # Recent performance tracking
        self._recent_outcomes: Dict[str, List[float]] = defaultdict(list)
        self._recent_pnl: Dict[str, List[float]] = defaultdict(list)
        
        # Load existing model
        self._load_model()
        
        logger.info(f"MetaLearner initialized for {len(strategy_names)} strategies")
    
    def predict_best_strategy(
        self,
        contexts: List[StrategyContext],
        features: Dict[str, float],
        regime: str
    ) -> Tuple[str, Dict[str, float]]:
        """
        Predict which strategy will perform best given current context.
        
        Returns:
            (best_strategy_name, {strategy: probability})
        """
        if not contexts:
            return None, {}
        
        if not SKLEARN_AVAILABLE or not self.models:
            # Fallback to highest confidence
            best = max(contexts, key=lambda c: c.confidence)
            probs = {c.strategy_name: c.confidence for c in contexts}
            return best.strategy_name, probs
        
        # Build meta-features
        meta_features = self._build_meta_features(contexts, features, regime)
        X = meta_features.to_vector(self.strategy_names).reshape(1, -1)
        
        # Get predictions from each strategy model
        strategy_scores = {}
        
        for strategy_name in self.strategy_names:
            if strategy_name not in self.models:
                # Default score based on recent performance
                recent_wr = np.mean(self._recent_outcomes.get(strategy_name, [0.5]))
                strategy_scores[strategy_name] = recent_wr
                continue
            
            model = self.models[strategy_name]
            scaler = self.scalers.get(strategy_name)
            
            try:
                X_scaled = scaler.transform(X) if scaler else X
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)[0]
                    # Probability of profitable trade
                    score = proba[1] if len(proba) > 1 else proba[0]
                else:
                    score = model.predict(X_scaled)[0]
                
                strategy_scores[strategy_name] = float(score)
            except Exception as e:
                logger.debug(f"Prediction error for {strategy_name}: {e}")
                strategy_scores[strategy_name] = 0.5
        
        # Weight by signal presence and confidence
        for ctx in contexts:
            if ctx.strategy_name in strategy_scores:
                # Boost score if strategy has a signal
                strategy_scores[ctx.strategy_name] *= (1 + ctx.confidence * 0.5)
        
        # Find best
        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
        else:
            best_strategy = contexts[0].strategy_name
        
        # Normalize to probabilities
        total = sum(strategy_scores.values()) or 1
        probabilities = {k: v / total for k, v in strategy_scores.items()}
        
        return best_strategy, probabilities
    
    def record_outcome(
        self,
        strategy_name: str,
        features: np.ndarray,
        pnl: float,
        profitable: bool
    ):
        """Record trade outcome for learning."""
        # Track recent performance
        self._recent_outcomes[strategy_name].append(1.0 if profitable else 0.0)
        self._recent_outcomes[strategy_name] = self._recent_outcomes[strategy_name][-self.lookback_trades:]
        
        self._recent_pnl[strategy_name].append(pnl)
        self._recent_pnl[strategy_name] = self._recent_pnl[strategy_name][-self.lookback_trades:]
        
        # Store training sample
        self.training_data.append((features, strategy_name, 1.0 if profitable else 0.0))
        self.training_data = self.training_data[-5000:]  # Keep last 5000 samples
        
        self._samples_since_train += 1
        
        # Retrain if enough new samples
        if self._samples_since_train >= self.retrain_interval:
            self._train_models()
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights based on recent performance."""
        weights = {}
        
        for name in self.strategy_names:
            recent_wr = self._recent_outcomes.get(name, [])
            recent_pnl = self._recent_pnl.get(name, [])
            
            if recent_wr:
                # Weight = win_rate * (1 + avg_pnl_sign)
                wr = np.mean(recent_wr)
                pnl_sign = 1 if np.mean(recent_pnl) > 0 else 0.5 if not recent_pnl else 0.8
                weights[name] = wr * pnl_sign
            else:
                weights[name] = 0.5  # Default
        
        # Normalize
        total = sum(weights.values()) or 1
        return {k: v / total for k, v in weights.items()}
    
    def _build_meta_features(
        self,
        contexts: List[StrategyContext],
        features: Dict[str, float],
        regime: str
    ) -> MetaFeatures:
        """Build meta-features from strategy contexts."""
        now = datetime.utcnow()
        
        # Count signals
        buy_signals = [c for c in contexts if c.signal_type == "BUY"]
        sell_signals = [c for c in contexts if c.signal_type == "SELL"]
        
        confidences = [c.confidence for c in contexts]
        
        # Strategy agreement (-1 = all sell, +1 = all buy)
        if contexts:
            agreement = (len(buy_signals) - len(sell_signals)) / len(contexts)
        else:
            agreement = 0
        
        # Recent win rates
        recent_wins = {
            name: np.mean(outcomes) if outcomes else 0.5
            for name, outcomes in self._recent_outcomes.items()
        }
        
        # Recent PnL (normalized)
        recent_pnl = {}
        for name, pnls in self._recent_pnl.items():
            if pnls:
                # Normalize to roughly -1 to 1
                recent_pnl[name] = np.tanh(np.mean(pnls) * 10)
            else:
                recent_pnl[name] = 0
        
        return MetaFeatures(
            regime=regime,
            volatility=features.get("volatility_20", 0),
            trend_strength=features.get("trend_strength", 0),
            rsi=features.get("rsi", 50),
            num_buy_signals=len(buy_signals),
            num_sell_signals=len(sell_signals),
            avg_confidence=np.mean(confidences) if confidences else 0,
            max_confidence=max(confidences) if confidences else 0,
            strategy_agreement=agreement,
            hour=now.hour,
            day_of_week=now.weekday(),
            strategy_recent_wins=recent_wins,
            strategy_recent_pnl=recent_pnl
        )
    
    def _train_models(self):
        """Train models for each strategy."""
        if not SKLEARN_AVAILABLE:
            return
        
        logger.info(f"Training meta-learner with {len(self.training_data)} samples")
        
        # Group by strategy
        strategy_data: Dict[str, List[Tuple]] = defaultdict(list)
        for features, strategy, outcome in self.training_data:
            strategy_data[strategy].append((features, outcome))
        
        for strategy_name in self.strategy_names:
            samples = strategy_data.get(strategy_name, [])
            
            if len(samples) < self.min_samples:
                continue
            
            X = np.array([s[0] for s in samples])
            y = np.array([s[1] for s in samples])
            
            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0, posinf=1, neginf=-1)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train gradient boosting
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            try:
                model.fit(X_scaled, y)
                
                # Cross-validation score
                if len(y) >= 20:
                    cv_scores = cross_val_score(model, X_scaled, y, cv=3)
                    logger.info(f"  {strategy_name}: CV accuracy = {np.mean(cv_scores):.3f}")
                
                self.models[strategy_name] = model
                self.scalers[strategy_name] = scaler
                
            except Exception as e:
                logger.error(f"Training error for {strategy_name}: {e}")
        
        self._samples_since_train = 0
        self._save_model()
    
    def _save_model(self):
        """Save models to disk."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "models": self.models,
                "scalers": self.scalers,
                "recent_outcomes": dict(self._recent_outcomes),
                "recent_pnl": dict(self._recent_pnl),
                "strategy_names": self.strategy_names
            }
            
            with open(self.model_path, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Meta-learner saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving meta-learner: {e}")
    
    def _load_model(self):
        """Load models from disk."""
        if not self.model_path.exists():
            return
        
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            
            self.models = data.get("models", {})
            self.scalers = data.get("scalers", {})
            self._recent_outcomes = defaultdict(list, data.get("recent_outcomes", {}))
            self._recent_pnl = defaultdict(list, data.get("recent_pnl", {}))
            
            logger.info(f"Meta-learner loaded: {len(self.models)} strategy models")
        except Exception as e:
            logger.error(f"Error loading meta-learner: {e}")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get meta-learner statistics."""
        return {
            "trained_models": len(self.models),
            "total_samples": len(self.training_data),
            "samples_since_train": self._samples_since_train,
            "strategy_weights": self.get_strategy_weights(),
            "recent_win_rates": {
                name: np.mean(outcomes) if outcomes else None
                for name, outcomes in self._recent_outcomes.items()
            }
        }
