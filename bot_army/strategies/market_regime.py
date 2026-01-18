"""Market Regime Strategy - Regime-specific trading."""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from ..core.events import SignalType


class MarketRegimeStrategy(BaseStrategy):
    """
    Regime-aware strategy that adapts behavior based on detected market regime.
    
    - Trending: Follow momentum
    - Ranging: Mean reversion
    - High volatility: Wider stops, smaller size
    - Low volatility: Tighter stops, fade extremes
    """
    
    def __init__(self, config: StrategyConfig = None):
        if config is None:
            config = StrategyConfig(
                name="market_regime",
                min_confidence=0.5,
                position_size_pct=0.08,
                stop_loss_pct=0.1,
                take_profit_pct=0.18,
                cooldown_seconds=600
            )
        super().__init__(config)
        
        self._current_regime = "unknown"
        self._regime_confidence = 0.0
    
    def _detect_regime(self, prices: np.ndarray, features: Dict[str, float]) -> tuple:
        """Detect current market regime."""
        if len(prices) < 20:
            return "unknown", 0.0
        
        # Volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        # Trend strength
        sma_5 = np.mean(prices[-5:])
        sma_20 = np.mean(prices[-20:])
        trend = (sma_5 - sma_20) / sma_20
        
        # Hurst exponent approximation (simplified)
        half = len(returns) // 2
        if half > 5:
            rs1 = (max(returns[:half]) - min(returns[:half])) / (np.std(returns[:half]) + 1e-8)
            rs2 = (max(returns[half:]) - min(returns[half:])) / (np.std(returns[half:]) + 1e-8)
            hurst_proxy = np.log(rs2 / (rs1 + 1e-8) + 1) / np.log(2)
        else:
            hurst_proxy = 0.5
        
        # Classify regime
        if volatility > 0.5:
            regime = "high_volatility"
            confidence = min(1, volatility / 0.8)
        elif volatility < 0.15:
            regime = "low_volatility"
            confidence = min(1, 0.15 / (volatility + 0.01))
        elif abs(trend) > 0.05:
            regime = "trending_up" if trend > 0 else "trending_down"
            confidence = min(1, abs(trend) / 0.1)
        elif hurst_proxy < 0.4:
            regime = "mean_reverting"
            confidence = min(1, (0.5 - hurst_proxy) / 0.2)
        else:
            regime = "ranging"
            confidence = 0.5
        
        return regime, confidence
    
    def analyze(
        self,
        market_id: str,
        features: Dict[str, float],
        price_history: List[float],
        orderbook: Dict = None,
        trades: List[Dict] = None
    ) -> Optional[StrategyResult]:
        
        if not self.should_generate_signal(market_id):
            return None
        
        if len(price_history) < 25:
            return None
        
        prices = np.array(price_history)
        current_price = prices[-1]
        
        # Detect regime
        regime, regime_conf = self._detect_regime(prices, features)
        self._current_regime = regime
        self._regime_confidence = regime_conf
        
        signal_type = None
        reasoning = [f"Regime: {regime} ({regime_conf:.1%})"]
        
        # Regime-specific logic
        if regime == "trending_up":
            # Buy pullbacks in uptrend
            pullback = (prices[-1] - prices[-5]) / prices[-5]
            if -0.03 < pullback < -0.01:
                signal_type = SignalType.BUY
                reasoning.append("Buying pullback in uptrend")
        
        elif regime == "trending_down":
            # Sell rallies in downtrend
            rally = (prices[-1] - prices[-5]) / prices[-5]
            if 0.01 < rally < 0.03:
                signal_type = SignalType.SELL
                reasoning.append("Selling rally in downtrend")
        
        elif regime == "mean_reverting" or regime == "ranging":
            # Mean reversion
            mean = np.mean(prices[-20:])
            std = np.std(prices[-20:])
            zscore = (current_price - mean) / (std + 1e-8)
            
            if zscore < -1.5:
                signal_type = SignalType.BUY
                reasoning.append(f"Mean reversion buy (z={zscore:.2f})")
            elif zscore > 1.5:
                signal_type = SignalType.SELL
                reasoning.append(f"Mean reversion sell (z={zscore:.2f})")
        
        elif regime == "high_volatility":
            # Fade extremes in high vol
            returns = np.diff(prices[-10:]) / prices[-11:-1]
            recent_move = np.sum(returns)
            
            if recent_move < -0.08:
                signal_type = SignalType.BUY
                reasoning.append("Fade extreme down move in high vol")
            elif recent_move > 0.08:
                signal_type = SignalType.SELL
                reasoning.append("Fade extreme up move in high vol")
        
        elif regime == "low_volatility":
            # Breakout anticipation
            bb_width = features.get("bb_width", 0.1)
            if bb_width < 0.05:
                momentum = features.get("momentum_5", 0)
                if momentum > 0.01:
                    signal_type = SignalType.BUY
                    reasoning.append("Low vol squeeze - bullish breakout")
                elif momentum < -0.01:
                    signal_type = SignalType.SELL
                    reasoning.append("Low vol squeeze - bearish breakout")
        
        if signal_type is None:
            return None
        
        # Confidence
        confidence = min(0.8, 0.4 + regime_conf * 0.3)
        
        signal = self.create_signal(
            market_id=market_id,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            features=features
        )
        
        return StrategyResult(
            signal=signal,
            strategy_name=self.config.name,
            score=confidence,
            reasoning=reasoning
        )
    
    def set_regime(self, regime: str, confidence: float = 0.5):
        """Externally set regime."""
        self._current_regime = regime
        self._regime_confidence = confidence
