"""
Multi-Timeframe Analysis
========================

Analyze markets across multiple timeframes:
- 1 minute (scalping signals)
- 5 minutes (short-term)
- 15 minutes (intraday)
- 1 hour (swing)
- 4 hours (position)
- 1 day (macro)

Combines signals from all timeframes for confluence.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger("ml.multi_timeframe")


class Timeframe(Enum):
    """Trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: Timeframe
    trend: str  # "up", "down", "neutral"
    trend_strength: float  # -1 to 1
    momentum: float  # -1 to 1
    overbought: bool
    oversold: bool
    support_nearby: bool
    resistance_nearby: bool
    volume_confirmation: bool
    signal_strength: float  # 0 to 1
    
    def to_dict(self) -> Dict:
        return {
            "timeframe": self.timeframe.value,
            "trend": self.trend,
            "trend_strength": self.trend_strength,
            "momentum": self.momentum,
            "overbought": self.overbought,
            "oversold": self.oversold,
            "support_nearby": self.support_nearby,
            "resistance_nearby": self.resistance_nearby,
            "volume_confirmation": self.volume_confirmation,
            "signal_strength": self.signal_strength
        }


@dataclass
class MultiTimeframeSignal:
    """Combined signal from all timeframes."""
    direction: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float  # 0 to 1
    confluence_score: float  # How many timeframes agree
    signals: Dict[str, TimeframeSignal]
    dominant_timeframe: Timeframe
    reasoning: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "confluence_score": self.confluence_score,
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
            "dominant_timeframe": self.dominant_timeframe.value,
            "reasoning": self.reasoning
        }


class TimeframeAnalyzer:
    """
    Analyzer for a single timeframe.
    """
    
    def __init__(self, timeframe: Timeframe, lookback: int = 100):
        self.timeframe = timeframe
        self.lookback = lookback
        
        # OHLCV data
        self._opens = deque(maxlen=lookback)
        self._highs = deque(maxlen=lookback)
        self._lows = deque(maxlen=lookback)
        self._closes = deque(maxlen=lookback)
        self._volumes = deque(maxlen=lookback)
        self._timestamps = deque(maxlen=lookback)
        
        # Aggregation state (for building candles from ticks)
        self._current_candle = None
        self._candle_start = None
    
    def add_tick(self, price: float, volume: float = 0, timestamp: datetime = None):
        """Add tick and potentially complete a candle."""
        timestamp = timestamp or datetime.utcnow()
        
        # Determine candle boundary
        candle_start = self._get_candle_start(timestamp)
        
        if self._candle_start is None or candle_start > self._candle_start:
            # Complete previous candle if exists
            if self._current_candle is not None:
                self._complete_candle()
            
            # Start new candle
            self._candle_start = candle_start
            self._current_candle = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume
            }
        else:
            # Update current candle
            self._current_candle["high"] = max(self._current_candle["high"], price)
            self._current_candle["low"] = min(self._current_candle["low"], price)
            self._current_candle["close"] = price
            self._current_candle["volume"] += volume
    
    def add_candle(self, open_: float, high: float, low: float, close: float, 
                   volume: float = 0, timestamp: datetime = None):
        """Add completed candle directly."""
        self._opens.append(open_)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._volumes.append(volume)
        self._timestamps.append(timestamp or datetime.utcnow())
    
    def _complete_candle(self):
        """Complete current candle and add to history."""
        if self._current_candle is None:
            return
        
        self._opens.append(self._current_candle["open"])
        self._highs.append(self._current_candle["high"])
        self._lows.append(self._current_candle["low"])
        self._closes.append(self._current_candle["close"])
        self._volumes.append(self._current_candle["volume"])
        self._timestamps.append(self._candle_start)
        
        self._current_candle = None
    
    def _get_candle_start(self, timestamp: datetime) -> datetime:
        """Get the start time of the candle containing this timestamp."""
        if self.timeframe == Timeframe.M1:
            return timestamp.replace(second=0, microsecond=0)
        elif self.timeframe == Timeframe.M5:
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif self.timeframe == Timeframe.M15:
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif self.timeframe == Timeframe.H1:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif self.timeframe == Timeframe.H4:
            hour = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif self.timeframe == Timeframe.D1:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        return timestamp
    
    def analyze(self) -> Optional[TimeframeSignal]:
        """Analyze current timeframe and generate signal."""
        if len(self._closes) < 20:
            return None
        
        closes = np.array(self._closes)
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        volumes = np.array(self._volumes) if self._volumes else None
        
        # Calculate indicators
        trend, trend_strength = self._calculate_trend(closes)
        momentum = self._calculate_momentum(closes)
        rsi = self._calculate_rsi(closes)
        support, resistance = self._find_sr_levels(highs, lows, closes)
        vol_confirm = self._check_volume_confirmation(volumes, closes)
        
        # Determine signal
        overbought = rsi > 70
        oversold = rsi < 30
        current_price = closes[-1]
        support_nearby = abs(current_price - support) / current_price < 0.02
        resistance_nearby = abs(current_price - resistance) / current_price < 0.02
        
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(
            trend_strength, momentum, rsi, vol_confirm,
            support_nearby, resistance_nearby
        )
        
        return TimeframeSignal(
            timeframe=self.timeframe,
            trend=trend,
            trend_strength=trend_strength,
            momentum=momentum,
            overbought=overbought,
            oversold=oversold,
            support_nearby=support_nearby,
            resistance_nearby=resistance_nearby,
            volume_confirmation=vol_confirm,
            signal_strength=signal_strength
        )
    
    def _calculate_trend(self, closes: np.ndarray) -> Tuple[str, float]:
        """Calculate trend direction and strength."""
        # Multiple moving averages
        ma_short = np.mean(closes[-10:])
        ma_medium = np.mean(closes[-20:])
        ma_long = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)
        
        current = closes[-1]
        
        # Calculate trend strength
        above_short = current > ma_short
        above_medium = current > ma_medium
        above_long = current > ma_long
        
        # Trend score (-3 to +3)
        trend_score = sum([
            1 if above_short else -1,
            1 if above_medium else -1,
            1 if above_long else -1
        ])
        
        # MA alignment bonus
        if ma_short > ma_medium > ma_long:
            trend_score += 0.5
        elif ma_short < ma_medium < ma_long:
            trend_score -= 0.5
        
        # Normalize to -1 to 1
        trend_strength = np.clip(trend_score / 3.5, -1, 1)
        
        if trend_strength > 0.3:
            trend = "up"
        elif trend_strength < -0.3:
            trend = "down"
        else:
            trend = "neutral"
        
        return trend, trend_strength
    
    def _calculate_momentum(self, closes: np.ndarray) -> float:
        """Calculate momentum (-1 to 1)."""
        if len(closes) < 14:
            return 0
        
        # Rate of change
        roc_5 = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        roc_10 = (closes[-1] / closes[-10] - 1) if len(closes) >= 10 else 0
        roc_14 = (closes[-1] / closes[-14] - 1) if len(closes) >= 14 else 0
        
        # Average weighted momentum
        momentum = roc_5 * 0.5 + roc_10 * 0.3 + roc_14 * 0.2
        
        # Normalize
        return np.clip(momentum * 20, -1, 1)
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50
        
        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _find_sr_levels(self, highs: np.ndarray, lows: np.ndarray, 
                        closes: np.ndarray) -> Tuple[float, float]:
        """Find support and resistance levels."""
        if len(closes) < 20:
            return closes[-1] * 0.98, closes[-1] * 1.02
        
        # Recent swing lows = support
        support_candidates = []
        for i in range(2, min(len(lows) - 2, 30)):
            if lows[-i] < lows[-i-1] and lows[-i] < lows[-i+1]:
                support_candidates.append(lows[-i])
        
        support = np.mean(support_candidates[-3:]) if support_candidates else lows.min()
        
        # Recent swing highs = resistance
        resistance_candidates = []
        for i in range(2, min(len(highs) - 2, 30)):
            if highs[-i] > highs[-i-1] and highs[-i] > highs[-i+1]:
                resistance_candidates.append(highs[-i])
        
        resistance = np.mean(resistance_candidates[-3:]) if resistance_candidates else highs.max()
        
        return support, resistance
    
    def _check_volume_confirmation(self, volumes: np.ndarray, closes: np.ndarray) -> bool:
        """Check if volume confirms price movement."""
        if volumes is None or len(volumes) < 10:
            return True  # Assume confirmed if no volume data
        
        # Price direction
        price_up = closes[-1] > closes[-5]
        
        # Volume trend
        recent_vol = np.mean(volumes[-3:])
        avg_vol = np.mean(volumes[-20:])
        high_volume = recent_vol > avg_vol * 1.2
        
        # Volume should increase with price moves
        return high_volume
    
    def _calculate_signal_strength(
        self,
        trend_strength: float,
        momentum: float,
        rsi: float,
        vol_confirm: bool,
        support_nearby: bool,
        resistance_nearby: bool
    ) -> float:
        """Calculate overall signal strength (0-1)."""
        # Base from trend and momentum alignment
        alignment = abs(trend_strength) * abs(momentum)
        
        # RSI extreme bonus
        rsi_bonus = 0
        if rsi > 70 or rsi < 30:
            rsi_bonus = 0.2
        
        # Volume confirmation bonus
        vol_bonus = 0.1 if vol_confirm else 0
        
        # S/R proximity bonus
        sr_bonus = 0.15 if (support_nearby or resistance_nearby) else 0
        
        signal_strength = min(alignment + rsi_bonus + vol_bonus + sr_bonus, 1.0)
        
        return signal_strength


class MultiTimeframeEngine:
    """
    Multi-timeframe analysis engine.
    Combines signals from all timeframes.
    """
    
    def __init__(self, timeframes: List[Timeframe] = None):
        self.timeframes = timeframes or [
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1
        ]
        
        # Timeframe weights (higher = more important)
        self.weights = {
            Timeframe.M1: 0.5,
            Timeframe.M5: 0.8,
            Timeframe.M15: 1.0,
            Timeframe.H1: 1.5,
            Timeframe.H4: 1.2,
            Timeframe.D1: 1.0
        }
        
        # Analyzers per timeframe
        self.analyzers: Dict[Timeframe, TimeframeAnalyzer] = {
            tf: TimeframeAnalyzer(tf) for tf in self.timeframes
        }
    
    def add_tick(self, price: float, volume: float = 0, timestamp: datetime = None):
        """Add tick to all timeframe analyzers."""
        for analyzer in self.analyzers.values():
            analyzer.add_tick(price, volume, timestamp)
    
    def add_candle(self, timeframe: Timeframe, open_: float, high: float, 
                   low: float, close: float, volume: float = 0,
                   timestamp: datetime = None):
        """Add candle to specific timeframe."""
        if timeframe in self.analyzers:
            self.analyzers[timeframe].add_candle(open_, high, low, close, volume, timestamp)
    
    def analyze(self) -> MultiTimeframeSignal:
        """
        Analyze all timeframes and generate combined signal.
        """
        signals: Dict[str, TimeframeSignal] = {}
        reasoning = []
        
        # Get signal from each timeframe
        for tf, analyzer in self.analyzers.items():
            signal = analyzer.analyze()
            if signal:
                signals[tf.value] = signal
        
        if not signals:
            return MultiTimeframeSignal(
                direction="NEUTRAL",
                confidence=0,
                confluence_score=0,
                signals={},
                dominant_timeframe=Timeframe.M15,
                reasoning=["Insufficient data for analysis"]
            )
        
        # Calculate confluence
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        for tf, signal in signals.items():
            tf_enum = Timeframe(tf)
            weight = self.weights.get(tf_enum, 1.0)
            total_weight += weight
            
            if signal.trend == "up":
                buy_score += weight * signal.signal_strength
                reasoning.append(f"{tf}: bullish ({signal.signal_strength:.0%})")
            elif signal.trend == "down":
                sell_score += weight * signal.signal_strength
                reasoning.append(f"{tf}: bearish ({signal.signal_strength:.0%})")
            else:
                reasoning.append(f"{tf}: neutral")
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine direction
        if buy_score > sell_score and buy_score > 0.3:
            direction = "BUY"
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            direction = "SELL"
            confidence = sell_score
        else:
            direction = "NEUTRAL"
            confidence = 0
        
        # Calculate confluence score (how many timeframes agree)
        if direction == "BUY":
            agreeing = sum(1 for s in signals.values() if s.trend == "up")
        elif direction == "SELL":
            agreeing = sum(1 for s in signals.values() if s.trend == "down")
        else:
            agreeing = 0
        
        confluence_score = agreeing / len(signals) if signals else 0
        
        # Find dominant timeframe
        dominant_tf = max(
            signals.items(),
            key=lambda x: self.weights.get(Timeframe(x[0]), 1) * x[1].signal_strength
        )
        dominant_timeframe = Timeframe(dominant_tf[0])
        
        return MultiTimeframeSignal(
            direction=direction,
            confidence=confidence,
            confluence_score=confluence_score,
            signals=signals,
            dominant_timeframe=dominant_timeframe,
            reasoning=reasoning
        )
    
    def get_features(self) -> Dict[str, float]:
        """Get features for ML models."""
        features = {}
        
        for tf, analyzer in self.analyzers.items():
            signal = analyzer.analyze()
            if signal:
                prefix = f"mtf_{tf.value}_"
                features[f"{prefix}trend_strength"] = signal.trend_strength
                features[f"{prefix}momentum"] = signal.momentum
                features[f"{prefix}signal_strength"] = signal.signal_strength
                features[f"{prefix}overbought"] = 1.0 if signal.overbought else 0.0
                features[f"{prefix}oversold"] = 1.0 if signal.oversold else 0.0
                features[f"{prefix}vol_confirm"] = 1.0 if signal.volume_confirmation else 0.0
        
        # Confluence features
        mtf_signal = self.analyze()
        features["mtf_confluence"] = mtf_signal.confluence_score
        features["mtf_confidence"] = mtf_signal.confidence
        features["mtf_is_buy"] = 1.0 if mtf_signal.direction == "BUY" else 0.0
        features["mtf_is_sell"] = 1.0 if mtf_signal.direction == "SELL" else 0.0
        
        return features
