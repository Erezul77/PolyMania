"""Pattern detection for technical analysis."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("analyzer.patterns")


class PatternType(Enum):
    """Types of patterns."""
    # Trend patterns
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"
    
    # Reversal patterns
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    HEAD_SHOULDERS = "HEAD_SHOULDERS"
    INVERSE_HEAD_SHOULDERS = "INVERSE_HEAD_SHOULDERS"
    
    # Continuation patterns
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRICAL_TRIANGLE = "SYMMETRICAL_TRIANGLE"
    FLAG = "FLAG"
    PENNANT = "PENNANT"
    
    # Candlestick patterns
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    DOJI = "DOJI"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    
    # Momentum patterns
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    SQUEEZE = "SQUEEZE"
    DIVERGENCE = "DIVERGENCE"


@dataclass
class Pattern:
    """Detected pattern."""
    pattern_type: PatternType
    market_id: str
    confidence: float
    start_idx: int
    end_idx: int
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    signal: str = "NEUTRAL"  # BUY, SELL, NEUTRAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern_type.value,
            "market_id": self.market_id,
            "confidence": self.confidence,
            "signal": self.signal,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class PatternDetector:
    """
    Detects technical patterns in price data.
    Uses statistical and heuristic methods.
    """
    
    def __init__(self):
        self._detected_patterns: Dict[str, List[Pattern]] = {}
        self._history_size = 50
    
    def detect_all(
        self,
        market_id: str,
        prices: List[float],
        volumes: List[float] = None
    ) -> List[Pattern]:
        """Detect all patterns in price data."""
        patterns = []
        
        if len(prices) < 20:
            return patterns
        
        # Convert to numpy
        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes) if volumes else None
        
        # Trend patterns
        trend = self._detect_trend(market_id, prices_arr)
        if trend:
            patterns.append(trend)
        
        # Reversal patterns
        patterns.extend(self._detect_reversal_patterns(market_id, prices_arr))
        
        # Breakout/Breakdown
        breakout = self._detect_breakout(market_id, prices_arr, volumes_arr)
        if breakout:
            patterns.append(breakout)
        
        # Squeeze (volatility contraction)
        squeeze = self._detect_squeeze(market_id, prices_arr)
        if squeeze:
            patterns.append(squeeze)
        
        # Divergence (price vs momentum)
        divergence = self._detect_divergence(market_id, prices_arr)
        if divergence:
            patterns.append(divergence)
        
        # Store patterns
        if market_id not in self._detected_patterns:
            self._detected_patterns[market_id] = []
        self._detected_patterns[market_id].extend(patterns)
        self._detected_patterns[market_id] = self._detected_patterns[market_id][-self._history_size:]
        
        return patterns
    
    def _detect_trend(
        self,
        market_id: str,
        prices: np.ndarray
    ) -> Optional[Pattern]:
        """Detect trend direction."""
        if len(prices) < 10:
            return None
        
        # Linear regression
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # R-squared for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine trend
        slope_normalized = slope / np.mean(prices)
        
        if slope_normalized > 0.001:
            pattern_type = PatternType.UPTREND
            signal = "BUY"
        elif slope_normalized < -0.001:
            pattern_type = PatternType.DOWNTREND
            signal = "SELL"
        else:
            pattern_type = PatternType.SIDEWAYS
            signal = "NEUTRAL"
        
        return Pattern(
            pattern_type=pattern_type,
            market_id=market_id,
            confidence=r_squared,
            start_idx=0,
            end_idx=len(prices) - 1,
            signal=signal,
            metadata={
                "slope": slope_normalized,
                "r_squared": r_squared
            }
        )
    
    def _detect_reversal_patterns(
        self,
        market_id: str,
        prices: np.ndarray
    ) -> List[Pattern]:
        """Detect reversal patterns."""
        patterns = []
        
        if len(prices) < 30:
            return patterns
        
        # Find local peaks and troughs
        peaks = self._find_peaks(prices)
        troughs = self._find_troughs(prices)
        
        # Double Top
        if len(peaks) >= 2:
            last_peaks = peaks[-2:]
            if abs(prices[last_peaks[0]] - prices[last_peaks[1]]) / prices[last_peaks[0]] < 0.03:
                # Two peaks at similar level
                neckline = min(prices[last_peaks[0]:last_peaks[1]+1])
                if prices[-1] < neckline:
                    patterns.append(Pattern(
                        pattern_type=PatternType.DOUBLE_TOP,
                        market_id=market_id,
                        confidence=0.7,
                        start_idx=last_peaks[0],
                        end_idx=len(prices) - 1,
                        price_target=neckline - (prices[last_peaks[0]] - neckline),
                        signal="SELL"
                    ))
        
        # Double Bottom
        if len(troughs) >= 2:
            last_troughs = troughs[-2:]
            if abs(prices[last_troughs[0]] - prices[last_troughs[1]]) / prices[last_troughs[0]] < 0.03:
                # Two troughs at similar level
                neckline = max(prices[last_troughs[0]:last_troughs[1]+1])
                if prices[-1] > neckline:
                    patterns.append(Pattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        market_id=market_id,
                        confidence=0.7,
                        start_idx=last_troughs[0],
                        end_idx=len(prices) - 1,
                        price_target=neckline + (neckline - prices[last_troughs[0]]),
                        signal="BUY"
                    ))
        
        return patterns
    
    def _detect_breakout(
        self,
        market_id: str,
        prices: np.ndarray,
        volumes: np.ndarray = None
    ) -> Optional[Pattern]:
        """Detect breakout/breakdown."""
        if len(prices) < 20:
            return None
        
        recent = prices[-5:]
        historical = prices[-25:-5]
        
        high = np.max(historical)
        low = np.min(historical)
        current = prices[-1]
        
        # Volume confirmation
        volume_surge = False
        if volumes is not None and len(volumes) >= 20:
            avg_volume = np.mean(volumes[-25:-5])
            recent_volume = np.mean(volumes[-5:])
            volume_surge = recent_volume > avg_volume * 1.5
        
        confidence = 0.6 + (0.2 if volume_surge else 0)
        
        # Breakout above resistance
        if current > high * 1.02:
            return Pattern(
                pattern_type=PatternType.BREAKOUT,
                market_id=market_id,
                confidence=confidence,
                start_idx=len(prices) - 5,
                end_idx=len(prices) - 1,
                price_target=current + (current - high),
                stop_loss=high,
                signal="BUY",
                metadata={"resistance": high, "volume_confirmed": volume_surge}
            )
        
        # Breakdown below support
        if current < low * 0.98:
            return Pattern(
                pattern_type=PatternType.BREAKDOWN,
                market_id=market_id,
                confidence=confidence,
                start_idx=len(prices) - 5,
                end_idx=len(prices) - 1,
                price_target=current - (low - current),
                stop_loss=low,
                signal="SELL",
                metadata={"support": low, "volume_confirmed": volume_surge}
            )
        
        return None
    
    def _detect_squeeze(
        self,
        market_id: str,
        prices: np.ndarray
    ) -> Optional[Pattern]:
        """Detect volatility squeeze."""
        if len(prices) < 30:
            return None
        
        # Calculate Bollinger Band width
        window = 20
        sma = np.convolve(prices, np.ones(window)/window, mode='valid')
        std = np.array([np.std(prices[i:i+window]) for i in range(len(prices)-window+1)])
        
        if len(std) < 10:
            return None
        
        bb_width = (2 * std) / sma
        
        # Recent vs historical width
        recent_width = np.mean(bb_width[-5:])
        historical_width = np.mean(bb_width[-20:-5])
        
        # Squeeze if width contracted significantly
        if recent_width < historical_width * 0.6:
            return Pattern(
                pattern_type=PatternType.SQUEEZE,
                market_id=market_id,
                confidence=0.65,
                start_idx=len(prices) - 10,
                end_idx=len(prices) - 1,
                signal="NEUTRAL",  # Direction unknown until breakout
                metadata={
                    "width_ratio": recent_width / historical_width,
                    "potential_move": historical_width - recent_width
                }
            )
        
        return None
    
    def _detect_divergence(
        self,
        market_id: str,
        prices: np.ndarray
    ) -> Optional[Pattern]:
        """Detect price/momentum divergence."""
        if len(prices) < 20:
            return None
        
        # Calculate RSI
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.convolve(gains, np.ones(14)/14, mode='valid')
        avg_loss = np.convolve(losses, np.ones(14)/14, mode='valid')
        
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) < 5:
            return None
        
        # Compare price and RSI trends
        price_trend = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        rsi_trend = (rsi[-1] - rsi[-5]) / 100 if len(rsi) >= 5 else 0
        
        # Bullish divergence: price down, RSI up
        if price_trend < -0.02 and rsi_trend > 0.05:
            return Pattern(
                pattern_type=PatternType.DIVERGENCE,
                market_id=market_id,
                confidence=0.6,
                start_idx=len(prices) - 10,
                end_idx=len(prices) - 1,
                signal="BUY",
                metadata={
                    "type": "bullish",
                    "price_change": price_trend,
                    "rsi_change": rsi_trend
                }
            )
        
        # Bearish divergence: price up, RSI down
        if price_trend > 0.02 and rsi_trend < -0.05:
            return Pattern(
                pattern_type=PatternType.DIVERGENCE,
                market_id=market_id,
                confidence=0.6,
                start_idx=len(prices) - 10,
                end_idx=len(prices) - 1,
                signal="SELL",
                metadata={
                    "type": "bearish",
                    "price_change": price_trend,
                    "rsi_change": rsi_trend
                }
            )
        
        return None
    
    def _find_peaks(self, prices: np.ndarray, order: int = 3) -> List[int]:
        """Find local maxima."""
        peaks = []
        for i in range(order, len(prices) - order):
            if all(prices[i] > prices[i-j] for j in range(1, order+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, order+1)):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, prices: np.ndarray, order: int = 3) -> List[int]:
        """Find local minima."""
        troughs = []
        for i in range(order, len(prices) - order):
            if all(prices[i] < prices[i-j] for j in range(1, order+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, order+1)):
                troughs.append(i)
        return troughs
    
    def get_patterns(self, market_id: str) -> List[Pattern]:
        """Get detected patterns for market."""
        return self._detected_patterns.get(market_id, [])
    
    def get_active_signals(self) -> List[Pattern]:
        """Get all active trading signals from patterns."""
        signals = []
        for patterns in self._detected_patterns.values():
            for p in patterns[-5:]:  # Recent patterns only
                if p.signal != "NEUTRAL":
                    signals.append(p)
        return signals
