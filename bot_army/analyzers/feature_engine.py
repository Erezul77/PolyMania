"""Feature engineering for ML models."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("analyzer.features")


@dataclass
class FeatureSet:
    """Container for computed features."""
    market_id: str
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_vector(self, feature_names: List[str] = None) -> np.ndarray:
        """Convert to numpy array."""
        if feature_names:
            return np.array([self.features.get(f, 0.0) for f in feature_names])
        return np.array(list(self.features.values()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "features": self.features,
            "timestamp": self.timestamp.isoformat()
        }


class FeatureEngine:
    """
    Advanced feature engineering engine.
    Computes 100+ features across multiple dimensions.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
        self._feature_cache: Dict[str, FeatureSet] = {}
    
    def compute_all_features(
        self,
        market_id: str,
        price_history: List[Dict],
        orderbook: Dict = None,
        trades: List[Dict] = None,
        events: Dict = None
    ) -> FeatureSet:
        """Compute all features for a market."""
        
        features = {}
        
        # Price features
        if price_history:
            features.update(self._compute_price_features(price_history))
            features.update(self._compute_technical_features(price_history))
            features.update(self._compute_volatility_features(price_history))
            features.update(self._compute_momentum_features(price_history))
        
        # Orderbook features
        if orderbook:
            features.update(self._compute_orderbook_features(orderbook))
        
        # Trade flow features
        if trades:
            features.update(self._compute_trade_features(trades))
        
        # Event features
        if events:
            features.update(self._compute_event_features(events))
        
        # Time features
        features.update(self._compute_time_features())
        
        feature_set = FeatureSet(
            market_id=market_id,
            features=features
        )
        
        self._feature_cache[market_id] = feature_set
        self.feature_names = list(features.keys())
        
        return feature_set
    
    def _compute_price_features(self, history: List[Dict]) -> Dict[str, float]:
        """Compute price-based features."""
        if not history:
            return {}
        
        prices = [h.get("price", h.get("close", 0)) for h in history]
        prices = [p for p in prices if p > 0]
        
        if len(prices) < 2:
            return {}
        
        current = prices[-1]
        
        features = {
            # Current price level
            "price_current": current,
            "price_distance_from_50": abs(current - 0.5),
            
            # Price returns
            "return_1": (current - prices[-2]) / prices[-2] if prices[-2] > 0 else 0,
            "return_5": (current - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))] if len(prices) >= 5 else 0,
            "return_10": (current - prices[-min(10, len(prices))]) / prices[-min(10, len(prices))] if len(prices) >= 10 else 0,
            "return_20": (current - prices[-min(20, len(prices))]) / prices[-min(20, len(prices))] if len(prices) >= 20 else 0,
            
            # Price statistics
            "price_mean": np.mean(prices),
            "price_std": np.std(prices) if len(prices) > 1 else 0,
            "price_min": np.min(prices),
            "price_max": np.max(prices),
            "price_range": np.max(prices) - np.min(prices),
            
            # Z-score
            "price_zscore": (current - np.mean(prices)) / np.std(prices) if np.std(prices) > 0 else 0,
            
            # Percentile
            "price_percentile": sum(1 for p in prices if p <= current) / len(prices),
        }
        
        return features
    
    def _compute_technical_features(self, history: List[Dict]) -> Dict[str, float]:
        """Compute technical indicator features."""
        prices = [h.get("price", h.get("close", 0)) for h in history]
        prices = [p for p in prices if p > 0]
        
        if len(prices) < 20:
            return {}
        
        current = prices[-1]
        
        # Moving averages
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        
        # EMA
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        
        # MACD
        macd = ema_12 - ema_26
        signal = self._ema([macd], 9) if macd else 0
        
        # RSI
        rsi = self._rsi(prices, 14)
        
        # Bollinger Bands
        bb_mid = sma_20
        bb_std = np.std(prices[-20:])
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        bb_position = (current - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        features = {
            # Moving averages
            "sma_5": sma_5,
            "sma_10": sma_10,
            "sma_20": sma_20,
            "price_vs_sma5": current / sma_5 - 1 if sma_5 > 0 else 0,
            "price_vs_sma20": current / sma_20 - 1 if sma_20 > 0 else 0,
            "sma5_vs_sma20": sma_5 / sma_20 - 1 if sma_20 > 0 else 0,
            
            # MACD
            "macd": macd,
            "macd_signal": signal,
            "macd_histogram": macd - signal,
            
            # RSI
            "rsi": rsi,
            "rsi_overbought": 1 if rsi > 70 else 0,
            "rsi_oversold": 1 if rsi < 30 else 0,
            
            # Bollinger Bands
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "bb_position": bb_position,
            "bb_squeeze": 1 if bb_width < 0.1 else 0,
        }
        
        return features
    
    def _compute_volatility_features(self, history: List[Dict]) -> Dict[str, float]:
        """Compute volatility features."""
        prices = [h.get("price", h.get("close", 0)) for h in history]
        prices = [p for p in prices if p > 0]
        
        if len(prices) < 10:
            return {}
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 5:
            return {}
        
        # Historical volatility
        vol_5 = np.std(returns[-5:]) * np.sqrt(252) if len(returns) >= 5 else 0
        vol_10 = np.std(returns[-10:]) * np.sqrt(252) if len(returns) >= 10 else 0
        vol_20 = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0
        
        # ATR (Average True Range) - simplified
        highs = [h.get("high", h.get("price", 0)) for h in history]
        lows = [h.get("low", h.get("price", 0)) for h in history]
        
        if len(highs) >= 14:
            true_ranges = [
                max(h - l, abs(h - prices[i-1]) if i > 0 else 0, abs(l - prices[i-1]) if i > 0 else 0)
                for i, (h, l) in enumerate(zip(highs[-14:], lows[-14:]))
            ]
            atr = np.mean(true_ranges)
        else:
            atr = 0
        
        # Volatility regime
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1
        
        features = {
            "volatility_5": vol_5,
            "volatility_10": vol_10,
            "volatility_20": vol_20,
            "volatility_ratio": vol_ratio,
            "atr": atr,
            "vol_expanding": 1 if vol_ratio > 1.2 else 0,
            "vol_contracting": 1 if vol_ratio < 0.8 else 0,
            "high_volatility": 1 if vol_20 > 0.5 else 0,
            "low_volatility": 1 if vol_20 < 0.2 else 0,
        }
        
        return features
    
    def _compute_momentum_features(self, history: List[Dict]) -> Dict[str, float]:
        """Compute momentum features."""
        prices = [h.get("price", h.get("close", 0)) for h in history]
        prices = [p for p in prices if p > 0]
        
        if len(prices) < 10:
            return {}
        
        # Rate of change
        roc_5 = (prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))] * 100 if len(prices) >= 5 else 0
        roc_10 = (prices[-1] - prices[-min(10, len(prices))]) / prices[-min(10, len(prices))] * 100 if len(prices) >= 10 else 0
        
        # Momentum
        mom_5 = prices[-1] - prices[-min(5, len(prices))] if len(prices) >= 5 else 0
        mom_10 = prices[-1] - prices[-min(10, len(prices))] if len(prices) >= 10 else 0
        
        # Trend strength
        up_moves = sum(1 for i in range(1, min(20, len(prices))) if prices[-i] > prices[-i-1])
        down_moves = min(20, len(prices)) - 1 - up_moves
        trend_strength = (up_moves - down_moves) / (up_moves + down_moves) if (up_moves + down_moves) > 0 else 0
        
        # Acceleration
        returns = np.diff(prices[-10:]) if len(prices) >= 10 else []
        acceleration = np.diff(returns).mean() if len(returns) > 1 else 0
        
        features = {
            "roc_5": roc_5,
            "roc_10": roc_10,
            "momentum_5": mom_5,
            "momentum_10": mom_10,
            "trend_strength": trend_strength,
            "trend_up": 1 if trend_strength > 0.3 else 0,
            "trend_down": 1 if trend_strength < -0.3 else 0,
            "acceleration": acceleration,
            "accelerating": 1 if acceleration > 0.001 else 0,
            "decelerating": 1 if acceleration < -0.001 else 0,
        }
        
        return features
    
    def _compute_orderbook_features(self, orderbook: Dict) -> Dict[str, float]:
        """Compute orderbook features."""
        features = {
            "ob_best_bid": orderbook.get("best_bid", 0),
            "ob_best_ask": orderbook.get("best_ask", 0),
            "ob_spread": orderbook.get("spread", 0),
            "ob_spread_bps": orderbook.get("spread_bps", 0),
            "ob_mid_price": orderbook.get("mid_price", 0),
            "ob_bid_depth": orderbook.get("bid_depth", 0),
            "ob_ask_depth": orderbook.get("ask_depth", 0),
            "ob_imbalance": orderbook.get("imbalance", 0),
            "ob_total_depth": orderbook.get("bid_depth", 0) + orderbook.get("ask_depth", 0),
        }
        
        # Derived features
        total_depth = features["ob_total_depth"]
        if total_depth > 0:
            features["ob_bid_ratio"] = features["ob_bid_depth"] / total_depth
            features["ob_ask_ratio"] = features["ob_ask_depth"] / total_depth
        else:
            features["ob_bid_ratio"] = 0.5
            features["ob_ask_ratio"] = 0.5
        
        return features
    
    def _compute_trade_features(self, trades: List[Dict]) -> Dict[str, float]:
        """Compute trade flow features."""
        if not trades:
            return {}
        
        buy_volume = sum(t.get("value", 0) for t in trades if t.get("side", "").upper() == "BUY")
        sell_volume = sum(t.get("value", 0) for t in trades if t.get("side", "").upper() == "SELL")
        total_volume = buy_volume + sell_volume
        
        buy_count = sum(1 for t in trades if t.get("side", "").upper() == "BUY")
        sell_count = len(trades) - buy_count
        
        avg_size = total_volume / len(trades) if trades else 0
        
        # Large trades (> $1000)
        large_trades = [t for t in trades if t.get("value", 0) > 1000]
        large_buy = sum(t.get("value", 0) for t in large_trades if t.get("side", "").upper() == "BUY")
        large_sell = sum(t.get("value", 0) for t in large_trades if t.get("side", "").upper() == "SELL")
        
        features = {
            "trade_count": len(trades),
            "trade_buy_volume": buy_volume,
            "trade_sell_volume": sell_volume,
            "trade_total_volume": total_volume,
            "trade_net_flow": buy_volume - sell_volume,
            "trade_flow_ratio": buy_volume / sell_volume if sell_volume > 0 else 1,
            "trade_buy_count": buy_count,
            "trade_sell_count": sell_count,
            "trade_avg_size": avg_size,
            "trade_large_count": len(large_trades),
            "trade_large_buy": large_buy,
            "trade_large_sell": large_sell,
            "trade_large_flow": large_buy - large_sell,
            "trade_buy_pressure": 1 if buy_volume > sell_volume * 1.5 else 0,
            "trade_sell_pressure": 1 if sell_volume > buy_volume * 1.5 else 0,
        }
        
        return features
    
    def _compute_event_features(self, event: Dict) -> Dict[str, float]:
        """Compute event-related features."""
        now = datetime.utcnow()
        
        features = {
            "event_volume": event.get("volume", 0),
            "event_liquidity": event.get("liquidity", 0),
            "event_outcome_count": len(event.get("outcomes", [])),
        }
        
        # Time to event end
        end_date = event.get("end_date")
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z", ""))
            time_to_end = (end_date - now).total_seconds() / 3600
            features["event_hours_to_end"] = max(0, time_to_end)
            features["event_ending_soon"] = 1 if 0 < time_to_end < 24 else 0
            features["event_ending_imminent"] = 1 if 0 < time_to_end < 4 else 0
        
        return features
    
    def _compute_time_features(self) -> Dict[str, float]:
        """Compute time-based features."""
        now = datetime.utcnow()
        
        features = {
            "time_hour": now.hour,
            "time_day_of_week": now.weekday(),
            "time_is_weekend": 1 if now.weekday() >= 5 else 0,
            "time_is_us_trading": 1 if 14 <= now.hour <= 21 else 0,
            "time_is_asia_trading": 1 if 0 <= now.hour <= 8 else 0,
            "time_is_europe_trading": 1 if 8 <= now.hour <= 16 else 0,
        }
        
        # Cyclical encoding
        features["time_hour_sin"] = np.sin(2 * np.pi * now.hour / 24)
        features["time_hour_cos"] = np.cos(2 * np.pi * now.hour / 24)
        features["time_dow_sin"] = np.sin(2 * np.pi * now.weekday() / 7)
        features["time_dow_cos"] = np.cos(2 * np.pi * now.weekday() / 7)
        
        return features
    
    def _ema(self, data: List[float], period: int) -> float:
        """Calculate EMA."""
        if not data or len(data) < period:
            return data[-1] if data else 0
        
        multiplier = 2 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50
        
        changes = np.diff(prices[-(period+1):])
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names
    
    def get_cached_features(self, market_id: str) -> Optional[FeatureSet]:
        """Get cached features for market."""
        return self._feature_cache.get(market_id)
