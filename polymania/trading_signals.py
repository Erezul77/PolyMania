import csv, os, time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from .technical_analysis import TechnicalIndicators, analyze_price_series, get_support_resistance_levels

class SignalType(Enum):
    STRONG_BUY = 'STRONG_BUY'
    BUY = 'BUY'
    WEAK_BUY = 'WEAK_BUY'
    HOLD = 'HOLD'
    WEAK_SELL = 'WEAK_SELL'
    SELL = 'SELL'
    STRONG_SELL = 'STRONG_SELL'

class PatternType(Enum):
    BREAKOUT_UP = 'BREAKOUT_UP'
    BREAKOUT_DOWN = 'BREAKOUT_DOWN'
    REVERSAL_BULLISH = 'REVERSAL_BULLISH'
    REVERSAL_BEARISH = 'REVERSAL_BEARISH'
    NONE = 'NONE'

@dataclass
class TradingSignal:
    timestamp: int
    event_id: str
    event_title: str
    outcome: str
    signal_type: SignalType
    confidence: float
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    pattern: PatternType = PatternType.NONE
    indicators: Optional[TechnicalIndicators] = None
    support_levels: List[float] = None
    resistance_levels: List[float] = None
    reasons: List[str] = None
    
    def __post_init__(self):
        self.support_levels = self.support_levels or []
        self.resistance_levels = self.resistance_levels or []
        self.reasons = self.reasons or []
    
    @property
    def risk_reward_ratio(self):
        if self.target_price is None or self.stop_loss is None: return None
        reward = abs(self.target_price - self.current_price)
        risk = abs(self.current_price - self.stop_loss)
        return reward / risk if risk else None

def detect_pattern(prices, ind, sup, res):
    if len(prices) < 10: return PatternType.NONE, 'Insufficient data'
    cur = prices[-1]
    if res and cur > max(res) * 0.98: return PatternType.BREAKOUT_UP, 'Breaking resistance'
    if sup and cur < min(sup) * 1.02: return PatternType.BREAKOUT_DOWN, 'Breaking support'
    if ind.rsi:
        if ind.rsi < 30 and prices[-1] > prices[-2]: return PatternType.REVERSAL_BULLISH, 'Oversold'
        if ind.rsi > 70 and prices[-1] < prices[-2]: return PatternType.REVERSAL_BEARISH, 'Overbought'
    return PatternType.NONE, 'No clear pattern'

def determine_signal_type(ind, pattern, strength):
    reasons = []
    if strength > 0.5: base = SignalType.STRONG_BUY
    elif strength > 0.25: base = SignalType.BUY
    elif strength > 0.1: base = SignalType.WEAK_BUY
    elif strength > -0.1: base = SignalType.HOLD
    elif strength > -0.25: base = SignalType.WEAK_SELL
    elif strength > -0.5: base = SignalType.SELL
    else: base = SignalType.STRONG_SELL
    reasons.append('Strength: ' + str(round(strength, 2)))
    if pattern in [PatternType.BREAKOUT_UP, PatternType.REVERSAL_BULLISH]:
        if base in [SignalType.HOLD, SignalType.WEAK_BUY]: base = SignalType.BUY
    if pattern in [PatternType.BREAKOUT_DOWN, PatternType.REVERSAL_BEARISH]:
        if base in [SignalType.HOLD, SignalType.WEAK_SELL]: base = SignalType.SELL
    if ind.rsi: reasons.append('RSI: ' + str(round(ind.rsi, 1)))
    if ind.trend: reasons.append('Trend: ' + ind.trend)
    return base, min(0.95, 0.5 + abs(strength) * 0.5), reasons

def generate_trading_signal(event_id, event_title, prices, timestamps=None, outcome='Yes'):
    if len(prices) < 5: return None
    ind = analyze_price_series(prices)
    sup, res = get_support_resistance_levels(prices)
    pat, pat_desc = detect_pattern(prices, ind, sup, res)
    sig_type, conf, reasons = determine_signal_type(ind, pat, ind.signal_strength or 0)
    if pat != PatternType.NONE: reasons.insert(0, pat_desc)
    t = min([r for r in res if r > prices[-1]], default=prices[-1]*1.1) if 'BUY' in sig_type.value else max([s for s in sup if s < prices[-1]], default=prices[-1]*0.9)
    s = max([x for x in sup if x < prices[-1]], default=prices[-1]*0.95) if 'BUY' in sig_type.value else min([x for x in res if x > prices[-1]], default=prices[-1]*1.05)
    return TradingSignal(int(time.time()), event_id, event_title, outcome, sig_type, conf, prices[-1], t, s, pat, ind, sup, res, reasons)

SIGNALS_CSV = 'data/trading_signals.csv'

def log_trading_signal(signal):
    os.makedirs(os.path.dirname(SIGNALS_CSV) or '.', exist_ok=True)
    hdr = ['timestamp', 'event_id', 'signal_type', 'confidence', 'price', 'target', 'stop', 'pattern', 'reasons']
    if not os.path.exists(SIGNALS_CSV) or os.path.getsize(SIGNALS_CSV) == 0:
        with open(SIGNALS_CSV, 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerow(hdr)
    row = [datetime.utcfromtimestamp(signal.timestamp).isoformat(), signal.event_id, signal.signal_type.value, round(signal.confidence,2), round(signal.current_price,4), round(signal.target_price,4) if signal.target_price else '', round(signal.stop_loss,4) if signal.stop_loss else '', signal.pattern.value, '|'.join(signal.reasons)]
    with open(SIGNALS_CSV, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)

def format_signal_for_telegram(signal):
    if 'BUY' in signal.signal_type.value:
        sym = 'BUY'
    elif 'SELL' in signal.signal_type.value:
        sym = 'SELL'
    else:
        sym = 'HOLD'
    title = signal.event_title[:50]
    lines = [
        sym + ' - ' + signal.signal_type.value,
        title,
        '',
        'Price: ' + str(round(signal.current_price,4)),
    ]
    if signal.target_price:
        pct = (signal.target_price - signal.current_price) / signal.current_price * 100
        lines.append('Target: ' + str(round(signal.target_price,4)) + ' (' + ('+' if pct>0 else '') + str(round(pct,1)) + '%)')
    if signal.stop_loss:
        pct = (signal.stop_loss - signal.current_price) / signal.current_price * 100
        lines.append('Stop: ' + str(round(signal.stop_loss,4)) + ' (' + ('+' if pct>0 else '') + str(round(pct,1)) + '%)')
    lines.append('')
    lines.append('Confidence: ' + str(int(signal.confidence*100)) + '%')
    return chr(10).join(lines)
