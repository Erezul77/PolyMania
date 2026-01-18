from dataclasses import dataclass
from typing import List, Optional, Tuple
from statistics import mean, stdev

@dataclass
class TechnicalIndicators:
    sma_short: Optional[float] = None
    sma_long: Optional[float] = None
    ema_short: Optional[float] = None
    ema_long: Optional[float] = None
    rsi: Optional[float] = None
    momentum: Optional[float] = None
    roc: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    current_price: Optional[float] = None
    price_change_pct: Optional[float] = None
    trend: Optional[str] = None
    signal_strength: Optional[float] = None

def calculate_sma(prices, period):
    return mean(prices[-period:]) if len(prices) >= period else None

def calculate_ema(prices, period):
    if len(prices) < period: return None
    mult = 2 / (period + 1)
    ema = mean(prices[:period])
    for p in prices[period:]: ema = (p * mult) + (ema * (1 - mult))
    return ema

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(d if d > 0 else 0)
        losses.append(abs(d) if d < 0 else 0)
    if len(gains) < period: return None
    ag, al = mean(gains[-period:]), mean(losses[-period:])
    if al == 0: return 100 if ag > 0 else 50
    return 100 - (100 / (1 + ag/al))

def calculate_momentum(prices, period=10):
    return prices[-1] - prices[-period-1] if len(prices) > period else None

def calculate_macd(prices, fast=12, slow=26, sig=9):
    if len(prices) < slow + sig: return None, None, None
    ef, es = calculate_ema(prices, fast), calculate_ema(prices, slow)
    if ef is None or es is None: return None, None, None
    ml = ef - es
    hist = []
    for i in range(slow, len(prices)+1):
        a, b = calculate_ema(prices[:i], fast), calculate_ema(prices[:i], slow)
        if a and b: hist.append(a - b)
    if len(hist) < sig: return ml, None, None
    sl = calculate_ema(hist, sig)
    return (ml, sl, ml - sl) if sl else (ml, None, None)

def calculate_bollinger(prices, period=20, nstd=2.0):
    if len(prices) < period: return None, None, None, None
    r = prices[-period:]
    m = mean(r)
    s = stdev(r) if len(r) > 1 else 0
    u, l = m + nstd*s, m - nstd*s
    return u, m, l, ((u-l)/m)*100 if m else 0

def determine_trend(prices, sma_s, sma_l):
    if sma_s and sma_l:
        if sma_s > sma_l * 1.01: return 'BULLISH'
        if sma_s < sma_l * 0.99: return 'BEARISH'
    return 'NEUTRAL'

def calculate_signal_strength(rsi, macd_h, trend):
    s = []
    if rsi is not None:
        if rsi > 70: s.append(-0.5 - (rsi-70)/60)
        elif rsi < 30: s.append(0.5 + (30-rsi)/60)
        else: s.append((50-rsi)/40)
    if macd_h is not None: s.append(max(-1, min(1, macd_h*10)))
    s.append(0.5 if trend == 'BULLISH' else (-0.5 if trend == 'BEARISH' else 0))
    return sum(s)/len(s) if s else 0.0

def analyze_price_series(prices, short_p=5, long_p=20):
    if not prices: return TechnicalIndicators()
    ind = TechnicalIndicators()
    ind.current_price = prices[-1]
    if len(prices) >= 2 and prices[0]: ind.price_change_pct = ((prices[-1]-prices[0])/prices[0])*100
    ind.sma_short = calculate_sma(prices, short_p)
    ind.sma_long = calculate_sma(prices, long_p)
    ind.ema_short = calculate_ema(prices, short_p)
    ind.ema_long = calculate_ema(prices, long_p)
    ind.rsi = calculate_rsi(prices)
    ind.momentum = calculate_momentum(prices, min(10, len(prices)-1))
    ind.macd_line, ind.macd_signal, ind.macd_histogram = calculate_macd(prices)
    ind.bb_upper, ind.bb_middle, ind.bb_lower, ind.bb_width = calculate_bollinger(prices, long_p)
    ind.trend = determine_trend(prices, ind.sma_short, ind.sma_long)
    ind.signal_strength = calculate_signal_strength(ind.rsi, ind.macd_histogram, ind.trend)
    return ind

def get_support_resistance_levels(prices, sens=0.02):
    if len(prices) < 10: return [], []
    sup, res = [], []
    for i in range(2, len(prices)-2):
        if prices[i] < min(prices[i-1], prices[i-2], prices[i+1], prices[i+2]): sup.append(prices[i])
        if prices[i] > max(prices[i-1], prices[i-2], prices[i+1], prices[i+2]): res.append(prices[i])
    return sup, res