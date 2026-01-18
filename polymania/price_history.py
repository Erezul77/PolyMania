import csv, os, time, json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict

PRICE_HISTORY_CSV = 'data/price_history.csv'
HEADERS = ['timestamp', 'timestamp_iso', 'event_id', 'event_title', 'event_slug', 'market_id', 'outcome', 'price', 'volume_24h', 'liquidity']

def _ensure_csv_header(path, headers):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(headers)

def _parse_prices(val):
    if val is None: return []
    if isinstance(val, list): return val
    if isinstance(val, str):
        try: return json.loads(val)
        except: return []
    return []

def _parse_outcomes(val):
    if val is None: return ['Yes', 'No']
    if isinstance(val, list): return val
    if isinstance(val, str):
        try: return json.loads(val)
        except: return ['Yes', 'No']
    return ['Yes', 'No']

@dataclass
class PricePoint:
    timestamp: int
    event_id: str
    market_id: str  
    outcome: str
    price: float
    volume_24h: float = 0.0
    liquidity: float = 0.0

@dataclass
class MarketSnapshot:
    timestamp: int
    event_id: str
    event_title: str
    event_slug: str
    outcomes: Dict[str, PricePoint] = field(default_factory=dict)

    @property
    def yes_price(self):
        for k, p in self.outcomes.items():
            if 'yes' in k.lower(): return p.price
        return None

class PriceHistoryCollector:
    def __init__(self, csv_path=PRICE_HISTORY_CSV, max_memory=10000):
        self.csv_path = csv_path
        self.max_memory_points = max_memory
        self._history: Dict[str, List[MarketSnapshot]] = defaultdict(list)
        _ensure_csv_header(self.csv_path, HEADERS)

    def record_event(self, event, markets):
        now = int(time.time())
        event_id = str(event.get('id', ''))
        snap = MarketSnapshot(now, event_id, str(event.get('title', '')), str(event.get('slug', '')))
        rows = []
        for mkt in markets:
            mid = str(mkt.get('id', ''))
            prices = _parse_prices(mkt.get('outcomePrices'))
            names = _parse_outcomes(mkt.get('outcomes'))
            for i, px in enumerate(prices):
                try:
                    price = float(px) if px else 0.0
                except (ValueError, TypeError):
                    price = 0.0
                name = names[i] if i < len(names) else f'O{i}'
                snap.outcomes[f'{mid}_{name}'] = PricePoint(now, event_id, mid, name, price)
                rows.append([now, datetime.utcfromtimestamp(now).isoformat()+'Z', event_id, snap.event_title, snap.event_slug, mid, name, f'{price:.6f}', '0', '0'])
        if rows:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)
        self._history[event_id].append(snap)
        return snap

    def get_history(self, event_id, lookback_sec=3600):
        cutoff = int(time.time()) - lookback_sec
        return [s for s in self._history.get(event_id, []) if s.timestamp >= cutoff]

    def get_price_series(self, event_id, outcome_contains='Yes', lookback_sec=3600):
        series = []
        for snap in self.get_history(event_id, lookback_sec):
            for k, pt in snap.outcomes.items():
                if outcome_contains.lower() in pt.outcome.lower():
                    series.append((pt.timestamp, pt.price))
                    break
        return sorted(series, key=lambda x: x[0])

_collector = None
def get_price_collector():
    global _collector
    if _collector is None: _collector = PriceHistoryCollector()
    return _collector
