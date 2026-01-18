import os
import json
import csv
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
from itertools import product

logger = logging.getLogger('polymania.multi_strategy')

STRATEGIES_FILE = 'data/strategies.json'
STRATEGY_RESULTS_FILE = 'data/strategy_results.csv'

@dataclass
class Strategy:
    id: int
    name: str
    take_profit: float
    stop_loss: float
    position_size: float
    min_confidence: float
    event_types: List[str]
    cash: float = 1000.0
    positions: Dict = field(default_factory=dict)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0

def generate_strategies(total_cash=100000):
    """Generate OPTIMIZED strategies based on 7-day backtesting results.
    Winners: Small positions ($50-100), High confidence (70%), Tight TP/SL (5-10%/3-5%)
    """
    strategies = []
    
    # OPTIMIZED parameters based on actual performance data:
    # - Small positions ONLY (no $500 which lost money)
    # - High confidence ONLY (70%+ performed best)
    # - Tight take-profits (5-10% worked, 15-20% too greedy)
    # - Tight stop-losses (3-5% performed well)
    take_profits = [0.05, 0.08, 0.10]        # Winners: 5-10%
    stop_losses = [0.03, 0.05, 0.08]         # Winners: tight stops
    position_sizes = [50, 100]               # Winners: small positions ONLY
    min_confidences = [0.70]                 # Winners: high confidence ONLY
    event_filters = [['all'], ['crypto'], ['sports'], ['politics'], ['other']]
    
    combo_id = 0
    num_strategies = len(take_profits) * len(stop_losses) * len(position_sizes) * len(min_confidences) * len(event_filters)
    cash_per = total_cash / num_strategies
    
    for tp, sl, ps, mc, et in product(take_profits, stop_losses, position_sizes, min_confidences, event_filters):
        name = 'TP' + str(int(tp*100)) + '_SL' + str(int(sl*100)) + '_PS' + str(ps) + '_MC' + str(int(mc*100))
        if et != ['all']:
            name += '_' + et[0].upper()[:4]
        strategies.append(Strategy(id=combo_id, name=name, take_profit=tp, stop_loss=sl,
            position_size=ps, min_confidence=mc, event_types=et, cash=cash_per))
        combo_id += 1
    
    logger.info('Generated ' + str(len(strategies)) + ' optimized strategies with $' + str(round(cash_per,2)) + ' each')
    return strategies

def save_strategies(strategies):
    os.makedirs('data', exist_ok=True)
    data = []
    for s in strategies:
        data.append({'id': s.id, 'name': s.name, 'take_profit': s.take_profit, 'stop_loss': s.stop_loss,
            'position_size': s.position_size, 'min_confidence': s.min_confidence, 'event_types': s.event_types,
            'cash': s.cash, 'positions': s.positions, 'total_trades': s.total_trades, 'wins': s.wins,
            'losses': s.losses, 'realized_pnl': s.realized_pnl})
    with open(STRATEGIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def load_strategies():
    if not os.path.exists(STRATEGIES_FILE):
        return []
    with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    strategies = []
    for d in data:
        strategies.append(Strategy(id=d['id'], name=d['name'], take_profit=d['take_profit'],
            stop_loss=d['stop_loss'], position_size=d['position_size'], min_confidence=d['min_confidence'],
            event_types=d['event_types'], cash=d['cash'], positions=d.get('positions', {}),
            total_trades=d.get('total_trades', 0), wins=d.get('wins', 0), losses=d.get('losses', 0),
            realized_pnl=d.get('realized_pnl', 0.0)))
    return strategies

def classify_event(title):
    t = title.lower()
    if any(x in t for x in ['bitcoin', 'btc', 'ethereum', 'solana', 'xrp', 'crypto', 'price']):
        return 'crypto'
    if any(x in t for x in ['fc', 'vs.', 'united', 'city', 'league', 'cup', 'nba', 'nfl']):
        return 'sports'
    if any(x in t for x in ['trump', 'biden', 'election', 'president', 'congress']):
        return 'politics'
    return 'other'

class MultiStrategyTrader:
    def __init__(self, force_reset=False):
        self.strategies = load_strategies() if not force_reset else []
        if not self.strategies:
            logger.info('Initializing OPTIMIZED strategies with $100,000')
            self.strategies = generate_strategies(100000)
            save_strategies(self.strategies)
            logger.info('Created ' + str(len(self.strategies)) + ' optimized strategies')
    
    def execute_buy(self, signal, price):
        results = []
        for s in self.strategies:
            if signal.confidence < s.min_confidence:
                continue
            if 'all' not in s.event_types:
                if classify_event(signal.event_title) not in s.event_types:
                    continue
            key = str(signal.event_id) + '_' + signal.outcome
            if key in s.positions:
                continue
            trade_size = min(s.position_size, s.cash * 0.9)
            if trade_size < 10:
                continue
            shares = trade_size / price
            s.cash -= trade_size
            s.positions[key] = {'event_id': signal.event_id, 'title': signal.event_title,
                'outcome': signal.outcome, 'shares': shares, 'entry_price': price,
                'entry_time': datetime.utcnow().isoformat()}
            s.total_trades += 1
            results.append('S' + str(s.id) + ':BUY')
        if results:
            save_strategies(self.strategies)
        return results
    
    def check_positions(self, current_prices):
        results = []
        for s in self.strategies:
            for key, pos in list(s.positions.items()):
                eid = pos['event_id']
                if eid not in current_prices:
                    continue
                curr = current_prices[eid]
                entry = pos['entry_price']
                pnl_pct = (curr - entry) / entry
                action = None
                if pnl_pct >= s.take_profit:
                    action = 'TP'
                elif pnl_pct <= -s.stop_loss:
                    action = 'SL'
                if action:
                    shares = pos['shares']
                    pnl = shares * (curr - entry)
                    s.cash += shares * curr
                    if pnl >= 0:
                        s.wins += 1
                    else:
                        s.losses += 1
                    s.realized_pnl += pnl
                    del s.positions[key]
                    results.append('S' + str(s.id) + ':' + action + ' $' + str(round(pnl,2)))
        if results:
            save_strategies(self.strategies)
        return results
    
    def get_summary(self):
        num = len(self.strategies)
        lines = ['=== ' + str(num) + ' OPTIMIZED STRATEGIES ===']
        total_cash = sum(s.cash for s in self.strategies)
        total_pos = sum(len(s.positions) for s in self.strategies)
        total_trades = sum(s.total_trades for s in self.strategies)
        total_wins = sum(s.wins for s in self.strategies)
        total_losses = sum(s.losses for s in self.strategies)
        total_pnl = sum(s.realized_pnl for s in self.strategies)
        lines.append('Cash: $' + str(round(total_cash,2)))
        lines.append('Positions: ' + str(total_pos))
        lines.append('Trades: ' + str(total_trades))
        lines.append('W/L: ' + str(total_wins) + '/' + str(total_losses))
        lines.append('PnL: $' + str(round(total_pnl,2)))
        lines.append('')
        lines.append('TOP 5:')
        top = sorted(self.strategies, key=lambda x: x.realized_pnl, reverse=True)[:5]
        for s in top:
            pv = sum(p['shares'] * p['entry_price'] for p in s.positions.values())
            lines.append('  ' + s.name + ': $' + str(round(s.cash+pv,0)) + ' PnL=$' + str(round(s.realized_pnl,2)))
        lines.append('')
        lines.append('BOTTOM 5:')
        bottom = sorted(self.strategies, key=lambda x: x.realized_pnl)[:5]
        for s in bottom:
            pv = sum(p['shares'] * p['entry_price'] for p in s.positions.values())
            lines.append('  ' + s.name + ': $' + str(round(s.cash+pv,0)) + ' PnL=$' + str(round(s.realized_pnl,2)))
        return chr(10).join(lines)

_multi_trader = None
def get_multi_trader():
    global _multi_trader
    if _multi_trader is None:
        _multi_trader = MultiStrategyTrader()
    return _multi_trader

def reset_strategies():
    """Reset and regenerate with OPTIMIZED strategy parameters."""
    global _multi_trader
    if os.path.exists(STRATEGIES_FILE):
        os.remove(STRATEGIES_FILE)
    _multi_trader = None
    logger.info('Resetting to optimized strategies')
    return generate_strategies(100000)

def reconfigure_trader():
    """Force reconfigure trader with new optimized strategies."""
    global _multi_trader
    if os.path.exists(STRATEGIES_FILE):
        os.remove(STRATEGIES_FILE)
    _multi_trader = MultiStrategyTrader(force_reset=True)
    return _multi_trader
