import json, os, time, csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

PORTFOLIO_FILE = 'data/paper_portfolio.json'
TRADES_LOG = 'data/paper_trades.csv'

@dataclass
class Position:
    event_id: str
    event_title: str
    outcome: str
    shares: float
    avg_price: float
    opened_at: int
    current_price: float = 0.0

@dataclass 
class Portfolio:
    initial_balance: float = 1000.0
    cash_balance: float = 1000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_realized_pnl: float = 0.0

def save_portfolio(p):
    os.makedirs('data', exist_ok=True)
    data = {
        'cash': p.cash_balance,
        'trades': p.total_trades,
        'wins': p.winning_trades,
        'losses': p.losing_trades,
        'realized_pnl': p.total_realized_pnl,
        'positions': {}
    }
    for key, pos in p.positions.items():
        data['positions'][key] = {
            'event_id': pos.event_id,
            'event_title': pos.event_title,
            'outcome': pos.outcome,
            'shares': pos.shares,
            'avg_price': pos.avg_price,
            'opened_at': pos.opened_at,
            'current_price': pos.current_price
        }
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return Portfolio()
    try:
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        p = Portfolio()
        p.cash_balance = data.get('cash', 1000)
        p.total_trades = data.get('trades', 0)
        p.winning_trades = data.get('wins', 0)
        p.losing_trades = data.get('losses', 0)
        p.total_realized_pnl = data.get('realized_pnl', 0)
        for key, pos_data in data.get('positions', {}).items():
            p.positions[key] = Position(
                pos_data['event_id'],
                pos_data['event_title'],
                pos_data['outcome'],
                pos_data['shares'],
                pos_data['avg_price'],
                pos_data['opened_at'],
                pos_data.get('current_price', 0)
            )
        return p
    except Exception as e:
        print('Error loading portfolio:', e)
        return Portfolio()

def log_trade(action, event_id, title, outcome, shares, price, balance, pnl=0):
    os.makedirs('data', exist_ok=True)
    hdr = ['timestamp', 'action', 'event_id', 'title', 'outcome', 'shares', 'price', 'value', 'pnl', 'balance']
    if not os.path.exists(TRADES_LOG) or os.path.getsize(TRADES_LOG) == 0:
        with open(TRADES_LOG, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(hdr)
    row = [datetime.utcnow().isoformat(), action, event_id, title[:40], outcome,
           round(shares, 2), round(price, 4), round(shares * price, 2), round(pnl, 2), round(balance, 2)]
    with open(TRADES_LOG, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)
