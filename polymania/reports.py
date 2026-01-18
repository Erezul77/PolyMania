import os, csv, json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict

SIGNALS_CSV = 'data/trading_signals.csv'
TRADES_CSV = 'data/paper_trades.csv'
PORTFOLIO_FILE = 'data/paper_portfolio.json'

def load_signals(days=1):
    if not os.path.exists(SIGNALS_CSV): return []
    cutoff = datetime.utcnow() - timedelta(days=days)
    signals = []
    with open(SIGNALS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row['timestamp'].replace('Z',''))
                if ts >= cutoff:
                    signals.append(row)
            except: pass
    return signals

def load_trades(days=1):
    if not os.path.exists(TRADES_CSV): return []
    cutoff = datetime.utcnow() - timedelta(days=days)
    trades = []
    with open(TRADES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row.get('timestamp','').replace('Z',''))
                if ts >= cutoff:
                    trades.append(row)
            except: pass
    return trades

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE): return None
    with open(PORTFOLIO_FILE, 'r') as f:
        return json.load(f)

def analyze_signals(signals):
    by_type = defaultdict(int)
    by_confidence = {'high': 0, 'medium': 0, 'low': 0}
    for s in signals:
        by_type[s.get('signal_type', 'UNKNOWN')] += 1
        conf = float(s.get('confidence', 0))
        if conf >= 0.7: by_confidence['high'] += 1
        elif conf >= 0.5: by_confidence['medium'] += 1
        else: by_confidence['low'] += 1
    return by_type, by_confidence

def analyze_trades(trades):
    buys, sells = 0, 0
    total_pnl = 0
    for t in trades:
        action = t.get('action', '')
        if action == 'BUY': buys += 1
        elif action == 'SELL':
            sells += 1
            reason = t.get('reason', '')
            if 'PnL:' in reason:
                try:
                    pnl = float(reason.split('PnL:')[1].replace('$','').strip())
                    total_pnl += pnl
                except: pass
    return buys, sells, total_pnl

def generate_daily_report():
    signals = load_signals(days=1)
    trades = load_trades(days=1)
    portfolio = load_portfolio()
    
    by_type, by_conf = analyze_signals(signals)
    buys, sells, pnl = analyze_trades(trades)
    
    lines = ['=== DAILY REPORT ===', '']
    
    # Portfolio
    if portfolio:
        cash = portfolio.get('cash', 1000)
        total_trades = portfolio.get('trades', 0)
        lines.append('PORTFOLIO:')
        lines.append('  Cash: $' + str(round(cash, 2)))
        lines.append('  Total Trades: ' + str(total_trades))
        lines.append('')
    
    # Signals
    lines.append('SIGNALS (24h): ' + str(len(signals)))
    for sig_type, count in sorted(by_type.items()):
        lines.append('  ' + sig_type + ': ' + str(count))
    lines.append('')
    lines.append('By Confidence:')
    lines.append('  High (70%+): ' + str(by_conf['high']))
    lines.append('  Medium (50-70%): ' + str(by_conf['medium']))
    lines.append('  Low (<50%): ' + str(by_conf['low']))
    lines.append('')
    
    # Trades
    lines.append('TRADES (24h):')
    lines.append('  Buys: ' + str(buys))
    lines.append('  Sells: ' + str(sells))
    lines.append('  Realized PnL: $' + str(round(pnl, 2)))
    
    return chr(10).join(lines)

def generate_weekly_report():
    signals = load_signals(days=7)
    trades = load_trades(days=7)
    portfolio = load_portfolio()
    
    by_type, by_conf = analyze_signals(signals)
    buys, sells, pnl = analyze_trades(trades)
    
    lines = ['=== WEEKLY REPORT ===', '']
    lines.append('Signals: ' + str(len(signals)))
    lines.append('Trades: ' + str(buys + sells))
    lines.append('Realized PnL: $' + str(round(pnl, 2)))
    
    if portfolio:
        lines.append('')
        lines.append('Portfolio Cash: $' + str(round(portfolio.get('cash', 1000), 2)))
    
    return chr(10).join(lines)

if __name__ == '__main__':
    print(generate_daily_report())
