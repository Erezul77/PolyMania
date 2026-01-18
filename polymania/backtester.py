import csv, os
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class BacktestResult:
    total_signals: int
    trades_executed: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float

class Backtester:
    def __init__(self, initial_balance=1000, position_size=50):
        self.initial_balance = initial_balance
        self.position_size = position_size
    
    def load_price_history(self, csv_path='data/price_history.csv'):
        if not os.path.exists(csv_path): return {}
        prices = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_id = row.get('event_id', '')
                ts = int(row.get('timestamp', 0))
                price = float(row.get('price', 0))
                if event_id not in prices:
                    prices[event_id] = []
                prices[event_id].append((ts, price))
        for event_id in prices:
            prices[event_id].sort(key=lambda x: x[0])
        return prices
    
    def load_signals(self, csv_path='data/trading_signals.csv'):
        if not os.path.exists(csv_path): return []
        signals = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                signals.append({
                    'timestamp': row.get('timestamp', ''),
                    'event_id': row.get('event_id', ''),
                    'signal_type': row.get('signal_type', ''),
                    'confidence': float(row.get('confidence', 0)),
                    'price': float(row.get('price', 0)),
                    'target': float(row.get('target', 0) or 0),
                    'stop': float(row.get('stop', 0) or 0),
                })
        return signals
    
    def run_backtest(self, signals, prices, min_confidence=0.5):
        balance = self.initial_balance
        positions = {}
        trades = []
        equity_curve = [balance]
        
        for sig in signals:
            if sig['confidence'] < min_confidence:
                continue
            
            event_id = sig['event_id']
            sig_type = sig['signal_type']
            price = sig['price']
            
            # BUY signal
            if 'BUY' in sig_type and event_id not in positions:
                shares = min(self.position_size, balance * 0.9) / price
                if shares > 0.1:
                    cost = shares * price
                    balance -= cost
                    positions[event_id] = {
                        'shares': shares,
                        'entry_price': price,
                        'target': sig['target'],
                        'stop': sig['stop']
                    }
            
            # SELL signal
            elif 'SELL' in sig_type and event_id in positions:
                pos = positions[event_id]
                proceeds = pos['shares'] * price
                pnl = proceeds - (pos['shares'] * pos['entry_price'])
                balance += proceeds
                trades.append({
                    'event_id': event_id,
                    'entry': pos['entry_price'],
                    'exit': price,
                    'pnl': pnl,
                    'pnl_pct': (price - pos['entry_price']) / pos['entry_price'] * 100
                })
                del positions[event_id]
            
            equity_curve.append(balance + sum(p['shares'] * price for p in positions.values()))
        
        # Calculate metrics
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        max_dd = 0
        peak = equity_curve[0]
        for eq in equity_curve:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd: max_dd = dd
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        return BacktestResult(
            total_signals=len(signals),
            trades_executed=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=sum(t['pnl'] for t in trades),
            win_rate=len(wins) / len(trades) * 100 if trades else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_dd * 100,
            sharpe_ratio=0  # Simplified
        )
    
    def format_report(self, result):
        lines = [
            '=== BACKTEST RESULTS ===',
            '',
            'Total Signals: ' + str(result.total_signals),
            'Trades Executed: ' + str(result.trades_executed),
            '',
            'Winning: ' + str(result.winning_trades),
            'Losing: ' + str(result.losing_trades),
            'Win Rate: ' + str(round(result.win_rate, 1)) + '%',
            '',
            'Total PnL: $' + str(round(result.total_pnl, 2)),
            'Avg Win: $' + str(round(result.avg_win, 2)),
            'Avg Loss: $' + str(round(result.avg_loss, 2)),
            'Max Drawdown: ' + str(round(result.max_drawdown, 1)) + '%',
        ]
        return chr(10).join(lines)

def run_backtest_cli():
    bt = Backtester()
    prices = bt.load_price_history()
    signals = bt.load_signals()
    if not signals:
        print('No signals found for backtesting')
        return
    result = bt.run_backtest(signals, prices)
    print(bt.format_report(result))

if __name__ == '__main__':
    run_backtest_cli()
