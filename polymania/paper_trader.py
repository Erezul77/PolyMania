import logging
from datetime import datetime
from .paper_portfolio import Portfolio, Position, load_portfolio, save_portfolio, log_trade
from .trading_signals import TradingSignal, SignalType
from .ai_learner import get_ai_learner

logger = logging.getLogger('polymania.paper_trader')

# OPTIMIZED SETTINGS based on 7 days of multi-strategy testing
MIN_CONFIDENCE = 0.70      # Only trade high-confidence signals
POSITION_SIZE = 50         # Small positions = lower risk
TAKE_PROFIT_PCT = 0.05     # 5% take profit (proven winner)
STOP_LOSS_PCT = 0.03       # 3% stop loss (tight risk management)

class PaperTrader:
    def __init__(self, position_size=POSITION_SIZE):
        self.portfolio = load_portfolio()
        self.position_size = position_size
        self.ai = get_ai_learner()
    
    def execute_signal(self, signal, price):
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            return self._buy(signal, price)
        elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            return self._sell(signal, price)
        return None
    
    def _buy(self, signal, price):
        key = signal.event_id + '_' + signal.outcome
        if key in self.portfolio.positions:
            logger.debug('Already have position in ' + signal.event_id)
            return None
        
        # OPTIMIZED: Only trade high-confidence signals (70%+)
        if signal.confidence < MIN_CONFIDENCE:
            logger.debug('Skipping low confidence: ' + str(round(signal.confidence*100)) + '% < ' + str(int(MIN_CONFIDENCE*100)) + '%')
            return None
        
        # AI check - should we trade?
        should, adjusted_conf, factors = self.ai.should_trade(
            signal.event_title, signal.signal_type.value, price, signal.confidence
        )
        
        if not should:
            logger.info('AI blocked trade: ' + signal.event_title[:30] + ' - ' + ', '.join(factors))
            return None
        
        available = self.portfolio.cash_balance * 0.9
        trade_size = min(self.position_size, available)
        if trade_size < 5:
            logger.debug('Not enough cash for trade')
            return None
        
        shares = trade_size / price
        cost = shares * price
        
        self.portfolio.cash_balance -= cost
        self.portfolio.positions[key] = Position(
            signal.event_id, signal.event_title, signal.outcome,
            shares, price, signal.timestamp, price
        )
        self.portfolio.total_trades += 1
        save_portfolio(self.portfolio)
        log_trade('BUY', signal.event_id, signal.event_title, signal.outcome,
                  shares, price, self.portfolio.cash_balance)
        
        ai_note = ' (AI conf: ' + str(round(adjusted_conf*100)) + '%)' if factors else ''
        msg = 'BUY ' + str(round(shares, 1)) + ' shares at $' + str(round(price, 4)) + ai_note
        logger.info(msg)
        return msg
    
    def _sell(self, signal, price):
        key = signal.event_id + '_' + signal.outcome
        if key not in self.portfolio.positions:
            return None
        
        pos = self.portfolio.positions[key]
        proceeds = pos.shares * price
        pnl = proceeds - (pos.shares * pos.avg_price)
        
        self.portfolio.cash_balance += proceeds
        if pnl >= 0:
            self.portfolio.winning_trades += 1
        else:
            self.portfolio.losing_trades += 1
        self.portfolio.total_realized_pnl += pnl
        
        # AI Learning - record outcome
        entry_time = datetime.utcfromtimestamp(pos.opened_at)
        self.ai.record_outcome(
            pos.event_title,
            'BUY',  # original signal that opened position
            pos.avg_price,
            price,
            pnl,
            entry_time
        )
        
        del self.portfolio.positions[key]
        save_portfolio(self.portfolio)
        log_trade('SELL', signal.event_id, signal.event_title, signal.outcome,
                  pos.shares, price, self.portfolio.cash_balance, pnl)
        
        result = 'WIN' if pnl > 0 else 'LOSS'
        msg = 'SELL ' + str(round(pos.shares, 1)) + ' shares PnL=$' + str(round(pnl, 2)) + ' [' + result + ']'
        logger.info(msg)
        return msg
    
    def check_positions(self, current_prices):
        """Check all positions and sell if target/stop hit"""
        sells = []
        for key, pos in list(self.portfolio.positions.items()):
            event_id = pos.event_id
            if event_id not in current_prices:
                continue
            
            current_price = current_prices[event_id]
            pos.current_price = current_price
            
            # Calculate P&L percentage
            pnl_pct = (current_price - pos.avg_price) / pos.avg_price
            
            # OPTIMIZED: Take profit at 5% gain (proven winner)
            if pnl_pct >= TAKE_PROFIT_PCT:
                result = self._close_position(key, current_price, 'TAKE_PROFIT')
                if result:
                    sells.append(result)
            # OPTIMIZED: Stop loss at 3% (tight risk management)
            elif pnl_pct <= -STOP_LOSS_PCT:
                result = self._close_position(key, current_price, 'STOP_LOSS')
                if result:
                    sells.append(result)
        
        save_portfolio(self.portfolio)
        return sells
    
    def _close_position(self, key, price, reason):
        if key not in self.portfolio.positions:
            return None
        
        pos = self.portfolio.positions[key]
        proceeds = pos.shares * price
        pnl = proceeds - (pos.shares * pos.avg_price)
        
        self.portfolio.cash_balance += proceeds
        if pnl >= 0:
            self.portfolio.winning_trades += 1
        else:
            self.portfolio.losing_trades += 1
        self.portfolio.total_realized_pnl += pnl
        
        # AI Learning
        entry_time = datetime.utcfromtimestamp(pos.opened_at)
        self.ai.record_outcome(pos.event_title, 'BUY', pos.avg_price, price, pnl, entry_time)
        
        del self.portfolio.positions[key]
        log_trade(reason, pos.event_id, pos.event_title, pos.outcome,
                  pos.shares, price, self.portfolio.cash_balance, pnl)
        
        result = 'WIN' if pnl > 0 else 'LOSS'
        msg = reason + ' ' + pos.event_title[:25] + ' PnL=$' + str(round(pnl, 2)) + ' [' + result + ']'
        logger.info(msg)
        return msg
    
    def get_summary(self):
        p = self.portfolio
        pos_value = sum(pos.shares * pos.current_price for pos in p.positions.values())
        total = p.cash_balance + pos_value
        return 'Cash: $' + str(round(p.cash_balance, 2)) + ' | Positions: $' + str(round(pos_value, 2)) + ' | Total: $' + str(round(total, 2)) + ' | Trades: ' + str(p.total_trades) + ' | W/L: ' + str(p.winning_trades) + '/' + str(p.losing_trades)
    
    def get_ai_insights(self):
        return self.ai.get_insights()

_trader = None
def get_paper_trader():
    global _trader
    if _trader is None:
        _trader = PaperTrader()
    return _trader

def reset_portfolio():
    global _trader
    import os
    if os.path.exists('data/paper_portfolio.json'):
        os.remove('data/paper_portfolio.json')
    _trader = None
    return Portfolio()
