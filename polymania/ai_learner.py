import json, os, pickle, csv
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import statistics

MODEL_FILE = 'data/ai_model.pkl'
OUTCOMES_FILE = 'data/trade_outcomes.csv'

class AILearner:
    def __init__(self):
        self.patterns = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        self.event_type_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        self.hour_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        self.price_range_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        self.signal_type_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        self.feature_weights = {
            'rsi_weight': 1.0,
            'momentum_weight': 1.0,
            'price_position_weight': 1.0,
            'volume_weight': 1.0,
        }
        self.total_trades = 0
        self.total_wins = 0
        self.total_pnl = 0
        self.load_model()
    
    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                self.patterns = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0}, data.get('patterns', {}))
                self.event_type_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}, data.get('event_types', {}))
                self.hour_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}, data.get('hours', {}))
                self.price_range_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}, data.get('price_ranges', {}))
                self.signal_type_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}, data.get('signal_types', {}))
                self.feature_weights = data.get('weights', self.feature_weights)
                self.total_trades = data.get('total_trades', 0)
                self.total_wins = data.get('total_wins', 0)
                self.total_pnl = data.get('total_pnl', 0)
            except Exception as e:
                print('Error loading AI model:', e)
    
    def save_model(self):
        os.makedirs(os.path.dirname(MODEL_FILE) or '.', exist_ok=True)
        data = {
            'patterns': dict(self.patterns),
            'event_types': dict(self.event_type_performance),
            'hours': dict(self.hour_performance),
            'price_ranges': dict(self.price_range_performance),
            'signal_types': dict(self.signal_type_performance),
            'weights': self.feature_weights,
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'total_pnl': self.total_pnl,
        }
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(data, f)
    
    def classify_event_type(self, title):
        title_lower = title.lower()
        if any(x in title_lower for x in ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'xrp', 'crypto', 'price']):
            return 'crypto'
        if any(x in title_lower for x in ['fc', 'vs.', 'united', 'city', 'match', 'league', 'cup']):
            return 'sports'
        if any(x in title_lower for x in ['trump', 'biden', 'election', 'president', 'congress', 'senate']):
            return 'politics'
        if any(x in title_lower for x in ['elon', 'musk', 'tweet', 'twitter', 'x.com']):
            return 'social'
        if any(x in title_lower for x in ['temperature', 'weather', 'rain', 'snow']):
            return 'weather'
        return 'other'
    
    def classify_price_range(self, price):
        if price < 0.2: return 'very_low'
        if price < 0.4: return 'low'
        if price < 0.6: return 'mid'
        if price < 0.8: return 'high'
        return 'very_high'
    
    def get_pattern_key(self, signal_type, event_type, price_range, hour):
        return signal_type + '_' + event_type + '_' + price_range + '_' + str(hour)
    
    def record_outcome(self, event_title, signal_type, entry_price, exit_price, pnl, entry_time):
        is_win = pnl > 0
        event_type = self.classify_event_type(event_title)
        price_range = self.classify_price_range(entry_price)
        hour = entry_time.hour if isinstance(entry_time, datetime) else datetime.utcnow().hour
        
        # Update pattern stats
        pattern_key = self.get_pattern_key(signal_type, event_type, price_range, hour)
        if is_win:
            self.patterns[pattern_key]['wins'] += 1
        else:
            self.patterns[pattern_key]['losses'] += 1
        self.patterns[pattern_key]['total_pnl'] += pnl
        
        # Update event type stats
        if is_win:
            self.event_type_performance[event_type]['wins'] += 1
        else:
            self.event_type_performance[event_type]['losses'] += 1
        self.event_type_performance[event_type]['pnl'] += pnl
        
        # Update hour stats
        if is_win:
            self.hour_performance[hour]['wins'] += 1
        else:
            self.hour_performance[hour]['losses'] += 1
        self.hour_performance[hour]['pnl'] += pnl
        
        # Update price range stats
        if is_win:
            self.price_range_performance[price_range]['wins'] += 1
        else:
            self.price_range_performance[price_range]['losses'] += 1
        self.price_range_performance[price_range]['pnl'] += pnl
        
        # Update signal type stats
        if is_win:
            self.signal_type_performance[signal_type]['wins'] += 1
        else:
            self.signal_type_performance[signal_type]['losses'] += 1
        self.signal_type_performance[signal_type]['pnl'] += pnl
        
        # Update totals
        self.total_trades += 1
        if is_win:
            self.total_wins += 1
        self.total_pnl += pnl
        
        # Log outcome
        self._log_outcome(event_title, signal_type, entry_price, exit_price, pnl, event_type, hour)
        
        # Update weights based on learning
        self._update_weights()
        self.save_model()
    
    def _log_outcome(self, title, signal, entry, exit_p, pnl, event_type, hour):
        os.makedirs(os.path.dirname(OUTCOMES_FILE) or '.', exist_ok=True)
        headers = ['timestamp', 'title', 'signal', 'entry', 'exit', 'pnl', 'event_type', 'hour', 'win']
        if not os.path.exists(OUTCOMES_FILE) or os.path.getsize(OUTCOMES_FILE) == 0:
            with open(OUTCOMES_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(headers)
        row = [datetime.utcnow().isoformat(), title[:50], signal, round(entry, 4), round(exit_p, 4), round(pnl, 2), event_type, hour, 1 if pnl > 0 else 0]
        with open(OUTCOMES_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)
    
    def _update_weights(self):
        # Simple adaptive weight update based on performance
        if self.total_trades < 10:
            return
        
        # Adjust weights based on what is working
        for event_type, stats in self.event_type_performance.items():
            total = stats['wins'] + stats['losses']
            if total >= 5:
                win_rate = stats['wins'] / total
                # Could use this to adjust strategy
    
    def get_confidence_adjustment(self, event_title, signal_type, price):
        event_type = self.classify_event_type(event_title)
        price_range = self.classify_price_range(price)
        hour = datetime.utcnow().hour
        
        adjustment = 0.0
        factors = []
        
        # Check event type performance
        et_stats = self.event_type_performance.get(event_type, {'wins': 0, 'losses': 0})
        et_total = et_stats['wins'] + et_stats['losses']
        if et_total >= 3:
            et_win_rate = et_stats['wins'] / et_total
            if et_win_rate > 0.6:
                adjustment += 0.1
                factors.append(event_type + ' has ' + str(round(et_win_rate*100)) + '% win rate')
            elif et_win_rate < 0.4:
                adjustment -= 0.1
                factors.append(event_type + ' has low win rate')
        
        # Check hour performance
        h_stats = self.hour_performance.get(hour, {'wins': 0, 'losses': 0})
        h_total = h_stats['wins'] + h_stats['losses']
        if h_total >= 3:
            h_win_rate = h_stats['wins'] / h_total
            if h_win_rate > 0.6:
                adjustment += 0.05
            elif h_win_rate < 0.4:
                adjustment -= 0.05
        
        # Check signal type performance
        st_stats = self.signal_type_performance.get(signal_type, {'wins': 0, 'losses': 0})
        st_total = st_stats['wins'] + st_stats['losses']
        if st_total >= 3:
            st_win_rate = st_stats['wins'] / st_total
            if st_win_rate > 0.6:
                adjustment += 0.1
            elif st_win_rate < 0.4:
                adjustment -= 0.1
        
        return adjustment, factors
    
    def should_trade(self, event_title, signal_type, price, base_confidence):
        adjustment, factors = self.get_confidence_adjustment(event_title, signal_type, price)
        adjusted_confidence = base_confidence + adjustment
        
        # AI recommendation
        if adjusted_confidence >= 0.7:
            return True, adjusted_confidence, factors
        elif adjusted_confidence < 0.4:
            return False, adjusted_confidence, factors + ['Low adjusted confidence']
        else:
            return True, adjusted_confidence, factors
    
    def get_insights(self):
        lines = ['=== AI LEARNING INSIGHTS ===', '']
        
        lines.append('Total Trades Analyzed: ' + str(self.total_trades))
        if self.total_trades > 0:
            lines.append('Overall Win Rate: ' + str(round(self.total_wins / self.total_trades * 100, 1)) + '%')
            lines.append('Total PnL: $' + str(round(self.total_pnl, 2)))
        
        lines.append('')
        lines.append('Event Type Performance:')
        for et, stats in sorted(self.event_type_performance.items()):
            total = stats['wins'] + stats['losses']
            if total > 0:
                wr = stats['wins'] / total * 100
                lines.append('  ' + et + ': ' + str(round(wr, 0)) + '% win (' + str(total) + ' trades, $' + str(round(stats['pnl'], 2)) + ')')
        
        lines.append('')
        lines.append('Best Hours (UTC):')
        best_hours = sorted(self.hour_performance.items(), key=lambda x: x[1]['wins'] / max(1, x[1]['wins'] + x[1]['losses']), reverse=True)[:5]
        for hour, stats in best_hours:
            total = stats['wins'] + stats['losses']
            if total > 0:
                wr = stats['wins'] / total * 100
                lines.append('  ' + str(hour) + ':00 - ' + str(round(wr, 0)) + '% win rate')
        
        lines.append('')
        lines.append('Signal Type Performance:')
        for st, stats in sorted(self.signal_type_performance.items()):
            total = stats['wins'] + stats['losses']
            if total > 0:
                wr = stats['wins'] / total * 100
                lines.append('  ' + st + ': ' + str(round(wr, 0)) + '% win (' + str(total) + ' trades)')
        
        return chr(10).join(lines)

_learner = None
def get_ai_learner():
    global _learner
    if _learner is None:
        _learner = AILearner()
    return _learner

if __name__ == '__main__':
    learner = get_ai_learner()
    print(learner.get_insights())
