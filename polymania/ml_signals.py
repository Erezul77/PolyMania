import csv, os, json, pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import statistics

MODEL_FILE = 'data/ml_model.pkl'
FEATURES_FILE = 'data/ml_features.json'

class PatternLearner:
    def __init__(self):
        self.patterns = defaultdict(list)
        self.signal_outcomes = []
        self.feature_weights = {
            'rsi_oversold': 0.0,
            'rsi_overbought': 0.0,
            'price_momentum': 0.0,
            'volume_spike': 0.0,
            'trend_strength': 0.0,
        }
        self.load_model()
    
    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = data.get('patterns', defaultdict(list))
                    self.signal_outcomes = data.get('outcomes', [])
                    self.feature_weights = data.get('weights', self.feature_weights)
            except: pass
    
    def save_model(self):
        os.makedirs(os.path.dirname(MODEL_FILE) or '.', exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'patterns': dict(self.patterns),
                'outcomes': self.signal_outcomes,
                'weights': self.feature_weights
            }, f)
    
    def extract_features(self, prices):
        if len(prices) < 10:
            return None
        
        # Calculate basic features
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        features = {
            'price_mean': statistics.mean(prices),
            'price_std': statistics.stdev(prices) if len(prices) > 1 else 0,
            'return_mean': statistics.mean(returns) if returns else 0,
            'return_std': statistics.stdev(returns) if len(returns) > 1 else 0,
            'momentum_5': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
            'momentum_10': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0,
            'price_position': (prices[-1] - min(prices)) / (max(prices) - min(prices)) if max(prices) != min(prices) else 0.5,
        }
        return features
    
    def record_outcome(self, signal_type, features, actual_return, was_profitable):
        outcome = {
            'signal_type': signal_type,
            'features': features,
            'actual_return': actual_return,
            'profitable': was_profitable,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.signal_outcomes.append(outcome)
        
        # Update pattern recognition
        pattern_key = self._get_pattern_key(features)
        self.patterns[pattern_key].append(was_profitable)
        
        # Simple weight update
        self._update_weights()
        self.save_model()
    
    def _get_pattern_key(self, features):
        mom = 'up' if features.get('momentum_5', 0) > 0.02 else ('down' if features.get('momentum_5', 0) < -0.02 else 'flat')
        pos = 'high' if features.get('price_position', 0.5) > 0.7 else ('low' if features.get('price_position', 0.5) < 0.3 else 'mid')
        return mom + '_' + pos
    
    def _update_weights(self):
        if len(self.signal_outcomes) < 10:
            return
        
        # Analyze which features correlate with profitable trades
        profitable = [o for o in self.signal_outcomes if o['profitable']]
        unprofitable = [o for o in self.signal_outcomes if not o['profitable']]
        
        if not profitable or not unprofitable:
            return
        
        # Simple correlation analysis
        for feature in ['momentum_5', 'momentum_10', 'price_position']:
            prof_avg = statistics.mean([o['features'].get(feature, 0) for o in profitable])
            unprof_avg = statistics.mean([o['features'].get(feature, 0) for o in unprofitable])
            self.feature_weights[feature] = prof_avg - unprof_avg
    
    def predict_confidence(self, signal_type, features):
        if not features:
            return 0.5
        
        base_confidence = 0.5
        
        # Pattern-based adjustment
        pattern_key = self._get_pattern_key(features)
        if pattern_key in self.patterns:
            outcomes = self.patterns[pattern_key]
            if len(outcomes) >= 3:
                win_rate = sum(outcomes) / len(outcomes)
                base_confidence = 0.3 + (win_rate * 0.6)
        
        # Feature-weight adjustment
        adjustment = 0
        for feature, weight in self.feature_weights.items():
            if feature in features:
                adjustment += features[feature] * weight * 0.1
        
        final_confidence = max(0.1, min(0.95, base_confidence + adjustment))
        return final_confidence
    
    def get_insights(self):
        if not self.signal_outcomes:
            return 'No data yet for analysis'
        
        total = len(self.signal_outcomes)
        profitable = sum(1 for o in self.signal_outcomes if o['profitable'])
        
        lines = [
            '=== ML INSIGHTS ===',
            '',
            'Total Outcomes Recorded: ' + str(total),
            'Profitable: ' + str(profitable) + ' (' + str(round(profitable/total*100, 1)) + '%)',
            '',
            'Pattern Performance:',
        ]
        
        for pattern, outcomes in sorted(self.patterns.items()):
            if len(outcomes) >= 3:
                wr = sum(outcomes) / len(outcomes) * 100
                lines.append('  ' + pattern + ': ' + str(round(wr, 0)) + '% (' + str(len(outcomes)) + ' trades)')
        
        lines.append('')
        lines.append('Feature Weights:')
        for feat, weight in self.feature_weights.items():
            if weight != 0:
                lines.append('  ' + feat + ': ' + str(round(weight, 4)))
        
        return chr(10).join(lines)

_learner = None
def get_pattern_learner():
    global _learner
    if _learner is None:
        _learner = PatternLearner()
    return _learner

if __name__ == '__main__':
    learner = get_pattern_learner()
    print(learner.get_insights())
