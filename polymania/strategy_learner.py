"""
Strategy Learner - Reverse Engineer Top Trader Strategies

This module follows successful Polymarket traders and learns:
1. WHAT they trade (market types, categories)
2. WHEN they trade (timing patterns)
3. HOW they trade (position sizes, entry/exit points)
4. WHY they win (pattern recognition)

Then applies these learnings to our own signals.
"""

import csv
import json
import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from statistics import mean, stdev

from .config import settings

logger = logging.getLogger('polymania.strategy_learner')

# Files
FOLLOWED_TRADERS_FILE = 'data/followed_traders.json'
TRADER_ANALYSIS_FILE = 'data/trader_analysis.json'
STRATEGY_LEARNINGS_FILE = 'data/strategy_learnings.json'

# Known successful traders (will be populated)
# These are example addresses - we'll discover real ones
SEED_TRADERS = [
    # Top traders are discovered via trade analysis
]


@dataclass
class TraderProfile:
    """Complete profile of a trader we're following"""
    address: str
    alias: str = ''
    
    # Performance metrics
    total_trades: int = 0
    total_volume: float = 0.0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    
    # Trading patterns
    avg_position_size: float = 0.0
    avg_hold_time_hours: float = 0.0
    preferred_markets: List[str] = field(default_factory=list)  # crypto, politics, sports, etc
    active_hours_utc: List[int] = field(default_factory=list)  # Most active hours
    
    # Strategy indicators
    avg_entry_price: float = 0.0  # Do they buy cheap or expensive?
    contrarian_score: float = 0.0  # Do they go against the crowd?
    early_mover_score: float = 0.0  # Do they enter before big moves?
    
    # Recent activity
    last_trade_time: str = ''
    recent_markets: List[str] = field(default_factory=list)
    
    # Tags
    tags: List[str] = field(default_factory=list)  # 'whale', 'smart_money', 'degen', etc
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    @property
    def avg_pnl_per_trade(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0
    
    def is_worth_following(self) -> bool:
        """Determine if this trader is worth following"""
        if self.total_trades < 10:
            return False
        if self.win_rate < 0.55:  # At least 55% win rate
            return False
        if self.total_pnl <= 0:  # Must be profitable
            return False
        return True


@dataclass
class StrategyInsight:
    """A learned strategy insight from top traders"""
    insight_type: str  # 'timing', 'market_type', 'entry_price', 'position_size'
    description: str
    confidence: float  # 0-1
    sample_size: int
    traders_using: int
    recommendation: str


class StrategyLearner:
    """
    Learns strategies from top traders by analyzing their patterns.
    """
    
    def __init__(self):
        self.traders: Dict[str, TraderProfile] = {}
        self.insights: List[StrategyInsight] = []
        self.trade_history: Dict[str, List[Dict]] = defaultdict(list)  # wallet -> trades
        self._load_data()
    
    def _load_data(self):
        """Load existing trader data"""
        if os.path.exists(FOLLOWED_TRADERS_FILE):
            try:
                with open(FOLLOWED_TRADERS_FILE, 'r') as f:
                    data = json.load(f)
                for addr, profile in data.items():
                    self.traders[addr] = TraderProfile(**profile)
                logger.info(f'Loaded {len(self.traders)} followed traders')
            except Exception as e:
                logger.error(f'Error loading traders: {e}')
        
        if os.path.exists(STRATEGY_LEARNINGS_FILE):
            try:
                with open(STRATEGY_LEARNINGS_FILE, 'r') as f:
                    data = json.load(f)
                self.insights = [StrategyInsight(**i) for i in data]
            except Exception as e:
                logger.error(f'Error loading insights: {e}')
    
    def _save_data(self):
        """Save trader data and insights"""
        os.makedirs(os.path.dirname(FOLLOWED_TRADERS_FILE) or '.', exist_ok=True)
        
        # Save traders
        data = {addr: asdict(p) for addr, p in self.traders.items()}
        with open(FOLLOWED_TRADERS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save insights
        insights_data = [asdict(i) for i in self.insights]
        with open(STRATEGY_LEARNINGS_FILE, 'w') as f:
            json.dump(insights_data, f, indent=2)
    
    def fetch_trades(self, limit: int = 1000) -> List[Dict]:
        """Fetch recent trades from Polymarket"""
        try:
            resp = requests.get(
                f"{settings.data_base}/trades",
                params={'limit': limit},
                timeout=15
            )
            resp.raise_for_status()
            return resp.json() if isinstance(resp.json(), list) else []
        except Exception as e:
            logger.error(f'Error fetching trades: {e}')
            return []
    
    def fetch_trader_history(self, wallet: str, limit: int = 500) -> List[Dict]:
        """Fetch trade history for a specific wallet"""
        try:
            resp = requests.get(
                f"{settings.data_base}/trades",
                params={'proxyWallet': wallet, 'limit': limit},
                timeout=15
            )
            resp.raise_for_status()
            trades = resp.json()
            return trades if isinstance(trades, list) else []
        except Exception as e:
            logger.error(f'Error fetching trader history: {e}')
            return []
    
    def discover_top_traders(self, min_trades: int = 20, min_volume: float = 10000) -> List[TraderProfile]:
        """
        Discover top traders from recent market activity.
        Looks for wallets with high volume and consistent activity.
        """
        logger.info('Discovering top traders...')
        trades = self.fetch_trades(limit=2000)
        
        # Aggregate by wallet
        wallet_stats = defaultdict(lambda: {
            'trades': 0, 'volume': 0, 'buys': 0, 'sells': 0,
            'markets': set(), 'hours': [], 'prices': []
        })
        
        for trade in trades:
            wallet = trade.get('proxyWallet', '')
            if not wallet:
                continue
            
            size = float(trade.get('size', 0))
            price = float(trade.get('price', 0))
            value = size * price
            
            stats = wallet_stats[wallet]
            stats['trades'] += 1
            stats['volume'] += value
            
            if trade.get('side') == 'BUY':
                stats['buys'] += 1
            else:
                stats['sells'] += 1
            
            # Track market categories
            title = trade.get('title', trade.get('eventTitle', '')).lower()
            if any(x in title for x in ['bitcoin', 'btc', 'eth', 'crypto', 'solana']):
                stats['markets'].add('crypto')
            elif any(x in title for x in ['trump', 'biden', 'election', 'president']):
                stats['markets'].add('politics')
            elif any(x in title for x in ['vs', 'match', 'game', 'league']):
                stats['markets'].add('sports')
            else:
                stats['markets'].add('other')
            
            # Track timing
            ts = trade.get('timestamp', 0)
            if ts:
                hour = datetime.fromtimestamp(ts, timezone.utc).hour
                stats['hours'].append(hour)
            
            stats['prices'].append(price)
        
        # Find top traders
        top_traders = []
        for wallet, stats in wallet_stats.items():
            if stats['trades'] >= min_trades and stats['volume'] >= min_volume:
                # Create or update profile
                if wallet not in self.traders:
                    self.traders[wallet] = TraderProfile(address=wallet)
                
                profile = self.traders[wallet]
                profile.total_trades = stats['trades']
                profile.total_volume = stats['volume']
                profile.avg_position_size = stats['volume'] / stats['trades']
                profile.preferred_markets = list(stats['markets'])
                
                # Find most active hours
                if stats['hours']:
                    hour_counts = defaultdict(int)
                    for h in stats['hours']:
                        hour_counts[h] += 1
                    top_hours = sorted(hour_counts.keys(), key=lambda h: hour_counts[h], reverse=True)[:3]
                    profile.active_hours_utc = top_hours
                
                # Analyze entry price preference
                if stats['prices']:
                    profile.avg_entry_price = mean(stats['prices'])
                
                # Tag the trader
                if stats['volume'] >= 50000:
                    if 'whale' not in profile.tags:
                        profile.tags.append('whale')
                
                if stats['trades'] >= 50:
                    if 'active' not in profile.tags:
                        profile.tags.append('active')
                
                top_traders.append(profile)
        
        # Sort by volume
        top_traders.sort(key=lambda t: t.total_volume, reverse=True)
        
        self._save_data()
        logger.info(f'Discovered {len(top_traders)} potential top traders')
        
        return top_traders[:20]  # Return top 20
    
    def analyze_trader_strategy(self, wallet: str) -> Dict[str, Any]:
        """
        Deep analysis of a specific trader's strategy.
        """
        trades = self.fetch_trader_history(wallet, limit=500)
        if not trades:
            return {}
        
        analysis = {
            'wallet': wallet[:16] + '...',
            'total_trades': len(trades),
            'market_preferences': defaultdict(int),
            'timing_patterns': defaultdict(int),
            'size_patterns': [],
            'entry_price_distribution': [],
            'side_preference': {'BUY': 0, 'SELL': 0},
        }
        
        for trade in trades:
            # Market type
            title = trade.get('title', trade.get('eventTitle', '')).lower()
            if any(x in title for x in ['bitcoin', 'btc', 'eth', 'crypto']):
                analysis['market_preferences']['crypto'] += 1
            elif any(x in title for x in ['trump', 'biden', 'election']):
                analysis['market_preferences']['politics'] += 1
            elif any(x in title for x in ['vs', 'match', 'league']):
                analysis['market_preferences']['sports'] += 1
            else:
                analysis['market_preferences']['other'] += 1
            
            # Timing
            ts = trade.get('timestamp', 0)
            if ts:
                hour = datetime.fromtimestamp(ts, timezone.utc).hour
                analysis['timing_patterns'][hour] += 1
            
            # Size and price
            size = float(trade.get('size', 0))
            price = float(trade.get('price', 0))
            analysis['size_patterns'].append(size * price)
            analysis['entry_price_distribution'].append(price)
            
            # Side
            side = trade.get('side', 'BUY')
            analysis['side_preference'][side] += 1
        
        # Calculate insights
        if analysis['size_patterns']:
            analysis['avg_trade_size'] = mean(analysis['size_patterns'])
            analysis['trade_size_consistency'] = stdev(analysis['size_patterns']) if len(analysis['size_patterns']) > 1 else 0
        
        if analysis['entry_price_distribution']:
            analysis['prefers_cheap'] = mean(analysis['entry_price_distribution']) < 0.4
            analysis['prefers_expensive'] = mean(analysis['entry_price_distribution']) > 0.6
        
        # Most active hours
        if analysis['timing_patterns']:
            top_hours = sorted(analysis['timing_patterns'].items(), key=lambda x: x[1], reverse=True)[:3]
            analysis['most_active_hours'] = [h for h, _ in top_hours]
        
        # Market focus
        if analysis['market_preferences']:
            top_market = max(analysis['market_preferences'].items(), key=lambda x: x[1])
            analysis['primary_market'] = top_market[0]
        
        return analysis
    
    def generate_strategy_insights(self) -> List[StrategyInsight]:
        """
        Analyze all followed traders and generate strategy insights.
        """
        logger.info('Generating strategy insights from top traders...')
        
        if len(self.traders) < 3:
            logger.info('Not enough traders to generate insights')
            return []
        
        insights = []
        
        # Aggregate patterns across top traders
        all_markets = defaultdict(int)
        all_hours = defaultdict(int)
        all_entry_prices = []
        all_sizes = []
        
        top_traders = [t for t in self.traders.values() if t.total_trades >= 10]
        
        for trader in top_traders:
            for market in trader.preferred_markets:
                all_markets[market] += 1
            for hour in trader.active_hours_utc:
                all_hours[hour] += 1
            if trader.avg_entry_price > 0:
                all_entry_prices.append(trader.avg_entry_price)
            if trader.avg_position_size > 0:
                all_sizes.append(trader.avg_position_size)
        
        # Generate market type insight
        if all_markets:
            top_market = max(all_markets.items(), key=lambda x: x[1])
            insights.append(StrategyInsight(
                insight_type='market_type',
                description=f'Top traders prefer {top_market[0]} markets',
                confidence=top_market[1] / len(top_traders) if top_traders else 0,
                sample_size=sum(all_markets.values()),
                traders_using=top_market[1],
                recommendation=f'Focus on {top_market[0]} markets for higher success rate'
            ))
        
        # Generate timing insight
        if all_hours:
            top_hours = sorted(all_hours.items(), key=lambda x: x[1], reverse=True)[:3]
            hours_str = ', '.join([f'{h}:00 UTC' for h, _ in top_hours])
            insights.append(StrategyInsight(
                insight_type='timing',
                description=f'Most active trading hours: {hours_str}',
                confidence=0.7,
                sample_size=sum(all_hours.values()),
                traders_using=len([t for t in top_traders if t.active_hours_utc]),
                recommendation=f'Prioritize trading during {hours_str}'
            ))
        
        # Generate entry price insight
        if all_entry_prices:
            avg_entry = mean(all_entry_prices)
            if avg_entry < 0.35:
                entry_pref = 'low-priced (under 35¢) markets'
            elif avg_entry > 0.65:
                entry_pref = 'high-probability (over 65¢) markets'
            else:
                entry_pref = 'mid-range (35-65¢) markets'
            
            insights.append(StrategyInsight(
                insight_type='entry_price',
                description=f'Top traders prefer {entry_pref}',
                confidence=0.6,
                sample_size=len(all_entry_prices),
                traders_using=len(all_entry_prices),
                recommendation=f'Look for opportunities in {entry_pref}'
            ))
        
        # Generate position size insight
        if all_sizes:
            avg_size = mean(all_sizes)
            insights.append(StrategyInsight(
                insight_type='position_size',
                description=f'Average position size: ${avg_size:,.0f}',
                confidence=0.8,
                sample_size=len(all_sizes),
                traders_using=len(all_sizes),
                recommendation=f'Consider position sizes around ${avg_size:,.0f}'
            ))
        
        self.insights = insights
        self._save_data()
        
        return insights
    
    def add_trader_to_follow(self, address: str, alias: str = '', tags: List[str] = None):
        """Manually add a trader to follow"""
        if address not in self.traders:
            self.traders[address] = TraderProfile(address=address)
        
        profile = self.traders[address]
        if alias:
            profile.alias = alias
        if tags:
            for tag in tags:
                if tag not in profile.tags:
                    profile.tags.append(tag)
        
        self._save_data()
        logger.info(f'Added trader to follow: {address[:16]}... alias={alias}')
    
    def get_trading_recommendations(self, event_title: str, current_price: float) -> Tuple[float, List[str]]:
        """
        Get recommendations based on learned strategies.
        
        Returns:
            (confidence_adjustment, reasons)
        """
        adjustment = 0.0
        reasons = []
        
        # Check if this market type is preferred by top traders
        title_lower = event_title.lower()
        market_type = 'other'
        if any(x in title_lower for x in ['bitcoin', 'btc', 'eth', 'crypto']):
            market_type = 'crypto'
        elif any(x in title_lower for x in ['trump', 'biden', 'election']):
            market_type = 'politics'
        elif any(x in title_lower for x in ['vs', 'match', 'league']):
            market_type = 'sports'
        
        # Check insights
        for insight in self.insights:
            if insight.insight_type == 'market_type' and market_type in insight.description:
                adjustment += 0.05
                reasons.append(f'📚 Top traders prefer {market_type} markets')
            
            if insight.insight_type == 'entry_price':
                if 'low-priced' in insight.description and current_price < 0.35:
                    adjustment += 0.05
                    reasons.append('📚 Price in preferred range for top traders')
                elif 'high-probability' in insight.description and current_price > 0.65:
                    adjustment += 0.05
                    reasons.append('📚 Price in preferred range for top traders')
        
        # Check current hour
        current_hour = datetime.now(timezone.utc).hour
        for insight in self.insights:
            if insight.insight_type == 'timing' and f'{current_hour}:00' in insight.description:
                adjustment += 0.03
                reasons.append(f'📚 Trading during top trader active hours')
        
        return min(0.15, adjustment), reasons
    
    def get_summary(self) -> str:
        """Get summary of strategy learner status"""
        lines = [
            '=== STRATEGY LEARNER STATUS ===',
            f'Followed Traders: {len(self.traders)}',
            f'Strategy Insights: {len(self.insights)}',
            '',
        ]
        
        # Top followed traders
        top = sorted(self.traders.values(), key=lambda t: t.total_volume, reverse=True)[:5]
        if top:
            lines.append('TOP FOLLOWED TRADERS:')
            for t in top:
                alias = f' ({t.alias})' if t.alias else ''
                tags = f' [{", ".join(t.tags)}]' if t.tags else ''
                lines.append(f'  {t.address[:12]}...{alias} Vol:${t.total_volume:,.0f} Trades:{t.total_trades}{tags}')
        
        lines.append('')
        
        # Insights
        if self.insights:
            lines.append('LEARNED STRATEGIES:')
            for i in self.insights[:5]:
                lines.append(f'  • {i.description}')
                lines.append(f'    → {i.recommendation}')
        
        return '\n'.join(lines)


# Singleton
_learner: Optional[StrategyLearner] = None

def get_strategy_learner() -> StrategyLearner:
    global _learner
    if _learner is None:
        _learner = StrategyLearner()
    return _learner


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    learner = get_strategy_learner()
    
    print('Discovering top traders...')
    # Use lower thresholds to find more traders
    top = learner.discover_top_traders(min_trades=3, min_volume=500)
    print(f'Found {len(top)} top traders')
    
    if top:
        print('\nAnalyzing top trader strategy...')
        analysis = learner.analyze_trader_strategy(top[0].address)
        print(f'Primary market: {analysis.get("primary_market", "N/A")}')
        print(f'Most active hours: {analysis.get("most_active_hours", [])}')
    
    print('\nGenerating insights...')
    insights = learner.generate_strategy_insights()
    
    print('\n' + learner.get_summary())
