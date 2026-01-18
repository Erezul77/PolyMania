"""
Whale Tracker - Follow the Smart Money on Polymarket

This module tracks large/profitable traders and generates signals when they trade.

Strategy:
1. IDENTIFY whales: Large trade sizes, consistent profits
2. TRACK their wallets: Build a database of smart money
3. FOLLOW their moves: Generate signals when they trade
4. REACT FAST: Get in before the price fully moves

The edge: Whales often have better information or analysis.
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

from .config import settings

logger = logging.getLogger('polymania.whale')

# Files
WHALE_DB_FILE = 'data/whale_wallets.json'
WHALE_TRADES_CSV = 'data/whale_trades.csv'
WHALE_SIGNALS_CSV = 'data/whale_signals.csv'

# Thresholds
MIN_TRADE_SIZE_USD = 500  # Minimum trade to be considered "whale"
WHALE_THRESHOLD_USD = 5000  # Single trade this big = definitely a whale
MIN_TRADES_FOR_SCORING = 5  # Need this many trades to score a wallet
HIGH_WIN_RATE = 0.6  # 60%+ win rate = smart money
SIGNAL_COOLDOWN_SEC = 300  # Don't signal same wallet+event within 5 min


@dataclass
class WalletStats:
    """Statistics for a tracked wallet"""
    address: str
    total_trades: int = 0
    total_volume: float = 0.0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    avg_trade_size: float = 0.0
    last_seen: str = ''
    tags: List[str] = field(default_factory=list)  # 'whale', 'smart_money', 'degen', etc.
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    @property
    def score(self) -> float:
        """
        Score wallet quality (0-100)
        Based on: win rate, volume, consistency
        """
        if self.total_trades < MIN_TRADES_FOR_SCORING:
            return 0.0
        
        # Win rate component (0-40 points)
        wr_score = min(40, self.win_rate * 50)
        
        # Volume component (0-30 points) - log scale
        import math
        vol_score = min(30, math.log10(max(1, self.total_volume)) * 5)
        
        # Consistency component (0-30 points) - more trades = more reliable
        cons_score = min(30, self.total_trades * 1.5)
        
        return wr_score + vol_score + cons_score
    
    def is_whale(self) -> bool:
        return self.avg_trade_size >= MIN_TRADE_SIZE_USD or self.total_volume >= WHALE_THRESHOLD_USD * 3
    
    def is_smart_money(self) -> bool:
        return self.win_rate >= HIGH_WIN_RATE and self.total_trades >= MIN_TRADES_FOR_SCORING


@dataclass
class WhaleSignal:
    """Signal generated when a tracked whale makes a trade"""
    timestamp: datetime
    wallet: str
    wallet_score: float
    event_id: str
    event_title: str
    side: str  # 'BUY' or 'SELL'
    outcome: str  # 'Yes' or 'No'
    size: float
    price: float
    confidence_boost: float  # How much to boost our signal confidence
    reasons: List[str] = field(default_factory=list)


class WhaleTracker:
    """
    Tracks whale wallets and generates signals when they trade.
    """
    
    def __init__(self):
        self.wallets: Dict[str, WalletStats] = {}
        self.signal_cooldowns: Dict[str, int] = {}  # wallet+event -> last signal timestamp
        self._load_wallet_db()
    
    def _load_wallet_db(self):
        """Load wallet database from file"""
        if os.path.exists(WHALE_DB_FILE):
            try:
                with open(WHALE_DB_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for addr, stats in data.items():
                    self.wallets[addr] = WalletStats(**stats)
                logger.info(f'Loaded {len(self.wallets)} tracked wallets')
            except Exception as e:
                logger.error(f'Error loading whale DB: {e}')
    
    def _save_wallet_db(self):
        """Save wallet database to file"""
        os.makedirs(os.path.dirname(WHALE_DB_FILE) or '.', exist_ok=True)
        data = {addr: asdict(stats) for addr, stats in self.wallets.items()}
        with open(WHALE_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def fetch_recent_trades(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Fetch recent trades across all markets from Polymarket Data API.
        """
        try:
            resp = requests.get(
                f"{settings.data_base}/trades",
                params={'limit': limit},
                timeout=15
            )
            resp.raise_for_status()
            trades = resp.json()
            return trades if isinstance(trades, list) else []
        except Exception as e:
            logger.error(f'Error fetching trades: {e}')
            return []
    
    def fetch_trades_for_wallet(self, wallet: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch trades for a specific wallet"""
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
            logger.error(f'Error fetching wallet trades: {e}')
            return []
    
    def process_trade(self, trade: Dict[str, Any]) -> Optional[WalletStats]:
        """
        Process a single trade and update wallet stats.
        Returns updated WalletStats if this is a whale trade.
        """
        wallet = trade.get('proxyWallet', '')
        if not wallet:
            return None
        
        size = float(trade.get('size', 0))
        price = float(trade.get('price', 0))
        trade_value = size * price
        
        # Update or create wallet stats
        if wallet not in self.wallets:
            self.wallets[wallet] = WalletStats(address=wallet)
        
        stats = self.wallets[wallet]
        stats.total_trades += 1
        stats.total_volume += trade_value
        stats.avg_trade_size = stats.total_volume / stats.total_trades
        stats.last_seen = datetime.now(timezone.utc).isoformat()
        
        # Tag as whale if trade is large enough
        if trade_value >= WHALE_THRESHOLD_USD:
            if 'whale' not in stats.tags:
                stats.tags.append('whale')
                logger.info(f'New whale identified: {wallet[:10]}... (${trade_value:.0f} trade)')
        
        # Return stats if this is a significant trade
        if trade_value >= MIN_TRADE_SIZE_USD:
            return stats
        
        return None
    
    def scan_for_whales(self, min_trades: int = 10, min_volume: float = 10000) -> List[WalletStats]:
        """
        Scan recent trades to identify whale wallets.
        """
        trades = self.fetch_recent_trades(limit=1000)
        logger.info(f'Scanning {len(trades)} trades for whales...')
        
        # Process all trades
        for trade in trades:
            self.process_trade(trade)
        
        # Find whales
        whales = []
        for wallet, stats in self.wallets.items():
            if stats.total_trades >= min_trades and stats.total_volume >= min_volume:
                if stats.is_whale():
                    whales.append(stats)
        
        # Save updated database
        self._save_wallet_db()
        
        logger.info(f'Found {len(whales)} whale wallets')
        return sorted(whales, key=lambda w: w.score, reverse=True)
    
    def add_wallet_to_track(self, address: str, tags: List[str] = None):
        """Manually add a wallet to track (e.g., from leaderboard)"""
        if address not in self.wallets:
            self.wallets[address] = WalletStats(address=address)
        
        if tags:
            for tag in tags:
                if tag not in self.wallets[address].tags:
                    self.wallets[address].tags.append(tag)
        
        self._save_wallet_db()
        logger.info(f'Added wallet to track: {address[:10]}... tags={tags}')
    
    def check_for_whale_signals(self, lookback_sec: int = 300) -> List[WhaleSignal]:
        """
        Check recent trades for whale activity and generate signals.
        
        Args:
            lookback_sec: Only look at trades from last N seconds
        
        Returns:
            List of WhaleSignal objects
        """
        signals = []
        trades = self.fetch_recent_trades(limit=200)
        
        now = int(time.time())
        cutoff = now - lookback_sec
        
        for trade in trades:
            # Check timestamp
            ts = int(trade.get('timestamp', 0))
            if ts < cutoff:
                continue
            
            wallet = trade.get('proxyWallet', '')
            if not wallet:
                continue
            
            # Check if we're tracking this wallet
            if wallet not in self.wallets:
                # Process new trade to potentially add wallet
                stats = self.process_trade(trade)
                if stats is None or not stats.is_whale():
                    continue
            else:
                stats = self.wallets[wallet]
            
            # Only signal on whales or smart money
            if not (stats.is_whale() or stats.is_smart_money()):
                continue
            
            # Check cooldown
            event_id = trade.get('eventId', trade.get('market', ''))
            cooldown_key = f"{wallet}_{event_id}"
            last_signal = self.signal_cooldowns.get(cooldown_key, 0)
            if now - last_signal < SIGNAL_COOLDOWN_SEC:
                continue
            
            # Generate signal
            size = float(trade.get('size', 0))
            price = float(trade.get('price', 0))
            trade_value = size * price
            
            if trade_value < MIN_TRADE_SIZE_USD:
                continue
            
            # Calculate confidence boost based on wallet quality
            base_boost = 0.05
            if stats.is_smart_money():
                base_boost += 0.1
            if trade_value >= WHALE_THRESHOLD_USD:
                base_boost += 0.1
            if stats.score >= 70:
                base_boost += 0.05
            
            confidence_boost = min(0.25, base_boost)
            
            reasons = []
            if stats.is_whale():
                reasons.append(f'Whale wallet (${stats.total_volume:,.0f} volume)')
            if stats.is_smart_money():
                reasons.append(f'Smart money ({stats.win_rate:.0%} win rate)')
            reasons.append(f'Trade size: ${trade_value:,.0f}')
            reasons.append(f'Wallet score: {stats.score:.0f}/100')
            
            signal = WhaleSignal(
                timestamp=datetime.fromtimestamp(ts, timezone.utc),
                wallet=wallet,
                wallet_score=stats.score,
                event_id=event_id,
                event_title=trade.get('title', trade.get('eventTitle', 'Unknown')),
                side=trade.get('side', 'UNKNOWN'),
                outcome=trade.get('outcome', 'Yes'),
                size=size,
                price=price,
                confidence_boost=confidence_boost,
                reasons=reasons,
            )
            
            signals.append(signal)
            self.signal_cooldowns[cooldown_key] = now
            
            # Log signal
            self._log_whale_signal(signal)
            logger.info(f'Whale signal: {signal.side} {signal.outcome} on {signal.event_title[:30]}... (${trade_value:.0f})')
        
        return signals
    
    def _log_whale_signal(self, signal: WhaleSignal):
        """Log whale signal to CSV"""
        os.makedirs(os.path.dirname(WHALE_SIGNALS_CSV) or '.', exist_ok=True)
        
        headers = ['timestamp', 'wallet', 'score', 'event_id', 'event_title', 
                   'side', 'outcome', 'size', 'price', 'boost', 'reasons']
        
        if not os.path.exists(WHALE_SIGNALS_CSV) or os.path.getsize(WHALE_SIGNALS_CSV) == 0:
            with open(WHALE_SIGNALS_CSV, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(headers)
        
        row = [
            signal.timestamp.isoformat(),
            signal.wallet[:16] + '...',
            round(signal.wallet_score, 1),
            signal.event_id,
            signal.event_title[:40],
            signal.side,
            signal.outcome,
            round(signal.size, 2),
            round(signal.price, 4),
            round(signal.confidence_boost, 2),
            '|'.join(signal.reasons[:3]),
        ]
        
        with open(WHALE_SIGNALS_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)
    
    def get_top_whales(self, n: int = 10) -> List[WalletStats]:
        """Get top N whales by score"""
        whales = [w for w in self.wallets.values() if w.is_whale() or w.is_smart_money()]
        return sorted(whales, key=lambda w: w.score, reverse=True)[:n]
    
    def get_summary(self) -> str:
        """Get summary of whale tracker status"""
        total = len(self.wallets)
        whales = sum(1 for w in self.wallets.values() if w.is_whale())
        smart = sum(1 for w in self.wallets.values() if w.is_smart_money())
        
        lines = [
            '=== WHALE TRACKER STATUS ===',
            f'Tracked Wallets: {total}',
            f'Identified Whales: {whales}',
            f'Smart Money: {smart}',
            '',
            'TOP WHALES:',
        ]
        
        for w in self.get_top_whales(5):
            wr = f'{w.win_rate:.0%}' if w.total_trades >= MIN_TRADES_FOR_SCORING else 'N/A'
            lines.append(f'  {w.address[:10]}... Score:{w.score:.0f} Vol:${w.total_volume:,.0f} WR:{wr}')
        
        return '\n'.join(lines)


# Singleton instance
_tracker: Optional[WhaleTracker] = None

def get_whale_tracker() -> WhaleTracker:
    """Get singleton whale tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = WhaleTracker()
    return _tracker


def get_whale_signals_for_event(event_id: str, event_title: str) -> Tuple[float, List[str]]:
    """
    Get whale signal adjustment for a specific event.
    
    Returns:
        (confidence_adjustment, reasons)
    """
    tracker = get_whale_tracker()
    signals = tracker.check_for_whale_signals(lookback_sec=600)
    
    adjustment = 0.0
    reasons = []
    
    for signal in signals:
        if signal.event_id == event_id:
            # Whale is trading this event!
            if signal.side == 'BUY':
                adjustment += signal.confidence_boost
                reasons.append(f'🐋 Whale BUY: +{signal.confidence_boost:.0%} ({signal.reasons[0]})')
            else:
                adjustment -= signal.confidence_boost * 0.5  # Sells are less informative
                reasons.append(f'🐋 Whale SELL detected')
    
    return min(0.3, adjustment), reasons


if __name__ == '__main__':
    # Test whale tracker
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    tracker = get_whale_tracker()
    
    print('Scanning for whales...')
    whales = tracker.scan_for_whales(min_trades=3, min_volume=1000)
    
    print(f'\nFound {len(whales)} whales')
    print(tracker.get_summary())
    
    print('\nChecking for whale signals...')
    signals = tracker.check_for_whale_signals(lookback_sec=3600)
    print(f'Generated {len(signals)} whale signals')
