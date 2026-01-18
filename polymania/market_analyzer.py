import logging, time, requests
from datetime import datetime
from typing import Any, Dict, List, Optional
from .config import settings
from .price_history import get_price_collector, PriceHistoryCollector
from .technical_analysis import analyze_price_series
from .trading_signals import generate_trading_signal, log_trading_signal, format_signal_for_telegram, TradingSignal, SignalType
from .polymarket_client import fetch_active_events, current_timestamp
from .notifier import send_telegram_message
from .paper_trader import get_paper_trader
from .multi_strategy import get_multi_trader
from .alpha_signals import get_alpha_signals_cached, calculate_alpha_adjustment
from .contrarian_logic import analyze_crowd_state, should_skip_trade, format_contrarian_warning, get_position_size_multiplier
from .whale_tracker import get_whale_tracker, get_whale_signals_for_event
from .strategy_learner import get_strategy_learner
from .segment_router import detect_segment, get_config_for_event, adjust_signal_for_segment, MarketSegment
from .market_regime import analyze_market_regime, adjust_for_regime
# === NEW: Advanced Profit Optimization Modules ===
from .profit_optimizer import get_profit_optimizer
from .correlation_matrix import get_correlation_matrix
from .opportunity_scanner import get_opportunity_scanner
from .ensemble_brain import get_ensemble_brain, SourceSignal, SignalSource
from .risk_manager import get_risk_manager

logger = logging.getLogger('polymania.analyzer')
_signal_cooldown: Dict[str, int] = {}
SIGNAL_COOLDOWN_SEC = 900
PAPER_TRADING_ENABLED = True
MULTI_STRATEGY_ENABLED = True

def fetch_events_with_markets(limit=100):
    try:
        resp = requests.get(settings.gamma_base + '/events', params={'order':'id','ascending':'false','closed':'false','limit':limit}, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        return events.get('events', events) if isinstance(events, dict) else events
    except Exception as e:
        logger.error('Error fetching events: ' + str(e))
        return []

def should_generate_signal(event_id):
    return current_timestamp() - _signal_cooldown.get(event_id, 0) >= SIGNAL_COOLDOWN_SEC

def record_signal_sent(event_id):
    _signal_cooldown[event_id] = current_timestamp()

def is_signal_actionable(signal):
    if signal.signal_type == SignalType.HOLD or signal.confidence < 0.4: return False
    if signal.signal_type in [SignalType.WEAK_BUY, SignalType.WEAK_SELL]: return signal.confidence >= 0.6
    return signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.STRONG_SELL, SignalType.SELL]

def analyze_single_event(event, collector, lookback_sec=3600):
    """
    Analyze a single event with MAXIMUM PROFIT OPTIMIZATION.
    
    This is where we beat other bots:
    1. Segment routing: Different strategies for crypto/sports/politics
    2. Market regime: Adapt to volatility, trends, pre-event states
    3. Alpha signals: External intelligence from Telegram/News/Whales
    4. Contrarian logic: Don't be the last buyer/seller
    5. Strategy learner: Apply patterns from successful traders
    6. NEW: Multi-dimensional scoring across all signal sources
    7. NEW: Cross-market correlation analysis
    8. NEW: Pattern & opportunity scanning
    9. NEW: Ensemble brain for final decision
    10. NEW: Dynamic risk management
    """
    event_id = str(event.get('id', ''))
    event_title = str(event.get('title', ''))
    if not event_id: return None
    
    ps = collector.get_price_series(event_id, 'Yes', lookback_sec)
    if len(ps) < 5: return None
    
    prices = [p[1] for p in ps]
    
    # === SEGMENT DETECTION: Route to appropriate strategy ===
    segment, segment_config = get_config_for_event(event_title)
    
    # === MARKET REGIME: Detect current conditions ===
    regime_analysis = analyze_market_regime(prices, event_title)
    
    # Check if regime says don't trade
    if not regime_analysis.regime.value == 'pre_event':
        pass  # Continue
    elif regime_analysis.time_to_resolution_hours and regime_analysis.time_to_resolution_hours < segment_config.min_time_to_resolution_hours:
        logger.info(f'PRE-EVENT SKIP: {event_title[:40]} - {regime_analysis.time_to_resolution_hours:.1f}h to resolution')
        return None
    
    # Generate base signal from technical analysis
    signal = generate_trading_signal(event_id, event_title, prices, outcome='Yes')
    if signal is None: return None
    
    # Store segment and regime info on signal
    signal._segment = segment
    signal._regime = regime_analysis
    
    # Apply segment-specific adjustments
    rsi = signal.indicators.rsi if signal.indicators else None
    macd_hist = signal.indicators.macd_histogram if signal.indicators else None
    price_change = signal.indicators.price_change_pct if signal.indicators else None
    has_alpha = False
    has_whale = False
    
    seg_adjusted, seg_action, seg_reasons = adjust_signal_for_segment(
        signal.confidence, signal.signal_type.value, event_title, rsi
    )
    
    if seg_action == 'SKIP':
        logger.debug(f'SEGMENT SKIP: {event_title[:40]} - {seg_reasons}')
        return None
    
    signal.confidence = seg_adjusted
    signal.reasons.insert(0, f'Segment: {segment.value}')
    
    # Apply regime adjustments
    regime_adjusted, should_trade, regime_reasons = adjust_for_regime(
        signal.signal_type.value, signal.confidence, regime_analysis
    )
    
    if not should_trade:
        logger.info(f'REGIME SKIP: {event_title[:40]} - {regime_analysis.regime.value}')
        return None
    
    signal.confidence = regime_adjusted
    signal.reasons.append(f'Regime: {regime_analysis.regime.value}')
    
    # =========================================================================
    # === NEW: ADVANCED PROFIT OPTIMIZATION PIPELINE ===
    # =========================================================================
    
    # Collect signals from all sources for ensemble decision
    source_signals = []
    
    # 1. Technical signal
    source_signals.append(SourceSignal(
        source=SignalSource.TECHNICAL,
        direction='BUY' if signal.signal_type.value in ['STRONG_BUY', 'BUY', 'WEAK_BUY'] else ('SELL' if signal.signal_type.value in ['STRONG_SELL', 'SELL', 'WEAK_SELL'] else 'NEUTRAL'),
        strength=signal.confidence,
        confidence=0.7,
        reasons=[f'Technical: {signal.signal_type.value}'],
    ))
    
    # 2. Alpha signals: External Intelligence
    alpha_signals = get_alpha_signals_cached(event_title, event_id)
    if alpha_signals:
        alpha_adj, alpha_reasons = calculate_alpha_adjustment(alpha_signals, signal.signal_type.value)
        if alpha_adj != 0:
            has_alpha = True
            old_conf = signal.confidence
            signal.confidence = max(0.1, min(0.95, signal.confidence + alpha_adj))
            signal.reasons.append(f'Alpha adjustment: {alpha_adj:+.0%}')
            signal.reasons.extend(alpha_reasons[:2])
            
            source_signals.append(SourceSignal(
                source=SignalSource.NEWS if 'news' in str(alpha_reasons).lower() else SignalSource.TELEGRAM,
                direction='BUY' if alpha_adj > 0 else 'SELL',
                strength=abs(alpha_adj),
                confidence=0.6,
                reasons=alpha_reasons[:2],
            ))
    
    # 3. Whale signals: Follow the Smart Money
    try:
        whale_adj, whale_reasons = get_whale_signals_for_event(event_id, event_title)
        if whale_adj != 0:
            has_whale = True
            old_conf = signal.confidence
            signal.confidence = max(0.1, min(0.95, signal.confidence + whale_adj))
            signal.reasons.extend(whale_reasons)
            
            source_signals.append(SourceSignal(
                source=SignalSource.WHALE,
                direction='BUY' if whale_adj > 0 else 'SELL',
                strength=abs(whale_adj) * 2,  # Whales are important
                confidence=0.75,
                reasons=whale_reasons,
            ))
            logger.info(f'🐋 Whale signal for {event_title[:30]}: {old_conf:.2f} -> {signal.confidence:.2f}')
    except Exception as e:
        logger.debug(f'Whale tracker error: {e}')
    
    # 4. Strategy learner: Patterns from top traders
    try:
        learner = get_strategy_learner()
        strategy_adj, strategy_reasons = learner.get_trading_recommendations(event_title, signal.current_price)
        if strategy_adj != 0:
            old_conf = signal.confidence
            signal.confidence = max(0.1, min(0.95, signal.confidence + strategy_adj))
            signal.reasons.extend(strategy_reasons)
            
            source_signals.append(SourceSignal(
                source=SignalSource.STRATEGY_LEARNED,
                direction='BUY' if strategy_adj > 0 else 'SELL',
                strength=abs(strategy_adj) * 1.5,
                confidence=0.65,
                reasons=strategy_reasons,
            ))
    except Exception as e:
        logger.debug(f'Strategy learner error: {e}')
    
    # 5. NEW: Cross-market correlation analysis
    try:
        corr_matrix = get_correlation_matrix()
        corr_adj, corr_reasons = corr_matrix.get_cross_market_boost(
            event_title, segment.value, 
            'BUY' if signal.signal_type.value in ['STRONG_BUY', 'BUY', 'WEAK_BUY'] else 'SELL'
        )
        if corr_adj != 0:
            old_conf = signal.confidence
            signal.confidence = max(0.1, min(0.95, signal.confidence + corr_adj))
            signal.reasons.extend(corr_reasons[:1])
            
            source_signals.append(SourceSignal(
                source=SignalSource.CROSS_MARKET,
                direction='BUY' if corr_adj > 0 else 'SELL',
                strength=abs(corr_adj),
                confidence=0.55,
                reasons=corr_reasons,
            ))
        
        # Record this price move for correlation tracking
        if len(prices) >= 2:
            corr_matrix.record_move(event_id, segment.value, prices[-2], prices[-1])
    except Exception as e:
        logger.debug(f'Correlation matrix error: {e}')
    
    # 6. NEW: Opportunity scanning (patterns, breakouts, etc.)
    try:
        scanner = get_opportunity_scanner()
        scan = scanner.scan_event(
            event_id, event_title, prices,
            rsi=rsi,
            segment=segment.value,
        )
        
        if scan.patterns:
            best_pattern = max(scan.patterns, key=lambda p: p.strength * p.confidence)
            if best_pattern.strength > 0.5:
                pattern_adj = best_pattern.strength * 0.1 * (1 if best_pattern.direction == 'BULLISH' else -1)
                signal.confidence = max(0.1, min(0.95, signal.confidence + pattern_adj))
                signal.reasons.append(f'Pattern: {best_pattern.pattern_type.value}')
                
                source_signals.append(SourceSignal(
                    source=SignalSource.PATTERN,
                    direction='BUY' if best_pattern.direction == 'BULLISH' else 'SELL',
                    strength=best_pattern.strength,
                    confidence=best_pattern.confidence,
                    reasons=best_pattern.reasons[:2],
                ))
    except Exception as e:
        logger.debug(f'Opportunity scanner error: {e}')
    
    # 7. NEW: Multi-dimensional profit optimization
    try:
        optimizer = get_profit_optimizer()
        opt_result = optimizer.optimize_signal(
            event_title=event_title,
            segment=segment.value,
            regime=regime_analysis.regime.value,
            prices=prices,
            rsi=rsi,
            macd_histogram=macd_hist,
            trend=signal.indicators.trend if signal.indicators else 'NEUTRAL',
            price_change_pct=price_change,
            has_news_alpha=has_alpha,
            news_sentiment=0.3 if has_alpha else 0,
            whale_signal=has_whale,
            whale_direction='BUY' if signal.signal_type.value in ['STRONG_BUY', 'BUY'] else 'SELL',
        )
        
        # Apply multi-dimensional adjustment
        if opt_result['action'] != 'NEUTRAL':
            md_adj = (opt_result['multi_dimensional_score'] - 0.5) * 0.2
            if md_adj != 0:
                signal.confidence = max(0.1, min(0.95, signal.confidence + md_adj))
                if opt_result['agreement_level'] > 0.7:
                    signal.reasons.append(f"Strong agreement ({opt_result['agreement_level']:.0%})")
        
        # Store for position sizing
        signal._optimizer_result = opt_result
    except Exception as e:
        logger.debug(f'Profit optimizer error: {e}')
    
    # 8. NEW: Ensemble brain final decision
    try:
        brain = get_ensemble_brain()
        volatility = (max(prices) - min(prices)) / min(prices) if min(prices) > 0 else 0.1
        
        ensemble_decision = brain.make_decision(
            event_id=event_id,
            event_title=event_title,
            current_price=signal.current_price,
            recent_prices=prices,
            signals=source_signals,
            regime=regime_analysis.regime.value,
            volatility=volatility,
        )
        
        # Apply ensemble confidence calibration
        if ensemble_decision.action != 'HOLD':
            # Blend our confidence with ensemble confidence
            signal.confidence = 0.6 * signal.confidence + 0.4 * ensemble_decision.confidence
            signal.reasons.append(f"Ensemble: {ensemble_decision.consensus_level:.0%} consensus")
        
        # Store for risk management
        signal._ensemble_decision = ensemble_decision
    except Exception as e:
        logger.debug(f'Ensemble brain error: {e}')
    
    # 9. NEW: Risk management check
    try:
        risk_mgr = get_risk_manager()
        risk_adj = risk_mgr.get_risk_adjustment(
            event_id=event_id,
            segment=segment.value,
            proposed_size=0.05,  # 5% default
        )
        
        if not risk_adj.should_trade:
            logger.info(f'RISK SKIP: {event_title[:40]} - {risk_adj.reason}')
            signal.signal_type = SignalType.HOLD
            signal.reasons.insert(0, f'RISK: {risk_adj.reason}')
            return signal
        
        # Apply risk multipliers
        signal._risk_adjustment = risk_adj
    except Exception as e:
        logger.debug(f'Risk manager error: {e}')
    
    # === CONTRARIAN LOGIC: Don't be the last buyer ===
    rsi = signal.indicators.rsi if signal.indicators else None
    price_change = signal.indicators.price_change_pct if signal.indicators else None
    
    contrarian = analyze_crowd_state(rsi, price_change, signal.signal_type.value)
    
    # Check if we should skip this trade entirely
    skip, skip_reason = should_skip_trade(contrarian)
    if skip:
        logger.info(f'CONTRARIAN SKIP: {event_title[:40]} - {skip_reason}')
        # Convert to HOLD instead of executing
        signal.signal_type = SignalType.HOLD
        signal.reasons.insert(0, f'CONTRARIAN: {skip_reason}')
        return signal
    
    # Apply contrarian confidence penalty
    if contrarian.confidence_penalty != 0:
        old_conf = signal.confidence
        signal.confidence = max(0.1, min(0.95, signal.confidence + contrarian.confidence_penalty))
        signal.reasons.append(f'Contrarian: {contrarian.recommendation}')
        logger.debug(f'Contrarian adjusted {event_title[:30]}: {old_conf:.2f} -> {signal.confidence:.2f}')
    
    # Store contrarian analysis for position sizing
    signal._contrarian = contrarian
    
    return signal

def run_analysis_cycle(collector, notify=True, paper_trade=True):
    signals = []
    trader = get_paper_trader() if paper_trade else None
    events = fetch_events_with_markets(limit=50)
    logger.debug('Fetched ' + str(len(events)) + ' events for analysis')
    
    # Build price map for position checking
    current_prices = {}
    for event in events:
        eid = str(event.get('id', ''))
        markets = event.get('markets', [])
        if markets and len(markets) > 0:
            m = markets[0]
            prices = m.get('outcomePrices', '[]')
            if isinstance(prices, str):
                import json
                try:
                    prices = json.loads(prices)
                except:
                    prices = []
            if prices and len(prices) > 0:
                try:
                    current_prices[eid] = float(prices[0])
                except:
                    pass
    
    # Check existing positions for take-profit/stop-loss
    if trader and paper_trade and current_prices:
        sells = trader.check_positions(current_prices)
        for sell_msg in sells:
            logger.info(sell_msg)
            if notify and settings.telegram_bot_token and settings.telegram_chat_id:
                try:
                    send_telegram_message('AUTO: ' + sell_msg + chr(10) + trader.get_summary())
                except Exception as e:
                    logger.error('Telegram error: ' + str(e))
    
    # Multi-strategy trading (100 strategies with $100k)
    multi_trader = get_multi_trader() if MULTI_STRATEGY_ENABLED else None
    if multi_trader and current_prices:
        multi_sells = multi_trader.check_positions(current_prices)
        if multi_sells:
            logger.info('Multi-strategy: ' + str(len(multi_sells)) + ' closes')
    
    for event in events:
        event_id = str(event.get('id', ''))
        markets = event.get('markets', [])
        if markets: collector.record_event(event, markets)
        if not should_generate_signal(event_id): continue
        signal = analyze_single_event(event, collector)
        if signal is None: continue
        log_trading_signal(signal)
        if not is_signal_actionable(signal): continue
        record_signal_sent(event_id)
        signals.append(signal)
        
        # Paper trading execution
        trade_msg = None
        if trader and paper_trade:
            trade_msg = trader.execute_signal(signal, signal.current_price)
        
        # Multi-strategy execution
        if multi_trader and signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            multi_buys = multi_trader.execute_buy(signal, signal.current_price)
            if multi_buys:
                logger.debug('Multi-strategy: ' + str(len(multi_buys)) + ' buys')
        
        log_msg = 'Signal: ' + signal.signal_type.value + ' for ' + signal.event_title[:40]
        if trade_msg:
            log_msg += ' | TRADE: ' + trade_msg
        logger.info(log_msg)
        
        # Telegram notification with contrarian warnings
        if notify and settings.telegram_bot_token and settings.telegram_chat_id:
            try:
                msg = format_signal_for_telegram(signal)
                
                # Add contrarian warning if applicable
                contrarian = getattr(signal, '_contrarian', None)
                if contrarian:
                    warning = format_contrarian_warning(contrarian)
                    if warning:
                        msg += chr(10) + chr(10) + warning
                
                if trade_msg:
                    msg += chr(10) + chr(10) + 'PAPER TRADE: ' + trade_msg
                    msg += chr(10) + 'Portfolio: ' + trader.get_summary()
                send_telegram_message(msg)
            except Exception as e:
                logger.error('Telegram error: ' + str(e))
    
    return signals

def analysis_loop(interval_sec=60, notify=True, paper_trade=True):
    collector = get_price_collector()
    logger.info('Starting market analyzer: interval=' + str(interval_sec) + 's, paper_trading=' + str(paper_trade))
    while True:
        try:
            signals = run_analysis_cycle(collector, notify=notify, paper_trade=paper_trade)
            if signals:
                logger.info('Generated ' + str(len(signals)) + ' actionable signals')
        except Exception as e:
            logger.exception('Error in analysis loop: ' + str(e))
        time.sleep(interval_sec)

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG if '--debug' in sys.argv else logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    paper = '--no-paper' not in sys.argv
    if '--once' in sys.argv:
        collector = get_price_collector()
        signals = run_analysis_cycle(collector, notify='--notify' in sys.argv, paper_trade=paper)
        print('Generated ' + str(len(signals)) + ' signals')
        for s in signals:
            print('  ' + s.signal_type.value + ': ' + s.event_title[:50])
        if paper:
            print('Portfolio: ' + get_paper_trader().get_summary())
    else:
        analysis_loop(interval_sec=60, notify=True, paper_trade=paper)
