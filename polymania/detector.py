cuimport logging
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

from .config import settings
from .polymarket_client import current_timestamp

logger = logging.getLogger("polymania.detector")


def normalize_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize trade dict from Data API to a compact structure.
    
    Example from docs:
    {
        "proxyWallet": "...",
        "side": "BUY",
        "size": 123,
        "price": 0.57,
        "timestamp": 1730731984,
        "title": "...",
        "slug": "...",
        "eventSlug": "...",
        "outcome": "Yes",
        ...
    }
    """
    return {
        "timestamp": int(trade["timestamp"]),
        "price": float(trade["price"]),
        "size": float(trade["size"]),
        "side": str(trade.get("side", "")),
        "outcome": str(trade.get("outcome", "")),
    }


def detect_run_for_event(event: Dict[str, Any], trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Core detection logic.
    
    Returns a signal dict if a strong directional run is detected,
    otherwise returns None.
    """
    event_id = event.get("id")
    event_title = event.get("title", "(no title)")
    
    logger.debug(
        "detect_run_for_event: event_id=%s title=%s trades=%d",
        event_id,
        event_title,
        len(trades),
    )
    
    if not trades:
        logger.debug("event_id=%s: no trades at all -> no signal", event_id)
        return None

    now = current_timestamp()

    recent_trades = [t for t in trades if now - t["timestamp"] <= settings.recent_window_sec]
    base_trades = [t for t in trades if now - t["timestamp"] <= settings.base_window_sec]

    logger.debug(
        "event_id=%s: base_trades=%d, recent_trades=%d",
        event_id,
        len(base_trades),
        len(recent_trades),
    )

    # Need at least a few trades in the recent window
    if len(recent_trades) < settings.min_recent_trades:
        logger.debug(
            "event_id=%s: recent_trades=%d < min_recent_trades=%d -> no signal",
            event_id,
            len(recent_trades),
            settings.min_recent_trades,
        )
        return None

    # Last trade price
    last_trade = sorted(trades, key=lambda t: t["timestamp"])[-1]
    last_price = last_trade["price"]

    # Base price – average over longer window
    base_price = mean(t["price"] for t in base_trades) if base_trades else last_price
    
    # Signed price jump: positive = up, negative = down
    price_jump = last_price - base_price
    abs_price_jump = abs(price_jump)
    
    # Simple direction label for later use
    move_direction = "UP" if price_jump >= 0 else "DOWN"

    # Volume in recent window
    recent_volume = sum(t["size"] for t in recent_trades)
    if recent_volume < settings.min_recent_volume:
        logger.debug(
            "event_id=%s: recent_volume=%.3f < min_recent_volume=%.3f -> no signal",
            event_id,
            recent_volume,
            settings.min_recent_volume,
        )
        return None

    # Dominance by outcome + side
    volume_by_key = defaultdict(float)
    for t in recent_trades:
        key = f"{t['outcome']}_{t['side']}"
        volume_by_key[key] += t["size"]

    if not volume_by_key:
        logger.debug("event_id=%s: no volume_by_key -> no signal", event_id)
        return None

    dominant_key, dominant_vol = max(volume_by_key.items(), key=lambda kv: kv[1])
    total_vol = sum(volume_by_key.values())
    dominance = dominant_vol / total_vol if total_vol > 0 else 0.0

    logger.debug(
        "event_id=%s: price_jump=%.4f (abs=%.4f) (min=%.4f), dominance=%.4f (min=%.4f)",
        event_id,
        price_jump,
        abs_price_jump,
        settings.min_price_jump,
        dominance,
        settings.dominance_threshold,
    )

    # Up-only: require positive jump
    if price_jump >= settings.min_price_jump and dominance >= settings.dominance_threshold:
        outcome, side = dominant_key.split("_", 1)
        
        logger.debug(
            (
                "SIGNAL event_id=%s outcome=%s side=%s dir=%s "
                "base_price=%.4f last_price=%.4f "
                "price_jump=%.4f abs_price_jump=%.4f "
                "recent_volume=%.3f dominance=%.4f"
            ),
            event_id,
            outcome,
            side,
            move_direction,
            base_price,
            last_price,
            price_jump,
            abs_price_jump,
            recent_volume,
            dominance,
        )
        
        return {
            "event_id": event.get("id"),
            "event_title": event.get("title"),
            "event_slug": event.get("slug") or event.get("eventSlug"),
            "last_price": last_price,
            "base_price": base_price,
            # signed jump: positive = up, negative = down
            "price_jump": price_jump,
            # magnitude of the jump, for convenience
            "abs_price_jump": abs_price_jump,
            # simple label for direction
            "move_direction": move_direction,
            "recent_volume": recent_volume,
            "dominant_outcome": outcome,
            "dominant_side": side,
            "dominance": dominance,
        }

    # If we got here, either price_jump or dominance was too low
    if price_jump < settings.min_price_jump:
        logger.debug(
            (
                "event_id=%s: price_jump=%.4f (abs=%.4f) "
                "< min_price_jump=%.4f -> no signal"
            ),
            event_id,
            price_jump,
            abs_price_jump,
            settings.min_price_jump,
        )
    elif dominance < settings.dominance_threshold:
        logger.debug(
            "event_id=%s: dominance=%.4f < dominance_threshold=%.4f -> no signal",
            event_id,
            dominance,
            settings.dominance_threshold,
        )

    return None

