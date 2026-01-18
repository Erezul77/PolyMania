from typing import Any, Dict, List, Optional

from .config import settings


def decide_signal_type(signal: Dict[str, Any], news: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Very simple, heuristic, DRY-RUN ONLY signal classifier.

    It does NOT trade, does NOT place orders, and is NOT financial advice.
    It only labels the run with a rough "signal type" that you can inspect later.

    Returns a dict with:
    - "type": one of:
        - "AGGRESSIVE_BULLISH_SIGNAL"
        - "CAUTION_BEARISH_SIGNAL"
        - "NEUTRAL_WATCH"
    - "reason": short human-readable explanation
    """
    last_price = float(signal.get("last_price", 0.0))
    base_price = float(signal.get("base_price", 0.0))
    price_jump = float(signal.get("price_jump", 0.0))
    dominance = float(signal.get("dominance", 0.0))
    recent_volume = float(signal.get("recent_volume", 0.0))

    outcome = str(signal.get("dominant_outcome") or "")
    side = str(signal.get("dominant_side") or "").upper()

    # Default
    signal_type = "NEUTRAL_WATCH"
    reason_parts = []

    # Common info
    reason_parts.append(f"side={side}, outcome={outcome}")
    reason_parts.append(f"jump={price_jump:.3f}, last={last_price:.3f}, dominance={dominance:.2f}, vol={recent_volume:.1f}")

    # Heuristic: strong BUY flow with decent jump and price not extremely high
    if side == "BUY" and price_jump >= settings.min_price_jump and dominance >= settings.dominance_threshold:
        if last_price < 0.7:
            signal_type = "AGGRESSIVE_BULLISH_SIGNAL"
            reason_parts.append("strong BUY run, price still below 0.7")
        else:
            signal_type = "NEUTRAL_WATCH"
            reason_parts.append("BUY run but price already relatively high")

    # Heuristic: strong SELL flow during run (could indicate panic / unwind)
    elif side == "SELL" and dominance >= settings.dominance_threshold:
        signal_type = "CAUTION_BEARISH_SIGNAL"
        reason_parts.append("dominant SELL flow during run window")

    else:
        reason_parts.append("no clear directional bias; treat as neutral watch")

    reason = "; ".join(reason_parts)

    return {
        "type": signal_type,
        "reason": reason,
    }

