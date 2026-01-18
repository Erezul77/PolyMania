from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeIdea:
    """
    Logical description of what *could* be traded, not an actual order.

    This is used in dry-run / advisory modes to describe:
    - which event
    - which outcome/direction
    - approximate size and price boundaries
    """
    event_id: str
    outcome: str
    direction: str  # e.g. "BUY" / "SELL" / "LONG" / "SHORT"
    size_shares: float
    max_price: Optional[float] = None
    time_in_force: str = "GTC"
    rationale: str = ""


@dataclass
class Decision:
    """
    Decision about whether a TradeIdea would be allowed under current rules.

    This is purely logical and does not execute any real trade.
    """
    approved: bool
    reason: str
    adjusted_size_shares: Optional[float] = None
    risk_estimate: Optional[float] = None

