from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Position:
    """
    Logical representation of a single position in a Polymarket event.

    This is a local model only. It is NOT automatically synchronized
    with any real exchange account.
    """
    event_id: str
    outcome: str
    direction: str  # e.g. "LONG" or "SHORT" / "YES"/"NO" semantics
    size_shares: float
    avg_entry_price: float
    current_mark_price: float = 0.0


@dataclass
class PortfolioState:
    """
    High-level portfolio snapshot.

    For now this is a purely logical object that can be:
    - updated manually
    - or later synchronized with a real account through a separate client.
    """
    cash_balance: float = 0.0
    positions: List[Position] = field(default_factory=list)

    def get_position(self, event_id: str, outcome: str) -> Optional[Position]:
        for p in self.positions:
            if p.event_id == event_id and p.outcome == outcome:
                return p
        return None

    def total_exposure(self) -> float:
        # Simple placeholder for future use
        total = 0.0
        for p in self.positions:
            total += p.size_shares * p.current_mark_price
        return total


def empty_portfolio() -> PortfolioState:
    """
    Convenience function to create an empty portfolio state.
    """
    return PortfolioState(cash_balance=0.0, positions=[])

