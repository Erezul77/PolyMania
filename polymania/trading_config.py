from dataclasses import dataclass
from typing import Literal, Optional


TradingMode = Literal["OBSERVER", "ADVISOR", "SEMI_AUTO", "AUTO"]


@dataclass
class TradingConfig:
    """
    High-level configuration for PolyMania trading behavior.

    This does NOT execute trades. It only defines risk and behavior parameters
    for other modules to consult.
    """
    mode: TradingMode = "OBSERVER"

    # Risk and exposure limits (conceptual, not enforced yet)
    max_daily_loss: Optional[float] = None
    max_trades_per_day: Optional[int] = None
    max_risk_per_trade: Optional[float] = None  # e.g. as fraction of bankroll
    max_exposure_per_market: Optional[float] = None
    max_exposure_per_theme: Optional[float] = None

    # Which signal types are even eligible to become TradeIdeas
    # e.g. ["AGGRESSIVE_BULLISH_SIGNAL"]
    eligible_signal_types: Optional[list[str]] = None


def default_trading_config() -> TradingConfig:
    """
    Provide a default trading configuration for development / dry-run mode.

    In production you might load this from a dedicated config file or env vars.
    """
    return TradingConfig(
        mode="ADVISOR",
        max_daily_loss=None,
        max_trades_per_day=None,
        max_risk_per_trade=None,
        max_exposure_per_market=None,
        max_exposure_per_theme=None,
        eligible_signal_types=["AGGRESSIVE_BULLISH_SIGNAL"],
    )

