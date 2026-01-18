from typing import Any, Dict, Optional

from .portfolio_state import PortfolioState
from .trading_config import TradingConfig
from .trading_models import TradeIdea


def build_trade_idea(
    signal: Dict[str, Any],
    portfolio: PortfolioState,
    config: TradingConfig,
) -> Optional[TradeIdea]:
    """
    Skeleton implementation: given a signal and current portfolio/config,
    decide whether to construct a TradeIdea.

    In this initial skeleton, we do NOT implement any real logic and
    simply return None. This keeps behavior safe until you're ready
    to codify explicit rules.
    """
    # Example of where you *would* look at:
    # - signal["signal_type"]
    # - existing positions in `portfolio`
    # - risk limits in `config`
    #
    # For now, return None to indicate "no action".
    return None

