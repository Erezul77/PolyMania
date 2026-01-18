from .portfolio_state import PortfolioState
from .trading_config import TradingConfig
from .trading_models import TradeIdea, Decision


def evaluate_trade_idea(
    idea: TradeIdea,
    portfolio: PortfolioState,
    config: TradingConfig,
) -> Decision:
    """
    Skeleton execution guard.

    This function is responsible for deciding whether a TradeIdea would be allowed
    under current risk rules. In this initial skeleton, it always returns a
    non-approved Decision with a placeholder reason, so that no real trading
    logic is accidentally executed.
    """
    return Decision(
        approved=False,
        reason="ExecutionGuard not implemented yet (safety default)",
        adjusted_size_shares=None,
        risk_estimate=None,
    )

