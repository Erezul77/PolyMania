"""
PolyMania ML Engine
===================

World-class machine learning components for algorithmic trading:
- Regime Detection (HMM, volatility clustering)
- Risk Management (Kelly, Risk Parity, VaR)
- Multi-Timeframe Analysis
- Sentiment Integration
- Portfolio Optimization (Markowitz, Black-Litterman)
"""

from .regime_detector import (
    AdvancedRegimeDetector,
    MarketRegime,
    RegimeState,
    HiddenMarkovRegime,
    VolatilityRegimeDetector,
    TrendStrengthAnalyzer
)

from .risk_manager import (
    AdvancedRiskManager,
    KellyCriterion,
    RiskParityAllocator,
    DrawdownManager,
    PositionSize,
    RiskMetrics
)

from .multi_timeframe import (
    MultiTimeframeEngine,
    TimeframeAnalyzer,
    Timeframe,
    TimeframeSignal,
    MultiTimeframeSignal
)

from .sentiment import (
    SentimentEngine,
    SentimentAggregator,
    TextSentimentAnalyzer,
    FearGreedIndicator,
    SentimentSource,
    SentimentReading,
    AggregateSentiment
)

from .portfolio_optimizer import (
    PortfolioOptimizationEngine,
    MeanVarianceOptimizer,
    CovarianceEstimator,
    OptimizationMethod,
    PortfolioWeights,
    EfficientFrontier
)


__all__ = [
    # Regime Detection
    "AdvancedRegimeDetector",
    "MarketRegime",
    "RegimeState",
    "HiddenMarkovRegime",
    "VolatilityRegimeDetector",
    "TrendStrengthAnalyzer",
    
    # Risk Management
    "AdvancedRiskManager",
    "KellyCriterion",
    "RiskParityAllocator",
    "DrawdownManager",
    "PositionSize",
    "RiskMetrics",
    
    # Multi-Timeframe
    "MultiTimeframeEngine",
    "TimeframeAnalyzer",
    "Timeframe",
    "TimeframeSignal",
    "MultiTimeframeSignal",
    
    # Sentiment
    "SentimentEngine",
    "SentimentAggregator",
    "TextSentimentAnalyzer",
    "FearGreedIndicator",
    "SentimentSource",
    "SentimentReading",
    "AggregateSentiment",
    
    # Portfolio Optimization
    "PortfolioOptimizationEngine",
    "MeanVarianceOptimizer",
    "CovarianceEstimator",
    "OptimizationMethod",
    "PortfolioWeights",
    "EfficientFrontier",
]
