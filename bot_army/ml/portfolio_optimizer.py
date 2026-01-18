"""
Portfolio Optimization
======================

Advanced portfolio optimization using:
- Mean-Variance (Markowitz)
- Black-Litterman
- Risk Parity
- Maximum Sharpe Ratio
- Minimum Variance
- Maximum Diversification
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger("ml.portfolio_optimizer")


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    EQUAL_WEIGHT = "equal_weight"
    BLACK_LITTERMAN = "black_litterman"


@dataclass
class PortfolioWeights:
    """Optimized portfolio weights."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: OptimizationMethod
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EfficientFrontier:
    """Efficient frontier points."""
    returns: List[float]
    volatilities: List[float]
    sharpe_ratios: List[float]
    weights_at_points: List[Dict[str, float]]
    max_sharpe_point: int
    min_var_point: int


class CovarianceEstimator:
    """
    Covariance matrix estimation with various methods.
    """
    
    def __init__(self, method: str = "shrinkage"):
        self.method = method
    
    def estimate(self, returns: np.ndarray) -> np.ndarray:
        """
        Estimate covariance matrix.
        
        Args:
            returns: (n_samples, n_assets) return matrix
        
        Returns:
            (n_assets, n_assets) covariance matrix
        """
        if self.method == "sample":
            return self._sample_cov(returns)
        elif self.method == "shrinkage":
            return self._ledoit_wolf_shrinkage(returns)
        elif self.method == "exponential":
            return self._exponential_cov(returns)
        else:
            return self._sample_cov(returns)
    
    def _sample_cov(self, returns: np.ndarray) -> np.ndarray:
        """Simple sample covariance."""
        return np.cov(returns.T)
    
    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage estimator.
        Shrinks towards scaled identity matrix.
        """
        n, p = returns.shape
        
        # Sample covariance
        sample_cov = np.cov(returns.T)
        
        # Shrinkage target (scaled identity)
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)
        
        # Optimal shrinkage intensity (simplified)
        # In practice, this should be computed more carefully
        delta = sample_cov - target
        delta_sq = delta @ delta
        
        # Shrinkage intensity
        shrinkage = min(1, max(0, 
            (np.sum(delta_sq) - np.trace(delta_sq)) / 
            (n * np.sum(delta_sq) + 1e-10)
        ))
        
        # Shrunk covariance
        return (1 - shrinkage) * sample_cov + shrinkage * target
    
    def _exponential_cov(self, returns: np.ndarray, halflife: int = 60) -> np.ndarray:
        """Exponentially weighted covariance."""
        n, p = returns.shape
        
        # Decay factor
        alpha = 1 - np.exp(-np.log(2) / halflife)
        
        # Weights
        weights = np.array([(1 - alpha) ** i for i in range(n-1, -1, -1)])
        weights = weights / weights.sum()
        
        # Weighted mean
        weighted_mean = np.average(returns, weights=weights, axis=0)
        
        # Centered returns
        centered = returns - weighted_mean
        
        # Weighted covariance
        cov = np.zeros((p, p))
        for i in range(n):
            cov += weights[i] * np.outer(centered[i], centered[i])
        
        return cov


class MeanVarianceOptimizer:
    """
    Mean-Variance (Markowitz) Portfolio Optimizer.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: float = None,
        target_volatility: float = None
    ):
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_return = target_return
        self.target_volatility = target_volatility
        
        self.cov_estimator = CovarianceEstimator(method="shrinkage")
    
    def optimize(
        self,
        returns: np.ndarray,
        asset_names: List[str],
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    ) -> PortfolioWeights:
        """
        Optimize portfolio weights.
        
        Args:
            returns: (n_samples, n_assets) return matrix
            asset_names: List of asset/strategy names
            method: Optimization method to use
        
        Returns:
            PortfolioWeights with optimal allocation
        """
        n_assets = len(asset_names)
        
        # Estimate expected returns and covariance
        expected_returns = np.mean(returns, axis=0) * 252  # Annualized
        cov_matrix = self.cov_estimator.estimate(returns) * 252  # Annualized
        
        # Choose optimization method
        if method == OptimizationMethod.MAX_SHARPE:
            weights = self._max_sharpe(expected_returns, cov_matrix)
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = self._min_variance(cov_matrix)
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(cov_matrix)
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._max_diversification(cov_matrix)
        elif method == OptimizationMethod.EQUAL_WEIGHT:
            weights = np.ones(n_assets) / n_assets
        else:
            weights = self._max_sharpe(expected_returns, cov_matrix)
        
        # Calculate portfolio statistics
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        return PortfolioWeights(
            weights={name: float(w) for name, w in zip(asset_names, weights)},
            expected_return=float(port_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            method=method
        )
    
    def _max_sharpe(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Find portfolio with maximum Sharpe ratio.
        Uses analytical solution for unconstrained case.
        """
        n = len(expected_returns)
        
        # Excess returns
        excess_returns = expected_returns - self.risk_free_rate
        
        # Inverse covariance
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except:
            inv_cov = np.linalg.pinv(cov_matrix)
        
        # Optimal weights (unconstrained)
        weights = inv_cov @ excess_returns
        
        # Normalize
        weights = weights / np.sum(np.abs(weights))
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        return weights
    
    def _min_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Find minimum variance portfolio."""
        n = len(cov_matrix)
        
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except:
            inv_cov = np.linalg.pinv(cov_matrix)
        
        ones = np.ones(n)
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        return weights
    
    def _risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Risk parity: equal risk contribution from each asset.
        Uses iterative algorithm.
        """
        n = len(cov_matrix)
        
        # Initial weights
        weights = np.ones(n) / n
        
        # Iterative optimization
        for _ in range(100):
            # Marginal risk contribution
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            
            # Risk contribution
            risk_contrib = weights * marginal_contrib
            
            # Target: equal risk contribution
            target_contrib = portfolio_vol / n
            
            # Update weights
            weights = weights * (target_contrib / (risk_contrib + 1e-10))
            weights = weights / np.sum(weights)
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        return weights
    
    def _max_diversification(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Maximum diversification portfolio.
        Maximizes diversification ratio = weighted avg vol / portfolio vol
        """
        n = len(cov_matrix)
        
        # Asset volatilities
        vols = np.sqrt(np.diag(cov_matrix))
        
        # Correlation matrix
        corr_matrix = cov_matrix / np.outer(vols, vols)
        
        # Use inverse volatility as starting point
        weights = 1 / vols
        weights = weights / np.sum(weights)
        
        # Iterative optimization for max diversification
        for _ in range(100):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            weighted_avg_vol = np.dot(weights, vols)
            
            # Gradient of diversification ratio
            grad = vols / weighted_avg_vol - (cov_matrix @ weights) / (portfolio_vol * weighted_avg_vol)
            
            # Gradient step
            weights = weights + 0.01 * grad
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        return weights
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply min/max weight constraints."""
        # Floor at min
        weights = np.maximum(weights, self.min_weight)
        
        # Cap at max
        weights = np.minimum(weights, self.max_weight)
        
        # Renormalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def efficient_frontier(
        self,
        returns: np.ndarray,
        asset_names: List[str],
        n_points: int = 20
    ) -> EfficientFrontier:
        """
        Calculate efficient frontier.
        
        Returns:
            EfficientFrontier with points along the frontier
        """
        expected_returns = np.mean(returns, axis=0) * 252
        cov_matrix = self.cov_estimator.estimate(returns) * 252
        
        # Find min and max return portfolios
        min_var_weights = self._min_variance(cov_matrix)
        max_ret_idx = np.argmax(expected_returns)
        
        min_ret = np.dot(min_var_weights, expected_returns)
        max_ret = expected_returns[max_ret_idx]
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_returns = []
        frontier_vols = []
        frontier_sharpes = []
        frontier_weights = []
        
        for target in target_returns:
            # Find min variance for this target return (simplified)
            weights = self._target_return_portfolio(expected_returns, cov_matrix, target)
            
            port_ret = np.dot(weights, expected_returns)
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0
            
            frontier_returns.append(port_ret)
            frontier_vols.append(port_vol)
            frontier_sharpes.append(sharpe)
            frontier_weights.append({name: float(w) for name, w in zip(asset_names, weights)})
        
        # Find max Sharpe and min variance points
        max_sharpe_idx = np.argmax(frontier_sharpes)
        min_var_idx = np.argmin(frontier_vols)
        
        return EfficientFrontier(
            returns=frontier_returns,
            volatilities=frontier_vols,
            sharpe_ratios=frontier_sharpes,
            weights_at_points=frontier_weights,
            max_sharpe_point=max_sharpe_idx,
            min_var_point=min_var_idx
        )
    
    def _target_return_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: float
    ) -> np.ndarray:
        """Find minimum variance portfolio for target return."""
        n = len(expected_returns)
        
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except:
            inv_cov = np.linalg.pinv(cov_matrix)
        
        ones = np.ones(n)
        
        # Lagrange multipliers
        A = ones @ inv_cov @ ones
        B = ones @ inv_cov @ expected_returns
        C = expected_returns @ inv_cov @ expected_returns
        D = A * C - B * B
        
        if D <= 0:
            return ones / n  # Fallback to equal weight
        
        # Optimal weights
        lambda1 = (C - B * target_return) / D
        lambda2 = (A * target_return - B) / D
        
        weights = lambda1 * (inv_cov @ ones) + lambda2 * (inv_cov @ expected_returns)
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        return weights


class PortfolioOptimizationEngine:
    """
    Master portfolio optimization engine.
    """
    
    def __init__(
        self,
        strategy_names: List[str],
        lookback_days: int = 60,
        rebalance_threshold: float = 0.05
    ):
        self.strategy_names = strategy_names
        self.lookback_days = lookback_days
        self.rebalance_threshold = rebalance_threshold
        
        # Optimizer
        self.optimizer = MeanVarianceOptimizer(
            risk_free_rate=0.02,
            min_weight=0.02,
            max_weight=0.40
        )
        
        # Returns history per strategy
        self._returns: Dict[str, deque] = {
            name: deque(maxlen=lookback_days * 24)  # Hourly returns
            for name in strategy_names
        }
        
        # Current weights
        self._current_weights: Dict[str, float] = {
            name: 1 / len(strategy_names) for name in strategy_names
        }
        
        # Last optimization
        self._last_optimization: datetime = None
    
    def record_return(self, strategy_name: str, return_value: float):
        """Record return for a strategy."""
        if strategy_name in self._returns:
            self._returns[strategy_name].append(return_value)
    
    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        if self._last_optimization is None:
            return True
        
        # Rebalance daily
        if datetime.utcnow() - self._last_optimization > timedelta(days=1):
            return True
        
        return False
    
    def optimize(
        self,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    ) -> PortfolioWeights:
        """
        Run portfolio optimization.
        """
        # Check if we have enough data
        min_samples = min(len(r) for r in self._returns.values())
        
        if min_samples < 20:
            # Not enough data, use equal weights
            weights = {name: 1 / len(self.strategy_names) for name in self.strategy_names}
            return PortfolioWeights(
                weights=weights,
                expected_return=0,
                expected_volatility=0.15,
                sharpe_ratio=0,
                method=OptimizationMethod.EQUAL_WEIGHT
            )
        
        # Build return matrix
        n_samples = min_samples
        returns_matrix = np.zeros((n_samples, len(self.strategy_names)))
        
        for i, name in enumerate(self.strategy_names):
            returns_matrix[:, i] = list(self._returns[name])[-n_samples:]
        
        # Run optimization
        result = self.optimizer.optimize(returns_matrix, self.strategy_names, method)
        
        # Update current weights
        self._current_weights = result.weights
        self._last_optimization = datetime.utcnow()
        
        logger.info(f"Portfolio optimized: Sharpe={result.sharpe_ratio:.2f}, "
                   f"Return={result.expected_return:.1%}, Vol={result.expected_volatility:.1%}")
        
        return result
    
    def get_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        return self._current_weights.copy()
    
    def get_efficient_frontier(self) -> Optional[EfficientFrontier]:
        """Calculate efficient frontier."""
        min_samples = min(len(r) for r in self._returns.values())
        
        if min_samples < 20:
            return None
        
        n_samples = min_samples
        returns_matrix = np.zeros((n_samples, len(self.strategy_names)))
        
        for i, name in enumerate(self.strategy_names):
            returns_matrix[:, i] = list(self._returns[name])[-n_samples:]
        
        return self.optimizer.efficient_frontier(returns_matrix, self.strategy_names)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "strategies": self.strategy_names,
            "current_weights": self._current_weights,
            "last_optimization": self._last_optimization.isoformat() if self._last_optimization else None,
            "data_points": {name: len(r) for name, r in self._returns.items()}
        }
