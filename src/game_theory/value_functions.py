"""
Coalition value functions for LCFG.

Implements the coalition value function from Equation 1:
v(S) = φ(⊕_{a_i ∈ S} c_i) - ψ(|S|)

where:
- φ: [0,1]^d → R aggregates capabilities
- ⊕: componentwise maximum (coverage-based tasks)
- ψ: N → R captures coordination costs
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ValueFunction(ABC):
    """Abstract base class for value functions."""
    
    @abstractmethod
    def __call__(self, capabilities: np.ndarray) -> float:
        """
        Compute value from capability profiles.
        
        Args:
            capabilities: Array of shape (n_agents, d) with capability profiles
        
        Returns:
            Aggregated value
        """
        pass


class CoverageValueFunction(ValueFunction):
    """
    Coverage-based value function φ.
    
    Uses componentwise maximum (modeling coverage-based tasks)
    and L1-normalization:
    
    φ(⊕_{a_i ∈ S} c_i) = ||max(c_1, ..., c_k)||_1 / d
    """
    
    def __init__(
        self,
        normalize: bool = True,
        aggregation: str = "max",
    ):
        """
        Initialize coverage value function.
        
        Args:
            normalize: Whether to L1-normalize (divide by d)
            aggregation: How to combine capabilities ("max", "mean", "sum")
        """
        self.normalize = normalize
        self.aggregation = aggregation
    
    def __call__(self, capabilities: np.ndarray) -> float:
        """
        Compute coverage value.
        
        Args:
            capabilities: Array of shape (n_agents, d) or (d,) for single agent
        
        Returns:
            Coverage value φ(⊕ c_i)
        """
        capabilities = np.atleast_2d(capabilities)
        
        if capabilities.size == 0:
            return 0.0
        
        # Aggregate across agents (componentwise)
        if self.aggregation == "max":
            # ⊕ = componentwise maximum
            aggregated = capabilities.max(axis=0)
        elif self.aggregation == "mean":
            aggregated = capabilities.mean(axis=0)
        elif self.aggregation == "sum":
            aggregated = capabilities.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Compute coverage: L1 norm (optionally normalized)
        coverage = float(np.sum(aggregated))
        
        if self.normalize:
            d = aggregated.shape[0]
            coverage = coverage / d
        
        return coverage
    
    def __repr__(self) -> str:
        return f"CoverageValueFunction(normalize={self.normalize}, agg={self.aggregation})"


class CoordinationCost:
    """
    Coordination cost function ψ(k).
    
    Models superlinear coordination overhead:
    ψ(k) = α · k^β
    
    with α = 0.15, β = 1.3 (empirically calibrated).
    """
    
    def __init__(
        self,
        alpha: float = 0.15,
        beta: float = 1.3,
    ):
        """
        Initialize coordination cost function.
        
        Args:
            alpha: Cost coefficient (default: 0.15)
            beta: Cost exponent (default: 1.3, superlinear)
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        
        self.alpha = alpha
        self.beta = beta
        
        logger.debug(f"CoordinationCost initialized: α={alpha}, β={beta}")
    
    def __call__(self, k: int) -> float:
        """
        Compute coordination cost for coalition of size k.
        
        Args:
            k: Coalition size
        
        Returns:
            Coordination cost ψ(k) = α · k^β
        """
        if k <= 0:
            return 0.0
        
        return self.alpha * (k ** self.beta)
    
    def marginal_cost(self, k: int) -> float:
        """
        Compute marginal cost of adding one agent.
        
        Args:
            k: Current coalition size
        
        Returns:
            ψ(k+1) - ψ(k)
        """
        return self(k + 1) - self(k)
    
    def __repr__(self) -> str:
        return f"CoordinationCost(α={self.alpha}, β={self.beta})"


class CompositeValueFunction(ValueFunction):
    """
    Composite value function combining coverage and coordination cost.
    
    v(S) = φ(⊕ c_i) - ψ(|S|)
    """
    
    def __init__(
        self,
        coverage: Optional[CoverageValueFunction] = None,
        cost: Optional[CoordinationCost] = None,
    ):
        """
        Initialize composite value function.
        
        Args:
            coverage: Coverage function φ (default: CoverageValueFunction)
            cost: Coordination cost ψ (default: CoordinationCost)
        """
        self.coverage = coverage or CoverageValueFunction()
        self.cost = cost or CoordinationCost()
    
    def __call__(self, capabilities: np.ndarray) -> float:
        """
        Compute coalition value v(S).
        
        Args:
            capabilities: Array of shape (n_agents, d)
        
        Returns:
            Coalition value v(S) = φ(⊕ c_i) - ψ(|S|)
        """
        capabilities = np.atleast_2d(capabilities)
        
        if capabilities.size == 0:
            return 0.0
        
        n_agents = capabilities.shape[0]
        
        coverage_value = self.coverage(capabilities)
        coordination_cost = self.cost(n_agents)
        
        return coverage_value - coordination_cost
    
    def __repr__(self) -> str:
        return f"CompositeValueFunction({self.coverage}, {self.cost})"


def compute_worked_example() -> dict:
    """
    Compute the worked example from Example 1 in the paper.
    
    Three agents with capabilities:
    - c1 = (0.68, 0.30, 0.40) (math, facts, logic)
    - c2 = (0.40, 0.65, 0.30)
    - c3 = (0.30, 0.40, 0.76)
    
    Returns:
        Dict with computed values matching Example 1
    """
    c1 = np.array([0.68, 0.30, 0.40])
    c2 = np.array([0.40, 0.65, 0.30])
    c3 = np.array([0.30, 0.40, 0.76])
    
    phi = CoverageValueFunction(normalize=True, aggregation="max")
    psi = CoordinationCost(alpha=0.15, beta=1.3)
    
    # Coalition {a1, a2}
    c12 = np.array([c1, c2])
    coverage_12 = phi(c12)  # max(0.68,0.40)=0.68, max(0.30,0.65)=0.65, max(0.40,0.30)=0.40
    # (0.68 + 0.65 + 0.40) / 3 = 0.577
    cost_12 = psi(2)  # 0.15 * 2^1.3 ≈ 0.37
    v_12 = coverage_12 - cost_12
    per_capita_12 = v_12 / 2
    
    # Grand coalition {a1, a2, a3}
    c123 = np.array([c1, c2, c3])
    coverage_123 = phi(c123)  # max values: 0.68, 0.65, 0.76
    # (0.68 + 0.65 + 0.76) / 3 = 0.697
    cost_123 = psi(3)  # 0.15 * 3^1.3 ≈ 0.60
    v_123 = coverage_123 - cost_123
    per_capita_123 = v_123 / 3
    
    return {
        "coalition_12": {
            "coverage": coverage_12,
            "cost": cost_12,
            "value": v_12,
            "per_capita": per_capita_12,
        },
        "coalition_123": {
            "coverage": coverage_123,
            "cost": cost_123,
            "value": v_123,
            "per_capita": per_capita_123,
        },
        "analysis": (
            f"Per-capita: v_i({{a1,a2}}) = {per_capita_12:.3f} vs "
            f"v_i({{a1,a2,a3}}) = {per_capita_123:.3f}. "
            f"Grand coalition yields lower per-capita value due to coordination costs."
        ),
    }


if __name__ == "__main__":
    # Verify worked example matches paper
    result = compute_worked_example()
    print("Worked Example (Example 1):")
    print(f"  Coalition {{a1, a2}}: v = {result['coalition_12']['value']:.2f}")
    print(f"  Grand coalition: v = {result['coalition_123']['value']:.2f}")
    print(f"  {result['analysis']}")
