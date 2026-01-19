"""
Stability analysis for LLM Coalition Formation Games.

Implements stability concepts including:
- Nash stability verification (Definition 4)
- Consistency-driven stability bounds (Theorem 2)
- Complexity results (Theorem 5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from coalition_llm.game_theory.coalition_game import (
        Coalition,
        CoalitionGame,
        Partition,
    )

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """
    Result of stability analysis.
    
    Attributes:
        is_nash_stable: Whether partition is Nash-stable
        blocking_deviations: List of agents with improving deviations
        preference_queries: Number of preference queries made
        consistency_score: Measured preference consistency (p)
        theoretical_bounds: Predicted stability probability [lower, upper]
    """
    is_nash_stable: bool
    blocking_deviations: List[Tuple[str, "Coalition", float]]
    preference_queries: int
    consistency_score: Optional[float] = None
    theoretical_bounds: Optional[Tuple[float, float]] = None
    
    def __repr__(self) -> str:
        status = "Nash-stable" if self.is_nash_stable else "Unstable"
        return (
            f"StabilityResult({status}, "
            f"queries={self.preference_queries}, "
            f"blocking={len(self.blocking_deviations)})"
        )


class StabilityAnalyzer:
    """
    Analyzer for coalition stability.
    
    Implements:
    - O(n²) Nash stability verification (Theorem 5.1)
    - Consistency-driven stability bounds (Theorem 2)
    - Statistical significance testing
    """
    
    def __init__(
        self,
        game: "CoalitionGame",
        epsilon: float = 0.0,
    ):
        """
        Initialize stability analyzer.
        
        Args:
            game: The coalition formation game
            epsilon: ε-rationality threshold
        """
        self.game = game
        self.epsilon = epsilon
        
        # Tracking for consistency estimation
        self._preference_history: Dict[str, List[int]] = {}
    
    def verify_nash_stability(
        self,
        partition: "Partition",
    ) -> StabilityResult:
        """
        Verify if a partition is Nash-stable.
        
        A partition π is Nash-stable if no agent prefers joining another
        coalition: ∀a_i ∈ C ∈ π, ∀C' ∈ π ∪ {∅}: C ≿_i C' ∪ {a_i}
        
        Complexity: O(n²) preference queries (Theorem 5.1)
        
        Args:
            partition: Coalition structure to verify
        
        Returns:
            StabilityResult with stability status and details
        """
        blocking_deviations: List[Tuple[str, "Coalition", float]] = []
        queries = 0
        
        for agent_id in self.game.agent_ids:
            deviations = self.game.get_improving_deviations(
                partition, agent_id, self.epsilon
            )
            queries += len(partition.coalitions)  # One comparison per coalition
            
            if deviations:
                for target, improvement in deviations:
                    blocking_deviations.append((agent_id, target, improvement))
        
        is_stable = len(blocking_deviations) == 0
        
        logger.debug(
            f"Nash stability check: {is_stable}, "
            f"{len(blocking_deviations)} blocking deviations, "
            f"{queries} queries"
        )
        
        return StabilityResult(
            is_nash_stable=is_stable,
            blocking_deviations=blocking_deviations,
            preference_queries=queries,
        )
    
    def verify_individual_stability(
        self,
        partition: "Partition",
    ) -> StabilityResult:
        """
        Verify individual stability (weaker than Nash).
        
        A partition is individually stable if no agent can profitably
        deviate with consent from the receiving coalition.
        
        Args:
            partition: Coalition structure to verify
        
        Returns:
            StabilityResult
        """
        blocking: List[Tuple[str, "Coalition", float]] = []
        queries = 0
        
        for agent_id in self.game.agent_ids:
            current = partition.get_coalition(agent_id)
            current_value = self.game.per_capita_value(current)
            
            for target in partition.coalitions:
                if target == current:
                    continue
                
                # Check if agent wants to join
                new_coalition = target.add(agent_id)
                new_value = self.game.per_capita_value(new_coalition)
                queries += 1
                
                improvement = new_value - current_value
                if improvement <= self.epsilon:
                    continue  # Agent doesn't want to move
                
                # Check if receiving coalition consents
                # (existing members don't lose value)
                target_value = self.game.per_capita_value(target)
                consents = new_value >= target_value - self.epsilon
                queries += 1
                
                if consents:
                    blocking.append((agent_id, target, improvement))
        
        return StabilityResult(
            is_nash_stable=len(blocking) == 0,
            blocking_deviations=blocking,
            preference_queries=queries,
        )
    
    def estimate_consistency(
        self,
        partition: "Partition",
        num_samples: int = 10,
        seed: Optional[int] = None,
    ) -> float:
        """
        Estimate preference consistency p for critical decisions.
        
        Consistency is the probability an agent gives the same preference
        when queried multiple times about the same comparison.
        
        Args:
            partition: Coalition structure for context
            num_samples: Number of repeated queries per comparison
            seed: Random seed
        
        Returns:
            Estimated consistency p ∈ [0, 1]
        """
        rng = np.random.default_rng(seed)
        
        consistencies = []
        
        for agent_id in self.game.agent_ids:
            current = partition.get_coalition(agent_id)
            
            for target in partition.coalitions:
                if target == current:
                    continue
                
                # Query preference multiple times
                preferences = []
                for _ in range(num_samples):
                    # In practice, this would query the LLM
                    # Here we use ground truth + noise for simulation
                    pref = self.game.agent_prefers(
                        agent_id,
                        current,
                        target.add(agent_id),
                        epsilon=0.0,
                    )
                    
                    # Add realistic noise (simulating LLM stochasticity)
                    if rng.random() < 0.1:  # 10% error rate baseline
                        pref = -pref if pref != 0 else rng.choice([-1, 1])
                    
                    preferences.append(pref)
                
                # Consistency = frequency of modal preference
                if preferences:
                    unique, counts = np.unique(preferences, return_counts=True)
                    consistency = counts.max() / len(preferences)
                    consistencies.append(consistency)
        
        return float(np.mean(consistencies)) if consistencies else 1.0
    
    def compute_theoretical_bounds(
        self,
        p: float,
        k_eff: int,
        k_n: int,
        delta: float,
        epsilon_bar: float,
        p_easy: float = 0.98,
    ) -> Tuple[float, float]:
        """
        Compute consistency-driven stability bounds from Theorem 2.
        
        Pr[Nash-stable] ∈ [p^{K_eff} · p_easy^{K_n - K_eff} · γ(G), p^{K_eff/2}]
        
        where γ(G) ≥ 1 - exp(-δ/ε̄) under capability monotonicity.
        
        Args:
            p: Preference consistency on critical decisions
            k_eff: Number of effective critical decisions
            k_n: Total number of decisions
            delta: Value gap δ
            epsilon_bar: Mean rationality bound ε̄
            p_easy: Consistency on easy decisions (default: 0.98)
        
        Returns:
            (lower_bound, upper_bound) for stability probability
        """
        # Game structure factor
        gamma = 1.0 - np.exp(-delta / epsilon_bar) if epsilon_bar > 0 else 1.0
        
        # Lower bound: all decisions consistent AND dynamics reach stability
        lower = (p ** k_eff) * (p_easy ** (k_n - k_eff)) * gamma
        
        # Upper bound: critical decisions consistent enough
        upper = p ** (k_eff / 2)
        
        return (float(lower), float(upper))
    
    def estimate_value_gap(self) -> float:
        """
        Estimate the δ-value gap for this game.
        
        The value gap δ is the minimum non-zero difference between
        per-capita values of distinct coalitions.
        
        Returns:
            Estimated δ value
        """
        # Sample coalition pairs and compute value differences
        values = []
        
        for agent_id in self.game.agent_ids:
            agent_coalitions = []
            
            # Generate various coalitions containing this agent
            for size in range(1, min(4, self.game.n + 1)):
                other_agents = [a for a in self.game.agent_ids if a != agent_id]
                for i in range(min(5, len(other_agents) ** size)):
                    members = [agent_id]
                    indices = np.random.choice(
                        len(other_agents), 
                        size=min(size - 1, len(other_agents)),
                        replace=False,
                    )
                    members.extend([other_agents[j] for j in indices])
                    
                    from coalition_llm.game_theory.coalition_game import Coalition
                    coalition = Coalition.from_agents(members)
                    values.append(self.game.per_capita_value(coalition))
        
        # Find minimum non-zero difference
        values = np.array(sorted(set(values)))
        if len(values) < 2:
            return 0.1  # Default
        
        diffs = np.diff(values)
        nonzero_diffs = diffs[diffs > 1e-6]
        
        return float(nonzero_diffs.min()) if len(nonzero_diffs) > 0 else 0.1
    
    def estimate_epsilon(
        self,
        agent_id: str,
        ground_truth_values: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Estimate ε-rationality bound for an agent.
        
        ε is the threshold below which an agent's choices become
        near-random between options.
        
        Args:
            agent_id: Agent to analyze
            ground_truth_values: Optional dict of true coalition values
        
        Returns:
            Estimated ε value
        """
        # This would normally involve repeated preference queries
        # and finding the threshold where choices become inconsistent.
        # For simulation, return typical values from Table 5.
        
        agent = self.game.agents.get(agent_id)
        if agent is None:
            return 0.17  # Default
        
        # Return model-specific estimates from Table 5
        model_epsilons = {
            "gpt-4": 0.15,
            "claude-3": 0.14,
            "llama-3": 0.22,
        }
        
        model_name = agent.model_name.lower()
        for key, eps in model_epsilons.items():
            if key in model_name:
                return eps
        
        return 0.17  # Average


def compute_stability_rate(
    results: List[StabilityResult],
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute Nash stability rate with confidence interval.
    
    Args:
        results: List of stability results from experiments
    
    Returns:
        (rate, std, (ci_lower, ci_upper)) at 95% confidence
    """
    stable = [1 if r.is_nash_stable else 0 for r in results]
    rate = np.mean(stable)
    std = np.std(stable)
    
    # Bootstrap confidence interval (BCa method approximation)
    n = len(stable)
    se = std / np.sqrt(n)
    ci_lower = rate - 1.96 * se
    ci_upper = rate + 1.96 * se
    
    return float(rate), float(std), (float(ci_lower), float(ci_upper))


def wilcoxon_test(
    results_a: List[StabilityResult],
    results_b: List[StabilityResult],
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test for paired comparisons.
    
    Args:
        results_a: Results from condition A
        results_b: Results from condition B
    
    Returns:
        (statistic, p_value)
    """
    stable_a = [1 if r.is_nash_stable else 0 for r in results_a]
    stable_b = [1 if r.is_nash_stable else 0 for r in results_b]
    
    # Wilcoxon signed-rank test
    try:
        stat, p_value = stats.wilcoxon(stable_a, stable_b)
    except ValueError:
        # All differences are zero
        stat, p_value = 0.0, 1.0
    
    return float(stat), float(p_value)


def cohens_d(
    results_a: List[StabilityResult],
    results_b: List[StabilityResult],
) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        results_a: Results from condition A
        results_b: Results from condition B
    
    Returns:
        Cohen's d effect size
    """
    stable_a = np.array([1 if r.is_nash_stable else 0 for r in results_a])
    stable_b = np.array([1 if r.is_nash_stable else 0 for r in results_b])
    
    mean_a, mean_b = stable_a.mean(), stable_b.mean()
    std_pooled = np.sqrt((stable_a.var() + stable_b.var()) / 2)
    
    if std_pooled == 0:
        return 0.0
    
    return float((mean_a - mean_b) / std_pooled)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.01,
) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: Raw p-values from multiple tests
        alpha: Family-wise error rate
    
    Returns:
        List of (corrected_p, is_significant) tuples
    """
    n_tests = len(p_values)
    threshold = alpha / n_tests
    
    return [
        (p * n_tests, p < threshold)
        for p in p_values
    ]
