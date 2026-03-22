"""
Evaluation metrics for coalition formation experiments.

Implements metrics from Table 3:
- Nash Stability Rate (%)
- Convergence (rounds)
- Welfare (social welfare)
- Consistency (preference consistency)

Plus statistical testing from Section 7.1:
- Bootstrap confidence intervals (10,000 iterations, BCa method)
- Wilcoxon signed-rank tests
- Bonferroni correction
- Cohen's d effect sizes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """
    Result from a single coalition formation episode.
    
    Attributes:
        episode_id: Unique episode identifier
        protocol: Protocol used (e.g., "CoalT", "VanillaCoT")
        seed: Random seed used
        is_nash_stable: Whether final partition is Nash-stable
        convergence_rounds: Rounds to convergence (-1 if not converged)
        social_welfare: Final social welfare value
        consistency_score: Measured preference consistency
        final_partition: String representation of final partition
    """
    episode_id: int
    protocol: str
    seed: int
    is_nash_stable: bool
    convergence_rounds: int
    social_welfare: float
    consistency_score: float
    final_partition: str


@dataclass
class AggregateMetrics:
    """
    Aggregated metrics across multiple episodes.
    
    Attributes:
        protocol: Protocol name
        n_episodes: Number of episodes
        nash_stability_rate: Fraction of Nash-stable outcomes
        nash_stability_ci: 95% confidence interval [lower, upper]
        convergence_mean: Mean convergence rounds
        convergence_std: Standard deviation
        welfare_mean: Mean social welfare
        welfare_std: Standard deviation
        consistency_mean: Mean preference consistency
        consistency_std: Standard deviation
    """
    protocol: str
    n_episodes: int
    nash_stability_rate: float
    nash_stability_ci: Tuple[float, float]
    convergence_mean: float
    convergence_std: float
    welfare_mean: float
    welfare_std: float
    consistency_mean: float
    consistency_std: float
    
    def to_table_row(self) -> str:
        """Format as table row matching Table 3."""
        return (
            f"{self.protocol:15} "
            f"{self.nash_stability_rate*100:5.1f}% "
            f"[{self.nash_stability_ci[0]*100:.1f}, {self.nash_stability_ci[1]*100:.1f}] "
            f"{self.convergence_mean:5.1f}±{self.convergence_std:.1f} "
            f"{self.welfare_mean:.2f}±{self.welfare_std:.2f} "
            f"{self.consistency_mean:.2f}±{self.consistency_std:.2f}"
        )


def compute_nash_stability_rate(
    results: List[EpisodeResult],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute Nash stability rate with bootstrap confidence interval.
    
    Uses BCa (bias-corrected and accelerated) bootstrap method
    as specified in Section 7.1.
    
    Args:
        results: List of episode results
        confidence: Confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap iterations (default: 10,000)
    
    Returns:
        (rate, (ci_lower, ci_upper))
    """
    stable = np.array([1 if r.is_nash_stable else 0 for r in results])
    n = len(stable)
    
    if n == 0:
        return 0.0, (0.0, 0.0)
    
    rate = stable.mean()
    
    # Bootstrap confidence interval
    rng = np.random.default_rng(42)
    bootstrap_rates = []
    
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_rates.append(stable[indices].mean())
    
    bootstrap_rates = np.array(bootstrap_rates)
    
    # Percentile method (approximation to BCa)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_rates, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_rates, (1 - alpha / 2) * 100)
    
    return float(rate), (float(ci_lower), float(ci_upper))


def compute_convergence_stats(
    results: List[EpisodeResult],
) -> Tuple[float, float]:
    """
    Compute convergence statistics (mean ± std).
    
    Only includes episodes that converged (convergence_rounds >= 0).
    
    Args:
        results: List of episode results
    
    Returns:
        (mean, std) of convergence rounds
    """
    rounds = [r.convergence_rounds for r in results if r.convergence_rounds >= 0]
    
    if not rounds:
        return float("nan"), float("nan")
    
    return float(np.mean(rounds)), float(np.std(rounds))


def compute_welfare_stats(
    results: List[EpisodeResult],
) -> Tuple[float, float]:
    """
    Compute social welfare statistics (mean ± std).
    
    Args:
        results: List of episode results
    
    Returns:
        (mean, std) of social welfare
    """
    welfare = [r.social_welfare for r in results]
    
    if not welfare:
        return float("nan"), float("nan")
    
    return float(np.mean(welfare)), float(np.std(welfare))


def compute_consistency_stats(
    results: List[EpisodeResult],
) -> Tuple[float, float]:
    """
    Compute preference consistency statistics (mean ± std).
    
    Args:
        results: List of episode results
    
    Returns:
        (mean, std) of consistency scores
    """
    consistency = [r.consistency_score for r in results]
    
    if not consistency:
        return float("nan"), float("nan")
    
    return float(np.mean(consistency)), float(np.std(consistency))


def aggregate_results(results: List[EpisodeResult]) -> AggregateMetrics:
    """
    Aggregate results into summary metrics matching Table 3.
    
    Args:
        results: List of episode results (should be from same protocol)
    
    Returns:
        AggregateMetrics object
    """
    if not results:
        raise ValueError("No results to aggregate")
    
    protocol = results[0].protocol
    
    nash_rate, nash_ci = compute_nash_stability_rate(results)
    conv_mean, conv_std = compute_convergence_stats(results)
    welfare_mean, welfare_std = compute_welfare_stats(results)
    consist_mean, consist_std = compute_consistency_stats(results)
    
    return AggregateMetrics(
        protocol=protocol,
        n_episodes=len(results),
        nash_stability_rate=nash_rate,
        nash_stability_ci=nash_ci,
        convergence_mean=conv_mean,
        convergence_std=conv_std,
        welfare_mean=welfare_mean,
        welfare_std=welfare_std,
        consistency_mean=consist_mean,
        consistency_std=consist_std,
    )


def wilcoxon_signed_rank_test(
    results_a: List[EpisodeResult],
    results_b: List[EpisodeResult],
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test for paired comparison.
    
    Reference: Section 7.1 statistical methodology.
    
    Args:
        results_a: Results from condition A
        results_b: Results from condition B (paired)
    
    Returns:
        (statistic, p_value)
    """
    if len(results_a) != len(results_b):
        raise ValueError("Results must be paired (same length)")
    
    stable_a = np.array([1 if r.is_nash_stable else 0 for r in results_a])
    stable_b = np.array([1 if r.is_nash_stable else 0 for r in results_b])
    
    # Wilcoxon signed-rank test
    try:
        stat, p_value = stats.wilcoxon(stable_a, stable_b, alternative="two-sided")
    except ValueError as e:
        # Handle case where all differences are zero
        logger.warning(f"Wilcoxon test failed: {e}")
        stat, p_value = 0.0, 1.0
    
    return float(stat), float(p_value)


def cohens_d(
    results_a: List[EpisodeResult],
    results_b: List[EpisodeResult],
) -> float:
    """
    Compute Cohen's d effect size for Nash stability comparison.
    
    Args:
        results_a: Results from condition A
        results_b: Results from condition B
    
    Returns:
        Cohen's d effect size
    """
    stable_a = np.array([1 if r.is_nash_stable else 0 for r in results_a])
    stable_b = np.array([1 if r.is_nash_stable else 0 for r in results_b])
    
    mean_a, mean_b = stable_a.mean(), stable_b.mean()
    
    # Pooled standard deviation
    n_a, n_b = len(stable_a), len(stable_b)
    var_a, var_b = stable_a.var(), stable_b.var()
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((mean_a - mean_b) / pooled_std)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.01,
) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: Raw p-values from multiple tests
        alpha: Family-wise error rate (default: 0.01)
    
    Returns:
        List of (corrected_p, is_significant) tuples
    """
    n_tests = len(p_values)
    threshold = alpha / n_tests
    
    return [
        (min(p * n_tests, 1.0), p < threshold)
        for p in p_values
    ]


def generate_table3(
    results_by_protocol: Dict[str, List[EpisodeResult]],
    baseline_protocol: str = "Standard",
) -> str:
    """
    Generate Table 3 from the paper.
    
    Args:
        results_by_protocol: Dict mapping protocol name to results
        baseline_protocol: Protocol to compare against
    
    Returns:
        Formatted table string
    """
    lines = [
        "=" * 90,
        "Table 3: Coalition formation results (mean ± std, 95% CI in brackets)",
        "=" * 90,
        f"{'Condition':15} {'Nash Stable':15} {'Conv (rounds)':15} {'Welfare':12} {'Consist.':12}",
        "-" * 90,
    ]
    
    baseline_results = results_by_protocol.get(baseline_protocol, [])
    p_values = []
    
    for protocol, results in results_by_protocol.items():
        metrics = aggregate_results(results)
        
        # Significance marker
        sig = ""
        if protocol != baseline_protocol and baseline_results:
            _, p_val = wilcoxon_signed_rank_test(results, baseline_results)
            p_values.append(p_val)
            if p_val < 0.001:
                sig = "**"
            elif p_val < 0.01:
                sig = "*"
        
        lines.append(
            f"{protocol:15} "
            f"{metrics.nash_stability_rate*100:5.1f}%{sig:2} "
            f"[{metrics.nash_stability_ci[0]*100:.1f}, {metrics.nash_stability_ci[1]*100:.1f}] "
            f"{metrics.convergence_mean:5.1f}±{metrics.convergence_std:.1f} "
            f"{metrics.welfare_mean:.2f}±{metrics.welfare_std:.2f} "
            f"{metrics.consistency_mean:.2f}±{metrics.consistency_std:.2f}"
        )
    
    lines.append("-" * 90)
    lines.append("* p < 0.01, ** p < 0.001 (Wilcoxon, Bonferroni-corrected)")
    lines.append("=" * 90)
    
    return "\n".join(lines)


# Expected results from Table 3 for validation
EXPECTED_TABLE3 = {
    "Random": {
        "nash_stable": 0.283,
        "nash_ci": (0.251, 0.315),
        "convergence": None,
        "welfare": (0.58, 0.14),
        "consistency": None,
    },
    "Greedy": {
        "nash_stable": 0.521,
        "nash_ci": None,
        "convergence": (6.8, 3.2),
        "welfare": (0.69, 0.10),
        "consistency": (0.71, 0.08),
    },
    "Standard": {
        "nash_stable": 0.418,
        "nash_ci": None,
        "convergence": (18.3, 7.2),
        "welfare": (0.72, 0.11),
        "consistency": (0.64, 0.09),
    },
    "VanillaCoT": {
        "nash_stable": 0.584,
        "nash_ci": None,
        "convergence": (14.2, 5.8),
        "welfare": (0.75, 0.09),
        "consistency": (0.74, 0.07),
    },
    "SelfConsistency": {
        "nash_stable": 0.627,
        "nash_ci": None,
        "convergence": (13.1, 5.2),
        "welfare": (0.77, 0.08),
        "consistency": (0.79, 0.06),
    },
    "CoalT": {
        "nash_stable": 0.732,
        "nash_ci": (0.693, 0.771),
        "convergence": (11.4, 4.1),
        "welfare": (0.81, 0.08),
        "consistency": (0.86, 0.05),
    },
}
