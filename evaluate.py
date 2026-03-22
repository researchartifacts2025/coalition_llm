#!/usr/bin/env python
"""
Evaluation script for coalition formation experiments.

Loads results from training runs and performs statistical analysis:
- Aggregates metrics across seeds
- Performs statistical significance tests (Wilcoxon signed-rank)
- Applies Bonferroni correction
- Computes effect sizes (Cohen's d)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from coalition_llm.evaluation.metrics import (
    EpisodeResult,
    AggregateMetrics,
    aggregate_results,
    wilcoxon_signed_rank_test,
    cohens_d,
    bonferroni_correction,
    generate_table3,
    EXPECTED_TABLE3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict[str, List[EpisodeResult]]:
    """
    Load results from all experiment runs in a directory.
    
    Args:
        results_dir: Directory containing experiment outputs
    
    Returns:
        Dict mapping protocol name to list of EpisodeResults
    """
    results_by_protocol: Dict[str, List[EpisodeResult]] = {}
    
    for results_file in results_dir.rglob("results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            protocol = data.get("config", {}).get("protocol", {}).get("name", "unknown")
            
            episodes = []
            for ep in data.get("episodes", []):
                result = EpisodeResult(
                    episode_id=ep["episode_id"],
                    protocol=protocol,
                    seed=ep["seed"],
                    is_nash_stable=ep["is_nash_stable"],
                    convergence_rounds=ep["convergence_rounds"],
                    social_welfare=ep["social_welfare"],
                    consistency_score=ep["consistency_score"],
                    final_partition="",  # Not saved in JSON
                )
                episodes.append(result)
            
            if protocol not in results_by_protocol:
                results_by_protocol[protocol] = []
            results_by_protocol[protocol].extend(episodes)
            
            logger.info(f"Loaded {len(episodes)} episodes for {protocol} from {results_file}")
        
        except Exception as e:
            logger.warning(f"Failed to load {results_file}: {e}")
    
    return results_by_protocol


def compute_pairwise_comparisons(
    results_by_protocol: Dict[str, List[EpisodeResult]],
    baseline: str = "Standard",
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise statistical comparisons against baseline.
    
    Args:
        results_by_protocol: Results organized by protocol
        baseline: Baseline protocol for comparison
    
    Returns:
        Dict with comparison statistics for each protocol
    """
    if baseline not in results_by_protocol:
        logger.warning(f"Baseline {baseline} not found in results")
        return {}
    
    baseline_results = results_by_protocol[baseline]
    comparisons = {}
    
    for protocol, results in results_by_protocol.items():
        if protocol == baseline:
            continue
        
        # Match by seed for paired comparison
        paired_baseline = []
        paired_protocol = []
        
        baseline_by_seed = {r.seed: r for r in baseline_results}
        
        for r in results:
            if r.seed in baseline_by_seed:
                paired_baseline.append(baseline_by_seed[r.seed])
                paired_protocol.append(r)
        
        if len(paired_baseline) < 10:
            logger.warning(
                f"Insufficient paired samples for {protocol} vs {baseline}: "
                f"{len(paired_baseline)}"
            )
            continue
        
        # Statistical tests
        stat, p_value = wilcoxon_signed_rank_test(paired_protocol, paired_baseline)
        d = cohens_d(paired_protocol, paired_baseline)
        
        # Compute improvement
        protocol_rate = sum(r.is_nash_stable for r in paired_protocol) / len(paired_protocol)
        baseline_rate = sum(r.is_nash_stable for r in paired_baseline) / len(paired_baseline)
        improvement = protocol_rate - baseline_rate
        
        comparisons[protocol] = {
            "wilcoxon_stat": stat,
            "p_value": p_value,
            "cohens_d": d,
            "improvement": improvement,
            "n_pairs": len(paired_baseline),
        }
    
    return comparisons


def generate_results_report(
    results_by_protocol: Dict[str, List[EpisodeResult]],
    comparisons: Dict[str, Dict[str, float]],
    baseline: str = "Standard",
) -> str:
    """
    Generate formatted results report.
    
    Args:
        results_by_protocol: Results organized by protocol
        comparisons: Pairwise comparison statistics
        baseline: Baseline protocol name
    
    Returns:
        Formatted markdown report
    """
    lines = [
        "# Coalition Formation Results",
        "",
        "| Condition | Nash Stable | Conv (rounds) | Welfare | Consistency |",
        "|-----------|-------------|---------------|---------|-------------|",
    ]
    
    # Order protocols for display
    protocol_order = ["Random", "Greedy", "Standard", "VanillaCoT", "SelfConsistency", "CoalT"]
    
    for protocol in protocol_order:
        if protocol not in results_by_protocol:
            continue
        
        results = results_by_protocol[protocol]
        metrics = aggregate_results(results)
        
        # Significance markers
        sig = ""
        if protocol in comparisons:
            p_val = comparisons[protocol]["p_value"]
            if p_val < 0.001:
                sig = "**"
            elif p_val < 0.01:
                sig = "*"
        
        lines.append(
            f"| {protocol} | "
            f"{metrics.nash_stability_rate*100:.1f}%{sig} "
            f"[{metrics.nash_stability_ci[0]*100:.1f}, {metrics.nash_stability_ci[1]*100:.1f}] | "
            f"{metrics.convergence_mean:.1f}±{metrics.convergence_std:.1f} | "
            f"{metrics.welfare_mean:.2f}±{metrics.welfare_std:.2f} | "
            f"{metrics.consistency_mean:.2f}±{metrics.consistency_std:.2f} |"
        )
    
    lines.extend([
        "",
        f"*Statistical significance vs. {baseline}: * p < 0.01, ** p < 0.001 "
        "(Wilcoxon signed-rank, Bonferroni-corrected)*",
        "",
        "## Effect Sizes",
        "",
        "| Protocol | Cohen's d | Improvement |",
        "|----------|-----------|-------------|",
    ])
    
    for protocol, stats in comparisons.items():
        lines.append(
            f"| {protocol} vs. {baseline} | "
            f"{stats['cohens_d']:.2f} | "
            f"{stats['improvement']*100:+.1f}pp |"
        )
    
    return "\n".join(lines)


def validate_against_expected(
    results_by_protocol: Dict[str, List[EpisodeResult]],
    tolerance: float = 0.05,
) -> Dict[str, bool]:
    """
    Validate results against expected values from paper.
    
    Args:
        results_by_protocol: Actual results
        tolerance: Acceptable deviation from expected
    
    Returns:
        Dict mapping protocol to validation status
    """
    validation = {}
    
    for protocol, expected in EXPECTED_TABLE3.items():
        if protocol not in results_by_protocol:
            validation[protocol] = None  # Not run
            continue
        
        metrics = aggregate_results(results_by_protocol[protocol])
        
        expected_rate = expected["nash_stable"]
        actual_rate = metrics.nash_stability_rate
        
        diff = abs(actual_rate - expected_rate)
        validation[protocol] = diff <= tolerance
        
        if not validation[protocol]:
            logger.warning(
                f"{protocol}: Expected {expected_rate*100:.1f}%, "
                f"got {actual_rate*100:.1f}% (diff: {diff*100:.1f}pp)"
            )
    
    return validation


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate coalition formation results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing experiment outputs",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="Standard",
        help="Baseline protocol for comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for report (default: stdout)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate against expected Table 3 values",
    )
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading results from {args.results_dir}")
    results_by_protocol = load_results(args.results_dir)
    
    if not results_by_protocol:
        logger.error("No results found!")
        return 1
    
    logger.info(f"Loaded results for protocols: {list(results_by_protocol.keys())}")
    
    # Compute comparisons
    comparisons = compute_pairwise_comparisons(results_by_protocol, args.baseline)
    
    # Apply Bonferroni correction
    if comparisons:
        p_values = [stats["p_value"] for stats in comparisons.values()]
        corrections = bonferroni_correction(p_values)
        for (protocol, stats), (corrected_p, sig) in zip(comparisons.items(), corrections):
            stats["p_value_corrected"] = corrected_p
            stats["significant"] = sig
    
    # Generate report
    report = generate_results_report(results_by_protocol, comparisons, args.baseline)
    
    if args.output:
        args.output.write_text(report)
        logger.info(f"Report saved to {args.output}")
    else:
        print(report)
    
    # Validation
    if args.validate:
        logger.info("Validating against expected values...")
        validation = validate_against_expected(results_by_protocol)
        
        all_valid = all(v for v in validation.values() if v is not None)
        if all_valid:
            logger.info("✓ All results within expected tolerance")
        else:
            logger.warning("✗ Some results deviate from expected values")
    
    return 0


if __name__ == "__main__":
    exit(main())
