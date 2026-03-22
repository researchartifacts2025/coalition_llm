#!/usr/bin/env python
"""
Training script for coalition formation experiments.

Runs coalition formation episodes with specified protocol and configuration.
Uses Hydra for configuration management and supports full reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from coalition_llm.agents.llm_agent import LLMAgent, AgentConfig
from coalition_llm.game_theory.coalition_game import CoalitionGame, Partition
from coalition_llm.game_theory.stability import StabilityAnalyzer, StabilityResult
from coalition_llm.game_theory.value_functions import CoverageValueFunction, CoordinationCost
from coalition_llm.prompts.baseline_protocols import create_protocol
from coalition_llm.prompts.coalt_protocol import CoalTProtocol, Preference
from coalition_llm.evaluation.metrics import EpisodeResult, aggregate_results
from coalition_llm.utils.reproducibility import set_seed, print_reproducibility_info

logger = logging.getLogger(__name__)


def create_agents(cfg: DictConfig) -> Dict[str, LLMAgent]:
    """Create LLM agents from configuration."""
    agent_config = AgentConfig(
        temperature=cfg.agent.temperature,
        max_tokens=cfg.agent.max_tokens,
    )
    
    if cfg.agent.get("use_default_profiles", True):
        return LLMAgent.create_default_agents(config=agent_config)
    
    # Custom agent configuration
    agents = {}
    for agent_id, profile in cfg.agent.profiles.items():
        agents[agent_id] = LLMAgent(
            agent_id=agent_id,
            model_name=profile.model,
            capabilities=np.array(profile.capabilities),
            config=agent_config,
        )
    return agents


def create_game(agents: Dict[str, LLMAgent], cfg: DictConfig) -> CoalitionGame:
    """Create coalition formation game from configuration."""
    value_function = CoverageValueFunction(
        normalize=cfg.game.value_function.normalize,
        aggregation=cfg.game.value_function.aggregation,
    )
    
    coordination_cost = CoordinationCost(
        alpha=cfg.game.coordination_cost.alpha,
        beta=cfg.game.coordination_cost.beta,
    )
    
    return CoalitionGame(
        agents=agents,
        value_function=value_function,
        coordination_cost=coordination_cost,
    )


def run_episode(
    game: CoalitionGame,
    protocol: CoalTProtocol,
    episode_id: int,
    cfg: DictConfig,
    seed: int,
) -> EpisodeResult:
    """
    Run a single coalition formation episode.
    
    Args:
        game: Coalition formation game
        protocol: Prompting protocol to use
        episode_id: Episode identifier
        cfg: Configuration
        seed: Random seed for this episode
    
    Returns:
        EpisodeResult with all metrics
    """
    set_seed(seed)
    
    # Initialize partition (singletons)
    agent_ids = list(game.agents.keys())
    partition = Partition.singletons(agent_ids)
    
    # Run improving dynamics
    max_rounds = cfg.training.max_rounds
    epsilon = cfg.training.epsilon
    
    history = [partition]
    convergence_round = -1
    
    for round_num in range(max_rounds):
        # Check stability
        analyzer = StabilityAnalyzer(game, epsilon=epsilon)
        stability_result = analyzer.verify_nash_stability(partition)
        
        if stability_result.is_nash_stable:
            convergence_round = round_num
            break
        
        # Find and execute improving deviation
        improved = False
        for agent_id in np.random.permutation(agent_ids):
            deviations = game.get_improving_deviations(partition, agent_id, epsilon)
            
            if deviations:
                # Query protocol for preference
                current = partition.get_coalition(agent_id)
                target, _ = deviations[0]  # Best deviation
                
                if cfg.training.get("use_protocol", True):
                    response = protocol.query_preference(
                        game.agents[agent_id],
                        current,
                        target,
                        game,
                    )
                    
                    if response.preference != Preference.ALTERNATIVE:
                        continue
                
                # Execute move
                target_idx = next(
                    (i for i, c in enumerate(partition.coalitions) if c == target),
                    len(partition.coalitions),
                )
                partition = partition.move_agent(agent_id, target_idx)
                history.append(partition)
                improved = True
                break
        
        if not improved:
            # No improving moves found (stable or stuck)
            convergence_round = round_num
            break
    
    # Final stability check
    analyzer = StabilityAnalyzer(game, epsilon=epsilon)
    final_stability = analyzer.verify_nash_stability(partition)
    
    # Compute metrics
    social_welfare = game.social_welfare(partition)
    consistency = analyzer.estimate_consistency(partition, seed=seed)
    
    return EpisodeResult(
        episode_id=episode_id,
        protocol=protocol.name if hasattr(protocol, "name") else str(type(protocol).__name__),
        seed=seed,
        is_nash_stable=final_stability.is_nash_stable,
        convergence_rounds=convergence_round if convergence_round >= 0 else max_rounds,
        social_welfare=social_welfare,
        consistency_score=consistency,
        final_partition=str(partition),
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def train(cfg: DictConfig) -> float:
    """
    Main training function.
    
    Runs coalition formation experiments and saves results.
    
    Args:
        cfg: Hydra configuration
    
    Returns:
        Nash stability rate (for hyperparameter optimization)
    """
    # Setup
    OmegaConf.resolve(cfg)
    print_reproducibility_info()
    
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create agents and game
    agents = create_agents(cfg)
    game = create_game(agents, cfg)
    
    logger.info(f"Created game with {len(agents)} agents")
    for agent_id, agent in agents.items():
        logger.info(f"  {agent}")
    
    # Create protocol
    protocol = create_protocol(cfg.protocol.name, **cfg.protocol.get("params", {}))
    logger.info(f"Using protocol: {protocol}")
    
    # Run episodes
    n_episodes = cfg.training.n_episodes
    base_seed = cfg.seed
    results: List[EpisodeResult] = []
    
    start_time = time.time()
    
    for episode_id in range(n_episodes):
        seed = base_seed + episode_id
        
        result = run_episode(game, protocol, episode_id, cfg, seed)
        results.append(result)
        
        if (episode_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (episode_id + 1) / elapsed
            logger.info(
                f"Episode {episode_id + 1}/{n_episodes} "
                f"({rate:.1f} eps/s, "
                f"Nash stable: {sum(r.is_nash_stable for r in results)}/{len(results)})"
            )
    
    # Aggregate results
    metrics = aggregate_results(results)
    
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Protocol: {cfg.protocol.name}")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Nash Stability Rate: {metrics.nash_stability_rate*100:.1f}% "
                f"CI: [{metrics.nash_stability_ci[0]*100:.1f}, "
                f"{metrics.nash_stability_ci[1]*100:.1f}]")
    logger.info(f"Convergence: {metrics.convergence_mean:.1f} ± {metrics.convergence_std:.1f} rounds")
    logger.info(f"Welfare: {metrics.welfare_mean:.2f} ± {metrics.welfare_std:.2f}")
    logger.info(f"Consistency: {metrics.consistency_mean:.2f} ± {metrics.consistency_std:.2f}")
    logger.info("=" * 60)
    
    # Save results
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # Save detailed results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": OmegaConf.to_container(cfg),
                "metrics": {
                    "nash_stability_rate": metrics.nash_stability_rate,
                    "nash_stability_ci": metrics.nash_stability_ci,
                    "convergence_mean": metrics.convergence_mean,
                    "convergence_std": metrics.convergence_std,
                    "welfare_mean": metrics.welfare_mean,
                    "welfare_std": metrics.welfare_std,
                    "consistency_mean": metrics.consistency_mean,
                    "consistency_std": metrics.consistency_std,
                },
                "episodes": [
                    {
                        "episode_id": r.episode_id,
                        "seed": r.seed,
                        "is_nash_stable": r.is_nash_stable,
                        "convergence_rounds": r.convergence_rounds,
                        "social_welfare": r.social_welfare,
                        "consistency_score": r.consistency_score,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    
    logger.info(f"Saved results to {results_file}")
    
    return metrics.nash_stability_rate


if __name__ == "__main__":
    train()
