"""
LLM Coalition Formation Game (LCFG) implementation.

This module implements Definition 2 from the paper:
G = (N, v, {≿_i}_{i∈N}) where:
- N is a set of LLM agents
- v: 2^N → R is a coalition value function
- ≿_i is agent a_i's preference relation over coalitions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Iterator, List, Optional, Set, Tuple

import numpy as np

from coalition_llm.game_theory.value_functions import (
    CoverageValueFunction,
    CoordinationCost,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Coalition:
    """
    A coalition of LLM agents.
    
    A coalition S ⊆ N is a subset of agents that work together.
    Immutable to enable use as dict keys and set members.
    
    Attributes:
        members: Frozenset of agent IDs in this coalition.
    """
    members: FrozenSet[str]
    
    def __post_init__(self) -> None:
        """Validate coalition has at least one member."""
        if not self.members:
            # Empty coalition is valid (represents agents who want to be alone)
            pass
    
    @classmethod
    def from_agents(cls, agent_ids: List[str] | Set[str]) -> Coalition:
        """Create coalition from a list or set of agent IDs."""
        return cls(frozenset(agent_ids))
    
    @classmethod
    def singleton(cls, agent_id: str) -> Coalition:
        """Create a singleton coalition containing one agent."""
        return cls(frozenset([agent_id]))
    
    @classmethod
    def empty(cls) -> Coalition:
        """Create an empty coalition."""
        return cls(frozenset())
    
    def __contains__(self, agent_id: str) -> bool:
        """Check if agent is in this coalition."""
        return agent_id in self.members
    
    def __len__(self) -> int:
        """Return the size of the coalition."""
        return len(self.members)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over agent IDs in the coalition."""
        return iter(self.members)
    
    def __repr__(self) -> str:
        """String representation."""
        if not self.members:
            return "Coalition(∅)"
        return f"Coalition({{{', '.join(sorted(self.members))}}})"
    
    def add(self, agent_id: str) -> Coalition:
        """Return a new coalition with the agent added."""
        return Coalition(self.members | {agent_id})
    
    def remove(self, agent_id: str) -> Coalition:
        """Return a new coalition with the agent removed."""
        return Coalition(self.members - {agent_id})
    
    def union(self, other: Coalition) -> Coalition:
        """Return union of two coalitions."""
        return Coalition(self.members | other.members)
    
    def intersection(self, other: Coalition) -> Coalition:
        """Return intersection of two coalitions."""
        return Coalition(self.members & other.members)


@dataclass
class Partition:
    """
    A coalition structure (partition) of the agent set.
    
    A partition π = {C_1, ..., C_k} partitions N such that:
    - ∪_j C_j = N (every agent is in exactly one coalition)
    - C_i ∩ C_j = ∅ for i ≠ j (coalitions are disjoint)
    
    Reference: Section 3.3 of the paper.
    """
    coalitions: List[Coalition]
    
    def __post_init__(self) -> None:
        """Validate this is a valid partition."""
        # Check for overlaps
        all_agents: Set[str] = set()
        for coalition in self.coalitions:
            overlap = all_agents & coalition.members
            if overlap:
                raise ValueError(
                    f"Invalid partition: agents {overlap} appear in multiple coalitions"
                )
            all_agents |= coalition.members
    
    @classmethod
    def from_assignment(cls, assignment: dict[str, int]) -> Partition:
        """
        Create partition from agent-to-coalition assignment.
        
        Args:
            assignment: Dict mapping agent_id -> coalition_index
        
        Returns:
            Partition object
        """
        coalition_members: dict[int, Set[str]] = {}
        for agent_id, coalition_idx in assignment.items():
            if coalition_idx not in coalition_members:
                coalition_members[coalition_idx] = set()
            coalition_members[coalition_idx].add(agent_id)
        
        coalitions = [
            Coalition.from_agents(members)
            for members in coalition_members.values()
        ]
        return cls(coalitions)
    
    @classmethod
    def singletons(cls, agent_ids: List[str]) -> Partition:
        """Create partition where each agent is in their own coalition."""
        return cls([Coalition.singleton(aid) for aid in agent_ids])
    
    @classmethod
    def grand_coalition(cls, agent_ids: List[str]) -> Partition:
        """Create partition with all agents in one coalition."""
        return cls([Coalition.from_agents(agent_ids)])
    
    def get_coalition(self, agent_id: str) -> Coalition:
        """Get the coalition containing a specific agent."""
        for coalition in self.coalitions:
            if agent_id in coalition:
                return coalition
        raise ValueError(f"Agent {agent_id} not found in partition")
    
    def get_coalition_index(self, agent_id: str) -> int:
        """Get the index of the coalition containing an agent."""
        for idx, coalition in enumerate(self.coalitions):
            if agent_id in coalition:
                return idx
        raise ValueError(f"Agent {agent_id} not found in partition")
    
    def move_agent(self, agent_id: str, target_coalition_idx: int) -> Partition:
        """
        Create new partition with agent moved to target coalition.
        
        Args:
            agent_id: Agent to move
            target_coalition_idx: Index of target coalition
        
        Returns:
            New Partition with the agent moved
        """
        new_coalitions: List[Coalition] = []
        
        for idx, coalition in enumerate(self.coalitions):
            if agent_id in coalition:
                # Remove from current coalition
                new_coal = coalition.remove(agent_id)
                if len(new_coal) > 0:  # Keep non-empty coalitions
                    new_coalitions.append(new_coal)
            elif idx == target_coalition_idx:
                # Add to target coalition
                new_coalitions.append(coalition.add(agent_id))
            else:
                new_coalitions.append(coalition)
        
        # Handle case where target was empty (new singleton)
        if target_coalition_idx == len(self.coalitions):
            new_coalitions.append(Coalition.singleton(agent_id))
        
        return Partition(new_coalitions)
    
    def __len__(self) -> int:
        """Return number of coalitions."""
        return len(self.coalitions)
    
    def __iter__(self) -> Iterator[Coalition]:
        """Iterate over coalitions."""
        return iter(self.coalitions)
    
    def __repr__(self) -> str:
        """String representation."""
        coal_strs = [str(c) for c in self.coalitions]
        return f"Partition([{', '.join(coal_strs)}])"
    
    def to_assignment(self) -> dict[str, int]:
        """Convert to agent-to-coalition-index mapping."""
        assignment = {}
        for idx, coalition in enumerate(self.coalitions):
            for agent_id in coalition:
                assignment[agent_id] = idx
        return assignment
    
    @property
    def all_agents(self) -> Set[str]:
        """Get all agents in this partition."""
        agents: Set[str] = set()
        for coalition in self.coalitions:
            agents |= coalition.members
        return agents


class CoalitionGame:
    """
    LLM Coalition Formation Game (LCFG).
    
    Implements Definition 2: G = (N, v, {≿_i}_{i∈N})
    
    The game consists of:
    - A set of LLM agents N with capability profiles
    - A coalition value function v (Equation 1)
    - Agent preferences derived from per-capita value
    
    Attributes:
        agents: Dict mapping agent_id to agent instance
        value_function: Callable computing coalition value
        coordination_cost: Coordination cost function ψ(k)
    """
    
    def __init__(
        self,
        agents: dict[str, "LLMAgent"],  # Forward reference
        value_function: Optional[CoverageValueFunction] = None,
        coordination_cost: Optional[CoordinationCost] = None,
    ):
        """
        Initialize coalition formation game.
        
        Args:
            agents: Dict mapping agent_id to LLMAgent instances
            value_function: Coalition value function (default: coverage-based)
            coordination_cost: Coordination cost function (default: superlinear)
        """
        self.agents = agents
        self.value_function = value_function or CoverageValueFunction()
        self.coordination_cost = coordination_cost or CoordinationCost()
        
        # Cache for computed values
        self._value_cache: dict[FrozenSet[str], float] = {}
        
        logger.info(
            f"Initialized CoalitionGame with {len(agents)} agents, "
            f"α={self.coordination_cost.alpha}, β={self.coordination_cost.beta}"
        )
    
    @property
    def agent_ids(self) -> List[str]:
        """Get list of all agent IDs."""
        return list(self.agents.keys())
    
    @property
    def n(self) -> int:
        """Number of agents."""
        return len(self.agents)
    
    def coalition_value(self, coalition: Coalition) -> float:
        """
        Compute coalition value v(S) using Equation 1.
        
        v(S) = φ(⊕_{a_i ∈ S} c_i) - ψ(|S|)
        
        where φ aggregates capabilities (coverage), ⊕ is componentwise max,
        and ψ is the coordination cost.
        
        Args:
            coalition: Coalition to evaluate
        
        Returns:
            Coalition value v(S)
        """
        if not coalition.members:
            return 0.0
        
        # Check cache
        if coalition.members in self._value_cache:
            return self._value_cache[coalition.members]
        
        # Get capability profiles
        capabilities = []
        for agent_id in coalition.members:
            if agent_id not in self.agents:
                raise ValueError(f"Unknown agent: {agent_id}")
            capabilities.append(self.agents[agent_id].capabilities)
        
        # Compute coverage value: φ(⊕ c_i)
        capabilities_array = np.array(capabilities)
        coverage = self.value_function(capabilities_array)
        
        # Subtract coordination cost: ψ(|S|)
        cost = self.coordination_cost(len(coalition))
        
        value = coverage - cost
        
        # Cache result
        self._value_cache[coalition.members] = value
        
        return value
    
    def per_capita_value(self, coalition: Coalition) -> float:
        """
        Compute per-capita value v_i(S) = v(S) / |S|.
        
        This is what agents use to compare coalitions (Section 3.2).
        
        Args:
            coalition: Coalition to evaluate
        
        Returns:
            Per-capita value
        """
        if not coalition.members:
            return 0.0
        return self.coalition_value(coalition) / len(coalition)
    
    def agent_prefers(
        self,
        agent_id: str,
        coalition_a: Coalition,
        coalition_b: Coalition,
        epsilon: float = 0.0,
    ) -> int:
        """
        Determine if agent prefers coalition A over B.
        
        Uses ε-rational preferences (Definition 3):
        v_i(A) > v_i(B) + ε ⟹ A ≻_i B
        
        Args:
            agent_id: Agent making the comparison
            coalition_a: First coalition (must contain agent)
            coalition_b: Second coalition (must contain agent)
            epsilon: Rationality threshold (ε)
        
        Returns:
            1 if agent prefers A, -1 if prefers B, 0 if indifferent
        """
        if agent_id not in coalition_a or agent_id not in coalition_b:
            raise ValueError(f"Agent {agent_id} must be in both coalitions")
        
        value_a = self.per_capita_value(coalition_a)
        value_b = self.per_capita_value(coalition_b)
        
        diff = value_a - value_b
        
        if diff > epsilon:
            return 1  # Prefers A
        elif diff < -epsilon:
            return -1  # Prefers B
        else:
            return 0  # Indifferent
    
    def get_improving_deviations(
        self,
        partition: Partition,
        agent_id: str,
        epsilon: float = 0.0,
    ) -> List[Tuple[Coalition, float]]:
        """
        Find all coalitions agent would prefer to join.
        
        Args:
            partition: Current coalition structure
            agent_id: Agent considering deviation
            epsilon: Minimum improvement threshold
        
        Returns:
            List of (target_coalition, improvement) tuples
        """
        current = partition.get_coalition(agent_id)
        current_value = self.per_capita_value(current)
        
        improving: List[Tuple[Coalition, float]] = []
        
        for coalition in partition.coalitions:
            if coalition == current:
                continue
            
            # Consider joining this coalition
            new_coalition = coalition.add(agent_id)
            new_value = self.per_capita_value(new_coalition)
            
            improvement = new_value - current_value
            if improvement > epsilon:
                improving.append((coalition, improvement))
        
        # Also consider forming singleton
        singleton = Coalition.singleton(agent_id)
        singleton_value = self.per_capita_value(singleton)
        singleton_improvement = singleton_value - current_value
        if singleton_improvement > epsilon:
            improving.append((Coalition.empty(), singleton_improvement))
        
        return sorted(improving, key=lambda x: -x[1])  # Best first
    
    def potential(self, partition: Partition) -> float:
        """
        Compute potential function Φ(π) = Σ_{C∈π} v(C).
        
        Used in convergence proof (Theorem 1).
        
        Args:
            partition: Coalition structure
        
        Returns:
            Potential value
        """
        return sum(self.coalition_value(c) for c in partition.coalitions)
    
    def social_welfare(self, partition: Partition) -> float:
        """
        Compute social welfare (sum of per-capita values).
        
        Args:
            partition: Coalition structure
        
        Returns:
            Social welfare
        """
        total = 0.0
        for coalition in partition.coalitions:
            total += self.per_capita_value(coalition) * len(coalition)
        return total / self.n
    
    def run_improving_dynamics(
        self,
        initial_partition: Optional[Partition] = None,
        max_rounds: int = 30,
        epsilon: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[Partition, List[Partition], int]:
        """
        Run improving dynamics until convergence.
        
        At each round, a random agent with an improving deviation
        moves to their preferred coalition.
        
        Args:
            initial_partition: Starting partition (default: singletons)
            max_rounds: Maximum rounds before termination
            epsilon: Minimum improvement threshold
            seed: Random seed for agent selection
        
        Returns:
            (final_partition, history, convergence_round)
        """
        rng = np.random.default_rng(seed)
        
        if initial_partition is None:
            initial_partition = Partition.singletons(self.agent_ids)
        
        current = initial_partition
        history = [current]
        
        for round_num in range(max_rounds):
            # Collect all improving deviations
            all_deviations: List[Tuple[str, Coalition, float]] = []
            
            for agent_id in self.agent_ids:
                deviations = self.get_improving_deviations(
                    current, agent_id, epsilon
                )
                for target, improvement in deviations:
                    all_deviations.append((agent_id, target, improvement))
            
            if not all_deviations:
                # No improving deviations = stable
                logger.info(f"Converged at round {round_num}")
                return current, history, round_num
            
            # Select random improving agent
            idx = rng.integers(len(all_deviations))
            agent_id, target_coalition, _ = all_deviations[idx]
            
            # Find target coalition index
            if target_coalition.members:
                for cidx, c in enumerate(current.coalitions):
                    if c == target_coalition:
                        target_idx = cidx
                        break
            else:
                target_idx = len(current.coalitions)  # New singleton
            
            current = current.move_agent(agent_id, target_idx)
            history.append(current)
        
        logger.warning(f"Did not converge within {max_rounds} rounds")
        return current, history, -1
    
    def clear_cache(self) -> None:
        """Clear the value cache."""
        self._value_cache.clear()


# Type alias for preference function
PreferenceFunction = Callable[[str, Coalition, Coalition], int]
