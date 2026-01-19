"""
Baseline prompting protocols for coalition formation.

Implements the five baseline methods from Section 7.1:
1. Random: Uniformly random coalition assignment (28.3% stability)
2. Greedy: Maximize immediate per-capita value (52.1% stability)
3. Standard: Direct preference query (41.8% stability)
4. Vanilla CoT: Chain-of-thought prompting (58.4% stability)
5. Self-Consistency: Multiple CoT paths with majority voting (62.7% stability)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from coalition_llm.agents.llm_agent import LLMAgent
    from coalition_llm.game_theory.coalition_game import Coalition, CoalitionGame

from coalition_llm.prompts.coalt_protocol import Preference, CoalTResponse

logger = logging.getLogger(__name__)


# ============================================================================
# BASE PROTOCOL CLASS
# ============================================================================

class BaseProtocol(ABC):
    """Abstract base class for prompting protocols."""
    
    @abstractmethod
    def query_preference(
        self,
        agent: "LLMAgent",
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> CoalTResponse:
        """
        Query agent preference for coalition choice.
        
        Args:
            agent: Agent to query
            current_coalition: Agent's current coalition
            alternative_coalition: Coalition to potentially join
            game: The coalition formation game
        
        Returns:
            CoalTResponse with preference
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol name."""
        pass


# ============================================================================
# RANDOM PROTOCOL
# ============================================================================

class RandomProtocol(BaseProtocol):
    """
    Random coalition assignment baseline.
    
    Uniformly random preference - no reasoning involved.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed."""
        self.rng = np.random.default_rng(seed)
    
    def query_preference(
        self,
        agent: "LLMAgent",
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> CoalTResponse:
        """Return random preference."""
        choice = self.rng.choice([Preference.CURRENT, Preference.ALTERNATIVE])
        
        return CoalTResponse(
            preference=choice,
            reasoning="Random selection",
            step_outputs={},
            confidence=0.5,
        )
    
    @property
    def name(self) -> str:
        return "Random"
    
    def __repr__(self) -> str:
        return "RandomProtocol()"


# ============================================================================
# GREEDY PROTOCOL
# ============================================================================

class GreedyProtocol(BaseProtocol):
    """
    Greedy coalition formation baseline.
    
    Each agent joins the coalition maximizing immediate per-capita value.
    No reasoning - pure value calculation.
    """
    
    def __init__(self, epsilon: float = 0.001):
        """
        Initialize greedy protocol.
        
        Args:
            epsilon: Minimum improvement threshold to switch
        """
        self.epsilon = epsilon
    
    def query_preference(
        self,
        agent: "LLMAgent",
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> CoalTResponse:
        """Choose coalition with higher per-capita value."""
        # Compute per-capita values
        current_value = game.per_capita_value(current_coalition)
        
        # Add agent to alternative
        new_alt = alternative_coalition.add(agent.agent_id)
        alt_value = game.per_capita_value(new_alt)
        
        # Greedy choice
        if alt_value > current_value + self.epsilon:
            preference = Preference.ALTERNATIVE
        elif current_value > alt_value + self.epsilon:
            preference = Preference.CURRENT
        else:
            preference = Preference.INDIFFERENT
        
        return CoalTResponse(
            preference=preference,
            reasoning=f"Greedy: current={current_value:.3f}, alt={alt_value:.3f}",
            step_outputs={"current_value": str(current_value), "alt_value": str(alt_value)},
            confidence=abs(alt_value - current_value) * 5,  # Scale to ~confidence
        )
    
    @property
    def name(self) -> str:
        return "Greedy"
    
    def __repr__(self) -> str:
        return f"GreedyProtocol(ε={self.epsilon})"


# ============================================================================
# STANDARD PROMPTING PROTOCOL
# ============================================================================

STANDARD_PROMPT_TEMPLATE = """You are evaluating coalition options.

Current coalition: {current_members}
Alternative coalition (if you join): {alternative_members}

Which coalition do you prefer? Answer with:
- "PREFER CURRENT" to stay in your current coalition
- "PREFER ALTERNATIVE" to join the alternative coalition
- "INDIFFERENT" if you have no strong preference

Your preference:"""


class StandardProtocol(BaseProtocol):
    """
    Standard direct prompting baseline.
    
    Simple query without structured reasoning.
    """
    
    def __init__(self, system_prompt: str = "You are a helpful AI agent."):
        """Initialize with system prompt."""
        self.system_prompt = system_prompt
    
    def query_preference(
        self,
        agent: "LLMAgent",
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> CoalTResponse:
        """Query with simple direct prompt."""
        # Format coalition members
        current_members = ", ".join(current_coalition.members) or "just you"
        
        alt_members = list(alternative_coalition.members)
        if agent.agent_id not in alt_members:
            alt_members.append(agent.agent_id)
        alternative_members = ", ".join(alt_members)
        
        # Build prompt
        prompt = STANDARD_PROMPT_TEMPLATE.format(
            current_members=current_members,
            alternative_members=alternative_members,
        )
        
        # Query agent
        response_text = agent.query(prompt, system_prompt=self.system_prompt)
        
        # Parse preference
        response_lower = response_text.lower()
        if "prefer current" in response_lower or "stay" in response_lower:
            preference = Preference.CURRENT
        elif "prefer alternative" in response_lower or "join" in response_lower:
            preference = Preference.ALTERNATIVE
        else:
            preference = Preference.INDIFFERENT
        
        return CoalTResponse(
            preference=preference,
            reasoning=response_text,
            step_outputs={},
            confidence=0.6,
        )
    
    @property
    def name(self) -> str:
        return "Standard"
    
    def __repr__(self) -> str:
        return "StandardProtocol()"


# ============================================================================
# VANILLA CHAIN-OF-THOUGHT PROTOCOL
# ============================================================================

VANILLA_COT_PROMPT_TEMPLATE = """You are evaluating coalition options.

Current coalition: {current_members}
Alternative coalition (if you join): {alternative_members}

Think step by step about which coalition would be better for you.
Consider the pros and cons of each option.

After your reasoning, conclude with:
- "PREFER CURRENT" to stay
- "PREFER ALTERNATIVE" to join
- "INDIFFERENT" if equal

Your step-by-step analysis:"""


class VanillaCoTProtocol(BaseProtocol):
    """
    Vanilla Chain-of-Thought prompting baseline.
    
    Generic "think step by step" without game-theoretic framing.
    Expected stability: 58.4% (Table 3)
    
    Key difference from CoalT: No explicit game-theoretic concepts
    (capability complementarity, coordination costs, per-capita value).
    """
    
    def __init__(self, system_prompt: str = "You are a helpful AI agent."):
        """Initialize with system prompt."""
        self.system_prompt = system_prompt
    
    def query_preference(
        self,
        agent: "LLMAgent",
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> CoalTResponse:
        """Query with vanilla CoT prompt."""
        # Format coalition members
        current_members = ", ".join(current_coalition.members) or "just you"
        
        alt_members = list(alternative_coalition.members)
        if agent.agent_id not in alt_members:
            alt_members.append(agent.agent_id)
        alternative_members = ", ".join(alt_members)
        
        # Build prompt
        prompt = VANILLA_COT_PROMPT_TEMPLATE.format(
            current_members=current_members,
            alternative_members=alternative_members,
        )
        
        # Query agent
        response_text = agent.query(prompt, system_prompt=self.system_prompt)
        
        # Parse preference
        response_lower = response_text.lower()
        if "prefer current" in response_lower:
            preference = Preference.CURRENT
        elif "prefer alternative" in response_lower:
            preference = Preference.ALTERNATIVE
        elif "indifferent" in response_lower:
            preference = Preference.INDIFFERENT
        else:
            # Default based on keywords
            if "stay" in response_lower or "remain" in response_lower:
                preference = Preference.CURRENT
            elif "join" in response_lower or "switch" in response_lower:
                preference = Preference.ALTERNATIVE
            else:
                preference = Preference.CURRENT  # Conservative default
        
        return CoalTResponse(
            preference=preference,
            reasoning=response_text,
            step_outputs={"cot_reasoning": response_text},
            confidence=0.7,
        )
    
    @property
    def name(self) -> str:
        return "VanillaCoT"
    
    def __repr__(self) -> str:
        return "VanillaCoTProtocol()"


# ============================================================================
# SELF-CONSISTENCY PROTOCOL
# ============================================================================

class SelfConsistencyProtocol(BaseProtocol):
    """
    Self-Consistency prompting baseline.
    
    Multiple CoT reasoning paths with majority voting.
    """
    
    def __init__(
        self,
        n_samples: int = 5,
        temperature: float = 0.7,
        base_protocol: Optional[VanillaCoTProtocol] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize self-consistency protocol.
        
        Args:
            n_samples: Number of reasoning paths to sample
            temperature: Sampling temperature for diversity
            base_protocol: Base CoT protocol to use
            seed: Random seed
        """
        self.n_samples = n_samples
        self.temperature = temperature
        self.base_protocol = base_protocol or VanillaCoTProtocol()
        self.rng = np.random.default_rng(seed)
    
    def query_preference(
        self,
        agent: "LLMAgent",
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> CoalTResponse:
        """Query with multiple paths and majority voting."""
        # Store original temperature
        original_temp = agent.config.temperature
        
        # Sample multiple reasoning paths
        preferences: List[Preference] = []
        reasonings: List[str] = []
        
        try:
            agent.config.temperature = self.temperature
            
            for _ in range(self.n_samples):
                response = self.base_protocol.query_preference(
                    agent, current_coalition, alternative_coalition, game
                )
                preferences.append(response.preference)
                reasonings.append(response.reasoning)
        finally:
            # Restore original temperature
            agent.config.temperature = original_temp
        
        # Majority voting
        pref_counts = {}
        for p in preferences:
            pref_counts[p] = pref_counts.get(p, 0) + 1
        
        majority_pref = max(pref_counts.items(), key=lambda x: x[1])[0]
        majority_count = pref_counts[majority_pref]
        
        # Confidence based on agreement
        confidence = majority_count / self.n_samples
        
        return CoalTResponse(
            preference=majority_pref,
            reasoning=f"Self-consistency: {majority_pref.value} ({majority_count}/{self.n_samples})",
            step_outputs={
                "preferences": str([p.value for p in preferences]),
                "samples": str(self.n_samples),
            },
            confidence=confidence,
        )
    
    @property
    def name(self) -> str:
        return "SelfConsistency"
    
    def __repr__(self) -> str:
        return f"SelfConsistencyProtocol(n={self.n_samples}, T={self.temperature})"


# ============================================================================
# PROTOCOL FACTORY
# ============================================================================

def create_protocol(name: str, **kwargs) -> BaseProtocol:
    """
    Factory function to create protocols by name.
    
    Args:
        name: Protocol name (random, greedy, standard, vanilla_cot,
              self_consistency, coalt)
        **kwargs: Protocol-specific arguments
    
    Returns:
        Protocol instance
    """
    from coalition_llm.prompts.coalt_protocol import CoalTProtocol
    
    protocols = {
        "random": RandomProtocol,
        "greedy": GreedyProtocol,
        "standard": StandardProtocol,
        "vanilla_cot": VanillaCoTProtocol,
        "self_consistency": SelfConsistencyProtocol,
        "coalt": CoalTProtocol,
    }
    
    name_lower = name.lower().replace("-", "_")
    
    if name_lower not in protocols:
        raise ValueError(
            f"Unknown protocol: {name}. "
            f"Available: {list(protocols.keys())}"
        )
    
    return protocols[name_lower](**kwargs)
