"""
LLM Agent implementation for coalition formation.

Implements Definition 1: An LLM agent is a tuple a_i = (m_i, θ_i, c_i) where:
- m_i ∈ M is the model architecture (e.g., GPT-4, Claude-3)
- θ_i ∈ Θ = [0,2] × Σ* specifies configuration (temperature, system prompt)
- c_i ∈ [0,1]^d is a capability profile over d dimensions
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Default capability profiles from Table 2
# (Math, Facts, Logic) - estimated from benchmark subsets
DEFAULT_CAPABILITY_PROFILES = {
    # GPT-4 agents
    "a1": {"model": "gpt-4", "capabilities": [0.68, 0.73, 0.76]},
    "a2": {"model": "gpt-4", "capabilities": [0.65, 0.76, 0.73]},
    # Claude-3 agents
    "a3": {"model": "claude-3", "capabilities": [0.62, 0.78, 0.74]},
    "a4": {"model": "claude-3", "capabilities": [0.59, 0.81, 0.71]},
    # Llama-3 agents
    "a5": {"model": "llama-3", "capabilities": [0.58, 0.65, 0.79]},
    "a6": {"model": "llama-3", "capabilities": [0.55, 0.68, 0.76]},
}

# Capability dimensions
CAPABILITY_DIMENSIONS = ["math", "facts", "logic"]


@dataclass
class AgentConfig:
    """
    Agent configuration θ_i = (τ, s).
    
    Attributes:
        temperature: Sampling temperature τ ∈ [0, 2]
        system_prompt: System prompt s ∈ Σ*
        max_tokens: Maximum response tokens
        top_p: Nucleus sampling parameter
    """
    temperature: float = 0.0  # τ = 0 for reproducibility
    system_prompt: str = ""
    max_tokens: int = 1024
    top_p: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"Temperature must be in [0,2], got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class LLMAgent:
    """
    LLM Agent for coalition formation.
    
    Implements Definition 1: a_i = (m_i, θ_i, c_i)
    
    Attributes:
        agent_id: Unique identifier for this agent
        model_name: Model architecture m_i (e.g., "gpt-4", "claude-3")
        capabilities: Capability profile c_i ∈ [0,1]^d
        config: Agent configuration θ_i
    """
    agent_id: str
    model_name: str
    capabilities: np.ndarray
    config: AgentConfig = field(default_factory=AgentConfig)
    
    # API client (lazy initialization)
    _client: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Validate and normalize capabilities."""
        self.capabilities = np.array(self.capabilities, dtype=np.float64)
        
        if self.capabilities.ndim != 1:
            raise ValueError(f"Capabilities must be 1D, got shape {self.capabilities.shape}")
        
        if not np.all((self.capabilities >= 0) & (self.capabilities <= 1)):
            logger.warning(
                f"Agent {self.agent_id} capabilities outside [0,1]: "
                f"clipping to valid range"
            )
            self.capabilities = np.clip(self.capabilities, 0, 1)
    
    @classmethod
    def from_profile(
        cls,
        agent_id: str,
        profile: Optional[Dict[str, Any]] = None,
        config: Optional[AgentConfig] = None,
    ) -> LLMAgent:
        """
        Create agent from a capability profile.
        
        Args:
            agent_id: Agent identifier
            profile: Dict with "model" and "capabilities" keys
                    (defaults to DEFAULT_CAPABILITY_PROFILES)
            config: Agent configuration
        
        Returns:
            LLMAgent instance
        """
        if profile is None:
            if agent_id not in DEFAULT_CAPABILITY_PROFILES:
                raise ValueError(
                    f"Unknown agent_id '{agent_id}'. "
                    f"Available: {list(DEFAULT_CAPABILITY_PROFILES.keys())}"
                )
            profile = DEFAULT_CAPABILITY_PROFILES[agent_id]
        
        return cls(
            agent_id=agent_id,
            model_name=profile["model"],
            capabilities=np.array(profile["capabilities"]),
            config=config or AgentConfig(),
        )
    
    @classmethod
    def create_default_agents(
        cls,
        config: Optional[AgentConfig] = None,
    ) -> Dict[str, LLMAgent]:
        """
        Create the default 6 agents from Table 2.
        
        Returns:
            Dict mapping agent_id to LLMAgent
        """
        return {
            agent_id: cls.from_profile(agent_id, config=config)
            for agent_id in DEFAULT_CAPABILITY_PROFILES
        }
    
    @property
    def d(self) -> int:
        """Capability dimension."""
        return len(self.capabilities)
    
    @property
    def capability_dict(self) -> Dict[str, float]:
        """Capabilities as a labeled dict."""
        labels = CAPABILITY_DIMENSIONS[:self.d]
        return dict(zip(labels, self.capabilities.tolist()))
    
    def get_client(self) -> Any:
        """
        Get or create API client for this agent's model.
        
        Returns:
            API client (OpenAI, Anthropic, or Together)
        """
        if self._client is not None:
            return self._client
        
        model_lower = self.model_name.lower()
        
        if "gpt" in model_lower:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai package required for GPT models")
        
        elif "claude" in model_lower:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("anthropic package required for Claude models")
        
        elif "llama" in model_lower:
            try:
                from together import Together
                self._client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
            except ImportError:
                raise ImportError("together package required for Llama models")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
        
        return self._client
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Query the LLM agent.
        
        Args:
            prompt: User prompt
            system_prompt: Override system prompt (optional)
        
        Returns:
            Model response text
        """
        client = self.get_client()
        sys_prompt = system_prompt or self.config.system_prompt
        
        model_lower = self.model_name.lower()
        
        if "gpt" in model_lower:
            response = client.chat.completions.create(
                model=self._get_full_model_name(),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            return response.choices[0].message.content or ""
        
        elif "claude" in model_lower:
            response = client.messages.create(
                model=self._get_full_model_name(),
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.content[0].text
        
        elif "llama" in model_lower:
            response = client.chat.completions.create(
                model=self._get_full_model_name(),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            return response.choices[0].message.content or ""
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _get_full_model_name(self) -> str:
        """Get the full API model name."""
        model_lower = self.model_name.lower()
        
        # Map short names to full API names (from paper Section 7.1)
        model_mapping = {
            "gpt-4": "gpt-4-0125-preview",
            "claude-3": "claude-3-opus-20240229",
            "llama-3": "meta-llama/Llama-3-70b-chat-hf",
        }
        
        for short, full in model_mapping.items():
            if short in model_lower:
                return full
        
        # Return as-is if not found
        return self.model_name
    
    def __repr__(self) -> str:
        caps = ", ".join(f"{k}={v:.2f}" for k, v in self.capability_dict.items())
        return f"LLMAgent({self.agent_id}, {self.model_name}, [{caps}])"
    
    def __hash__(self) -> int:
        """Hash based on agent_id."""
        return hash(self.agent_id)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on agent_id."""
        if not isinstance(other, LLMAgent):
            return NotImplemented
        return self.agent_id == other.agent_id


def estimate_capabilities_from_benchmarks(
    model_name: str,
    math_accuracy: float,
    facts_accuracy: float,
    logic_accuracy: float,
) -> np.ndarray:
    """
    Create capability profile from benchmark accuracies.
    
    Note: These are relative estimates for our task domain,
    not official benchmark scores (see Section 7.1).
    
    Args:
        model_name: Model identifier
        math_accuracy: MATH subset accuracy
        facts_accuracy: MMLU knowledge subset accuracy
        logic_accuracy: LogiQA accuracy
    
    Returns:
        Capability profile c_i ∈ [0,1]^3
    """
    capabilities = np.array([
        math_accuracy,
        facts_accuracy,
        logic_accuracy,
    ])
    
    # Normalize to [0, 1] if needed
    capabilities = np.clip(capabilities, 0, 1)
    
    logger.info(
        f"Estimated capabilities for {model_name}: "
        f"math={capabilities[0]:.2f}, facts={capabilities[1]:.2f}, "
        f"logic={capabilities[2]:.2f}"
    )
    
    return capabilities


# Mock client for testing without API access
class MockLLMClient:
    """Mock client for testing coalition formation without API calls."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize mock client with optional seed."""
        self.rng = np.random.default_rng(seed)
        self.call_count = 0
    
    def generate_response(
        self,
        prompt: str,
        consistency: float = 0.86,
    ) -> str:
        """
        Generate mock response for coalition preference queries.
        
        Args:
            prompt: Input prompt
            consistency: Probability of consistent preference (default: CoalT level)
        
        Returns:
            Mock response string
        """
        self.call_count += 1
        
        # Parse preference query and return consistent response
        if "prefer" in prompt.lower() or "coalition" in prompt.lower():
            if self.rng.random() < consistency:
                return "I prefer the current coalition based on per-capita value analysis."
            else:
                return "I prefer the alternative coalition for capability complementarity."
        
        return f"Mock response to: {prompt[:50]}..."
