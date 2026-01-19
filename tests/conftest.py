"""
Pytest configuration and fixtures for Coalition LLM tests.

Provides mock LLM responses, sample game instances, and reproducibility fixtures.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK LLM RESPONSES
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_COALT_RESPONSE = """
Step 1 (Capability Analysis):
- Current coalition C: math=0.68, facts=0.73, logic=0.76
- Alternative coalition C': math=0.58, facts=0.78, logic=0.79

Step 2 (Complementarity Assessment):
- C has balanced capabilities with slight math advantage
- C' has stronger facts and logic but weaker math
- My math capability (0.68) would complement C' well

Step 3 (Value Estimation):
- v(C) = (0.68 + 0.73 + 0.76) / 3 - 0.15 * 2^1.3 = 0.72 - 0.37 = 0.35
- v(C') with me = (0.68 + 0.78 + 0.79) / 3 - 0.15 * 3^1.3 = 0.75 - 0.60 = 0.15

Step 4 (Coordination Cost Analysis):
- C (size 2): coordination cost = 0.37
- C' (size 3): coordination cost = 0.60
- Larger coalition incurs significantly higher overhead

Step 5 (Preference Declaration):
Based on per-capita value analysis:
- Per-capita v(C) = 0.35 / 2 = 0.175
- Per-capita v(C') = 0.15 / 3 = 0.050
I PREFER current coalition C over C'.
"""

MOCK_STANDARD_RESPONSE = "I prefer coalition C."

MOCK_VANILLA_COT_RESPONSE = """
Let me think step by step:
1. Coalition C has good math capabilities
2. Coalition C' has better logic but is larger
3. I should consider the tradeoffs
Therefore, I prefer C.
"""


@pytest.fixture
def mock_openai_response() -> dict[str, Any]:
    """Mock OpenAI API response for preference queries."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4-0125-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": MOCK_COALT_RESPONSE,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "total_tokens": 700,
        },
    }


@pytest.fixture
def mock_anthropic_response() -> dict[str, Any]:
    """Mock Anthropic API response for preference queries."""
    return {
        "id": "msg-test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": MOCK_COALT_RESPONSE}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 500,
            "output_tokens": 200,
        },
    }


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns deterministic responses."""
    client = MagicMock()
    client.chat.completions.create = MagicMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=MOCK_COALT_RESPONSE))]
        )
    )
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE GAME INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleCapabilityProfile:
    """Sample capability profile for testing."""
    math: float
    facts: float
    logic: float

    def to_array(self) -> np.ndarray:
        return np.array([self.math, self.facts, self.logic])


# Default capability profiles from Table 2
DEFAULT_CAPABILITY_PROFILES = {
    "a1_gpt4": SampleCapabilityProfile(0.68, 0.73, 0.76),
    "a2_gpt4": SampleCapabilityProfile(0.65, 0.76, 0.73),
    "a3_claude3": SampleCapabilityProfile(0.62, 0.78, 0.74),
    "a4_claude3": SampleCapabilityProfile(0.59, 0.81, 0.71),
    "a5_llama3": SampleCapabilityProfile(0.58, 0.65, 0.79),
    "a6_llama3": SampleCapabilityProfile(0.55, 0.68, 0.76),
}


@pytest.fixture
def sample_capabilities() -> dict[str, np.ndarray]:
    """Provide sample capability profiles as numpy arrays."""
    return {k: v.to_array() for k, v in DEFAULT_CAPABILITY_PROFILES.items()}


@pytest.fixture
def sample_6_agent_game(sample_capabilities):
    """Create a sample 6-agent LCFG instance."""
    from coalition_llm.game_theory.coalition_game import CoalitionGame
    from coalition_llm.game_theory.value_functions import CoverageValueFunction
    from coalition_llm.agents.llm_agent import LLMAgent

    value_fn = CoverageValueFunction(alpha=0.15, beta=1.3)
    agents = []
    
    model_mapping = {
        "a1_gpt4": ("gpt-4-0125-preview", "gpt-4"),
        "a2_gpt4": ("gpt-4-0125-preview", "gpt-4"),
        "a3_claude3": ("claude-3-opus-20240229", "claude-3"),
        "a4_claude3": ("claude-3-opus-20240229", "claude-3"),
        "a5_llama3": ("meta-llama/Llama-3-70b-chat-hf", "llama-3"),
        "a6_llama3": ("meta-llama/Llama-3-70b-chat-hf", "llama-3"),
    }

    for agent_id, cap in sample_capabilities.items():
        model_id, arch = model_mapping[agent_id]
        agent = LLMAgent(
            agent_id=agent_id,
            model_id=model_id,
            architecture=arch,
            capability_profile=cap,
            temperature=0.0,
        )
        agents.append(agent)

    return CoalitionGame(agents=agents, value_function=value_fn)


@pytest.fixture
def sample_partition():
    """Create a sample partition for testing."""
    # Partition: {a1, a3}, {a2, a5, a6}, {a4}
    return [
        frozenset(["a1_gpt4", "a3_claude3"]),
        frozenset(["a2_gpt4", "a5_llama3", "a6_llama3"]),
        frozenset(["a4_claude3"]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def seed() -> int:
    """Default random seed for tests."""
    return 42


@pytest.fixture
def set_deterministic_seed(seed: int) -> Generator[None, None, None]:
    """Set all random seeds for reproducible tests."""
    import torch
    
    # Save current state
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    yield
    
    # Restore state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE FUNCTION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def coverage_value_fn():
    """Create a CoverageValueFunction with paper parameters."""
    from coalition_llm.game_theory.value_functions import CoverageValueFunction
    return CoverageValueFunction(alpha=0.15, beta=1.3)


@pytest.fixture
def coordination_cost_params() -> dict[str, float]:
    """Parameters for coordination cost function."""
    return {"alpha": 0.15, "beta": 1.3}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_questions() -> list[dict[str, Any]]:
    """Sample questions for coalition QA task."""
    return [
        {
            "id": "q1",
            "question": "What is the derivative of x^2?",
            "answer": "2x",
            "difficulty": {"math": 0.3, "facts": 0.1, "logic": 0.2},
            "category": "math",
        },
        {
            "id": "q2",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "difficulty": {"math": 0.0, "facts": 0.2, "logic": 0.1},
            "category": "facts",
        },
        {
            "id": "q3",
            "question": "If all A are B, and all B are C, are all A C?",
            "answer": "Yes",
            "difficulty": {"math": 0.1, "facts": 0.1, "logic": 0.5},
            "category": "logic",
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_episode_results() -> list[dict[str, Any]]:
    """Sample episode results for metrics testing."""
    return [
        {
            "episode_id": 0,
            "nash_stable": True,
            "convergence_rounds": 5,
            "welfare": 0.81,
            "consistency": 0.86,
            "final_partition": [["a1", "a3"], ["a2", "a5", "a6"], ["a4"]],
        },
        {
            "episode_id": 1,
            "nash_stable": True,
            "convergence_rounds": 8,
            "welfare": 0.78,
            "consistency": 0.82,
            "final_partition": [["a1", "a2"], ["a3", "a4"], ["a5", "a6"]],
        },
        {
            "episode_id": 2,
            "nash_stable": False,
            "convergence_rounds": 30,  # Max rounds
            "welfare": 0.65,
            "consistency": 0.71,
            "final_partition": [["a1"], ["a2", "a3"], ["a4", "a5", "a6"]],
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def coalt_prompt_template() -> str:
    """CoalT prompt template for testing."""
    return """You are evaluating whether to join coalition C' instead of staying in C.

Current coalition C: {current_coalition}
Alternative coalition C': {alternative_coalition}
Your capabilities: {agent_capabilities}

Analyze systematically following these steps:

Step 1 (Capability Analysis): List capabilities of members in each coalition.
Step 2 (Complementarity Assessment): Identify capability gaps and overlaps.
Step 3 (Value Estimation): Estimate task performance for each coalition.
Step 4 (Coordination Cost Analysis): Assess communication/coordination overhead.
Step 5 (Preference Declaration): State "I PREFER C" or "I PREFER C'" based on per-capita value.

Begin your analysis:"""


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test."""
    yield
    # Clean up any global state if needed
    import gc
    gc.collect()
