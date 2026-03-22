"""
Coalition Formation in LLM Agent Networks.

A game-theoretic framework for analyzing coalition dynamics among LLM agents,
with stability analysis and the Coalition-of-Thought (CoalT) protocol.

Paper: "Coalition Formation in LLM Agent Networks: A Game-Theoretic Framework
with Stability Analysis"
"""

__version__ = "1.0.0"
__author__ = "Anonymous"
__email__ = "anonymous@example.com"

from coalition_llm.game_theory.coalition_game import (
    Coalition,
    CoalitionGame,
    Partition,
)
from coalition_llm.game_theory.stability import StabilityAnalyzer
from coalition_llm.game_theory.value_functions import (
    CoverageValueFunction,
    CoordinationCost,
)
from coalition_llm.agents.llm_agent import LLMAgent
from coalition_llm.prompts.coalt_protocol import CoalTProtocol
from coalition_llm.prompts.baseline_protocols import (
    StandardProtocol,
    VanillaCoTProtocol,
    SelfConsistencyProtocol,
    GreedyProtocol,
    RandomProtocol,
)

__all__ = [
    # Core classes
    "Coalition",
    "CoalitionGame",
    "Partition",
    "StabilityAnalyzer",
    "LLMAgent",
    # Value functions
    "CoverageValueFunction",
    "CoordinationCost",
    # Protocols
    "CoalTProtocol",
    "StandardProtocol",
    "VanillaCoTProtocol",
    "SelfConsistencyProtocol",
    "GreedyProtocol",
    "RandomProtocol",
]
