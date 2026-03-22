"""
Coalition-of-Thought (CoalT) prompting protocol.

Implements Algorithm 1: A structured 5-step protocol that guides agents
through game-theoretic coalition reasoning:

1. Capability Analysis
2. Complementarity Assessment
3. Value Estimation
4. Coordination Cost Analysis
5. Preference Declaration

Key insight: CoalT differs from vanilla CoT by incorporating game-theoretic
concepts (capability complementarity, coordination costs, per-capita value)
rather than generic step-by-step reasoning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from coalition_llm.agents.llm_agent import LLMAgent
    from coalition_llm.game_theory.coalition_game import Coalition, CoalitionGame

logger = logging.getLogger(__name__)


class Preference(Enum):
    """Coalition preference outcome."""
    CURRENT = "current"
    ALTERNATIVE = "alternative"
    INDIFFERENT = "indifferent"


@dataclass
class CoalTResponse:
    """
    Structured response from CoalT protocol.
    
    Attributes:
        preference: Final preference (CURRENT, ALTERNATIVE, INDIFFERENT)
        reasoning: Full reasoning trace
        step_outputs: Dict mapping step name to its output
        confidence: Estimated confidence in preference
    """
    preference: Preference
    reasoning: str
    step_outputs: Dict[str, str]
    confidence: float = 0.0
    
    def __repr__(self) -> str:
        return f"CoalTResponse({self.preference.value}, conf={self.confidence:.2f})"


# ============================================================================
# COALT PROMPT TEMPLATE
# ============================================================================

COALT_SYSTEM_PROMPT = """You are an AI agent participating in a coalition formation game.
Your goal is to form coalitions that maximize your per-capita value while considering
coordination costs. Analyze coalition options systematically using game-theoretic reasoning."""

COALT_PROMPT_TEMPLATE = """You are evaluating whether to join coalition C' instead of staying in your current coalition C.

**Current Coalition C:**
{current_coalition_info}

**Alternative Coalition C' (if you join):**
{alternative_coalition_info}

Analyze systematically following these 5 steps:

## Step 1: Capability Analysis
List the capabilities of members in each coalition. What strengths does each group have?

## Step 2: Complementarity Assessment
Identify capability gaps and overlaps. Are there complementary strengths or redundant capabilities?

## Step 3: Value Estimation
Estimate the expected task performance (coverage) for each coalition based on capabilities.

## Step 4: Coordination Cost Analysis
Assess the communication/coordination overhead for each coalition size.
Coordination cost grows superlinearly with size: ψ(k) = 0.15 × k^1.3

## Step 5: Preference Declaration
Based on your per-capita expected value (value / coalition_size), state your preference:
- "PREFER CURRENT" if you prefer staying in C
- "PREFER ALTERNATIVE" if you prefer joining C'
- "INDIFFERENT" if values are approximately equal

Provide your analysis:"""


# ============================================================================
# COALT PROTOCOL CLASS
# ============================================================================

class CoalTProtocol:
    """
    Coalition-of-Thought (CoalT) prompting protocol.
    
    Achieves 73.2% Nash stability rate compared to:
    - 58.4% for vanilla CoT (+14.8pp improvement, p < 0.001)
    - 41.8% for standard prompting (+31.4pp improvement)
    """
    
    def __init__(
        self,
        system_prompt: str = COALT_SYSTEM_PROMPT,
        include_capability_values: bool = True,
        include_coordination_formula: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize CoalT protocol.
        
        Args:
            system_prompt: System prompt for agents
            include_capability_values: Include numerical capability values
            include_coordination_formula: Include ψ(k) formula
            verbose: Enable verbose logging
        """
        self.system_prompt = system_prompt
        self.include_capability_values = include_capability_values
        self.include_coordination_formula = include_coordination_formula
        self.verbose = verbose
    
    def format_coalition_info(
        self,
        coalition_members: List[str],
        agents: Dict[str, "LLMAgent"],
        include_agent: Optional[str] = None,
    ) -> str:
        """
        Format coalition information for the prompt.
        
        Args:
            coalition_members: List of agent IDs in coalition
            agents: Dict of all agents
            include_agent: Agent to hypothetically add
        
        Returns:
            Formatted coalition description
        """
        members = list(coalition_members)
        if include_agent and include_agent not in members:
            members.append(include_agent)
        
        if not members:
            return "Empty coalition (you would be alone)"
        
        lines = [f"Members: {', '.join(members)} (size: {len(members)})"]
        
        if self.include_capability_values:
            lines.append("Capabilities:")
            for agent_id in members:
                agent = agents[agent_id]
                caps = agent.capability_dict
                cap_str = ", ".join(f"{k}={v:.2f}" for k, v in caps.items())
                lines.append(f"  - {agent_id} ({agent.model_name}): {cap_str}")
        
        return "\n".join(lines)
    
    def build_prompt(
        self,
        agent_id: str,
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> str:
        """
        Build the CoalT prompt for a preference query.
        
        Args:
            agent_id: Agent making the decision
            current_coalition: Agent's current coalition
            alternative_coalition: Coalition to potentially join
            game: The coalition formation game
        
        Returns:
            Formatted prompt string
        """
        # Format current coalition (excluding agent for comparison)
        current_info = self.format_coalition_info(
            list(current_coalition.members),
            game.agents,
        )
        
        # Format alternative coalition (including agent)
        alt_members = list(alternative_coalition.members)
        if agent_id not in alt_members:
            alt_members.append(agent_id)
        
        alternative_info = self.format_coalition_info(
            alt_members,
            game.agents,
        )
        
        # Build prompt
        prompt = COALT_PROMPT_TEMPLATE.format(
            current_coalition_info=current_info,
            alternative_coalition_info=alternative_info,
        )
        
        return prompt
    
    def parse_response(self, response: str) -> CoalTResponse:
        """
        Parse LLM response into structured CoalTResponse.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            Parsed CoalTResponse
        """
        # Extract step outputs
        step_outputs = {}
        step_patterns = [
            (r"Step 1[:\s]*Capability Analysis\s*(.*?)(?=Step 2|$)", "capability_analysis"),
            (r"Step 2[:\s]*Complementarity Assessment\s*(.*?)(?=Step 3|$)", "complementarity"),
            (r"Step 3[:\s]*Value Estimation\s*(.*?)(?=Step 4|$)", "value_estimation"),
            (r"Step 4[:\s]*Coordination Cost\s*(.*?)(?=Step 5|$)", "coordination_cost"),
            (r"Step 5[:\s]*Preference Declaration\s*(.*?)$", "preference"),
        ]
        
        for pattern, step_name in step_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                step_outputs[step_name] = match.group(1).strip()
        
        # Determine preference from response
        response_lower = response.lower()
        
        if "prefer current" in response_lower or "stay" in response_lower:
            preference = Preference.CURRENT
        elif "prefer alternative" in response_lower or "join" in response_lower:
            preference = Preference.ALTERNATIVE
        elif "indifferent" in response_lower:
            preference = Preference.INDIFFERENT
        else:
            # Default to current if unclear
            logger.warning(f"Could not parse preference from response, defaulting to CURRENT")
            preference = Preference.CURRENT
        
        # Estimate confidence based on language
        confidence = 0.5
        if "clearly" in response_lower or "definitely" in response_lower:
            confidence = 0.9
        elif "slightly" in response_lower or "marginally" in response_lower:
            confidence = 0.6
        elif "uncertain" in response_lower or "close" in response_lower:
            confidence = 0.4
        
        return CoalTResponse(
            preference=preference,
            reasoning=response,
            step_outputs=step_outputs,
            confidence=confidence,
        )
    
    def query_preference(
        self,
        agent: "LLMAgent",
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> CoalTResponse:
        """
        Query agent preference using CoalT protocol.
        
        Args:
            agent: Agent to query
            current_coalition: Agent's current coalition
            alternative_coalition: Coalition to potentially join
            game: The coalition formation game
        
        Returns:
            CoalTResponse with preference and reasoning
        """
        prompt = self.build_prompt(
            agent.agent_id,
            current_coalition,
            alternative_coalition,
            game,
        )
        
        if self.verbose:
            logger.info(f"CoalT prompt for {agent.agent_id}:\n{prompt}")
        
        # Query the agent
        response_text = agent.query(prompt, system_prompt=self.system_prompt)
        
        if self.verbose:
            logger.info(f"CoalT response from {agent.agent_id}:\n{response_text}")
        
        return self.parse_response(response_text)
    
    def __repr__(self) -> str:
        return "CoalTProtocol(5-step game-theoretic reasoning)"


# ============================================================================
# COALT ABLATION VARIANTS
# ============================================================================
# For Table 4 ablation study

class CoalTAblation(CoalTProtocol):
    """
    CoalT with specific components removed for ablation study.
    """
    
    def __init__(
        self,
        remove_capability_analysis: bool = False,
        remove_complementarity: bool = False,
        remove_value_estimation: bool = False,
        remove_coordination_cost: bool = False,
        **kwargs,
    ):
        """
        Initialize ablated CoalT.
        
        Args:
            remove_capability_analysis: Remove Step 1
            remove_complementarity: Remove Step 2
            remove_value_estimation: Remove Step 3
            remove_coordination_cost: Remove Step 4
        """
        super().__init__(**kwargs)
        self.remove_capability_analysis = remove_capability_analysis
        self.remove_complementarity = remove_complementarity
        self.remove_value_estimation = remove_value_estimation
        self.remove_coordination_cost = remove_coordination_cost
    
    def build_prompt(
        self,
        agent_id: str,
        current_coalition: "Coalition",
        alternative_coalition: "Coalition",
        game: "CoalitionGame",
    ) -> str:
        """Build ablated prompt with specified steps removed."""
        # Get base info
        current_info = self.format_coalition_info(
            list(current_coalition.members),
            game.agents,
        )
        
        alt_members = list(alternative_coalition.members)
        if agent_id not in alt_members:
            alt_members.append(agent_id)
        alternative_info = self.format_coalition_info(alt_members, game.agents)
        
        # Build steps based on ablation config
        steps = []
        step_num = 1
        
        if not self.remove_capability_analysis:
            steps.append(f"""## Step {step_num}: Capability Analysis
List the capabilities of members in each coalition.""")
            step_num += 1
        
        if not self.remove_complementarity:
            steps.append(f"""## Step {step_num}: Complementarity Assessment
Identify capability gaps and overlaps.""")
            step_num += 1
        
        if not self.remove_value_estimation:
            steps.append(f"""## Step {step_num}: Value Estimation
Estimate expected task performance for each coalition.""")
            step_num += 1
        
        if not self.remove_coordination_cost:
            steps.append(f"""## Step {step_num}: Coordination Cost Analysis
Assess coordination overhead. Cost = 0.15 × size^1.3""")
            step_num += 1
        
        steps.append(f"""## Step {step_num}: Preference Declaration
State: PREFER CURRENT, PREFER ALTERNATIVE, or INDIFFERENT""")
        
        steps_text = "\n\n".join(steps)
        
        prompt = f"""You are evaluating whether to join coalition C' instead of staying in C.

**Current Coalition C:**
{current_info}

**Alternative Coalition C' (if you join):**
{alternative_info}

Analyze following these steps:

{steps_text}

Provide your analysis:"""
        
        return prompt
    
    def __repr__(self) -> str:
        removed = []
        if self.remove_capability_analysis:
            removed.append("capability")
        if self.remove_complementarity:
            removed.append("complementarity")
        if self.remove_value_estimation:
            removed.append("value")
        if self.remove_coordination_cost:
            removed.append("cost")
        
        if removed:
            return f"CoalTAblation(removed={removed})"
        return "CoalTAblation(full)"


# ============================================================================
# EXAMPLE COALT REASONING
# ============================================================================
# Reference: Section 6 example

EXAMPLE_COALT_REASONING = """
Example CoalT Reasoning (from Section 6):

For agent a1 (GPT-4, math=0.68) comparing:
- Coalition C = {a1, a2} (both GPT-4)
- Coalition C' = {a3, a5, a6} (Claude + 2 Llama)

Step 1: C has math (0.68, 0.65) but limited logic (0.76, 0.73).
        C' has facts (0.78) and logic (0.79, 0.76).

Step 2: C has capability overlap; C' would add my math to their facts/logic.

Step 3: v(C∪{a1}) ≈ 0.42; v(C'∪{a1}) ≈ 0.51

Step 4: Size 3 vs. 4 means costs 0.44 vs. 0.64

Step 5: Per-capita: 0.14 vs. 0.13. PREFER CURRENT coalition C.
"""


def get_example_reasoning() -> str:
    """Return the example CoalT reasoning from Section 6."""
    return EXAMPLE_COALT_REASONING
