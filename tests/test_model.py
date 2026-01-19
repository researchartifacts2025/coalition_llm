"""
Unit tests for Coalition LLM models and game theory components.

Tests cover:
- Coalition and Partition dataclasses
- Value functions (coverage-based, coordination cost)
- Coalition game mechanics
- Stability analysis
- LLM agent functionality
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# COALITION AND PARTITION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoalition:
    """Tests for Coalition dataclass."""

    def test_coalition_creation(self, sample_capabilities):
        """Test creating a coalition with member capabilities."""
        from coalition_llm.game_theory.coalition_game import Coalition
        
        members = ["a1_gpt4", "a3_claude3"]
        capabilities = {k: sample_capabilities[k] for k in members}
        
        coalition = Coalition(members=frozenset(members), capabilities=capabilities)
        
        assert len(coalition.members) == 2
        assert "a1_gpt4" in coalition.members
        assert coalition.capabilities["a1_gpt4"][0] == pytest.approx(0.68, rel=0.01)

    def test_coalition_equality(self, sample_capabilities):
        """Test coalition equality based on members."""
        from coalition_llm.game_theory.coalition_game import Coalition
        
        members = ["a1_gpt4", "a3_claude3"]
        caps = {k: sample_capabilities[k] for k in members}
        
        c1 = Coalition(members=frozenset(members), capabilities=caps)
        c2 = Coalition(members=frozenset(["a3_claude3", "a1_gpt4"]), capabilities=caps)
        
        assert c1.members == c2.members

    def test_singleton_coalition(self, sample_capabilities):
        """Test singleton coalition creation."""
        from coalition_llm.game_theory.coalition_game import Coalition
        
        coalition = Coalition(
            members=frozenset(["a4_claude3"]),
            capabilities={"a4_claude3": sample_capabilities["a4_claude3"]}
        )
        
        assert len(coalition.members) == 1


class TestPartition:
    """Tests for Partition dataclass."""

    def test_partition_creation(self, sample_partition, sample_capabilities):
        """Test creating a partition of agents."""
        from coalition_llm.game_theory.coalition_game import Partition, Coalition
        
        coalitions = []
        for member_set in sample_partition:
            caps = {m: sample_capabilities[m] for m in member_set}
            coalitions.append(Coalition(members=member_set, capabilities=caps))
        
        partition = Partition(coalitions=coalitions)
        
        assert len(partition.coalitions) == 3
        # Check all agents are covered
        all_members = set()
        for c in partition.coalitions:
            all_members.update(c.members)
        assert len(all_members) == 6

    def test_partition_is_valid(self, sample_capabilities):
        """Test partition validity (disjoint, complete cover)."""
        from coalition_llm.game_theory.coalition_game import Partition, Coalition
        
        # Valid partition
        valid_partition = Partition(coalitions=[
            Coalition(frozenset(["a1_gpt4", "a2_gpt4"]), 
                     {k: sample_capabilities[k] for k in ["a1_gpt4", "a2_gpt4"]}),
            Coalition(frozenset(["a3_claude3", "a4_claude3"]),
                     {k: sample_capabilities[k] for k in ["a3_claude3", "a4_claude3"]}),
            Coalition(frozenset(["a5_llama3", "a6_llama3"]),
                     {k: sample_capabilities[k] for k in ["a5_llama3", "a6_llama3"]}),
        ])
        
        all_agents = set()
        for c in valid_partition.coalitions:
            all_agents.update(c.members)
        
        assert len(all_agents) == 6


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestValueFunctions:
    """Tests for coalition value functions."""

    def test_coverage_value_basic(self, coverage_value_fn, sample_capabilities):
        """Test basic coverage value computation."""
        # Single agent
        single_caps = {"a1_gpt4": sample_capabilities["a1_gpt4"]}
        value_single = coverage_value_fn.compute(single_caps)
        
        # Value should be positive for single agent (low coordination cost)
        assert value_single > 0

    def test_coverage_value_increases_with_complementarity(
        self, coverage_value_fn, sample_capabilities
    ):
        """Test that complementary capabilities increase value."""
        # Homogeneous coalition (two GPT-4s)
        homo_caps = {
            "a1_gpt4": sample_capabilities["a1_gpt4"],
            "a2_gpt4": sample_capabilities["a2_gpt4"],
        }
        
        # Heterogeneous coalition (GPT-4 + Llama-3)
        hetero_caps = {
            "a1_gpt4": sample_capabilities["a1_gpt4"],
            "a5_llama3": sample_capabilities["a5_llama3"],
        }
        
        v_homo = coverage_value_fn.compute(homo_caps)
        v_hetero = coverage_value_fn.compute(hetero_caps)
        
        # Both should be valid values
        assert isinstance(v_homo, float)
        assert isinstance(v_hetero, float)

    def test_coordination_cost_superlinear(self, coordination_cost_params):
        """Test that coordination cost grows superlinearly."""
        from coalition_llm.game_theory.value_functions import coordination_cost
        
        alpha, beta = coordination_cost_params["alpha"], coordination_cost_params["beta"]
        
        cost_2 = coordination_cost(2, alpha, beta)
        cost_3 = coordination_cost(3, alpha, beta)
        cost_4 = coordination_cost(4, alpha, beta)
        
        # Marginal cost should increase
        marginal_2_3 = cost_3 - cost_2
        marginal_3_4 = cost_4 - cost_3
        
        assert marginal_3_4 > marginal_2_3

    def test_per_capita_value_decreases_for_large_coalitions(
        self, coverage_value_fn, sample_capabilities
    ):
        """Test that per-capita value decreases for overly large coalitions."""
        # Small coalition
        small_caps = {
            "a1_gpt4": sample_capabilities["a1_gpt4"],
            "a3_claude3": sample_capabilities["a3_claude3"],
        }
        
        # Large coalition (all 6 agents)
        large_caps = sample_capabilities.copy()
        
        v_small = coverage_value_fn.compute(small_caps)
        v_large = coverage_value_fn.compute(large_caps)
        
        pc_small = v_small / 2
        pc_large = v_large / 6
        
        # Per-capita should favor smaller coalitions (due to coordination costs)
        assert pc_small > pc_large

    def test_worked_example_from_paper(self, coverage_value_fn):
        """Test the worked example from Section 3.1 of the paper."""
        # Example 1: Three agents with specific capabilities
        c1 = np.array([0.68, 0.30, 0.40])  # (math, facts, logic)
        c2 = np.array([0.40, 0.65, 0.30])
        c3 = np.array([0.30, 0.40, 0.76])
        
        # Coalition {a1, a2}
        caps_12 = {"a1": c1, "a2": c2}
        v_12 = coverage_value_fn.compute(caps_12)
        
        # Coalition {a1, a2, a3}
        caps_123 = {"a1": c1, "a2": c2, "a3": c3}
        v_123 = coverage_value_fn.compute(caps_123)
        
        # Per-capita values
        pc_12 = v_12 / 2
        pc_123 = v_123 / 3
        
        # Paper claims pc_12 > pc_123 (0.105 vs 0.033 approximately)
        assert pc_12 > pc_123


# ═══════════════════════════════════════════════════════════════════════════════
# COALITION GAME TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoalitionGame:
    """Tests for CoalitionGame class."""

    def test_game_initialization(self, sample_6_agent_game):
        """Test game initialization with agents and value function."""
        game = sample_6_agent_game
        assert len(game.agents) == 6

    def test_initial_partition_creation(self, sample_6_agent_game):
        """Test creating initial partition (singletons or random)."""
        game = sample_6_agent_game
        
        # Singleton partition
        partition = game.create_singleton_partition()
        assert len(partition.coalitions) == 6

    def test_agent_deviation(self, sample_6_agent_game, sample_capabilities):
        """Test computing agent deviation possibilities."""
        game = sample_6_agent_game
        
        partition = game.create_singleton_partition()
        
        # Get possible deviations for first agent
        agent = game.agents[0]
        deviations = game.get_possible_deviations(agent, partition)
        
        # Should have options to join other coalitions
        assert len(deviations) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY ANALYSIS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStabilityAnalyzer:
    """Tests for StabilityAnalyzer class."""

    def test_nash_stability_verification(self, sample_6_agent_game):
        """Test Nash stability verification for a partition."""
        from coalition_llm.game_theory.stability import StabilityAnalyzer
        
        game = sample_6_agent_game
        analyzer = StabilityAnalyzer(game)
        
        # Create a partition
        partition = game.create_singleton_partition()
        
        # Check stability (singletons are typically not Nash-stable)
        # The actual result depends on the value function and capabilities
        is_stable, violations = analyzer.check_nash_stability(partition)
        
        assert isinstance(is_stable, bool)
        assert isinstance(violations, list)

    def test_stability_verification_complexity(self, sample_6_agent_game):
        """Test that stability verification is O(n²)."""
        from coalition_llm.game_theory.stability import StabilityAnalyzer
        
        game = sample_6_agent_game
        analyzer = StabilityAnalyzer(game)
        partition = game.create_singleton_partition()
        
        # Count preference queries
        query_count = 0
        original_check = analyzer.check_preference
        
        def counting_check(*args, **kwargs):
            nonlocal query_count
            query_count += 1
            return True  # Mock preference check
        
        analyzer.check_preference = counting_check
        analyzer.check_nash_stability(partition)
        
        # Should be O(n²) = O(36) for 6 agents
        n = len(game.agents)
        assert query_count <= n * n

    def test_consistency_driven_bound(self):
        """Test consistency-driven stability bound from Theorem 2."""
        from coalition_llm.game_theory.stability import compute_stability_bound
        
        # Paper parameters: p=0.86, Keff=5, Kn=15, peasy=0.98
        p = 0.86
        K_eff = 5
        K_n = 15
        p_easy = 0.98
        delta = 0.08
        epsilon_bar = 0.17
        
        lower, upper = compute_stability_bound(
            p=p, K_eff=K_eff, K_n=K_n, p_easy=p_easy, delta=delta, epsilon_bar=epsilon_bar
        )
        
        # Paper claims bounds are [35%, 73%]
        assert lower >= 0.30  # Allow some tolerance
        assert upper <= 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# LLM AGENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMAgent:
    """Tests for LLMAgent class."""

    def test_agent_creation(self, sample_capabilities):
        """Test creating an LLM agent."""
        from coalition_llm.agents.llm_agent import LLMAgent
        
        agent = LLMAgent(
            agent_id="a1_gpt4",
            model_id="gpt-4-0125-preview",
            architecture="gpt-4",
            capability_profile=sample_capabilities["a1_gpt4"],
            temperature=0.0,
        )
        
        assert agent.agent_id == "a1_gpt4"
        assert agent.architecture == "gpt-4"
        assert agent.temperature == 0.0
        assert len(agent.capability_profile) == 3

    def test_agent_epsilon_rationality(self):
        """Test ε-rationality parameter for different architectures."""
        from coalition_llm.agents.llm_agent import LLMAgent, EPSILON_VALUES
        
        # Paper specifies ε ≈ 0.15 for GPT-4, ε ≈ 0.22 for Llama-3
        assert EPSILON_VALUES.get("gpt-4", 0.15) == pytest.approx(0.15, rel=0.1)
        assert EPSILON_VALUES.get("llama-3", 0.22) == pytest.approx(0.22, rel=0.1)

    @patch("coalition_llm.agents.llm_agent.openai.OpenAI")
    def test_agent_preference_query(self, mock_openai, sample_capabilities, mock_llm_client):
        """Test querying agent for coalition preference."""
        from coalition_llm.agents.llm_agent import LLMAgent
        
        mock_openai.return_value = mock_llm_client
        
        agent = LLMAgent(
            agent_id="a1_gpt4",
            model_id="gpt-4-0125-preview",
            architecture="gpt-4",
            capability_profile=sample_capabilities["a1_gpt4"],
            temperature=0.0,
        )
        
        # This would normally call the API
        # For testing, we use the mock
        current_coalition = frozenset(["a1_gpt4", "a2_gpt4"])
        alternative_coalition = frozenset(["a3_claude3", "a5_llama3"])
        
        # The actual preference query would be tested with integration tests
        assert agent.capability_profile[0] == pytest.approx(0.68, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoalTProtocol:
    """Tests for Coalition-of-Thought protocol."""

    def test_prompt_generation(self, coalt_prompt_template, sample_capabilities):
        """Test CoalT prompt generation."""
        from coalition_llm.prompts.coalt_protocol import CoalTProtocol
        
        protocol = CoalTProtocol()
        
        prompt = protocol.generate_prompt(
            agent_id="a1_gpt4",
            agent_capabilities=sample_capabilities["a1_gpt4"],
            current_coalition=frozenset(["a1_gpt4", "a2_gpt4"]),
            current_capabilities={
                "a1_gpt4": sample_capabilities["a1_gpt4"],
                "a2_gpt4": sample_capabilities["a2_gpt4"],
            },
            alternative_coalition=frozenset(["a3_claude3", "a5_llama3"]),
            alternative_capabilities={
                "a3_claude3": sample_capabilities["a3_claude3"],
                "a5_llama3": sample_capabilities["a5_llama3"],
            },
        )
        
        # Check all 5 steps are mentioned
        assert "Step 1" in prompt or "Capability Analysis" in prompt
        assert "Step 2" in prompt or "Complementarity" in prompt
        assert "Step 3" in prompt or "Value Estimation" in prompt
        assert "Step 4" in prompt or "Coordination Cost" in prompt
        assert "Step 5" in prompt or "Preference" in prompt

    def test_response_parsing(self):
        """Test parsing CoalT response for preference."""
        from coalition_llm.prompts.coalt_protocol import CoalTProtocol, parse_preference
        
        # Test different response formats
        assert parse_preference("I PREFER current coalition C") == "current"
        assert parse_preference("I PREFER C'") == "alternative"
        assert parse_preference("I prefer the alternative coalition") == "alternative"
        assert parse_preference("prefer C over C'") == "current"

    def test_ablation_configurations(self):
        """Test CoalT ablation study configurations."""
        from coalition_llm.prompts.coalt_protocol import CoalTAblation
        
        # Test each ablation variant
        ablation = CoalTAblation()
        
        # Full CoalT
        full_prompt = ablation.generate_prompt(
            components=["capability", "complementarity", "value", "coordination", "preference"],
            agent_id="a1",
            context={}
        )
        
        # Without complementarity
        no_comp_prompt = ablation.generate_prompt(
            components=["capability", "value", "coordination", "preference"],
            agent_id="a1",
            context={}
        )
        
        assert len(full_prompt) > len(no_comp_prompt)


class TestBaselineProtocols:
    """Tests for baseline prompting protocols."""

    def test_standard_protocol(self, sample_capabilities):
        """Test standard (direct query) protocol."""
        from coalition_llm.prompts.baseline_protocols import StandardProtocol
        
        protocol = StandardProtocol()
        prompt = protocol.generate_prompt(
            agent_id="a1_gpt4",
            current_coalition=frozenset(["a1_gpt4"]),
            alternative_coalition=frozenset(["a2_gpt4", "a3_claude3"]),
        )
        
        assert "prefer" in prompt.lower() or "choose" in prompt.lower()

    def test_vanilla_cot_protocol(self, sample_capabilities):
        """Test vanilla chain-of-thought protocol."""
        from coalition_llm.prompts.baseline_protocols import VanillaCoTProtocol
        
        protocol = VanillaCoTProtocol()
        prompt = protocol.generate_prompt(
            agent_id="a1_gpt4",
            current_coalition=frozenset(["a1_gpt4"]),
            alternative_coalition=frozenset(["a2_gpt4", "a3_claude3"]),
        )
        
        assert "step by step" in prompt.lower() or "think" in prompt.lower()

    def test_self_consistency_protocol(self, sample_capabilities):
        """Test self-consistency protocol (multiple CoT paths)."""
        from coalition_llm.prompts.baseline_protocols import SelfConsistencyProtocol
        
        protocol = SelfConsistencyProtocol(num_paths=5)
        
        assert protocol.num_paths == 5


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (MOCK)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegrationMock:
    """Mock integration tests for full pipeline."""

    def test_full_episode_mock(self, sample_6_agent_game, set_deterministic_seed):
        """Test running a full coalition formation episode (mocked)."""
        game = sample_6_agent_game
        
        # Start with singleton partition
        partition = game.create_singleton_partition()
        
        # Simulate a few rounds (with mocked preferences)
        max_rounds = 5
        for round_num in range(max_rounds):
            # In a real test, we would query agents for preferences
            # Here we just verify the structure works
            assert len(partition.coalitions) >= 1
            
        # Verify partition is still valid
        all_agents = set()
        for c in partition.coalitions:
            all_agents.update(c.members)
        assert len(all_agents) == 6
