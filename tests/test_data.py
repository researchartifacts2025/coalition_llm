"""
Unit tests for Coalition LLM data pipeline.

Tests cover:
- Dataset loading and preprocessing
- Question generation and formatting
- Data splits and reproducibility
- DataLoader functionality
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoalitionQADataset:
    """Tests for CoalitionQADataset class."""

    def test_dataset_creation(self, sample_questions):
        """Test creating a CoalitionQA dataset."""
        from coalition_llm.data.dataset import CoalitionQADataset
        
        dataset = CoalitionQADataset(questions=sample_questions)
        
        assert len(dataset) == 3
        assert dataset[0]["id"] == "q1"

    def test_dataset_getitem(self, sample_questions):
        """Test dataset indexing."""
        from coalition_llm.data.dataset import CoalitionQADataset
        
        dataset = CoalitionQADataset(questions=sample_questions)
        
        item = dataset[0]
        assert "question" in item
        assert "answer" in item
        assert "difficulty" in item

    def test_dataset_difficulty_scores(self, sample_questions):
        """Test that difficulty scores are properly formatted."""
        from coalition_llm.data.dataset import CoalitionQADataset
        
        dataset = CoalitionQADataset(questions=sample_questions)
        
        for i in range(len(dataset)):
            item = dataset[i]
            difficulty = item["difficulty"]
            
            assert "math" in difficulty
            assert "facts" in difficulty
            assert "logic" in difficulty
            assert all(0 <= v <= 1 for v in difficulty.values())

    def test_dataset_categories(self, sample_questions):
        """Test dataset category distribution."""
        from coalition_llm.data.dataset import CoalitionQADataset
        
        dataset = CoalitionQADataset(questions=sample_questions)
        
        categories = [dataset[i]["category"] for i in range(len(dataset))]
        assert "math" in categories
        assert "facts" in categories
        assert "logic" in categories

    def test_dataset_from_file(self, sample_questions):
        """Test loading dataset from JSON file."""
        from coalition_llm.data.dataset import CoalitionQADataset
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_questions, f)
            temp_path = f.name
        
        try:
            dataset = CoalitionQADataset.from_file(temp_path)
            assert len(dataset) == 3
        finally:
            Path(temp_path).unlink()

    def test_dataset_synthetic_generation(self):
        """Test synthetic question generation."""
        from coalition_llm.data.dataset import CoalitionQADataset
        
        dataset = CoalitionQADataset.generate_synthetic(
            n_questions=10,
            seed=42,
            categories=["math", "facts", "logic"],
        )
        
        assert len(dataset) == 10
        
        # Check difficulty distribution
        math_diffs = [dataset[i]["difficulty"]["math"] for i in range(len(dataset))]
        assert min(math_diffs) >= 0
        assert max(math_diffs) <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuestion:
    """Tests for Question dataclass."""

    def test_question_creation(self):
        """Test creating a Question object."""
        from coalition_llm.data.dataset import Question
        
        q = Question(
            id="test_q1",
            question="What is 2+2?",
            answer="4",
            difficulty={"math": 0.1, "facts": 0.0, "logic": 0.1},
            category="math",
        )
        
        assert q.id == "test_q1"
        assert q.answer == "4"

    def test_question_to_dict(self):
        """Test converting Question to dictionary."""
        from coalition_llm.data.dataset import Question
        
        q = Question(
            id="test_q1",
            question="What is the capital of France?",
            answer="Paris",
            difficulty={"math": 0.0, "facts": 0.3, "logic": 0.1},
            category="facts",
        )
        
        d = q.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "test_q1"
        assert d["answer"] == "Paris"

    def test_question_from_dict(self):
        """Test creating Question from dictionary."""
        from coalition_llm.data.dataset import Question
        
        data = {
            "id": "test_q1",
            "question": "Test question?",
            "answer": "Test answer",
            "difficulty": {"math": 0.5, "facts": 0.5, "logic": 0.5},
            "category": "math",
        }
        
        q = Question.from_dict(data)
        assert q.id == "test_q1"
        assert q.category == "math"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA SPLITS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataSplits:
    """Tests for data splitting functionality."""

    def test_stratified_split(self, sample_questions):
        """Test stratified train/val/test split."""
        from coalition_llm.data.dataset import create_stratified_split
        
        # Generate more questions for splitting
        questions = sample_questions * 10  # 30 questions
        
        train, val, test = create_stratified_split(
            questions=questions,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        
        total = len(train) + len(val) + len(test)
        assert total == len(questions)
        
        # Check approximate ratios
        assert len(train) / total >= 0.65
        assert len(test) / total >= 0.10

    def test_split_reproducibility(self, sample_questions):
        """Test that splits are reproducible with same seed."""
        from coalition_llm.data.dataset import create_stratified_split
        
        questions = sample_questions * 10
        
        train1, val1, test1 = create_stratified_split(questions, seed=42)
        train2, val2, test2 = create_stratified_split(questions, seed=42)
        
        assert [q["id"] for q in train1] == [q["id"] for q in train2]
        assert [q["id"] for q in val1] == [q["id"] for q in val2]
        assert [q["id"] for q in test1] == [q["id"] for q in test2]

    def test_split_no_overlap(self, sample_questions):
        """Test that splits have no overlapping questions."""
        from coalition_llm.data.dataset import create_stratified_split
        
        questions = sample_questions * 10
        train, val, test = create_stratified_split(questions, seed=42)
        
        train_ids = {q["id"] for q in train}
        val_ids = {q["id"] for q in val}
        test_ids = {q["id"] for q in test}
        
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# DATALOADER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoader:
    """Tests for DataLoader functionality."""

    def test_dataloader_creation(self, sample_questions):
        """Test creating a DataLoader."""
        from coalition_llm.data.dataset import CoalitionQADataset, create_dataloader
        
        dataset = CoalitionQADataset(questions=sample_questions)
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
        )
        
        batch = next(iter(dataloader))
        assert len(batch["question"]) == 2

    def test_dataloader_worker_seeding(self, sample_questions):
        """Test that DataLoader workers are properly seeded."""
        from coalition_llm.data.dataset import CoalitionQADataset, create_dataloader
        
        dataset = CoalitionQADataset(questions=sample_questions * 10)
        
        # Create two dataloaders with same seed
        dl1 = create_dataloader(dataset, batch_size=2, shuffle=True, seed=42, num_workers=0)
        dl2 = create_dataloader(dataset, batch_size=2, shuffle=True, seed=42, num_workers=0)
        
        # First batches should be identical
        batch1 = next(iter(dl1))
        batch2 = next(iter(dl2))
        
        assert batch1["question"] == batch2["question"]

    def test_dataloader_collate_fn(self, sample_questions):
        """Test custom collate function for coalition data."""
        from coalition_llm.data.dataset import CoalitionQADataset, collate_coalition_batch
        
        dataset = CoalitionQADataset(questions=sample_questions)
        
        # Manually collate a batch
        batch = [dataset[i] for i in range(len(dataset))]
        collated = collate_coalition_batch(batch)
        
        assert "question" in collated
        assert "difficulty" in collated
        assert len(collated["question"]) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK DATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkData:
    """Tests for benchmark data loading (MATH, MMLU, LogiQA)."""

    def test_math_format(self):
        """Test MATH dataset question format."""
        from coalition_llm.data.dataset import format_math_question
        
        raw_question = {
            "problem": "Find the derivative of f(x) = x^3",
            "solution": "f'(x) = 3x^2",
            "level": "Level 3",
            "type": "Calculus",
        }
        
        formatted = format_math_question(raw_question)
        
        assert "question" in formatted
        assert "answer" in formatted
        assert "difficulty" in formatted

    def test_mmlu_format(self):
        """Test MMLU dataset question format."""
        from coalition_llm.data.dataset import format_mmlu_question
        
        raw_question = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,  # Paris
            "subject": "geography",
        }
        
        formatted = format_mmlu_question(raw_question)
        
        assert "question" in formatted
        assert "Paris" in formatted["answer"]

    def test_logiqa_format(self):
        """Test LogiQA dataset question format."""
        from coalition_llm.data.dataset import format_logiqa_question
        
        raw_question = {
            "context": "All cats are mammals. All mammals are animals.",
            "question": "Are all cats animals?",
            "options": ["Yes", "No", "Cannot determine", "Sometimes"],
            "answer": "A",
        }
        
        formatted = format_logiqa_question(raw_question)
        
        assert "question" in formatted
        assert "difficulty" in formatted
        assert formatted["difficulty"]["logic"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFICULTY ESTIMATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifficultyEstimation:
    """Tests for question difficulty estimation."""

    def test_difficulty_from_keywords(self):
        """Test difficulty estimation from question keywords."""
        from coalition_llm.data.dataset import estimate_difficulty_from_text
        
        math_q = "Calculate the integral of x^2 dx"
        facts_q = "What year was the Declaration of Independence signed?"
        logic_q = "If P implies Q, and Q implies R, does P imply R?"
        
        math_diff = estimate_difficulty_from_text(math_q)
        facts_diff = estimate_difficulty_from_text(facts_q)
        logic_diff = estimate_difficulty_from_text(logic_q)
        
        assert math_diff["math"] > math_diff["facts"]
        assert facts_diff["facts"] > 0
        assert logic_diff["logic"] > logic_diff["math"]

    def test_difficulty_normalization(self):
        """Test that difficulty scores are normalized."""
        from coalition_llm.data.dataset import normalize_difficulty
        
        raw_diff = {"math": 2.5, "facts": 1.0, "logic": 3.5}
        normalized = normalize_difficulty(raw_diff)
        
        assert all(0 <= v <= 1 for v in normalized.values())


# ═══════════════════════════════════════════════════════════════════════════════
# DATA VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataValidation:
    """Tests for data validation utilities."""

    def test_validate_question_format(self):
        """Test question format validation."""
        from coalition_llm.data.dataset import validate_question
        
        valid_q = {
            "id": "q1",
            "question": "Test?",
            "answer": "Answer",
            "difficulty": {"math": 0.5, "facts": 0.5, "logic": 0.5},
            "category": "math",
        }
        
        invalid_q = {
            "id": "q1",
            "question": "Test?",
            # Missing answer
        }
        
        assert validate_question(valid_q) is True
        assert validate_question(invalid_q) is False

    def test_validate_difficulty_range(self):
        """Test difficulty value range validation."""
        from coalition_llm.data.dataset import validate_difficulty
        
        valid_diff = {"math": 0.5, "facts": 0.3, "logic": 0.8}
        invalid_diff = {"math": 1.5, "facts": -0.1, "logic": 0.5}
        
        assert validate_difficulty(valid_diff) is True
        assert validate_difficulty(invalid_diff) is False

    def test_dataset_statistics(self, sample_questions):
        """Test computing dataset statistics."""
        from coalition_llm.data.dataset import CoalitionQADataset
        
        dataset = CoalitionQADataset(questions=sample_questions * 10)
        stats = dataset.compute_statistics()
        
        assert "num_questions" in stats
        assert "category_distribution" in stats
        assert "difficulty_stats" in stats
        assert stats["num_questions"] == 30
