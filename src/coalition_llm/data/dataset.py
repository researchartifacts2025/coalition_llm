"""
Dataset for coalition formation tasks.

Implements the task domain:
- Collaborative question-answering requiring diverse expertise
- Mathematical reasoning, factual knowledge, logical analysis
- Ground-truth difficulty scores for coalition value computation

Benchmarks used:
- MATH (mathematical reasoning)
- MMLU (knowledge subset for factual knowledge)
- LogiQA (logical reasoning)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QuestionDomain(Enum):
    """Question domain categories."""
    MATH = "math"
    FACTS = "facts"
    LOGIC = "logic"


class Difficulty(Enum):
    """Question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Question:
    """
    A question in the coalition QA task.
    
    Attributes:
        id: Unique question identifier
        text: Question text
        answer: Correct answer
        domain: Primary domain (math, facts, logic)
        difficulty: Difficulty level
        capability_requirements: Required capabilities [math, facts, logic]
    """
    id: str
    text: str
    answer: str
    domain: QuestionDomain
    difficulty: Difficulty
    capability_requirements: np.ndarray
    
    def __post_init__(self):
        """Ensure capability_requirements is numpy array."""
        self.capability_requirements = np.array(self.capability_requirements)


class CoalitionQADataset:
    """
    Dataset for coalition formation question-answering tasks.
        
    Attributes:
        questions: List of Question objects
        domains: List of domain names
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        n_questions: int = 200,
        seed: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing benchmark data
            n_questions: Number of questions to use
            seed: Random seed for question selection
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/processed")
        self.n_questions = n_questions
        self.rng = np.random.default_rng(seed)
        
        self.domains = ["math", "facts", "logic"]
        self.questions: List[Question] = []
        
        logger.info(f"Initialized CoalitionQADataset with {n_questions} questions")
    
    def load(self) -> None:
        """Load questions from data files."""
        # Try to load from processed data
        processed_file = self.data_dir / "coalition_qa.json"
        
        if processed_file.exists():
            self._load_from_file(processed_file)
        else:
            # Generate synthetic questions for demo/testing
            logger.warning(
                f"Processed data not found at {processed_file}. "
                "Generating synthetic questions."
            )
            self._generate_synthetic()
    
    def _load_from_file(self, path: Path) -> None:
        """Load questions from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        for q_data in data["questions"]:
            question = Question(
                id=q_data["id"],
                text=q_data["text"],
                answer=q_data["answer"],
                domain=QuestionDomain(q_data["domain"]),
                difficulty=Difficulty(q_data["difficulty"]),
                capability_requirements=q_data["capability_requirements"],
            )
            self.questions.append(question)
        
        logger.info(f"Loaded {len(self.questions)} questions from {path}")
    
    def _generate_synthetic(self) -> None:
        """Generate synthetic questions for testing."""
        n_per_domain = self.n_questions // 3
        n_per_difficulty = n_per_domain // 3
        
        idx = 0
        for domain in QuestionDomain:
            for difficulty in Difficulty:
                for _ in range(n_per_difficulty):
                    # Generate capability requirements based on domain
                    cap_req = self._generate_capability_requirements(
                        domain, difficulty
                    )
                    
                    question = Question(
                        id=f"q_{idx:04d}",
                        text=f"[{domain.value}/{difficulty.value}] Sample question {idx}",
                        answer=f"Answer {idx}",
                        domain=domain,
                        difficulty=difficulty,
                        capability_requirements=cap_req,
                    )
                    self.questions.append(question)
                    idx += 1
        
        logger.info(f"Generated {len(self.questions)} synthetic questions")
    
    def _generate_capability_requirements(
        self,
        domain: QuestionDomain,
        difficulty: Difficulty,
    ) -> np.ndarray:
        """Generate capability requirements based on domain and difficulty."""
        # Base requirements by domain
        domain_weights = {
            QuestionDomain.MATH: [0.7, 0.1, 0.2],
            QuestionDomain.FACTS: [0.1, 0.7, 0.2],
            QuestionDomain.LOGIC: [0.2, 0.1, 0.7],
        }
        
        # Scale by difficulty
        difficulty_scale = {
            Difficulty.EASY: 0.4,
            Difficulty.MEDIUM: 0.6,
            Difficulty.HARD: 0.8,
        }
        
        base = np.array(domain_weights[domain])
        scale = difficulty_scale[difficulty]
        
        # Add noise
        noise = self.rng.uniform(-0.1, 0.1, size=3)
        requirements = np.clip(base * scale + noise, 0, 1)
        
        return requirements
    
    def get_batch(
        self,
        batch_size: int,
        domain: Optional[QuestionDomain] = None,
        difficulty: Optional[Difficulty] = None,
    ) -> List[Question]:
        """
        Get a batch of questions.
        
        Args:
            batch_size: Number of questions
            domain: Filter by domain (optional)
            difficulty: Filter by difficulty (optional)
        
        Returns:
            List of Question objects
        """
        candidates = self.questions
        
        if domain is not None:
            candidates = [q for q in candidates if q.domain == domain]
        
        if difficulty is not None:
            candidates = [q for q in candidates if q.difficulty == difficulty]
        
        if len(candidates) < batch_size:
            logger.warning(
                f"Only {len(candidates)} questions available, "
                f"requested {batch_size}"
            )
            batch_size = len(candidates)
        
        indices = self.rng.choice(len(candidates), size=batch_size, replace=False)
        return [candidates[i] for i in indices]
    
    def compute_coalition_value_for_question(
        self,
        question: Question,
        coalition_capabilities: np.ndarray,
    ) -> float:
        """
        Compute coalition value for answering a question.
        
        Value is based on how well coalition capabilities match
        question requirements.
        
        Args:
            question: The question to answer
            coalition_capabilities: Aggregated coalition capabilities [d,]
        
        Returns:
            Value in [0, 1]
        """
        # Compute match between capabilities and requirements
        # Using cosine similarity
        req = question.capability_requirements
        cap = coalition_capabilities
        
        dot_product = np.dot(req, cap)
        norm_product = np.linalg.norm(req) * np.linalg.norm(cap)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        
        # Scale by requirement magnitude (harder questions need more capability)
        requirement_magnitude = np.linalg.norm(req)
        
        return float(similarity * requirement_magnitude)
    
    def __len__(self) -> int:
        """Number of questions in dataset."""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Question:
        """Get question by index."""
        return self.questions[idx]
    
    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "n_questions": len(self.questions),
            "domains": self.domains,
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "answer": q.answer,
                    "domain": q.domain.value,
                    "difficulty": q.difficulty.value,
                    "capability_requirements": q.capability_requirements.tolist(),
                }
                for q in self.questions
            ],
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.questions)} questions to {path}")


def download_benchmarks(
    output_dir: Path,
    benchmarks: List[str] = ["math", "mmlu", "logiqa"],
) -> Dict[str, Path]:
    """
    Download benchmark datasets.
    
    Note: This is a placeholder. In practice, you would download from:
    - MATH: https://github.com/hendrycks/math
    - MMLU: https://github.com/hendrycks/test
    - LogiQA: https://github.com/lgw863/LogiQA-dataset
    
    Args:
        output_dir: Directory to save downloaded data
        benchmarks: List of benchmarks to download
    
    Returns:
        Dict mapping benchmark name to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    for benchmark in benchmarks:
        benchmark_dir = output_dir / benchmark
        benchmark_dir.mkdir(exist_ok=True)
        
        # Create placeholder file
        placeholder = benchmark_dir / "README.md"
        placeholder.write_text(
            f"# {benchmark.upper()} Benchmark\n\n"
            f"Download the actual data from the official source.\n"
        )
        
        paths[benchmark] = benchmark_dir
        logger.info(f"Created placeholder for {benchmark} at {benchmark_dir}")
    
    return paths
