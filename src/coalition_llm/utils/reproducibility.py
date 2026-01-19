"""
Reproducibility utilities for deterministic experiments.

Ensures full reproducibility by setting seeds for:
- Python's random module
- NumPy's random number generator
- PyTorch (CPU and CUDA)
- CUDA deterministic algorithms
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import random
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def set_seed(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic CUDA operations (slower but reproducible)
        warn_only: Only warn if deterministic ops can't be enabled
    
    Example:
        >>> set_seed(42, deterministic=True)
        >>> # All subsequent random operations are now reproducible
    """
    # Python's random module
    random.seed(seed)
    
    # Hash-based randomization (for dict ordering, etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy not available; skipping NumPy seed")
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        
        # CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            if deterministic:
                # Enable deterministic algorithms
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
                
                # cuDNN deterministic mode
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Required for some CUDA operations to be deterministic
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have the same deterministic controls
            logger.info("MPS backend detected; deterministic mode limited")
    
    except ImportError:
        logger.warning("PyTorch not available; skipping PyTorch seed")
    
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Gather environment information for reproducibility reporting.
    
    Returns:
        Dict with system, Python, and library version information
    """
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
    }
    
    # NumPy
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        info["numpy_version"] = "not installed"
    
    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
    
    # OpenAI
    try:
        import openai
        info["openai_version"] = openai.__version__
    except ImportError:
        info["openai_version"] = "not installed"
    
    # Anthropic
    try:
        import anthropic
        info["anthropic_version"] = anthropic.__version__
    except ImportError:
        info["anthropic_version"] = "not installed"
    
    # Together
    try:
        import together
        info["together_version"] = together.__version__
    except ImportError:
        info["together_version"] = "not installed"
    
    return info


def print_reproducibility_info() -> None:
    """Print reproducibility information to logger."""
    info = get_reproducibility_info()
    
    logger.info("=" * 60)
    logger.info("REPRODUCIBILITY INFORMATION")
    logger.info("=" * 60)
    
    for key, value in info.items():
        if isinstance(value, list):
            logger.info(f"  {key}:")
            for item in value:
                logger.info(f"    - {item}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of configuration for experiment tracking.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        SHA256 hash of sorted config string
    """
    import json
    
    # Sort keys for deterministic hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


class ReproducibilityContext:
    """
    Context manager for reproducible code blocks.
    
    Example:
        >>> with ReproducibilityContext(seed=42):
        ...     # All random operations here are reproducible
        ...     result = run_experiment()
    """
    
    def __init__(
        self,
        seed: int,
        deterministic: bool = True,
    ):
        """
        Initialize reproducibility context.
        
        Args:
            seed: Random seed
            deterministic: Enable deterministic mode
        """
        self.seed = seed
        self.deterministic = deterministic
        self._saved_states: Dict[str, Any] = {}
    
    def __enter__(self) -> "ReproducibilityContext":
        """Save current states and set new seed."""
        # Save Python random state
        self._saved_states["python"] = random.getstate()
        
        # Save NumPy state
        try:
            import numpy as np
            self._saved_states["numpy"] = np.random.get_state()
        except ImportError:
            pass
        
        # Save PyTorch state
        try:
            import torch
            self._saved_states["torch_cpu"] = torch.get_rng_state()
            if torch.cuda.is_available():
                self._saved_states["torch_cuda"] = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        
        # Set new seed
        set_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore previous states."""
        # Restore Python random state
        if "python" in self._saved_states:
            random.setstate(self._saved_states["python"])
        
        # Restore NumPy state
        if "numpy" in self._saved_states:
            import numpy as np
            np.random.set_state(self._saved_states["numpy"])
        
        # Restore PyTorch state
        if "torch_cpu" in self._saved_states:
            import torch
            torch.set_rng_state(self._saved_states["torch_cpu"])
            if "torch_cuda" in self._saved_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self._saved_states["torch_cuda"])


def verify_reproducibility(seed: int = 42, n_samples: int = 100) -> bool:
    """
    Verify that random operations are reproducible.
    
    Args:
        seed: Seed to test with
        n_samples: Number of samples to generate
    
    Returns:
        True if all tests pass
    """
    import numpy as np
    
    # Generate samples twice with same seed
    set_seed(seed)
    samples1_py = [random.random() for _ in range(n_samples)]
    samples1_np = np.random.random(n_samples).tolist()
    
    set_seed(seed)
    samples2_py = [random.random() for _ in range(n_samples)]
    samples2_np = np.random.random(n_samples).tolist()
    
    # Verify they match
    py_match = samples1_py == samples2_py
    np_match = samples1_np == samples2_np
    
    if not py_match:
        logger.error("Python random reproducibility FAILED")
    if not np_match:
        logger.error("NumPy random reproducibility FAILED")
    
    # PyTorch if available
    try:
        import torch
        set_seed(seed)
        samples1_torch = torch.rand(n_samples).tolist()
        set_seed(seed)
        samples2_torch = torch.rand(n_samples).tolist()
        torch_match = samples1_torch == samples2_torch
        
        if not torch_match:
            logger.error("PyTorch random reproducibility FAILED")
        
        return py_match and np_match and torch_match
    except ImportError:
        return py_match and np_match


if __name__ == "__main__":
    # Test reproducibility
    logging.basicConfig(level=logging.INFO)
    
    print("Testing reproducibility...")
    if verify_reproducibility():
        print("✓ All reproducibility tests passed")
    else:
        print("✗ Some reproducibility tests failed")
    
    print("\nReproducibility info:")
    print_reproducibility_info()
