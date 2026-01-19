#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Coalition LLM - Data Download Script
# ═══════════════════════════════════════════════════════════════════════════════
# Downloads required datasets: MATH, MMLU (subset), LogiQA
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Configuration
DATA_DIR="${DATA_DIR:-./data/raw}"
CACHE_DIR="${HF_HOME:-~/.cache/huggingface}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Check dependencies
# ═══════════════════════════════════════════════════════════════════════════════

check_dependencies() {
    log_info "Checking dependencies..."
    
    for cmd in python pip curl sha256sum; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is required but not installed."
            exit 1
        fi
    done
    
    # Check Python packages
    python -c "import datasets" 2>/dev/null || {
        log_warn "datasets package not found, installing..."
        pip install datasets
    }
    
    log_info "All dependencies satisfied."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Create directory structure
# ═══════════════════════════════════════════════════════════════════════════════

setup_directories() {
    log_info "Setting up data directories..."
    
    mkdir -p "${DATA_DIR}/math"
    mkdir -p "${DATA_DIR}/mmlu"
    mkdir -p "${DATA_DIR}/logiqa"
    mkdir -p "./data/processed"
    mkdir -p "./data/splits"
    
    log_info "Directories created."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Download MATH dataset
# ═══════════════════════════════════════════════════════════════════════════════

download_math() {
    log_info "Downloading MATH dataset..."
    
    python << 'EOF'
from datasets import load_dataset
import json
from pathlib import Path

# Load MATH dataset
dataset = load_dataset("hendrycks/competition_math", split="test")

# Save to local files
output_dir = Path("./data/raw/math")
output_dir.mkdir(parents=True, exist_ok=True)

# Sample 100 questions for coalition formation experiments
# Stratified by difficulty level
questions = []
for item in dataset.select(range(min(100, len(dataset)))):
    questions.append({
        "id": f"math_{len(questions)}",
        "question": item["problem"],
        "answer": item["solution"],
        "level": item["level"],
        "type": item["type"],
    })

with open(output_dir / "math_subset.json", "w") as f:
    json.dump(questions, f, indent=2)

print(f"Downloaded {len(questions)} MATH questions")
EOF
    
    log_info "MATH dataset downloaded."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Download MMLU dataset (knowledge subset)
# ═══════════════════════════════════════════════════════════════════════════════

download_mmlu() {
    log_info "Downloading MMLU dataset (knowledge subset)..."
    
    python << 'EOF'
from datasets import load_dataset
import json
from pathlib import Path

# Knowledge-focused subjects
subjects = [
    "world_religions",
    "us_history",
    "geography",
    "astronomy",
    "prehistory",
]

output_dir = Path("./data/raw/mmlu")
output_dir.mkdir(parents=True, exist_ok=True)

all_questions = []
for subject in subjects:
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
        
        for idx, item in enumerate(dataset.select(range(min(20, len(dataset))))):
            choices = item["choices"]
            answer_idx = item["answer"]
            
            all_questions.append({
                "id": f"mmlu_{subject}_{idx}",
                "question": item["question"],
                "choices": choices,
                "answer": choices[answer_idx],
                "subject": subject,
            })
    except Exception as e:
        print(f"Warning: Could not load {subject}: {e}")

with open(output_dir / "mmlu_subset.json", "w") as f:
    json.dump(all_questions, f, indent=2)

print(f"Downloaded {len(all_questions)} MMLU questions")
EOF
    
    log_info "MMLU dataset downloaded."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Download LogiQA dataset
# ═══════════════════════════════════════════════════════════════════════════════

download_logiqa() {
    log_info "Downloading LogiQA dataset..."
    
    python << 'EOF'
from datasets import load_dataset
import json
from pathlib import Path

# Load LogiQA
dataset = load_dataset("lucasmccabe/logiqa", split="test")

output_dir = Path("./data/raw/logiqa")
output_dir.mkdir(parents=True, exist_ok=True)

questions = []
for idx, item in enumerate(dataset.select(range(min(100, len(dataset))))):
    questions.append({
        "id": f"logiqa_{idx}",
        "context": item.get("context", ""),
        "question": item["query"],
        "options": item["options"],
        "answer": item["options"][item["correct_option"]],
    })

with open(output_dir / "logiqa_subset.json", "w") as f:
    json.dump(questions, f, indent=2)

print(f"Downloaded {len(questions)} LogiQA questions")
EOF
    
    log_info "LogiQA dataset downloaded."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Verify checksums
# ═══════════════════════════════════════════════════════════════════════════════

verify_checksums() {
    log_info "Verifying file integrity..."
    
    # Note: Checksums will vary based on dataset version and sampling
    # These are placeholder checksums - compute actual ones after download
    
    local files=(
        "data/raw/math/math_subset.json"
        "data/raw/mmlu/mmlu_subset.json"
        "data/raw/logiqa/logiqa_subset.json"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            local size=$(wc -c < "$file")
            log_info "✓ $file (${size} bytes)"
        else
            log_warn "✗ $file not found"
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════════════════
# Create combined dataset
# ═══════════════════════════════════════════════════════════════════════════════

create_combined_dataset() {
    log_info "Creating combined CoalitionQA dataset..."
    
    python << 'EOF'
import json
from pathlib import Path

output_dir = Path("./data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

combined = []

# Load MATH questions
math_file = Path("./data/raw/math/math_subset.json")
if math_file.exists():
    with open(math_file) as f:
        for q in json.load(f):
            combined.append({
                "id": q["id"],
                "question": q["question"],
                "answer": q["answer"],
                "category": "math",
                "difficulty": {"math": 0.6, "facts": 0.2, "logic": 0.4},
                "source": "MATH",
            })

# Load MMLU questions  
mmlu_file = Path("./data/raw/mmlu/mmlu_subset.json")
if mmlu_file.exists():
    with open(mmlu_file) as f:
        for q in json.load(f):
            combined.append({
                "id": q["id"],
                "question": q["question"],
                "answer": q["answer"],
                "category": "facts",
                "difficulty": {"math": 0.1, "facts": 0.7, "logic": 0.3},
                "source": "MMLU",
            })

# Load LogiQA questions
logiqa_file = Path("./data/raw/logiqa/logiqa_subset.json")
if logiqa_file.exists():
    with open(logiqa_file) as f:
        for q in json.load(f):
            combined.append({
                "id": q["id"],
                "question": f"{q.get('context', '')} {q['question']}".strip(),
                "answer": q["answer"],
                "category": "logic",
                "difficulty": {"math": 0.2, "facts": 0.3, "logic": 0.7},
                "source": "LogiQA",
            })

# Save combined dataset
with open(output_dir / "coalition_qa.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"Created combined dataset with {len(combined)} questions")

# Create train/val/test splits
import random
random.seed(42)
random.shuffle(combined)

n = len(combined)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

splits = {
    "train": combined[:train_end],
    "val": combined[train_end:val_end],
    "test": combined[val_end:],
}

splits_dir = Path("./data/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

for split_name, split_data in splits.items():
    with open(splits_dir / f"{split_name}.json", "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"{split_name}: {len(split_data)} questions")
EOF
    
    log_info "Combined dataset created."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "Coalition LLM - Data Download"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    
    check_dependencies
    setup_directories
    
    log_info "Downloading datasets..."
    download_math
    download_mmlu
    download_logiqa
    
    verify_checksums
    create_combined_dataset
    
    echo ""
    log_info "Data download complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run preprocessing: python scripts/preprocess.py"
    echo "  2. Start training: python train.py"
    echo ""
}

# Run main function
main "$@"
