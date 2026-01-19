#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Coalition Formation in LLM Agent Networks
# ═══════════════════════════════════════════════════════════════════════════════
#
# Requirements:
# - 8x NVIDIA A100 40GB GPUs (or equivalent)
# - API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY
# - ~8 hours runtime, ~$2,400 API cost
#
# Usage:
#   chmod +x scripts/reproduce_all.sh
#   ./scripts/reproduce_all.sh
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/reproduce_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/reproduce.log"

# Seeds used in paper
SEEDS=(0 42 123 456 1024)

# Number of episodes per condition
NUM_EPISODES=400

# Protocols to evaluate
PROTOCOLS=(random greedy standard vanilla_cot self_consistency coalt)

# Ablation configurations (Table 4)
ABLATIONS=(
    "coalt_no_capability"
    "coalt_no_complementarity"
    "coalt_no_value"
    "coalt_no_cost"
)

# Temperature settings (Table 5)
TEMPERATURES=(0.0 0.5 1.0)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

log_section() {
    local title="$1"
    log ""
    log "═══════════════════════════════════════════════════════════════════════════"
    log "${BLUE}${title}${NC}"
    log "═══════════════════════════════════════════════════════════════════════════"
}

check_env() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        log "${RED}ERROR: ${var_name} not set${NC}"
        return 1
    fi
    log "${GREEN}✓ ${var_name} is set${NC}"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log "${YELLOW}WARNING: nvidia-smi not found, GPU may not be available${NC}"
        return 0
    fi
    
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "Found ${gpu_count} GPU(s)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        log "  GPU: ${line}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight Checks
# ─────────────────────────────────────────────────────────────────────────────

preflight_checks() {
    log_section "Pre-flight Checks"
    
    # Check environment variables
    log "Checking API keys..."
    check_env "OPENAI_API_KEY"
    check_env "ANTHROPIC_API_KEY"
    check_env "TOGETHER_API_KEY"
    
    # Check GPU
    log ""
    log "Checking GPU availability..."
    check_gpu
    
    # Check Python environment
    log ""
    log "Checking Python environment..."
    python -c "import coalition_llm; print('✓ coalition_llm package installed')"
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
    python -c "import hydra; print(f'✓ Hydra {hydra.__version__}')"
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    log ""
    log "Output directory: ${OUTPUT_DIR}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Main Results
# ─────────────────────────────────────────────────────────────────────────────

run_table3() {
    log_section "Table 3: Main Results (400 episodes × 6 protocols × 5 seeds)"
    
    local table3_dir="${OUTPUT_DIR}/table3"
    mkdir -p "${table3_dir}"
    
    for protocol in "${PROTOCOLS[@]}"; do
        log ""
        log "${YELLOW}Protocol: ${protocol}${NC}"
        
        for seed in "${SEEDS[@]}"; do
            local run_dir="${table3_dir}/${protocol}/seed_${seed}"
            log "  Running seed ${seed}..."
            
            python "${PROJECT_ROOT}/train.py" \
                --config-name=experiment/reproduce_table3 \
                protocol="${protocol}" \
                experiment.seed="${seed}" \
                experiment.num_episodes="${NUM_EPISODES}" \
                output.dir="${run_dir}" \
                2>&1 | tee -a "${LOG_FILE}"
        done
    done
    
    log ""
    log "Aggregating Table 3 results..."
    python "${PROJECT_ROOT}/evaluate.py" \
        --results_dir "${table3_dir}" \
        --output_file "${OUTPUT_DIR}/table3_results.csv" \
        --format latex \
        2>&1 | tee -a "${LOG_FILE}"
    
    log "${GREEN}✓ Table 3 complete${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Ablation Study
# ─────────────────────────────────────────────────────────────────────────────

run_table4() {
    log_section "Table 4: Ablation Study"
    
    local table4_dir="${OUTPUT_DIR}/table4"
    mkdir -p "${table4_dir}"
    
    for ablation in "${ABLATIONS[@]}"; do
        log ""
        log "${YELLOW}Ablation: ${ablation}${NC}"
        
        for seed in "${SEEDS[@]}"; do
            local run_dir="${table4_dir}/${ablation}/seed_${seed}"
            log "  Running seed ${seed}..."
            
            python "${PROJECT_ROOT}/train.py" \
                --config-name=experiment/ablation \
                protocol="${ablation}" \
                experiment.seed="${seed}" \
                experiment.num_episodes="${NUM_EPISODES}" \
                output.dir="${run_dir}" \
                2>&1 | tee -a "${LOG_FILE}"
        done
    done
    
    log ""
    log "Aggregating Table 4 results..."
    python "${PROJECT_ROOT}/evaluate.py" \
        --results_dir "${table4_dir}" \
        --output_file "${OUTPUT_DIR}/table4_results.csv" \
        --format latex \
        2>&1 | tee -a "${LOG_FILE}"
    
    log "${GREEN}✓ Table 4 complete${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Table 5: Temperature Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

run_table5() {
    log_section "Table 5: Temperature Sensitivity Analysis"
    
    local table5_dir="${OUTPUT_DIR}/table5"
    mkdir -p "${table5_dir}"
    
    for temp in "${TEMPERATURES[@]}"; do
        log ""
        log "${YELLOW}Temperature: τ=${temp}${NC}"
        
        for seed in "${SEEDS[@]}"; do
            local run_dir="${table5_dir}/temp_${temp}/seed_${seed}"
            log "  Running seed ${seed}..."
            
            python "${PROJECT_ROOT}/train.py" \
                protocol=coalt \
                experiment.seed="${seed}" \
                experiment.num_episodes="${NUM_EPISODES}" \
                api.temperature="${temp}" \
                output.dir="${run_dir}" \
                2>&1 | tee -a "${LOG_FILE}"
        done
    done
    
    log ""
    log "Aggregating Table 5 results..."
    python "${PROJECT_ROOT}/evaluate.py" \
        --results_dir "${table5_dir}" \
        --output_file "${OUTPUT_DIR}/table5_results.csv" \
        --analyze_temperature \
        2>&1 | tee -a "${LOG_FILE}"
    
    log "${GREEN}✓ Table 5 complete${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Consistency vs. Stability
# ─────────────────────────────────────────────────────────────────────────────

run_figure2() {
    log_section "Figure 2: Consistency vs. Stability Analysis"
    
    log "Generating Figure 2 from Table 3 results..."
    python "${PROJECT_ROOT}/evaluate.py" \
        --results_dir "${OUTPUT_DIR}/table3" \
        --output_file "${OUTPUT_DIR}/figure2.pdf" \
        --plot_consistency_stability \
        2>&1 | tee -a "${LOG_FILE}"
    
    log "${GREEN}✓ Figure 2 complete${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Model-Specific Analysis
# ─────────────────────────────────────────────────────────────────────────────

run_model_analysis() {
    log_section "Model-Specific Analysis (Table 3 bottom)"
    
    local model_dir="${OUTPUT_DIR}/model_specific"
    mkdir -p "${model_dir}"
    
    local model_configs=("gpt4_only" "claude_only" "llama_only" "mixed")
    
    for config in "${model_configs[@]}"; do
        log ""
        log "${YELLOW}Configuration: ${config}${NC}"
        
        for seed in "${SEEDS[@]}"; do
            local run_dir="${model_dir}/${config}/seed_${seed}"
            log "  Running seed ${seed}..."
            
            python "${PROJECT_ROOT}/train.py" \
                protocol=coalt \
                experiment.seed="${seed}" \
                experiment.num_episodes=100 \
                game.model_config="${config}" \
                output.dir="${run_dir}" \
                2>&1 | tee -a "${LOG_FILE}"
        done
    done
    
    log "${GREEN}✓ Model analysis complete${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────

generate_summary() {
    log_section "Generating Final Summary"
    
    python "${PROJECT_ROOT}/evaluate.py" \
        --results_dir "${OUTPUT_DIR}" \
        --output_file "${OUTPUT_DIR}/full_results.csv" \
        --generate_all_tables \
        2>&1 | tee -a "${LOG_FILE}"
    
    log ""
    log "═══════════════════════════════════════════════════════════════════════════"
    log "${GREEN}REPRODUCTION COMPLETE${NC}"
    log "═══════════════════════════════════════════════════════════════════════════"
    log ""
    log "Results saved to: ${OUTPUT_DIR}"
    log ""
    log "Generated files:"
    log "  - table3_results.csv   : Main results (Table 3)"
    log "  - table4_results.csv   : Ablation study (Table 4)"
    log "  - table5_results.csv   : Temperature sensitivity (Table 5)"
    log "  - figure2.pdf          : Consistency vs. stability plot"
    log "  - full_results.csv     : Complete results"
    log "  - reproduce.log        : Full execution log"
    log ""
    log "To generate LaTeX tables:"
    log "  python evaluate.py --results_dir ${OUTPUT_DIR} --format latex"
}

# ─────────────────────────────────────────────────────────────────────────────
# Quick Validation Run (for testing)
# ─────────────────────────────────────────────────────────────────────────────

run_quick_validation() {
    log_section "Quick Validation Run (10 episodes, 1 seed)"
    
    local quick_dir="${OUTPUT_DIR}/quick_validation"
    mkdir -p "${quick_dir}"
    
    for protocol in "standard" "coalt"; do
        log "Running ${protocol}..."
        python "${PROJECT_ROOT}/train.py" \
            protocol="${protocol}" \
            experiment.seed=42 \
            experiment.num_episodes=10 \
            output.dir="${quick_dir}/${protocol}" \
            2>&1 | tee -a "${LOG_FILE}"
    done
    
    log "${GREEN}✓ Quick validation complete${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
    cd "${PROJECT_ROOT}"
    
    # Parse arguments
    local run_quick=false
    local run_all=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                run_quick=true
                run_all=false
                shift
                ;;
            --table3-only)
                run_all=false
                preflight_checks
                run_table3
                generate_summary
                exit 0
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --quick        Run quick validation (10 episodes)"
                echo "  --table3-only  Run only Table 3 experiments"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                log "${RED}Unknown option: $1${NC}"
                exit 1
                ;;
        esac
    done
    
    # Run pre-flight checks
    preflight_checks
    
    if [[ "${run_quick}" == "true" ]]; then
        run_quick_validation
        exit 0
    fi
    
    if [[ "${run_all}" == "true" ]]; then
        # Full reproduction
        log ""
        log "Starting full reproduction..."
        log "Estimated time: ~8 hours"
        log "Estimated cost: ~\$2,400"
        log ""
        read -p "Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Aborted."
            exit 0
        fi
        
        run_table3
        run_table4
        run_table5
        run_figure2
        run_model_analysis
        generate_summary
    fi
}

# Run main
main "$@"
