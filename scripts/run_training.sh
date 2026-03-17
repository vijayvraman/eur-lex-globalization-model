#!/bin/bash
# Training Orchestration Script - Phases 2 & 3
#
# Runs CPT and SFT training on 4x RTX Pro 6000 GPUs (96GB GDDR7 each)
# For Phase 5 advanced training on 4x B200 GPUs, use run_training_b200.sh

set -e

echo "=========================================="
echo "EUR-Lex Model Training (4x RTX Pro 6000)"
echo "=========================================="
echo ""

# Configuration
PHASE=${1:-"both"}  # cpt, sft, or both
USE_FSDP=${USE_FSDP:-"true"}  # Default to FSDP2 (NVIDIA Blackwell optimized), set to "false" for DeepSpeed
PRECISION=${PRECISION:-"fp8"}  # Precision mode: "fp8" only (NVFP4 is B200+ only, not supported on RTX Pro 6000)
BASE_DIR="$(pwd)"

# Verify GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPUs."
    exit 1
fi

# Count GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Detected GPUs: $GPU_COUNT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | awk '{print "GPU: " $0}'
echo ""

if [ "$GPU_COUNT" -ne 4 ]; then
    echo "Warning: Expected 4 RTX Pro 6000 GPUs, found $GPU_COUNT"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set environment variables for Transformer Engine quantization
export NVTE_FP8_DPA_BWD=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

echo "Environment variables set for Transformer Engine quantization:"
echo "  NVTE_FP8_DPA_BWD=$NVTE_FP8_DPA_BWD"
echo "  NVTE_ALLOW_NONDETERMINISTIC_ALGO=$NVTE_ALLOW_NONDETERMINISTIC_ALGO"
echo ""

# Verify data exists
if [ "$PHASE" == "cpt" ] || [ "$PHASE" == "both" ]; then
    if [ ! -d "data/cpt" ]; then
        echo "Error: CPT data not found in data/cpt/"
        echo "Please transfer data from Mac Studio first."
        exit 1
    fi
fi

if [ "$PHASE" == "sft" ] || [ "$PHASE" == "both" ]; then
    if [ ! -d "data/sft" ]; then
        echo "Error: SFT data not found in data/sft/"
        echo "Please transfer data from Mac Studio first."
        exit 1
    fi
fi

# Function to run CPT training
run_cpt_training() {
    echo "=========================================="
    echo "Phase 2: CPT Training (4x RTX Pro 6000)"
    echo "=========================================="
    echo "Started: $(date)"
    echo ""

    # Display backend and precision
    if [ "$USE_FSDP" == "true" ]; then
        echo "Backend: PyTorch FSDP2 + Transformer Engine (Blackwell optimized)"
    else
        echo "Backend: DeepSpeed ZeRO-3 + Transformer Engine"
    fi

    echo "Precision: FP8 (8-bit E4M3/E5M2)"
    echo "  → Expected memory: ~70GB per GPU (fits in 96GB GDDR7, ~73% utilization)"
    echo "  → Note: NVFP4 is B200+ only and not supported on RTX Pro 6000"
    echo ""

    echo "Configuration:"
    echo "  Model: LLaMA 3.3 70B"
    echo "  GPUs: 4x RTX Pro 6000 (96GB GDDR7 each)"
    echo "  Batch size: 2 per GPU × 4 GPUs × 16 grad_accum = 128"
    echo "  Learning rate: 2e-5"
    echo "  Steps: 4,260 (5 epochs over 446M tokens)"
    echo "  Estimated time: 12-18 hours"
    echo "  Checkpoint: checkpoints/cpt/final"
    echo ""
    read -p "Start CPT training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    START_TIME=$(date +%s)

    if [ "$USE_FSDP" == "true" ]; then
        # FSDP2 launcher (torchrun)
        torchrun \
            --nproc_per_node=4 \
            --master_port=29500 \
            scripts/train_cpt.py \
            --config configs/cpt_config.yaml \
            --fsdp_config configs/fsdp_config.json \
            --use_fp8 \
            --precision "$PRECISION"
    else
        # DeepSpeed launcher (original)
        deepspeed --num_gpus=4 \
            --master_port=29500 \
            scripts/train_cpt.py \
            --config configs/cpt_config.yaml \
            --deepspeed configs/ds_config_zero3.json \
            --use_fp8 \
            --precision "$PRECISION"
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "✓ CPT training complete in $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
    echo "Finished: $(date)"
    echo ""
    echo "Checkpoint: checkpoints/cpt/final"
    echo "Logs: logs/cpt_training/"
    echo ""
    echo "Next: Run Phase 3 SFT training, or transfer checkpoint for Phase 5 (B200)"
    echo ""
}

# Function to run SFT training
run_sft_training() {
    echo "=========================================="
    echo "Phase 3: SFT Training (4x RTX Pro 6000)"
    echo "=========================================="
    echo "Started: $(date)"
    echo ""

    # Display backend and precision
    if [ "$USE_FSDP" == "true" ]; then
        echo "Backend: PyTorch FSDP2 + Transformer Engine (Blackwell optimized)"
    else
        echo "Backend: DeepSpeed ZeRO-3 + Transformer Engine"
    fi

    echo "Precision: FP8 (8-bit E4M3/E5M2)"
    echo "  → Expected memory: ~40GB per GPU (fits in 96GB GDDR7, ~42% utilization)"
    echo "  → Note: NVFP4 is B200+ only and not supported on RTX Pro 6000"
    echo ""

    echo "Configuration:"
    echo "  Model: Phase 2 CPT checkpoint"
    echo "  GPUs: 4x RTX Pro 6000 (96GB GDDR7 each)"
    echo "  Batch size: 4 per GPU × 4 GPUs × 8 grad_accum = 128"
    echo "  Learning rate: 5e-6"
    echo "  Epochs: 3"
    echo "  Seq length: 2048 tokens"
    echo "  Input masking: Enabled"
    echo "  Estimated time: 6-8 hours"
    echo ""

    # Check if CPT checkpoint exists
    if [ ! -d "models/llama33-70b-eurlex-cpt-final" ] && [ ! -d "checkpoints/cpt/final" ]; then
        echo "Warning: Phase 2 CPT checkpoint not found"
        echo "Looking in:"
        echo "  - checkpoints/cpt/final"
        echo "  - models/llama33-70b-eurlex-cpt-final"
        echo ""
        echo "Run Phase 2 CPT training first:"
        echo "  ./scripts/run_training.sh cpt"
        echo ""
        read -p "Continue with base LLaMA model instead? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    read -p "Start SFT training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    START_TIME=$(date +%s)

    if [ "$USE_FSDP" == "true" ]; then
        # FSDP2 launcher (torchrun)
        torchrun \
            --nproc_per_node=4 \
            --master_port=29500 \
            scripts/train_sft.py \
            --config configs/sft_config.yaml \
            --fsdp_config configs/fsdp_config.json \
            --use_fp8 \
            --precision "$PRECISION"
    else
        # DeepSpeed launcher (original)
        deepspeed --num_gpus=4 \
            --master_port=29500 \
            scripts/train_sft.py \
            --config configs/sft_config.yaml \
            --deepspeed configs/ds_config_zero3_sft.json \
            --use_fp8 \
            --precision "$PRECISION"
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "✓ SFT training complete in $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
    echo "Finished: $(date)"
    echo ""
    echo "Checkpoint: checkpoints/sft/final"
    echo "Logs: logs/sft_training/"
    echo ""
    echo "Next: Run Phase 4 evaluation, or use checkpoints for Phase 5 (B200 advanced)"
    echo ""
}

# Run training based on phase
case "$PHASE" in
    cpt)
        run_cpt_training
        ;;
    sft)
        run_sft_training
        ;;
    both)
        run_cpt_training
        echo ""
        echo "Preparing for SFT training..."
        sleep 5
        run_sft_training
        ;;
    *)
        echo "Error: Invalid phase '$PHASE'"
        echo "Usage: $0 [cpt|sft|both]"
        exit 1
        ;;
esac

echo "=========================================="
echo "Phases 2 & 3 Complete! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "  Phase 4 - Evaluate model:"
echo "    python scripts/evaluate_model.py \\"
echo "      --model_path checkpoints/sft/final \\"
echo "      --eval_dataset data/sft/validation/sft_val.jsonl \\"
echo "      --output_file results/evaluation_report.json"
echo ""
echo "  Phase 5 - Advanced training on 4x B200 (optional):"
echo "    ./scripts/run_training_b200.sh both"
echo ""
echo "================================================================================"
echo "Usage (Phases 2 & 3 — 4x RTX Pro 6000):"
echo "================================================================================"
echo ""
echo "Phase 2 only (CPT):"
echo "  ./scripts/run_training.sh cpt"
echo ""
echo "Phase 3 only (SFT):"
echo "  ./scripts/run_training.sh sft"
echo ""
echo "Both phases:"
echo "  ./scripts/run_training.sh both"
echo ""
echo "Note: NVFP4 is NOT supported on RTX Pro 6000 (B200+ only)."
echo "  FP8 is the only quantization mode for Phases 2 & 3."
echo ""
echo "DeepSpeed ZeRO-3 (legacy fallback):"
echo "  USE_FSDP=false ./scripts/run_training.sh both"
echo ""
echo "For Phase 5 advanced training on 4x B200:"
echo "  ./scripts/run_training_b200.sh both"
echo ""
