#!/bin/bash
# Training Orchestration Script for GPU Cluster
#
# Runs CPT and SFT training on 4x B200 GPUs with FP4 quantization

set -e

echo "=========================================="
echo "EUR-Lex Model Training (4x B200 GPUs)"
echo "=========================================="
echo ""

# Configuration
PHASE=${1:-"both"}  # cpt, sft, or both
USE_FSDP=${USE_FSDP:-"true"}  # Default to FSDP2 (NVIDIA Blackwell optimized), set to "false" for DeepSpeed
PRECISION=${PRECISION:-"fp8"}  # Precision mode: "fp8" (default) or "nvfp4" (experimental)
BASE_DIR="$(pwd)"

# Verify GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPUs."
    exit 1
fi

# Count GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Detected GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -ne 4 ]; then
    echo "Warning: Expected 4 GPUs, found $GPU_COUNT"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display GPU info
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set environment variables for Transformer Engine FP4 quantization
export NVTE_FP8_DPA_BWD=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

echo "Environment variables set for FP4 quantization:"
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
    echo "Phase 1: CPT Training"
    echo "=========================================="
    echo "Started: $(date)"
    echo ""

    # Display backend and precision
    if [ "$USE_FSDP" == "true" ]; then
        echo "Backend: PyTorch FSDP2 + Transformer Engine (NVIDIA Blackwell optimized)"
    else
        echo "Backend: DeepSpeed ZeRO-3 + Transformer Engine"
    fi

    if [ "$PRECISION" == "nvfp4" ]; then
        echo "Precision: NVFP4 (4-bit E2M1) - Experimental"
        echo "  → Expected memory: ~35GB per GPU"
        echo "  → 50% memory savings vs FP8"
    else
        echo "Precision: FP8 (8-bit E4M3/E5M2) - Default"
        echo "  → Expected memory: ~70GB per GPU"
    fi

    echo "Configuration:"
    echo "  Model: LLaMA 3.3 70B"
    echo "  GPUs: 4x B200"
    echo "  Batch size: 2 per GPU × 4 GPUs × 16 grad_accum = 128"
    echo "  Learning rate: 2e-5"
    echo "  Steps: 40,000"
    echo "  Estimated time: 20-24 hours"
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
            --fsdp \
            --fsdp_config cpt \
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
    echo "Checkpoint location: checkpoints/cpt/final"
    echo "Logs: logs/cpt_training/"
    echo ""
}

# Function to run SFT training
run_sft_training() {
    echo "=========================================="
    echo "Phase 2: SFT Training"
    echo "=========================================="
    echo "Started: $(date)"
    echo ""

    # Display backend and precision
    if [ "$USE_FSDP" == "true" ]; then
        echo "Backend: PyTorch FSDP2 + Transformer Engine (NVIDIA Blackwell optimized)"
    else
        echo "Backend: DeepSpeed ZeRO-3 + Transformer Engine"
    fi

    if [ "$PRECISION" == "nvfp4" ]; then
        echo "Precision: NVFP4 (4-bit E2M1) - Experimental"
        echo "  → Expected memory: ~20GB per GPU"
        echo "  → 50% memory savings vs FP8"
    else
        echo "Precision: FP8 (8-bit E4M3/E5M2) - Default"
        echo "  → Expected memory: ~40GB per GPU"
    fi

    echo "Configuration:"
    echo "  Model: CPT checkpoint"
    echo "  GPUs: 4x B200"
    echo "  Batch size: 4 per GPU × 4 GPUs × 8 grad_accum = 128"
    echo "  Learning rate: 5e-6"
    echo "  Epochs: 3"
    echo "  Input masking: Enabled"
    echo "  Estimated time: 6-8 hours"
    echo ""

    # Check if CPT checkpoint exists
    if [ ! -d "models/llama33-70b-eurlex-cpt-final" ] && [ ! -d "checkpoints/cpt/final" ]; then
        echo "Warning: CPT checkpoint not found"
        echo "Looking in:"
        echo "  - models/llama33-70b-eurlex-cpt-final"
        echo "  - checkpoints/cpt/final"
        echo ""
        read -p "Continue with base model? (y/n) " -n 1 -r
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
            --fsdp \
            --fsdp_config sft \
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
    echo "Model location: checkpoints/sft/final"
    echo "Logs: logs/sft_training/"
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
echo "Training Complete! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate model:"
echo "     python scripts/evaluate_model.py \\"
echo "       --model_path checkpoints/sft/final \\"
echo "       --eval_dataset data/sft/validation/sft_val.jsonl \\"
echo "       --output_file results/evaluation_report.json"
echo ""
echo "  2. Convert checkpoint for inference:"
echo "     python src/utils/checkpoint_utils.py convert-fp4 \\"
echo "       --model_path checkpoints/sft/final \\"
echo "       --output_path models/llama33-70b-eurlex-sft-final"
echo ""
echo "================================================================================"
echo "Usage Examples:"
echo "================================================================================"
echo ""
echo "Default (FSDP2 + FP8):"
echo "  ./scripts/run_training.sh both"
echo ""
echo "Run only CPT or SFT:"
echo "  ./scripts/run_training.sh cpt"
echo "  ./scripts/run_training.sh sft"
echo ""
echo "Use NVFP4 (4-bit, experimental, 50% memory savings):"
echo "  PRECISION=nvfp4 ./scripts/run_training.sh both"
echo ""
echo "Use DeepSpeed ZeRO-3 (legacy):"
echo "  USE_FSDP=false ./scripts/run_training.sh both"
echo ""
echo "Combine options:"
echo "  PRECISION=nvfp4 USE_FSDP=true ./scripts/run_training.sh cpt"
echo ""
