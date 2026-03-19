#!/bin/bash
# Advanced Training Script — Phase 5
#
# Runs advanced CPT and SFT on 4x NVIDIA B200 GPUs (192GB HBM3e each)
# Starting from Phase 2/3 RTX Pro 6000 checkpoints, or independently from base model.
#
# Advantages over RTX Pro 6000 (Phases 2 & 3):
#   - 192GB HBM3e per GPU vs 96GB GDDR7 → 2x memory headroom
#   - ~8 TB/s memory bandwidth vs ~1.7 TB/s → ~4-5x throughput
#   - Larger effective batch sizes and longer context windows
#   - NVFP4 fully production-ready on B200
#
# Phase 5 capabilities:
#   - CPT with 8192-token sequences (vs 4096 on RTX Pro 6000)
#   - SFT with 4096-token sequences (vs 2048)
#   - Larger per-device batch sizes → better gradient estimates
#   - 10 CPT epochs + 5 SFT epochs for thorough domain adaptation

set -e

# Logging setup — log file named by Pacific time
LOG_DIR="$( cd "$(dirname "$0")/.." && pwd )/logs"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP=$(TZ="America/Los_Angeles" date +"%Y-%m-%d_%H-%M-%S_PT")
LOG_FILE="$LOG_DIR/training_b200_${LOG_TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"
echo ""

echo "=========================================="
echo "EUR-Lex Advanced Training (4x B200 GPUs)"
echo "Phase 5: Advanced CPT and SFT"
echo "=========================================="
echo ""

# Check for active virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No python virtual environment detected."
    echo "Please activate your environment first:"
    echo "  source venv/bin/activate"
    exit 1
fi

# Check for tmux session (recommended for long-running training)
if [ -z "$TMUX" ]; then
    echo "Error: Not running inside a tmux session."
    echo "Training can take several hours. Please run inside tmux:"
    echo "  tmux"
    echo "  ./scripts/run_training_b200.sh"
    exit 1
fi

# Configuration
PHASE=${1:-"both"}          # cpt, sft, or both
USE_FSDP=${USE_FSDP:-"true"}
PRECISION=${PRECISION:-"nvfp4"}  # Default NVFP4 on B200 — production-ready
BASE_DIR="$(pwd)"

# Checkpoint sources from Phases 2 & 3 (optional warm start)
CPT_WARMSTART=${CPT_WARMSTART:-"checkpoints/cpt/final"}   # Phase 2 output
SFT_WARMSTART=${SFT_WARMSTART:-"checkpoints/sft/final"}   # Phase 3 output

# Verify GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPUs."
    exit 1
fi

# Count GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPUs: $GPU_COUNT x $GPU_NAME"

if [ "$GPU_COUNT" -ne 4 ]; then
    echo "Warning: Expected 4 B200 GPUs, found $GPU_COUNT"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [[ "$GPU_NAME" != *"B200"* ]]; then
    echo "Warning: Expected B200 GPUs, detected: $GPU_NAME"
    echo "NVFP4 requires Blackwell (B200) hardware. Falling back to FP8."
    PRECISION="fp8"
    echo ""
fi

# Display GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Verify NVLink connectivity — on B200 all GPUs connect via NVSwitch
echo "Checking NVLink topology..."
if nvidia-smi topo -m 2>/dev/null | grep -q "NV"; then
    nvidia-smi topo -m
else
    echo "Warning: NVLink topology not confirmed. NCCL will fall back to PCIe."
    echo "  Expect slower all-reduce. Verify NVSwitch is functional."
fi
echo ""

# -----------------------------------------------------------------------
# NCCL environment variables optimized for NVLink + B200 NVSwitch
# -----------------------------------------------------------------------
# P2P and shared memory
export NCCL_P2P_DISABLE=0             # Ensure NVLink P2P is used (not PCIe fallback)
export NCCL_SHM_DISABLE=0            # Shared memory transport enabled

# NVLink SHARP (NVLS): Blackwell NVSwitch can execute collective ops in-fabric,
# turning a multi-hop tree all-reduce into a single-step operation.
export NCCL_NVLS_ENABLE=1

# GPU Direct RDMA through NVSwitch fabric (level 5 = optimal for B200)
export NCCL_NET_GDR_LEVEL=5

# Buffer and channel tuning to saturate NVLink (~1.8 TB/s bidirectional)
export NCCL_BUFFSIZE=8388608          # 8 MB buffer (default 4 MB is too small for NVLink BW)
export NCCL_NCHANNELS_PER_NET_PEER=8 # More channels per peer to maximize NVSwitch utilization

# Transformer Engine settings for FSDP2 + NVFP4/FP8
export NVTE_FP8_DPA_BWD=1             # FP8 backward pass for attention
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1              # Fused attention (Flash Attention 3 on B200)
export NVTE_FP8_AMAX_REDUCE=1         # All-reduce amax across FSDP ranks

if [ "$PRECISION" == "nvfp4" ]; then
    # Stochastic rounding improves accuracy at 4-bit without extra memory cost
    export NVTE_FP8_STOCHASTIC_ROUNDING=1
fi

echo "NCCL environment (NVLink + NVSwitch optimized):"
echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE            (NVLink P2P enabled)"
echo "  NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE              (NVLink SHARP in-fabric collectives)"
echo "  NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL           (GPU Direct through NVSwitch)"
echo "  NCCL_BUFFSIZE=$NCCL_BUFFSIZE          (8MB buffer)"
echo "  NCCL_NCHANNELS_PER_NET_PEER=$NCCL_NCHANNELS_PER_NET_PEER"
echo ""
echo "Transformer Engine:"
echo "  NVTE_FP8_DPA_BWD=$NVTE_FP8_DPA_BWD"
echo "  NVTE_ALLOW_NONDETERMINISTIC_ALGO=$NVTE_ALLOW_NONDETERMINISTIC_ALGO"
echo "  NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN"
echo "  NVTE_FP8_AMAX_REDUCE=$NVTE_FP8_AMAX_REDUCE"
if [ "$PRECISION" == "nvfp4" ]; then
    echo "  NVTE_FP8_STOCHASTIC_ROUNDING=$NVTE_FP8_STOCHASTIC_ROUNDING  (NVFP4 accuracy)"
fi
echo ""

# Verify data exists
if [ "$PHASE" == "cpt" ] || [ "$PHASE" == "both" ]; then
    if [ ! -d "data/cpt" ]; then
        echo "Error: CPT data not found in data/cpt/"
        echo "Run Phase 1 data processing first:"
        echo "  ./scripts/run_full_pipeline.sh"
        exit 1
    fi
fi

if [ "$PHASE" == "sft" ] || [ "$PHASE" == "both" ]; then
    if [ ! -d "data/sft" ]; then
        echo "Error: SFT data not found in data/sft/"
        echo "Run Phase 1 data processing first:"
        echo "  ./scripts/run_full_pipeline.sh"
        exit 1
    fi
fi

# --- Advanced CPT Training ---
run_cpt_b200() {
    echo "=========================================="
    echo "Phase 5a: Advanced CPT (4x B200)"
    echo "=========================================="
    echo "Started: $(date)"
    echo ""

    if [ "$USE_FSDP" == "true" ]; then
        echo "Backend: PyTorch FSDP2 + Transformer Engine (B200 optimized)"
    else
        echo "Backend: DeepSpeed ZeRO-3 + Transformer Engine"
    fi

    if [ "$PRECISION" == "nvfp4" ]; then
        echo "Precision: NVFP4 (4-bit E2M1) — Default on B200, production-ready"
        echo "  → Expected memory: ~50GB per GPU (well within 192GB HBM3e)"
        echo "  → Enables 8192-token sequences with headroom to spare"
    else
        echo "Precision: FP8 (8-bit E4M3/E5M2)"
        echo "  → Expected memory: ~90GB per GPU (fits in 192GB HBM3e)"
    fi

    echo ""
    echo "Configuration:"
    echo "  Model: LLaMA 3.3 70B"
    echo "  GPUs: 4x B200 (192GB HBM3e each)"
    echo "  Batch size: 4 per GPU × 4 GPUs × 8 grad_accum = 128"
    echo "  Sequence length: 8192 tokens (vs 4096 on RTX Pro 6000)"
    echo "  Learning rate: 1e-5"
    echo "  Steps: 8,520 (10 epochs over 446M tokens)"
    echo "  Estimated time: 3-4 hours"
    echo ""

    # Check for warm-start from Phase 2
    if [ -d "$CPT_WARMSTART" ]; then
        echo "Warm start: $CPT_WARMSTART (Phase 2 CPT checkpoint)"
    else
        echo "No Phase 2 checkpoint found — starting from base LLaMA model"
        echo "  (Set CPT_WARMSTART=<path> to warm-start from an existing checkpoint)"
    fi
    echo ""

    read -p "Start advanced CPT training on B200? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    START_TIME=$(date +%s)

    if [ "$USE_FSDP" == "true" ]; then
        torchrun \
            --nproc_per_node=4 \
            --master_port=29500 \
            scripts/train_cpt.py \
            --config configs/cpt_config_b200.yaml \
            --fsdp \
            --fsdp_config configs/fsdp_config.json \
            --use_fp8 \
            --precision "$PRECISION" \
            $([ -d "$CPT_WARMSTART" ] && echo "--resume_from_checkpoint $CPT_WARMSTART")
    else
        deepspeed --num_gpus=4 \
            --master_port=29500 \
            scripts/train_cpt.py \
            --config configs/cpt_config_b200.yaml \
            --deepspeed configs/ds_config_zero3.json \
            --use_fp8 \
            --precision "$PRECISION" \
            $([ -d "$CPT_WARMSTART" ] && echo "--resume_from_checkpoint $CPT_WARMSTART")
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "✓ Advanced CPT complete in $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
    echo "Finished: $(date)"
    echo ""
    echo "Checkpoint: checkpoints/cpt_b200/final"
    echo "Logs: logs/cpt_b200_training/"
    echo ""
}

# --- Advanced SFT Training ---
run_sft_b200() {
    echo "=========================================="
    echo "Phase 5b: Advanced SFT (4x B200)"
    echo "=========================================="
    echo "Started: $(date)"
    echo ""

    if [ "$USE_FSDP" == "true" ]; then
        echo "Backend: PyTorch FSDP2 + Transformer Engine (B200 optimized)"
    else
        echo "Backend: DeepSpeed ZeRO-3 + Transformer Engine"
    fi

    if [ "$PRECISION" == "nvfp4" ]; then
        echo "Precision: NVFP4 (4-bit E2M1) — Default on B200, production-ready"
        echo "  → Expected memory: ~30GB per GPU (well within 192GB HBM3e)"
    else
        echo "Precision: FP8 (8-bit E4M3/E5M2)"
        echo "  → Expected memory: ~55GB per GPU"
    fi

    echo ""
    echo "Configuration:"
    echo "  Model: Phase 5a CPT checkpoint (or Phase 2 CPT checkpoint)"
    echo "  GPUs: 4x B200 (192GB HBM3e each)"
    echo "  Batch size: 8 per GPU × 4 GPUs × 4 grad_accum = 128"
    echo "  Sequence length: 4096 tokens (vs 2048 on RTX Pro 6000)"
    echo "  Learning rate: 2e-6"
    echo "  Epochs: 5"
    echo "  Input masking: Enabled"
    echo "  Estimated time: 2-3 hours"
    echo ""

    # Determine which CPT checkpoint to use
    B200_CPT_CKPT="checkpoints/cpt_b200/final"
    FALLBACK_CKPT="$CPT_WARMSTART"

    if [ -d "$B200_CPT_CKPT" ]; then
        echo "Using Phase 5a CPT checkpoint: $B200_CPT_CKPT"
    elif [ -d "$FALLBACK_CKPT" ]; then
        echo "Phase 5a CPT not found — using Phase 2 CPT checkpoint: $FALLBACK_CKPT"
        B200_CPT_CKPT="$FALLBACK_CKPT"
    else
        echo "Warning: No CPT checkpoint found."
        echo "  Run Phase 5a CPT first:  ./scripts/run_training_b200.sh cpt"
        echo "  Or Phase 2 CPT:          ./scripts/run_training.sh cpt"
        echo ""
        read -p "Continue with base LLaMA model? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
        B200_CPT_CKPT=""
    fi
    echo ""

    read -p "Start advanced SFT training on B200? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    START_TIME=$(date +%s)

    if [ "$USE_FSDP" == "true" ]; then
        torchrun \
            --nproc_per_node=4 \
            --master_port=29500 \
            scripts/train_sft.py \
            --config configs/sft_config_b200.yaml \
            --fsdp \
            --fsdp_config configs/fsdp_config.json \
            --use_fp8 \
            --precision "$PRECISION" \
            $([ -n "$B200_CPT_CKPT" ] && echo "--model_path $B200_CPT_CKPT")
    else
        deepspeed --num_gpus=4 \
            --master_port=29500 \
            scripts/train_sft.py \
            --config configs/sft_config_b200.yaml \
            --deepspeed configs/ds_config_zero3_sft.json \
            --use_fp8 \
            --precision "$PRECISION" \
            $([ -n "$B200_CPT_CKPT" ] && echo "--model_path $B200_CPT_CKPT")
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "✓ Advanced SFT complete in $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
    echo "Finished: $(date)"
    echo ""
    echo "Checkpoint: checkpoints/sft_b200/final"
    echo "Logs: logs/sft_b200_training/"
    echo ""
}

# Run based on phase argument
case "$PHASE" in
    cpt)
        run_cpt_b200
        ;;
    sft)
        run_sft_b200
        ;;
    both)
        run_cpt_b200
        echo "Preparing for advanced SFT..."
        sleep 3
        run_sft_b200
        ;;
    *)
        echo "Error: Invalid phase '$PHASE'"
        echo "Usage: $0 [cpt|sft|both]"
        exit 1
        ;;
esac

echo "=========================================="
echo "Phase 5 Complete! ✓"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  CPT checkpoint: checkpoints/cpt_b200/final"
echo "  SFT checkpoint: checkpoints/sft_b200/final"
echo ""
echo "Next steps:"
echo ""
echo "  Phase 4 - Evaluate advanced model:"
echo "    python scripts/evaluate_model.py \\"
echo "      --model_path checkpoints/sft_b200/final \\"
echo "      --eval_dataset data/sft/validation/sft_val.jsonl \\"
echo "      --output_file results/evaluation_report_b200.json"
echo ""
echo "  Compare Phase 3 (RTX) vs Phase 5 (B200) models:"
echo "    python scripts/compare_models.py \\"
echo "      --base_model checkpoints/sft/final \\"
echo "      --finetuned_model checkpoints/sft_b200/final \\"
echo "      --test_dataset data/test/test_qna_100.jsonl \\"
echo "      --output_dir results/rtx_vs_b200"
echo ""
echo "================================================================================"
echo "Usage (Phase 5 — 4x B200):"
echo "================================================================================"
echo ""
echo "Default (FSDP2 + NVFP4, recommended for B200):"
echo "  ./scripts/run_training_b200.sh both"
echo ""
echo "Advanced CPT only:"
echo "  ./scripts/run_training_b200.sh cpt"
echo ""
echo "Advanced SFT only:"
echo "  ./scripts/run_training_b200.sh sft"
echo ""
echo "Warm-start from a specific Phase 2 checkpoint:"
echo "  CPT_WARMSTART=checkpoints/cpt/step-2000 ./scripts/run_training_b200.sh both"
echo ""
echo "FP8 precision instead of NVFP4:"
echo "  PRECISION=fp8 ./scripts/run_training_b200.sh both"
echo ""
echo "DeepSpeed ZeRO-3 (legacy):"
echo "  USE_FSDP=false ./scripts/run_training_b200.sh both"
echo ""
