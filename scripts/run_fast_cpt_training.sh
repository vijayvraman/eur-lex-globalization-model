#!/bin/bash
# Fast CPT Training Workflow — Phase 2 variant
#
# Implements the Hybrid Approach from docs/FAST_TRAINING_OPTIONS.md:
# 1. Filter corpus to top 40% quality documents (~2B tokens)
# 2. Build CPT corpus with 3072 token sequences
# 3. Train for 2 epochs with slightly higher learning rate
#
# Hardware: 4x RTX Pro 6000 (96GB GDDR7 each)
# For faster results on 4x B200, use run_training_b200.sh
#
# Expected results:
# - Training time: 4-6 hours (vs 12-18 hours for full CPT on RTX Pro 6000)
# - Quality: 75-80% of full CPT
# - Good enough for rapid prototyping

set -e

echo "=========================================="
echo "Fast CPT Training Workflow (4-6 hours)"
echo "=========================================="
echo ""

# Configuration
BASE_DIR="$(pwd)"
PARSED_DIR="$BASE_DIR/data/parsed"
FILTERED_DIR="$BASE_DIR/data/cpt_filtered"
TARGET_TOKENS=2000000000  # 2B tokens (40% of full corpus)
QUALITY_THRESHOLD=70      # Minimum quality score
NUM_SHARDS=16             # Fewer shards for smaller corpus
USE_FSDP=${USE_FSDP:-"true"}  # Default to FSDP2 (NVIDIA Blackwell optimized), set to "false" for DeepSpeed
PRECISION=${PRECISION:-"fp8"}  # Precision mode: "fp8" (default) or "nvfp4" (experimental)

# Parse command line arguments
SKIP_FILTERING=${1:-"false"}
SKIP_CORPUS_BUILD=${2:-"false"}

echo "Configuration:"
echo "  Parsed documents: $PARSED_DIR"
echo "  Filtered output: $FILTERED_DIR"
echo "  Target tokens: $TARGET_TOKENS (2B)"
echo "  Quality threshold: $QUALITY_THRESHOLD"
echo "  Sequence length: 3072"
echo "  Training steps: 12,000"
echo ""

# Check if parsed documents exist
if [ ! -d "$PARSED_DIR" ]; then
    echo "Error: Parsed documents not found at $PARSED_DIR"
    echo "Please run data processing first:"
    echo "  ./scripts/run_full_pipeline.sh"
    exit 1
fi

# Count parsed documents
DOC_COUNT=$(find "$PARSED_DIR" -name "*.json" | wc -l)
echo "Found $DOC_COUNT parsed documents"
echo ""

# Step 1: Filter documents by quality
if [ "$SKIP_FILTERING" == "false" ]; then
    echo "=========================================="
    echo "Step 1: Filtering High-Quality Documents"
    echo "=========================================="
    echo "This will select the top 40% of documents by quality score."
    echo "Criteria: recency, document type, length, subject matter"
    echo ""

    read -p "Start filtering? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping filtering..."
    else
        START_TIME=$(date +%s)

        python scripts/create_fast_cpt_corpus.py \
            --input_dir "$PARSED_DIR" \
            --output_dir "$FILTERED_DIR/documents" \
            --target_tokens "$TARGET_TOKENS" \
            --quality_threshold "$QUALITY_THRESHOLD"

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo "✓ Filtering complete in $(($DURATION / 60))m $(($DURATION % 60))s"
        echo ""

        # Display filtering statistics
        if [ -f "$FILTERED_DIR/documents/filtering_statistics.json" ]; then
            echo "Filtering statistics:"
            cat "$FILTERED_DIR/documents/filtering_statistics.json"
            echo ""
        fi
    fi
else
    echo "Skipping Step 1: Filtering (as requested)"
    echo ""
fi

# Step 2: Build CPT corpus
if [ "$SKIP_CORPUS_BUILD" == "false" ]; then
    echo "=========================================="
    echo "Step 2: Building CPT Corpus"
    echo "=========================================="
    echo "Creating training corpus with 3072-token sequences"
    echo "Shards: $NUM_SHARDS"
    echo ""

    if [ ! -d "$FILTERED_DIR/documents" ]; then
        echo "Error: Filtered documents not found at $FILTERED_DIR/documents"
        echo "Please run Step 1 (filtering) first."
        exit 1
    fi

    read -p "Start corpus building? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping corpus building..."
    else
        START_TIME=$(date +%s)

        python data_processing/dataset_builders/cpt_corpus_builder.py \
            --input_dir "$FILTERED_DIR/documents" \
            --output_dir "$FILTERED_DIR" \
            --max_seq_length 3072 \
            --num_shards "$NUM_SHARDS"

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo "✓ Corpus building complete in $(($DURATION / 60))m $(($DURATION % 60))s"
        echo ""

        # Display corpus statistics
        if [ -f "$FILTERED_DIR/corpus_statistics.json" ]; then
            echo "Corpus statistics:"
            cat "$FILTERED_DIR/corpus_statistics.json"
            echo ""
        fi
    fi
else
    echo "Skipping Step 2: Corpus building (as requested)"
    echo ""
fi

# Step 3: Transfer to GPU cluster (if needed)
echo "=========================================="
echo "Step 3: Data Transfer"
echo "=========================================="
echo ""

if [ -d "$FILTERED_DIR/train" ] && [ -d "$FILTERED_DIR/validation" ]; then
    echo "Filtered corpus ready for training:"
    echo "  Train: $FILTERED_DIR/train/"
    echo "  Validation: $FILTERED_DIR/validation/"
    echo ""

    # Check if we're on Mac or GPU cluster
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Running on macOS (Mac Studio)"
        echo ""
        echo "Next steps:"
        echo "  1. Transfer filtered corpus to GPU cluster:"
        echo ""
        echo "     # Using rsync (recommended):"
        echo "     rsync -avz --progress $FILTERED_DIR/ \\"
        echo "       user@gpu-cluster:/path/to/project/data/cpt_filtered/"
        echo ""
        echo "     # Or using tar + scp:"
        echo "     tar -czf cpt_filtered.tar.gz -C data cpt_filtered/"
        echo "     scp cpt_filtered.tar.gz user@gpu-cluster:/path/to/project/data/"
        echo ""
        echo "  2. On GPU cluster, run training:"
        echo "     ./scripts/run_fast_cpt_training.sh skip skip  # Skip steps 1-2"
        echo ""
    else
        echo "Detected Linux environment (likely GPU cluster)"
        echo "Proceeding to training..."
        echo ""

        # Step 4: Training on GPU cluster
        echo "=========================================="
        echo "Step 4: Fast CPT Training (4-6 hours)"
        echo "=========================================="
        echo ""

        # Verify GPU availability
        if ! command -v nvidia-smi &> /dev/null; then
            echo "Error: nvidia-smi not found. Are you on the GPU cluster?"
            exit 1
        fi

        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "Detected GPUs: $GPU_COUNT"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

        if [ "$GPU_COUNT" -lt 4 ]; then
            echo "Warning: Expected 4 RTX Pro 6000 GPUs, found $GPU_COUNT"
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi

        echo ""
        nvidia-smi --query-gpu=name,memory.total --format=csv
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

        echo "Training configuration:"
        echo "  Model: LLaMA 3.3 70B"
        echo "  Hardware: 4x RTX Pro 6000 (96GB GDDR7)"
        echo "  Steps: 2,270 (2 epochs over 446M tokens)"
        echo "  Sequence length: 3,072 (vs 4,096 full)"
        echo "  Learning rate: 3e-5 (vs 2e-5 full)"
        echo "  Batch size: 2 per GPU × 4 GPUs × 16 = 128"
        echo "  Expected time: 4-6 hours"
        echo ""

        read -p "Start fast CPT training? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Training cancelled."
            exit 0
        fi

        # Set environment variables for FP4 quantization
        export NVTE_FP8_DPA_BWD=1
        export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

        echo "Environment variables set for FP4 quantization"
        echo ""

        START_TIME=$(date +%s)

        if [ "$USE_FSDP" == "true" ]; then
            # FSDP2 launcher (torchrun)
            torchrun \
                --nproc_per_node=4 \
                --master_port=29500 \
                scripts/train_cpt.py \
                --config configs/cpt_config_fast.yaml \
            	--fsdp_config configs/fsdp_config.json \
                --use_fp8 \
                --precision "$PRECISION"
        else
            # DeepSpeed launcher (original)
            deepspeed --num_gpus=4 \
                --master_port=29500 \
                scripts/train_cpt.py \
                --config configs/cpt_config_fast.yaml \
                --deepspeed configs/ds_config_zero3.json \
                --use_fp8 \
                --precision "$PRECISION"
        fi

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo "=========================================="
        echo "✓ Fast CPT Training Complete!"
        echo "=========================================="
        echo ""
        echo "Training time: $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
        echo "Checkpoint: checkpoints/cpt_fast/final"
        echo ""
        echo "Next steps:"
        echo "  1. Evaluate checkpoint:"
        echo "     python scripts/evaluate_model.py \\"
        echo "       --model_path checkpoints/cpt_fast/final \\"
        echo "       --eval_dataset data/cpt_filtered/validation/cpt_val.parquet"
        echo ""
        echo "  2. Proceed to SFT training:"
        echo "     ./scripts/run_training.sh sft"
        echo ""
        echo "  3. Compare with full CPT (if available):"
        echo "     python scripts/compare_checkpoints.py \\"
        echo "       --fast checkpoints/cpt_fast/final \\"
        echo "       --full checkpoints/cpt/final"
        echo ""
        echo "================================================================================"
        echo "Usage (4x RTX Pro 6000 — Phase 2 fast variant):"
        echo "  Default (FSDP2 + FP8):"
        echo "    ./scripts/run_fast_cpt_training.sh"
        echo ""
        echo "  Note: NVFP4 is NOT supported on RTX Pro 6000 (B200+ only)."
        echo "  FP8 is the only quantization mode available on RTX Pro 6000."
        echo ""
        echo "  DeepSpeed ZeRO-3 (legacy):"
        echo "    USE_FSDP=false ./scripts/run_fast_cpt_training.sh"
        echo ""
        echo "  For full CPT (5 epochs, 12-18h):"
        echo "    ./scripts/run_training.sh cpt"
        echo ""
        echo "  For advanced CPT on 4x B200:"
        echo "    ./scripts/run_training_b200.sh cpt"
        echo "================================================================================"
        echo ""
    fi
else
    echo "Error: Filtered corpus not found."
    echo "Please complete Steps 1-2 first."
    exit 1
fi

echo ""
echo "=========================================="
echo "Fast CPT Workflow Complete!"
echo "=========================================="
echo ""
