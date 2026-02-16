#!/bin/bash
# Full Data Processing Pipeline
#
# Run complete data processing pipeline on 25GB FORMEX dataset
# This script should be run on Mac Studio for data processing

set -e  # Exit on error

echo "=========================================="
echo "EUR-Lex Full Data Processing Pipeline"
echo "=========================================="
echo ""
echo "WARNING: This will process ~25GB of FORMEX XML data"
echo "Estimated time: 24-36 hours on Mac Studio"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Configuration
BASE_DIR="$(pwd)"
RAW_DIR="$BASE_DIR/data/raw"
PARSED_DIR="$BASE_DIR/data/parsed"
CPT_DIR="$BASE_DIR/data/cpt"
SFT_DIR="$BASE_DIR/data/sft"
LOG_DIR="$BASE_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Get number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_WORKERS=$(sysctl -n hw.ncpu)
else
    NUM_WORKERS=$(nproc)
fi

echo "Detected $NUM_WORKERS CPU cores"
echo ""

# Verify input data exists
if [ ! -d "$RAW_DIR" ]; then
    echo "Error: Input directory not found: $RAW_DIR"
    echo "Please place your FORMEX XML files in data/raw/"
    exit 1
fi

# Count XML files
XML_COUNT=$(find "$RAW_DIR" -name "*.xml" | wc -l)
echo "Found $XML_COUNT XML files in $RAW_DIR"
echo ""

# Step 1: Parse FORMEX XML
echo "=========================================="
echo "Step 1/3: Parsing FORMEX XML"
echo "=========================================="
echo "Input: $RAW_DIR"
echo "Output: $PARSED_DIR"
echo "Workers: $NUM_WORKERS"
echo ""
echo "Started: $(date)"
START_TIME=$(date +%s)

python data_processing/parsers/formex_parser.py \
  --input "$RAW_DIR" \
  --output "$PARSED_DIR" \
  --workers "$NUM_WORKERS" \
  2>&1 | tee "$LOG_DIR/01_parsing.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "✓ Parsing complete in $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
echo "Finished: $(date)"
echo ""

# Step 2: Build CPT Corpus
echo "=========================================="
echo "Step 2/3: Building CPT Corpus"
echo "=========================================="
echo "Input: $PARSED_DIR"
echo "Output: $CPT_DIR"
echo "Shards: 32"
echo "Sequence length: 4096 tokens"
echo ""
echo "Started: $(date)"
START_TIME=$(date +%s)

python data_processing/dataset_builders/cpt_corpus_builder.py \
  --input_dir "$PARSED_DIR" \
  --output_dir "$CPT_DIR" \
  --max_seq_length 4096 \
  --num_shards 32 \
  2>&1 | tee "$LOG_DIR/02_cpt_building.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "✓ CPT corpus complete in $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
echo "Finished: $(date)"
echo ""

# Step 3: Build SFT Dataset
echo "=========================================="
echo "Step 3/3: Building SFT Dataset"
echo "=========================================="
echo "Input: $PARSED_DIR"
echo "Output: $SFT_DIR"
echo "Target pairs: 150,000"
echo "Shards: 8"
echo ""
echo "Started: $(date)"
START_TIME=$(date +%s)

python data_processing/dataset_builders/sft_dataset_builder.py \
  --input_dir "$PARSED_DIR" \
  --output_dir "$SFT_DIR" \
  --target_pairs 150000 \
  --num_shards 8 \
  2>&1 | tee "$LOG_DIR/03_sft_building.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "✓ SFT dataset complete in $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m"
echo "Finished: $(date)"
echo ""

# Generate summary report
echo "=========================================="
echo "Pipeline Complete! ✓"
echo "=========================================="
echo ""
echo "Output Summary:"
echo "  Parsed documents: $PARSED_DIR"
echo "  CPT corpus: $CPT_DIR"
echo "  SFT dataset: $SFT_DIR"
echo "  Logs: $LOG_DIR"
echo ""

# Display statistics
if [ -f "$CPT_DIR/corpus_statistics.json" ]; then
    echo "CPT Corpus Statistics:"
    cat "$CPT_DIR/corpus_statistics.json" | python -m json.tool
    echo ""
fi

if [ -f "$SFT_DIR/sft_statistics.json" ]; then
    echo "SFT Dataset Statistics:"
    cat "$SFT_DIR/sft_statistics.json" | python -m json.tool
    echo ""
fi

# Data transfer instructions
echo "=========================================="
echo "Next Steps: Transfer to GPU Cluster"
echo "=========================================="
echo ""
echo "Compress data for transfer:"
echo "  tar -czf cpt_data.tar.gz data/cpt/"
echo "  tar -czf sft_data.tar.gz data/sft/"
echo ""
echo "Transfer to GPU cluster:"
echo "  scp cpt_data.tar.gz user@gpu-cluster:/path/to/project/"
echo "  scp sft_data.tar.gz user@gpu-cluster:/path/to/project/"
echo ""
echo "Or use rsync:"
echo "  rsync -avz data/cpt/ user@gpu-cluster:/path/to/project/data/cpt/"
echo "  rsync -avz data/sft/ user@gpu-cluster:/path/to/project/data/sft/"
echo ""
