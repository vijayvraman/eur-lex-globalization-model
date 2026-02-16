#!/bin/bash
# Test Pipeline on Sample Data
#
# This script tests the complete data processing pipeline on a small sample dataset
# to verify that all components work correctly before running on the full 25GB dataset.

set -e  # Exit on error

echo "========================================"
echo "EUR-Lex Pipeline Test (Sample Data)"
echo "========================================"
echo ""

# Configuration
SAMPLE_SIZE=5  # Documents per language
BASE_DIR="$(pwd)"
SAMPLE_RAW_DIR="$BASE_DIR/data/raw_sample"
SAMPLE_PARSED_DIR="$BASE_DIR/data/parsed_sample"
SAMPLE_CPT_DIR="$BASE_DIR/data/cpt_sample"
SAMPLE_SFT_DIR="$BASE_DIR/data/sft_sample"

# Cleanup previous test data
echo "Cleaning up previous test data..."
rm -rf "$SAMPLE_RAW_DIR" "$SAMPLE_PARSED_DIR" "$SAMPLE_CPT_DIR" "$SAMPLE_SFT_DIR"

# Step 1: Generate sample FORMEX XML
echo ""
echo "Step 1/5: Generating sample FORMEX XML data"
echo "--------------------------------------------"
python scripts/generate_sample_data.py \
  --output_dir "$SAMPLE_RAW_DIR" \
  --num_docs $SAMPLE_SIZE

echo "✓ Sample data generated: $SAMPLE_RAW_DIR"
echo "  - Languages: EN, FR, DE, ES, PT"
echo "  - Documents per language: $SAMPLE_SIZE"
echo ""

# Step 2: Parse FORMEX XML
echo "Step 2/5: Parsing FORMEX XML"
echo "--------------------------------------------"
python data_processing/parsers/formex_parser.py \
  --input "$SAMPLE_RAW_DIR" \
  --output "$SAMPLE_PARSED_DIR" \
  --workers 4

echo "✓ XML parsed successfully"
echo "  Parsed files in: $SAMPLE_PARSED_DIR"
echo ""

# Step 3: Build CPT corpus
echo "Step 3/5: Building CPT corpus"
echo "--------------------------------------------"
python data_processing/dataset_builders/cpt_corpus_builder.py \
  --input_dir "$SAMPLE_PARSED_DIR" \
  --output_dir "$SAMPLE_CPT_DIR" \
  --max_seq_length 4096 \
  --num_shards 4

echo "✓ CPT corpus built successfully"
echo "  Corpus location: $SAMPLE_CPT_DIR"
echo ""

# Step 4: Build SFT dataset
echo "Step 4/5: Building SFT dataset"
echo "--------------------------------------------"
python data_processing/dataset_builders/sft_dataset_builder.py \
  --input_dir "$SAMPLE_PARSED_DIR" \
  --output_dir "$SAMPLE_SFT_DIR" \
  --target_pairs 100 \
  --num_shards 2

echo "✓ SFT dataset built successfully"
echo "  Dataset location: $SAMPLE_SFT_DIR"
echo ""

# Step 5: Verify outputs
echo "Step 5/5: Verifying outputs"
echo "--------------------------------------------"

# Check CPT corpus
if [ -f "$SAMPLE_CPT_DIR/corpus_statistics.json" ]; then
    echo "✓ CPT corpus statistics found"
    echo "  Statistics:"
    cat "$SAMPLE_CPT_DIR/corpus_statistics.json" | python -m json.tool | head -15
else
    echo "✗ CPT corpus statistics not found"
    exit 1
fi

echo ""

# Check SFT dataset
if [ -f "$SAMPLE_SFT_DIR/sft_statistics.json" ]; then
    echo "✓ SFT dataset statistics found"
    echo "  Statistics:"
    cat "$SAMPLE_SFT_DIR/sft_statistics.json" | python -m json.tool | head -15
else
    echo "✗ SFT dataset statistics not found"
    exit 1
fi

echo ""
echo "========================================"
echo "Pipeline Test Complete! ✓"
echo "========================================"
echo ""
echo "All components working correctly."
echo "You can now run the full pipeline on your 25GB dataset."
echo ""
echo "Next steps:"
echo "  1. Place your FORMEX XML files in data/raw/"
echo "  2. Run: ./scripts/run_full_pipeline.sh"
echo ""
