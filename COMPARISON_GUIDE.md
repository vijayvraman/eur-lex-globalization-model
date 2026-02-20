# EUR-Lex Model Comparison Guide

This guide explains how to use the QnA testing system to compare the base LLaMA 3.1 70B model with the fine-tuned EUR-Lex model.

## Overview

The comparison system consists of three main components:

1. **Test Set Generator** (`scripts/generate_test_set.py`) - Creates 100 Q&A pairs (20 per language)
2. **Model Comparator** (`scripts/compare_models.py`) - Runs both models and compares performance
3. **Report Generator** (`src/evaluation/report_generator.py`) - Generates readable reports

## Step 1: Generate Test Set

Create a test dataset with 100 Q&A pairs covering GDPR/RGPD across 5 languages:

```bash
python scripts/generate_test_set.py \
  --output_file data/test/test_qna_100.jsonl \
  --questions_per_language 20
```

**Output**: `data/test/test_qna_100.jsonl` (~50KB)

The test set includes:
- 20 English questions
- 20 French questions
- 20 German questions
- 20 Spanish questions
- 20 Portuguese questions

Question types:
- Definition questions (e.g., "What is 'personal data' according to GDPR?")
- Compliance questions (e.g., "What obligations does Article 33 impose?")
- Requirement questions (e.g., "What is required under Article 6?")
- Scope questions (e.g., "What is the territorial scope of GDPR?")
- Citation questions (e.g., "What does CELEX 32016R0679 refer to?")

## Step 2: Run Model Comparison

Compare base and fine-tuned models on the test set:

```bash
python scripts/compare_models.py \
  --base_model meta-llama/Llama-3.1-70B-Instruct \
  --finetuned_model ./checkpoints/sft/final \
  --test_dataset data/test/test_qna_100.jsonl \
  --output_dir results/model_comparison \
  --batch_size 8
```

### Command-line Arguments

- `--base_model`: Path to base model (default: `meta-llama/Llama-3.1-70B-Instruct`)
- `--finetuned_model`: Path to fine-tuned model (default: `./checkpoints/sft/final`)
- `--test_dataset`: Path to test JSONL file (required)
- `--output_dir`: Output directory for results (default: `results/model_comparison`)
- `--batch_size`: Batch size for inference (default: 8)
- `--max_new_tokens`: Max tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--device`: Device to use (default: `cuda`)

### Runtime

Expected runtime for 100 samples:
- Base model inference: ~2.5 minutes
- Fine-tuned model inference: ~2.5 minutes
- Report generation: ~10 seconds
- **Total**: ~5 minutes

### Memory Usage

- Peak GPU memory: ~55GB (single model + overhead)
- Safe with 4x 80GB B200 GPUs (~17% utilization)

## Step 3: Review Results

The comparison generates three output files:

### 1. JSON Results (`comparison_results.json`)

Complete results including:
- Overall metrics (base vs fine-tuned)
- Per-language breakdown
- Sample-level comparisons
- Win/loss/tie statistics

**Size**: ~150KB

### 2. Markdown Report (`comparison_report.md`)

Human-readable report with:
- Executive summary
- Overall performance table
- Per-language performance
- Win/loss/tie analysis
- Sample predictions
- Key observations

**Size**: ~10KB

### 3. Predictions CSV (`predictions.csv`)

Detailed predictions for error analysis:
- Sample ID, language, question
- Base and fine-tuned predictions
- All metrics for both models
- Winner identification

**Size**: ~200KB

## Example Output

### Console Summary

```
================================================================================
COMPARISON SUMMARY
================================================================================

Samples evaluated: 100

--------------------------------------------------------------------------------
OVERALL PERFORMANCE
--------------------------------------------------------------------------------

Metric               Base       Fine-tuned   Delta      Rel. Imp.
--------------------------------------------------------------------------------
Citation Accuracy    23.00%     87.00%       +64.00%    +278.3% ✓
Article Accuracy     31.00%     82.00%       +51.00%    +164.5% ✓
ROUGE-L              42.00%     71.00%       +29.00%    +69.0% ✓
Exact Match          15.00%     34.00%       +19.00%    +126.7% ✓

--------------------------------------------------------------------------------
WIN/LOSS/TIE ANALYSIS
--------------------------------------------------------------------------------

Fine-tuned wins: 93
Base wins:       4
Ties:            3
Win rate:        93.0%

--------------------------------------------------------------------------------
PER-LANGUAGE PERFORMANCE
--------------------------------------------------------------------------------

Language     Base ROUGE-L    FT ROUGE-L     Improvement
--------------------------------------------------------------------------------
EN            42.15%         71.23%         +29.08% ✓
FR            41.87%         70.89%         +29.02% ✓
DE            42.34%         71.45%         +29.11% ✓
ES            41.76%         70.56%         +28.80% ✓
PT            42.02%         71.01%         +28.99% ✓

================================================================================
✓ Comparison complete!
================================================================================
```

## Metrics Explained

### Citation Accuracy
Verifies that the generated answer contains the correct CELEX number (e.g., `32016R0679`).

**Score**: 1.0 if correct CELEX present, 0.0 otherwise

### Article Accuracy
Checks if the generated answer references the correct article number(s).

**Score**: Proportion of correct articles (0.0 to 1.0)

### ROUGE-L
Measures answer similarity to reference answer using longest common subsequence.

**Score**: F1 score (0.0 to 1.0), higher is better

### Exact Match
Checks for exact match after normalization (lowercasing, punctuation removal).

**Score**: 1.0 if exact match, 0.0 otherwise

## Customization

### Custom Test Questions

To use your own test questions, create a JSONL file with this format:

```json
{
  "question": "Your question here?",
  "answer": "Expected answer with citation (CELEX: XXXXX)",
  "language": "en",
  "metadata": {
    "celex": "32016R0679",
    "article": "4",
    "type": "definition"
  }
}
```

### Custom Inference Parameters

Adjust generation quality and speed:

```bash
# More creative responses
python scripts/compare_models.py \
  --temperature 0.9 \
  --max_new_tokens 1024 \
  ...

# Faster inference (smaller batch)
python scripts/compare_models.py \
  --batch_size 4 \
  --max_new_tokens 256 \
  ...
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:

1. **Reduce batch size**:
   ```bash
   --batch_size 4
   ```

2. **Reduce max tokens**:
   ```bash
   --max_new_tokens 256
   ```

3. **Check GPU memory**:
   ```bash
   nvidia-smi
   ```

### Model Loading Errors

If models fail to load:

1. **Verify model paths exist**:
   ```bash
   ls -la ./checkpoints/sft/final
   ```

2. **Check HuggingFace cache**:
   ```bash
   ls -la ~/.cache/huggingface/hub
   ```

3. **Ensure sufficient disk space** (models are ~35GB each)

### Inference Failures

If individual samples fail to generate:
- Errors are logged but don't stop the comparison
- Failed samples are counted and reported
- Check `comparison_results.json` for detailed error tracking

## Advanced Usage

### Compare Multiple Model Checkpoints

```bash
# Compare different SFT checkpoints
for checkpoint in checkpoint-500 checkpoint-1000 checkpoint-1500; do
  python scripts/compare_models.py \
    --finetuned_model ./checkpoints/sft/$checkpoint \
    --test_dataset data/test/test_qna_100.jsonl \
    --output_dir results/$checkpoint
done
```

### Language-Specific Evaluation

To focus on specific languages, filter the test dataset:

```bash
# Create English-only test set
cat data/test/test_qna_100.jsonl | \
  grep '"language": "en"' > data/test/test_en_only.jsonl

# Run comparison
python scripts/compare_models.py \
  --test_dataset data/test/test_en_only.jsonl \
  --output_dir results/english_only
```

## Files Created

```
eur-lex-globalization-model/
├── data/test/
│   └── test_qna_100.jsonl              # Test dataset
├── results/model_comparison/
│   ├── comparison_results.json         # Detailed metrics
│   ├── comparison_report.md            # Human-readable report
│   └── predictions.csv                 # Prediction details
├── scripts/
│   ├── generate_test_set.py            # Test set generator
│   └── compare_models.py               # Model comparator
└── src/evaluation/
    └── report_generator.py             # Report generator
```

## Dependencies

All required libraries are already in `requirements.txt`:
- `torch>=2.1.0`
- `transformers>=4.36.0`
- `datasets>=2.16.0`
- `rouge-score>=0.1.2`
- `tqdm>=4.66.0`

No additional dependencies needed!

## Next Steps

1. **Generate test set**: Run `generate_test_set.py`
2. **Run comparison**: Execute `compare_models.py` with your model paths
3. **Review results**: Check the Markdown report and CSV for detailed analysis
4. **Share findings**: The Markdown report is GitHub-friendly and ready to share

---

*For questions or issues, refer to the main project README or check the script docstrings.*
