# EUR-Lex Legal Q&A Model: LLaMA 3.3 70B Fine-Tuning

Train LLaMA 3.3 70B on 25GB FORMEX XML data (EN/FR/DE/ES/PT) for legal Q&A with citations using CPT and SFT with 
FP8 quantization on Nvidia RTX Pro 6000 and NVFP4 quantization on B200 GPUs.

## Overview

- **Base Model**: LLaMA 3.3 70B Instruct
- **Data**: 25GB FORMEX XML from EUR-Lex (5 languages)
- **Hardware**: Mac Studio (data processing) + 4x RTX Pro 6000 (standard training) + 4x B200 (advanced training)
- **Training**: CPT (domain adaptation) → SFT (instruction-tuning)
- **Optimization**: FP8 (default) or NVFP4 quantization via Transformer Engine
- **Distributed**: PyTorch FSDP2 (NVIDIA's recommended stack for Blackwell)
- **Workflow**: 5-phase pipeline — data processing, CPT, SFT, evaluation, and advanced B200 training

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Precision Modes](#precision-modes)
- [Dataset Status](#dataset-status)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Scripts Reference](#scripts-reference)
- [Usage](#usage)
  - [Phase 1: Data Processing (Mac Studio)](#phase-1-data-processing-mac-studio)
  - [Phase 2: CPT Training (4x RTX Pro 6000)](#phase-2-cpt-training-4x-rtx-pro-6000)
  - [Phase 3: SFT Training (4x RTX Pro 6000)](#phase-3-sft-training-4x-rtx-pro-6000)
  - [Phase 4: Evaluation](#phase-4-evaluation)
  - [Phase 5: Advanced CPT and SFT Training (4x B200)](#phase-5-advanced-cpt-and-sft-training-4x-b200)
- [Model Comparison & QnA Testing](#model-comparison--qna-testing)
- [Preventing Catastrophic Forgetting](#preventing-catastrophic-forgetting)
- [Configuration](#configuration)
- [Memory Optimization](#memory-optimization)
- [Success Metrics & Expected Results](#success-metrics--expected-results)
- [Monitoring](#monitoring)
- [Citation Format](#citation-format)
- [Troubleshooting](#troubleshooting)
- [Testing & Verification](#testing--verification)
- [License](#license)
- [Support](#support)
- [Acknowledgments](#acknowledgments)

## Key Features

✅ **Multilingual**: EN (35%), FR (25%), DE (20%), ES (12%), PT (8%)

✅ **Citation-Aware**: Automatic CELEX reference extraction and validation

✅ **Balanced Training**: 5 epochs to prevent catastrophic forgetting

✅ **Document Packing**: Efficient 4096-token sequence packing

✅ **FP8 Quantization**: 50% memory reduction vs BF16, production-ready on RTX Pro 6000 and B200

✅ **FSDP2 + Transformer Engine**: NVIDIA's recommended distributed training stack for Blackwell GPUs

✅ **NVFP4 Quantization** *(Phase 5 — B200 only)*: Additional 50% memory reduction vs FP8 (75% vs BF16)

## Precision Modes

This project supports two precision modes via NVIDIA Transformer Engine on Blackwell B200 GPUs. Each component of the neural network training pipeline uses different precision formats optimized for performance and accuracy.

**Reference**: [NVIDIA FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

### Precision by Component

| Neural Network Component | FP8 Mode | NVFP4 Mode | Notes |
|--------------------------|----------|------------|-------|
| **Forward Pass** | | | |
| → Model Weights | FP8 | NVFP4 + block scales FP16 | Quantized for compute |
| → Activations | FP8 | NVFP4 + block scales FP8 | 16-element blocks for NVFP4 |
| → Attention Scores | BF16 | BF16 | High precision for stability |
| → Layer Norms | FP32 | FP32 | Critical for numerical stability |
| **Backward Pass** | | | |
| → Gradients (dL/dW) | FP8 | NVFP4 + block scales FP8 | Wider range for gradients |
| → Activation Gradients | FP8 | NVFP4 + block scales FP8 | 16-element blocks for NVFP4 |
| → Gradient Accumulation | BF16 | BF16 | Accumulated over 8-16 steps |
| **Optimizer** | | | |
| → Master Weights | BF16 | BF16 | High precision copy |
| → Momentum (1st moment) | FP32 | FP32 | AdamW optimizer state |
| → Variance (2nd moment) | FP32 | FP32 | AdamW optimizer state |
| → Weight Updates | BF16 | BF16 | Applied to master weights |
| **Communication (FSDP)** | | | |
| → All-Reduce (gradients) | BF16 | BF16 | Cross-GPU synchronization |
| → All-Gather (weights) | FP8 | NVFP4 | Gathered as needed |
| → Reduce-Scatter | BF16 | BF16 | Gradient reduction |
| **Memory Storage** | | | |
| → Sharded Weights | FP8 | NVFP4 | Per-GPU shard |
| → Optimizer States | FP32 | FP32 | Largest memory consumer |
| → Activations (checkpointed) | BF16 | BF16 | Recomputed in backward |
| **Checkpointing** | | | |
| → Saved Model Weights | BF16 | BF16 | Full precision for safety |
| → Optimizer States | FP32 | FP32 | For resume capability |
| **Inference** | | | |
| → Loaded Weights | BF16 or FP8 | BF16 or NVFP4 | User configurable |
| → Activations | BF16 | BF16 | No quantization needed |

### Format Specifications

**FP8 Formats**:
- **E4M3**: 1 sign + 4 exponent + 3 mantissa bits (range: ±448, precision: ~0.1%)
- **E5M2**: 1 sign + 5 exponent + 2 mantissa bits (range: ±57,344, precision: ~1%)
- **Scaling**: Dynamic per-tensor scaling with AMAX tracking (1024-step history)
- **Status**: Production-ready (Hopper H100+, Blackwell B200+)

**NVFP4 Format**:
- **E2M1**: 1 sign + 2 exponent + 1 mantissa bit (range: ±6, precision: ~12.5% before scaling)
- **Block Scaling**: 16-element (activations/gradients) or 16×16 (weights) blocks
- **Scale Precision**: FP8 E4M3 (activations) or FP16 (weights)
- **Status**: Experimental (B200/B100 data center GPUs only — **not available on RTX Pro 6000**)

### Memory Breakdown (LLaMA 3.3 70B, CPT Training)

| Component | FP8 Mode | NVFP4 Mode | Savings |
|-----------|----------|------------|---------|
| Model Weights (sharded) | ~18GB | ~9GB | **50%** |
| Optimizer States (FP32) | ~36GB | ~36GB | 0% |
| Activations + Gradients | ~12GB | ~6GB | **50%** |
| FSDP Buffers | ~4GB | ~4GB | 0% |
| **Total per GPU** | **~70GB** | **~35GB** | **50%** |

### Quick Comparison

| Feature | FP8 | NVFP4 |
|---------|-----|-------|
| **Memory per GPU (CPT)** | ~70GB | ~35GB (-50%) |
| **Memory per GPU (SFT)** | ~40GB | ~20GB (-50%) |
| **Training Speed** | 90K tok/s | 88K tok/s (-2%) |
| **Model Quality** | 100% | 95-98% |
| **Stability** | Production-ready | Experimental |
| **Hardware** | H100+, B200+, **RTX Pro 6000** | **B200/B100 only** (GB100 die) |
| **Used in phases** | **2, 3** (RTX Pro 6000) and 5 (B200) | **5 only** (B200) |

### Usage by Hardware

**Phases 2 & 3 — RTX Pro 6000 (FP8 only)**:
```bash
# FP8 is the only quantization mode on RTX Pro 6000
./scripts/run_training.sh both
```

**Phase 5 — B200 (FP8 or NVFP4)**:
```bash
# NVFP4 default — production-ready on B200, 50% memory savings
./scripts/run_training_b200.sh both

# FP8 on B200 — maximum quality
PRECISION=fp8 ./scripts/run_training_b200.sh both
```

**When to Use**:
- **FP8**: Phases 2 & 3 (required on RTX Pro 6000); Phase 5 for maximum quality
- **NVFP4**: Phase 5 only (B200/B100 required); use when memory headroom or 2-5% quality tradeoff is acceptable

**See `docs/PRECISION_MODES.md` for detailed comparison, performance benchmarks, and troubleshooting.**

## Dataset Status

**Current Corpus (2 languages processed):**
- Languages: 2 of 5 (awaiting additional language data)
- Train tokens: 178,588,076 (~179M)
- Validation tokens: 6,519,849 (~6.5M)
- Total sequences: 17,588 (16,663 train + 925 validation)
- Total documents: 78,952 (75,004 train + 3,948 validation)

**Projected Corpus (5 languages):**
- Languages: All 5 (EN, FR, DE, ES, PT)
- Train tokens: ~446,470,190 (~446M)
- Validation tokens: ~16,299,623 (~16M)
- Total sequences: ~43,971 (41,658 train + 2,313 validation)
- Total documents: ~197,380

**Training configured for 5-language projection** - configurations use the projected 446M tokens to calculate optimal epoch counts (2 epochs for Fast CPT, 5 epochs for Full CPT).

## Installation

This project uses a **three-environment setup**:
1. **Mac Studio**: Data processing (FORMEX parsing, corpus building) — Phase 1
2. **4x RTX Pro 6000** (96GB GDDR7): Standard CPT + SFT training — Phases 2 & 3
3. **4x B200** (192GB HBM3e): Advanced CPT + SFT with larger context — Phase 5

### Setup on Mac Studio (Data Processing)

```bash
./setup_mac.sh
```

This installs:
- PyTorch (CPU/MPS)
- Data processing libraries (lxml, pandas, datasets)
- Transformers and tokenizers
- No CUDA dependencies (macOS doesn't support CUDA)

### Setup on GPU Cluster (Training)

```bash
./setup_gpu.sh
```

This installs:
- PyTorch with CUDA 12.1+ support
- NVIDIA Transformer Engine (FP8/NVFP4 quantization support)
- PyTorch FSDP2 (built into PyTorch 2.1+)
- Flash Attention 2
- All training dependencies

### Auto-Detection Setup (Recommended)

```bash
./setup.sh
```

Automatically detects your environment and runs the appropriate setup script.

## Project Structure

```
eur-lex-globalization-model/
├── configs/                     # Training configurations
├── data/                        # Data directory
│   ├── raw/                    # 25GB FORMEX XML files
│   ├── parsed/                 # Parsed documents
│   ├── cpt/                    # CPT corpus (32 shards)
│   └── sft/                    # SFT Q&A dataset (8 shards)
├── data_processing/             # Data processing pipeline
│   ├── parsers/                # FORMEX XML parser
│   ├── dataset_builders/       # CPT/SFT corpus builders
│   └── validators/             # Data quality validation
├── scripts/                     # Training scripts
├── src/                         # Source code
│   ├── training/               # Training logic
│   ├── evaluation/             # Evaluation metrics
│   └── utils/                  # Utilities
└── requirements.txt             # Dependencies
```

## Module Reference

### Core Source Modules (`/src/`)

#### Training Modules (`/src/training/`)

**`fsdp_config.py`**
- Configures PyTorch FSDP2 (Fully Sharded Data Parallel) for distributed training
- Sets up mixed precision policies (BF16 for parameters, FP16/FP8 for computation)
- Handles both FP8 and NVFP4 quantization modes via Transformer Engine
- Auto-wrap policy for LlamaDecoderLayer modules
- Configures sharding strategy (FULL_SHARD = ZeRO-3 equivalent)
- Activation checkpointing and backward prefetch optimization

**`data_collators.py`**
- `DataCollatorForCPT`: Standard language modeling collator for continued pretraining
  - Pads sequences to max length with left padding
  - Creates attention masks and labels for causal LM
- `DataCollatorForSFT`: Input masking collator for supervised fine-tuning
  - Masks instruction tokens (loss only computed on assistant responses)
  - Handles multi-turn conversations with proper masking
  - Preserves answer tokens for gradient computation

#### Evaluation Modules (`/src/evaluation/`)

**`metrics.py`**
- `LegalQAMetrics`: Comprehensive metrics for legal Q&A evaluation
  - **Citation Accuracy**: Extracts and validates CELEX numbers (e.g., `32016R0679`)
  - **Article Accuracy**: Validates article references (e.g., `Article 5`)
  - **ROUGE-L**: Measures answer quality and similarity to reference
  - **Exact Match**: Binary metric for perfect answer matching
  - **Hallucination Detection**: Identifies incorrect or fabricated citations
- Uses regex patterns for CELEX (`\d{4}[A-Z]\d{4}`) and article (`Article\s+\d+`) extraction
- Per-language metric tracking (EN, FR, DE, ES, PT)

**`report_generator.py`**
- Generates human-readable evaluation reports in Markdown format
- Creates comparison tables with delta calculations
- Produces win/loss/tie statistics for model comparisons
- Exports detailed prediction-level results to CSV

#### Utility Modules (`/src/utils/`)

**`fp8_utils.py`**
- FP8 and NVFP4 quantization utilities via NVIDIA Transformer Engine
- **FP8 Support**:
  - E4M3 format for forward pass (1 sign + 4 exp + 3 mantissa)
  - E5M2 format for backward pass (1 sign + 5 exp + 2 mantissa)
  - Dynamic per-tensor scaling with AMAX tracking
- **NVFP4 Support**:
  - E2M1 format with block scaling (1 sign + 2 exp + 1 mantissa)
  - 16-element blocks for activations/gradients
  - 16×16 blocks for weights
- Scaling factor management and numerical stability utilities

**`checkpoint_utils.py`**
- Checkpoint saving and loading for distributed training
- Model conversion utilities:
  - DeepSpeed checkpoint → FSDP checkpoint
  - FSDP checkpoint → DeepSpeed checkpoint
  - Handles state dict resharding and parameter redistribution
- Resume training from partial checkpoints
- Checkpoint consolidation for multi-GPU training

### Data Processing Modules (`/data_processing/`)

#### Parsers (`/data_processing/parsers/`)

**`formex_parser.py`**
- Parses FORMEX XML documents from EUR-Lex corpus
- **Data Structures**:
  - `DocumentMetadata`: CELEX number, language, document type, publication date, title
  - `Article`: Article ID, title, text content, paragraph list
  - `ParsedDocument`: Complete document with metadata, articles, and full text
- Extracts structured legal content from XML
- Handles multilingual documents (EN, FR, DE, ES, PT)
- Validates and normalizes CELEX references

#### Dataset Builders (`/data_processing/dataset_builders/`)

**`cpt_corpus_builder.py`**
- Builds Continued Pretraining corpus from parsed documents
- **Features**:
  - Document packing into 4096-token sequences for efficient training
  - Language distribution: EN 35%, FR 25%, DE 20%, ES 12%, PT 8%
  - Creates 32 training shards for distributed data loading
  - 95/5 train/validation split
  - Generates corpus statistics (token counts, sequence lengths, document counts)
- **Output**: Parquet files with columns `input_ids`, `attention_mask`, `labels`
- Tokenizes using LLaMA tokenizer with proper padding and truncation

**`sft_dataset_builder.py`**
- Builds Supervised Fine-Tuning dataset from Q&A pairs
- **Features**:
  - Converts documents into instruction-following Q&A format
  - Includes CELEX citations and article references in answers
  - Creates 8 training shards for efficient loading
  - Validates citation correctness (>95% accuracy)
  - Ensures multilingual balance across all languages
- **Output**: JSONL files with `question`, `answer`, `language`, `metadata` fields
- Supports conversation format with system/user/assistant roles

#### Validators (`/data_processing/validators/`)

**Data quality validation modules**:
- Citation validator: Verifies CELEX format and existence
- Language detector: Confirms document language matches metadata
- Completeness checker: Ensures all required fields are present
- Duplicate detector: Identifies and removes duplicate documents

## Scripts Reference

### Training & Orchestration Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **`scripts/train_cpt.py`** | Continued Pretraining script | - Trains LLaMA 3.3 70B on EUR-Lex corpus<br>- Supports FSDP2 and DeepSpeed backends<br>- FP8/NVFP4 quantization support<br>- 4,260 steps (5 epochs) for balanced learning<br>- Distributed training across 4x B200 GPUs |
| **`scripts/train_sft.py`** | Supervised Fine-Tuning script | - Instruction fine-tuning on Q&A pairs<br>- Input masking (loss only on responses)<br>- 3 epochs with early stopping<br>- Same distributed training support as CPT |
| **`scripts/run_training.sh`** | Phases 2 & 3 training orchestrator (RTX Pro 6000) | - Launches CPT, SFT, or both phases on 4x RTX Pro 6000<br>- Supports FP8/NVFP4 via `PRECISION` env var<br>- FSDP2 default; DeepSpeed via `USE_FSDP=false`<br>- Usage: `./scripts/run_training.sh [cpt\|sft\|both]` |
| **`scripts/run_training_b200.sh`** | Phase 5 advanced training orchestrator (B200) | - Advanced CPT + SFT on 4x B200 (192GB HBM3e)<br>- 8192-token CPT / 4096-token SFT sequences<br>- NVFP4 default (production-ready on B200)<br>- Warm-start from Phase 2/3 checkpoints<br>- Usage: `./scripts/run_training_b200.sh [cpt\|sft\|both]` |
| **`scripts/run_fast_cpt_training.sh`** | Fast CPT training variant (Phase 2) | - 4-6 hour training time on RTX Pro 6000<br>- 2 epochs through 446M tokens<br>- 3072-token sequences (vs 4096)<br>- Ideal for rapid prototyping |
| **`scripts/run_full_pipeline.sh`** | Full data processing pipeline | - Parses XML, builds CPT corpus, builds SFT dataset<br>- Maintained for batch data processing<br>- Use before Phases 2, 3, or 5 |

### Data Processing Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **`scripts/generate_sample_data.py`** | Generates sample EUR-Lex documents | - Creates synthetic FORMEX XML for testing<br>- Useful for pipeline validation without full corpus<br>- Generates 100+ sample documents across 5 languages |
| **`scripts/generate_test_set.py`** | Creates test Q&A pairs | - Generates 100+ Q&A pairs with citations<br>- 20 questions per language (EN, FR, DE, ES, PT)<br>- Includes definition, compliance, requirement questions<br>- Ground truth answers with proper CELEX citations<br>- Output: `data/test/test_qna_100.jsonl` |
| **`scripts/create_fast_cpt_corpus.py`** | Builds CPT corpus for fast training | - Creates smaller corpus for rapid iteration<br>- 3072-token sequences (vs 4096)<br>- Fewer shards for faster loading<br>- Same format as standard CPT corpus |

### Evaluation Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **`scripts/evaluate_model.py`** | Evaluates trained model | - Computes citation accuracy, ROUGE-L, exact match<br>- Per-language performance breakdown<br>- Hallucination detection<br>- Generates evaluation report (JSON + Markdown)<br>- Usage: `python scripts/evaluate_model.py --model_path <path> --eval_dataset <path>` |
| **`scripts/compare_models.py`** | Compares base vs fine-tuned models | - Runs both models on same test questions<br>- Detailed metric comparison with deltas<br>- Win/loss/tie analysis<br>- Exports predictions to CSV for error analysis<br>- Sequential model loading for memory safety<br>- Usage: `python scripts/compare_models.py --base_model <path> --finetuned_model <path> --test_dataset <path>` |

### Setup Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **`setup.sh`** | Auto-detection setup | - Detects Mac vs Linux environment<br>- Calls appropriate setup script<br>- Recommended for first-time setup |
| **`setup_mac.sh`** | Mac Studio setup | - PyTorch with CPU/MPS support<br>- Data processing libraries (lxml, pandas, datasets)<br>- No CUDA dependencies (macOS incompatible) |
| **`setup_gpu.sh`** | GPU cluster setup (RTX Pro 6000 or B200) | - PyTorch with CUDA 12.8+ support<br>- NVIDIA Transformer Engine (FP8/NVFP4)<br>- Flash Attention 2<br>- All training dependencies<br>- Works on both RTX Pro 6000 and B200 |

### Testing & Validation Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **`scripts/test_pipeline_sample.sh`** | Quick pipeline test | - Tests full pipeline on sample data<br>- Validates data processing → training → evaluation<br>- ~10 minute runtime<br>- Useful for CI/CD and development |

## Usage

The pipeline is organized into 5 phases across 3 hardware environments:

| Phase | Environment | Script | Est. Time |
|-------|-------------|--------|-----------|
| **1** Data Processing | Mac Studio | `run_full_pipeline.sh` | 4-6h |
| **2** CPT Training | 4x RTX Pro 6000 | `run_training.sh cpt` | 12-18h |
| **3** SFT Training | 4x RTX Pro 6000 | `run_training.sh sft` | 6-8h |
| **4** Evaluation | Any (GPU recommended) | `evaluate_model.py` | ~15 min |
| **5** Advanced CPT + SFT | 4x B200 | `run_training_b200.sh both` | 5-7h |

Phases 1–4 form the standard pipeline. Phase 5 is optional and produces a higher-quality model using larger context windows and more epochs on B200 hardware.

---

### Phase 1: Data Processing (Mac Studio)

**Location**: Mac Studio

#### 1.1 Quick pipeline test (recommended first)

```bash
# Test all components on synthetic data (~10 min)
./scripts/test_pipeline_sample.sh
```

#### 1.2 Parse FORMEX XML

```bash
python data_processing/parsers/formex_parser.py \
  --input data/raw \
  --output data/parsed \
  --workers 24
```

**What This Does**:
- Parses 25GB of FORMEX XML documents from EUR-Lex corpus
- Extracts structured content (CELEX numbers, article text, metadata)
- Validates CELEX references and language codes
- Handles 5 languages: EN, FR, DE, ES, PT
- Output: JSON files in `data/parsed/` (one per document)

**Expected Results**:
- Successfully parsed: >95% of documents
- Processing time: 4-6 hours with 24 workers
- Output size: ~8-10GB parsed JSON

#### 1.3 Build CPT Corpus

```bash
python data_processing/dataset_builders/cpt_corpus_builder.py \
  --input_dir data/parsed \
  --output_dir data/cpt \
  --max_seq_length 4096 \
  --num_shards 32
```

**What This Does**:
- Packs parsed documents into 4096-token sequences (>90% packing efficiency)
- Balanced language distribution: EN 35%, FR 25%, DE 20%, ES 12%, PT 8%
- Output: Parquet files with `input_ids`, `attention_mask`, `labels`
- 32 training shards + 1 validation shard (95/5 split)
- ~446M tokens, ~41,658 training sequences (5-language projection)

#### 1.4 Build SFT Dataset

```bash
python data_processing/dataset_builders/sft_dataset_builder.py \
  --input_dir data/parsed \
  --output_dir data/sft \
  --target_pairs 150000 \
  --num_shards 8
```

#### 1.5 Transfer Data to GPU Cluster

```bash
# Transfer to RTX Pro 6000 cluster (Phases 2 & 3)
rsync -avz --progress data/cpt/ user@rtx-cluster:/path/to/project/data/cpt/
rsync -avz --progress data/sft/ user@rtx-cluster:/path/to/project/data/sft/

# Or compress and scp
tar -czf cpt_data.tar.gz data/cpt/ && scp cpt_data.tar.gz user@rtx-cluster:/path/to/project/
tar -czf sft_data.tar.gz data/sft/ && scp sft_data.tar.gz user@rtx-cluster:/path/to/project/

# The same data is reused for Phase 5 on B200 — transfer once, use for both
```

---

### Phase 2: CPT Training (4x RTX Pro 6000)

**Location**: 4x RTX Pro 6000 (96GB GDDR7 each)

```bash
# FSDP2 + FP8 (only supported quantization mode on RTX Pro 6000)
./scripts/run_training.sh cpt
```

Or run directly:
```bash
torchrun --nproc_per_node=4 scripts/train_cpt.py \
  --config configs/cpt_config.yaml \
  --fsdp_config configs/fsdp_config.json \
  --use_fp8 --precision fp8
```

| | FP8 |
|--|-----|
| **Memory/GPU** | ~70GB (73% of 96GB GDDR7) |
| **Training time** | 12-18 hours |
| **Steps** | 4,260 (5 epochs, 446M tokens) |
| **Quality** | 100% |

> **Note**: NVFP4 is **not supported** on RTX Pro 6000 (GB202 die). NVFP4 requires the B200/B100 data center GPU (GB100 die). Use `run_training_b200.sh` for NVFP4.

**Note on Training Duration**: `max_steps: 4260` = 5 epochs through 446M tokens. This is the recommended balance to prevent catastrophic forgetting (see `docs/PREVENTING_CATASTROPHIC_FORGETTING.md`).

#### Fast CPT variant (4-6 hours on RTX Pro 6000)

```bash
# 2 epochs, 3072-token sequences, ~75-80% quality of full CPT
./scripts/run_fast_cpt_training.sh

# With NVFP4 (recommended on RTX Pro 6000)
PRECISION=nvfp4 ./scripts/run_fast_cpt_training.sh
```

Use fast CPT for rapid prototyping before committing to full training.
See `docs/FAST_TRAINING_OPTIONS.md` for a comparison of 7 approaches.

**Legacy DeepSpeed**: `USE_FSDP=false ./scripts/run_training.sh cpt`

---

### Phase 3: SFT Training (4x RTX Pro 6000)

**Location**: 4x RTX Pro 6000 (96GB GDDR7 each)

```bash
# FSDP2 + FP8 (only supported quantization mode on RTX Pro 6000)
./scripts/run_training.sh sft

# Run both Phase 2 + Phase 3 back-to-back
./scripts/run_training.sh both
```

Or run directly:
```bash
torchrun --nproc_per_node=4 scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --fsdp_config configs/fsdp_config.json \
  --use_fp8 --precision fp8
```

| | FP8 |
|--|-----|
| **Memory/GPU** | ~40GB (42% of 96GB GDDR7) |
| **Training time** | 6-8 hours |
| **Epochs** | 3 |
| **Seq length** | 2048 tokens |

> **Note**: NVFP4 is **not supported** on RTX Pro 6000. Use `run_training_b200.sh` for NVFP4 (Phase 5, B200).

SFT loads from the Phase 2 CPT checkpoint (`checkpoints/cpt/final`). Ensure Phase 2 has completed before running Phase 3.

**Legacy DeepSpeed**: `USE_FSDP=false ./scripts/run_training.sh sft`

---

### Phase 4: Evaluation

**Location**: Any machine with GPU access (or CPU for small test sets)

```bash
# Evaluate Phase 3 SFT model
python scripts/evaluate_model.py \
  --model_path checkpoints/sft/final \
  --eval_dataset data/sft/validation/sft_val.jsonl \
  --output_file results/evaluation_report.json

# Evaluate Phase 5 advanced model (if available)
python scripts/evaluate_model.py \
  --model_path checkpoints/sft_b200/final \
  --eval_dataset data/sft/validation/sft_val.jsonl \
  --output_file results/evaluation_report_b200.json
```

**Compare base model vs fine-tuned**:
```bash
python scripts/generate_test_set.py \
  --output_file data/test/test_qna_100.jsonl \
  --questions_per_language 20

python scripts/compare_models.py \
  --base_model meta-llama/Llama-3.3-70B-Instruct \
  --finetuned_model checkpoints/sft/final \
  --test_dataset data/test/test_qna_100.jsonl \
  --output_dir results/model_comparison
```

See [Model Comparison & QnA Testing](#model-comparison--qna-testing) for full details.

---

### Phase 5: Advanced CPT and SFT Training (4x B200)

**Location**: 4x NVIDIA B200 (192GB HBM3e each)

Phase 5 leverages B200's larger memory and bandwidth for:
- **Longer context**: 8192-token CPT (vs 4096) and 4096-token SFT (vs 2048)
- **More training**: 10 CPT epochs (vs 5) + 5 SFT epochs (vs 3)
- **NVFP4 as default**: Production-ready on B200 (vs experimental on RTX Pro 6000)
- **Warm start**: Optionally continues from Phase 2/3 checkpoints

```bash
# Run both advanced CPT + SFT (NVFP4 default, ~5-7 hours total)
./scripts/run_training_b200.sh both

# Advanced CPT only
./scripts/run_training_b200.sh cpt

# Advanced SFT only (requires CPT checkpoint)
./scripts/run_training_b200.sh sft

# Warm-start from Phase 2 CPT checkpoint
CPT_WARMSTART=checkpoints/cpt/final ./scripts/run_training_b200.sh both

# FP8 instead of NVFP4
PRECISION=fp8 ./scripts/run_training_b200.sh both
```

Or run directly:
```bash
# Advanced CPT: 8192-token sequences, 10 epochs
torchrun --nproc_per_node=4 scripts/train_cpt.py \
  --config configs/cpt_config_b200.yaml \
  --fsdp_config configs/fsdp_config.json \
  --use_fp8 --precision nvfp4

# Advanced SFT: 4096-token sequences, 5 epochs
torchrun --nproc_per_node=4 scripts/train_sft.py \
  --config configs/sft_config_b200.yaml \
  --fsdp_config configs/fsdp_config.json \
  --use_fp8 --precision nvfp4
```

| | Advanced CPT (B200) | Advanced SFT (B200) |
|--|---------------------|---------------------|
| **Config** | `cpt_config_b200.yaml` | `sft_config_b200.yaml` |
| **Memory/GPU (NVFP4)** | ~50GB (26% of 192GB) | ~30GB (16% of 192GB) |
| **Memory/GPU (FP8)** | ~90GB (47% of 192GB) | ~55GB (29% of 192GB) |
| **Seq length** | 8192 tokens | 4096 tokens |
| **Training** | 8,520 steps (10 epochs) | 5 epochs |
| **Est. time** | 3-4 hours | 2-3 hours |
| **Script** | `run_training_b200.sh cpt` | `run_training_b200.sh sft` |

**Outputs**:
- CPT checkpoint: `checkpoints/cpt_b200/final`
- SFT checkpoint: `checkpoints/sft_b200/final`

**Compare RTX vs B200 results**:
```bash
python scripts/compare_models.py \
  --base_model checkpoints/sft/final \
  --finetuned_model checkpoints/sft_b200/final \
  --test_dataset data/test/test_qna_100.jsonl \
  --output_dir results/rtx_vs_b200
```

## Model Comparison & QnA Testing

Compare base LLaMA 3.3 70B vs fine-tuned model performance on legal Q&A tasks. This system generates test questions, runs both models, and produces detailed comparison reports.

### Quick Start

**Step 1: Generate test set** (100 Q&A pairs, 20 per language):
```bash
python scripts/generate_test_set.py \
  --output_file data/test/test_qna_100.jsonl \
  --questions_per_language 20
```

**Step 2: Run comparison** (~5 minutes):
```bash
python scripts/compare_models.py \
  --base_model meta-llama/Llama-3.3-70B-Instruct \
  --finetuned_model ./checkpoints/sft/final \
  --test_dataset data/test/test_qna_100.jsonl \
  --output_dir results/model_comparison \
  --batch_size 8
```

### What Gets Measured

**Metrics**:
- **Citation Accuracy**: CELEX number correctness (e.g., `32016R0679`)
- **Article Accuracy**: Article reference correctness (e.g., `Article 5`)
- **ROUGE-L**: Answer quality and similarity to reference
- **Exact Match**: Perfect answer matching
- **Win/Loss/Tie**: Sample-level comparison statistics

**Per-Language Breakdown**: All metrics tracked separately for EN, FR, DE, ES, PT

### Output Files

1. **`comparison_results.json`**: Detailed metrics, deltas, and sample comparisons (~150KB)
2. **`comparison_report.md`**: Human-readable report with tables and examples (~10KB)
3. **`predictions.csv`**: Prediction-level details for error analysis (~200KB)

### Example Output

```
COMPARISON SUMMARY
================================================================================
Samples evaluated: 100

OVERALL PERFORMANCE
--------------------------------------------------------------------------------
Metric               Base       Fine-tuned   Delta      Rel. Imp.
--------------------------------------------------------------------------------
Citation Accuracy    23.00%     87.00%       +64.00%    +278.3% ✓
Article Accuracy     31.00%     82.00%       +51.00%    +164.5% ✓
ROUGE-L              42.00%     71.00%       +29.00%    +69.0% ✓
Exact Match          15.00%     34.00%       +19.00%    +126.7% ✓

WIN/LOSS/TIE ANALYSIS
--------------------------------------------------------------------------------
Fine-tuned wins: 93
Base wins:       4
Ties:            3
Win rate:        93.0%
```

### Test Set Contents

The generated test set includes realistic GDPR/RGPD questions:
- **Definition questions**: "What is 'personal data' according to GDPR?"
- **Compliance questions**: "What obligations does Article 33 impose?"
- **Requirement questions**: "What is required under Article 6?"
- **Scope questions**: "What is the territorial scope of GDPR?"
- **Citation questions**: "What does CELEX 32016R0679 refer to?"

All questions include proper ground truth answers with CELEX citations.

### Memory & Performance

- **Runtime**: ~5 minutes for 100 samples
  - Base model inference: ~2.5 minutes
  - Fine-tuned model inference: ~2.5 minutes
  - Report generation: ~10 seconds
- **Memory**: ~55GB peak GPU (safe with 4x 80GB B200 GPUs)
- **Sequential Loading**: Models loaded one at a time for memory safety

### Advanced Usage

**Custom test questions**:
```bash
# Create your own JSONL file with format:
{"question": "...", "answer": "...", "language": "en", "metadata": {...}}
```

**Adjust inference parameters**:
```bash
python scripts/compare_models.py \
  --batch_size 4 \
  --max_new_tokens 256 \
  --temperature 0.5 \
  ...
```

**Compare multiple checkpoints**:
```bash
for checkpoint in checkpoint-500 checkpoint-1000 checkpoint-1500; do
  python scripts/compare_models.py \
    --finetuned_model ./checkpoints/sft/$checkpoint \
    --test_dataset data/test/test_qna_100.jsonl \
    --output_dir results/$checkpoint
done
```

### Documentation

**Model Comparison & Evaluation:**
- **`docs/COMPARISON_GUIDE.md`**: Complete guide to comparing base vs fine-tuned models
  - Detailed usage instructions
  - Troubleshooting guide
  - Customization options
  - Metrics explanations
  - Example outputs

**Training Best Practices:**
- **`docs/PREVENTING_CATASTROPHIC_FORGETTING.md`**: Comprehensive guide to preventing catastrophic forgetting
  - Parameter tuning strategies
  - Configuration profiles (Conservative/Balanced/Aggressive)
  - Monitoring and evaluation protocols
  - Advanced anti-forgetting techniques
  - Recovery actions if forgetting occurs

## Preventing Catastrophic Forgetting

During continued pretraining, it's critical to prevent **catastrophic forgetting** - where the model loses general capabilities while learning domain-specific knowledge. Our training configurations are carefully balanced to achieve strong legal domain adaptation while preserving >90% of the base model's general capabilities.

**See detailed guide**: **[docs/PREVENTING_CATASTROPHIC_FORGETTING.md](docs/PREVENTING_CATASTROPHIC_FORGETTING.md)**

### Key Strategies

1. **Balanced Epochs**: 5 epochs (vs 47 in original config) prevents over-training
2. **Conservative Learning Rate**: 2e-5 (6.7% of pretraining LR)
3. **Weight Regularization**: 0.01 weight decay keeps weights close to pretrained values
4. **Gradient Clipping**: Max norm 1.0 prevents destabilizing updates
5. **Warmup Schedule**: 10% warmup with cosine decay for stable training

### Configuration Profiles Available

| Profile | Epochs | LR | Use Case | General Capability | Cost |
|---------|--------|----|---------|--------------------|------|
| Conservative | 4 | 1.5e-5 | Critical applications | >95% preserved | ~$65 |
| **Balanced (Current)** | **5** | **2.0e-5** | **Production use** | **>90% preserved** | **~$85** |
| Aggressive | 6 | 3.0e-5 | Maximum adaptation | ~85% preserved | ~$100 |

**For detailed parameter tuning, monitoring strategies, and advanced anti-forgetting techniques, see [docs/PREVENTING_CATASTROPHIC_FORGETTING.md](docs/PREVENTING_CATASTROPHIC_FORGETTING.md).**

---

## Configuration

### CPT Training Configurations

**Phase 2 — RTX Pro 6000** (`configs/cpt_config.yaml`):
- Sequence Length: 4096 tokens
- Batch Size: 2 per GPU × 4 GPUs × 16 grad_accum = 128 global
- Learning Rate: 2e-5 with cosine schedule
- **Training Steps: 4,260** (5 epochs through 446M tokens)
- Warmup Steps: 426 (10% of max_steps)
- Precision: `fp8` only (NVFP4 not supported on RTX Pro 6000)

**Phase 5 — B200 Advanced** (`configs/cpt_config_b200.yaml`):
- Sequence Length: **8192 tokens** (2x RTX Pro 6000)
- Batch Size: **4 per GPU** × 4 GPUs × 8 grad_accum = 128 global
- Learning Rate: **1e-5** (lower — refined continued pretraining)
- **Training Steps: 8,520** (10 epochs — deeper domain adaptation)
- Precision default: `nvfp4` (production-ready on B200)

**Why 5 epochs for Phase 2?** Balanced to prevent catastrophic forgetting — see `docs/PREVENTING_CATASTROPHIC_FORGETTING.md`.

### SFT Training Configurations

**Phase 3 — RTX Pro 6000** (`configs/sft_config.yaml`):
- Sequence Length: 2048 tokens
- Batch Size: 4 per GPU × 4 GPUs × 8 grad_accum = 128 global
- Learning Rate: 5e-6 with cosine schedule
- **Epochs: 3** (standard instruction fine-tuning)
- Precision: `fp8` only (NVFP4 not supported on RTX Pro 6000)

**Phase 5 — B200 Advanced** (`configs/sft_config_b200.yaml`):
- Sequence Length: **4096 tokens** (2x Phase 3)
- Batch Size: **8 per GPU** × 4 GPUs × 4 grad_accum = 128 global
- Learning Rate: **2e-6** (lower — refined SFT from stronger CPT base)
- **Epochs: 5**
- Precision default: `nvfp4`

### FSDP2 Configuration

PyTorch FSDP2 with Transformer Engine:
- **Sharding Strategy**: FULL_SHARD (equivalent to DeepSpeed ZeRO-3)
- **Precision Modes**:
  - **FP8 (default)**: E4M3 for forward pass, E5M2 for backward pass
  - **NVFP4 (experimental)**: E2M1 with block scaling for 50% memory savings
- **Mixed Precision**: BF16 for non-quantized operations
- **Memory**: No CPU offload needed (fits in GPU memory with quantization)
- **Activation Checkpointing**: Applied after FSDP wrapping
- **Communication Overlap**: Enabled via backward prefetch
- **Transformer Engine**: Automatic quantization of forward/backward passes

**NVIDIA's Recommended Stack for Blackwell**: This configuration (FSDP2 + Transformer Engine) is used in NeMo 2.0 and Megatron-Core for optimal performance on both RTX Pro 6000 and B200 GPUs.

**Legacy DeepSpeed Support**: DeepSpeed ZeRO-3 configs are maintained for backward compatibility. See `configs/ds_config_zero3.json`.

## Memory Optimization

### FP8 Mode (Default)

| Technique | Memory Saved | Speed Impact |
|-----------|--------------|--------------|
| FSDP2 FULL_SHARD | 210GB (3x sharding) | Minimal |
| FP8 Quantization | 70GB (50% reduction) | Similar to BF16 |
| Gradient Checkpointing | 50GB | -20% slower |
| Flash Attention 2 | 20GB | +2x faster |
| **Total** | **~280GB saved** | **Net: +40% faster** |

**Result**: 70B model fits in ~70GB per GPU for CPT, ~40GB for SFT

### NVFP4 Mode (Experimental)

| Technique | Memory Saved | Speed Impact |
|-----------|--------------|--------------|
| FSDP2 FULL_SHARD | 210GB (3x sharding) | Minimal |
| NVFP4 Quantization | 105GB (75% reduction) | ±5% vs FP8 |
| Gradient Checkpointing | 50GB | -20% slower |
| Flash Attention 2 | 20GB | +2x faster |
| **Total** | **~315GB saved** | **Net: +35% faster** |

**Result**: 70B model fits in ~35GB per GPU for CPT, ~20GB for SFT (50% savings vs FP8)

### Comparison

| Mode | CPT Memory/GPU | SFT Memory/GPU | Savings vs BF16 | Quality | Hardware |
|------|----------------|----------------|-----------------|---------|----------|
| BF16 (baseline) | ~140GB | ~80GB | 0% | 100% | All |
| **FP8** | **~70GB** | **~40GB** | **50%** | **100%** | All (H100+, RTX Pro 6000, B200) |
| NVFP4 | ~35GB | ~20GB | 75% | 95-98% | **B200/B100 only** |

**FSDP2 vs DeepSpeed**: FSDP2 provides equivalent or better performance with native PyTorch integration, better debugging tools, and improved communication overlap on Blackwell GPUs.

## Success Metrics & Expected Results

### Data Processing Results (Mac Studio)

**After XML Parsing:**
- ✓ Successfully parsed documents: >95%
- ✓ Average document size: 25-30KB
- ✓ Languages detected: EN, FR, DE, ES, PT
- ✓ CELEX numbers extracted: >90% of documents
- ✓ Processing time: 4-6 hours for 25GB

**After CPT Corpus Building (Current: 2 languages, Projected: 5 languages):**
- ✓ Total corpus size: ~5GB (projected for 5 languages)
- ✓ Training shards: 32 files
- ✓ Total tokens: ~446 million tokens (178.6M for current 2 languages)
- ✓ Training sequences: 16,663 (current), ~41,658 (projected for 5 languages)
- ✓ Validation sequences: 925 (current), ~2,313 (projected for 5 languages)
- ✓ Average sequence length: 10,718 tokens (train), 7,048 tokens (validation)
- ✓ Total documents: 75,004 train + 3,948 validation (current 2 languages)
- ✓ Document packing efficiency: >90%
- ✓ Language distribution (projected): EN 35%, FR 25%, DE 20%, ES 12%, PT 8%

**After SFT Dataset Building:**
- ✓ Total Q&A pairs: 150,000
- ✓ Training shards: 8 files
- ✓ Citation coverage: >90% of answers
- ✓ Average question length: 50-100 tokens
- ✓ Average answer length: 150-300 tokens
- ✓ Valid CELEX references: >95%
- ✓ Multilingual balance: All languages represented

### Training Performance

**Phase 2: CPT (4x RTX Pro 6000) — FP8 only**

| Metric | FP8 |
|--------|-----|
| Training time | 12-18 hours |
| Memory/GPU | ~70GB (73% of 96GB GDDR7) |
| Steps | 4,260 (5 epochs) |
| Throughput | ~20-30K tokens/sec |
| Quality | 100% |

> NVFP4 is not available on RTX Pro 6000 (GB202). FP8 is the only quantization mode.

**Phase 3: SFT (4x RTX Pro 6000) — FP8 only**

| Metric | FP8 |
|--------|-----|
| Training time | 6-8 hours |
| Memory/GPU | ~40GB (42% of 96GB GDDR7) |
| Epochs | 3 |
| Quality | 100% |

**Phase 5: Advanced CPT + SFT (4x B200)**

| Metric | Advanced CPT | Advanced SFT |
|--------|-------------|-------------|
| Training time | 3-4 hours | 2-3 hours |
| Memory/GPU (NVFP4) | ~50GB (26% of 192GB) | ~30GB (16% of 192GB) |
| Seq length | 8192 tokens | 4096 tokens |
| Training | 8,520 steps (10 epochs) | 5 epochs |
| Throughput | ~80-100K tokens/sec | ~150-180K tokens/sec |

### Model Quality Metrics

**CPT Success Criteria:**
- ✓ Legal domain perplexity: <15 (vs. base model ~25)
- ✓ General capabilities maintained: >90% on MMLU
- ✓ Legal terminology understanding: Significant improvement
- ✓ Cross-lingual consistency: Maintained across languages
- ✓ Training stability: No loss spikes or divergence

**SFT Success Criteria:**
- ✓ Citation accuracy: >85% (CELEX numbers correct)
- ✓ Article accuracy: >80% (Article numbers correct)
- ✓ ROUGE-L score: >0.6 (answer similarity)
- ✓ Exact match: >30% for factual questions
- ✓ Hallucination rate: <5% (incorrect citations)
- ✓ Instruction following: >90% adherence to prompts

**Language-Specific Performance:**
- ✓ English (EN): Baseline performance (100%)
- ✓ French (FR): Within 10% of EN
- ✓ German (DE): Within 10% of EN
- ✓ Spanish (ES): Within 10% of EN
- ✓ Portuguese (PT): Within 10% of EN
- ✓ Cross-lingual queries: Properly handled

**Citation Quality:**
- ✓ CELEX format accuracy: >95%
- ✓ Article reference accuracy: >85%
- ✓ Document type accuracy: >90%
- ✓ No hallucinated citations: <5% rate
- ✓ Proper citation placement: Consistent formatting

### System Performance

**Memory Efficiency (FP8 mode):**
- ✓ Model weights: 35-70GB (vs. 140GB in BF16)
- ✓ Total memory per GPU (CPT): ~70GB (vs. 140GB in BF16)
- ✓ Total memory per GPU (SFT): ~40GB (vs. 80GB in BF16)
- ✓ Peak memory usage: <75GB per GPU
- ✓ Memory headroom: 5-10GB available
- ✓ No CPU offload needed
- ✓ Stable memory profile throughout training

**Memory Efficiency (NVFP4 mode):**
- ✓ Model weights: 17.5-35GB (75% reduction vs BF16)
- ✓ Total memory per GPU (CPT): ~35GB (50% savings vs FP8)
- ✓ Total memory per GPU (SFT): ~20GB (50% savings vs FP8)
- ✓ Peak memory usage: <40GB per GPU
- ✓ Memory headroom: 40-45GB available
- ✓ Quality tradeoff: 95-98% of FP8 performance

**Training Speed (with Quantization + Balanced Epochs):**
- ✓ CPT throughput: 80-100K tokens/sec (similar for FP8/NVFP4)
- ✓ SFT throughput: 150-180K tokens/sec (similar for FP8/NVFP4)
- ✓ FP8 vs BF16: ~40% faster overall (Flash Attention benefits)
- ✓ NVFP4 vs FP8: ±5% difference (within margin of error)
- ✓ **Total CPT cost: ~$85** (Fast CPT: ~$35)
- ✓ **Time savings**: 2.5-3 hours (vs 20-24 hours for over-trained approach)
- ✓ **Prevents catastrophic forgetting**: 5 epochs (vs 47 epochs at original config)

**System Stability:**
- ✓ Zero OOM errors during training
- ✓ Gradient norms stable (<10.0)
- ✓ FP4 scaling factors stable
- ✓ No numerical instability
- ✓ Checkpoint saving successful
- ✓ Resumable from any checkpoint

### Final Model Characteristics

**Model Capabilities:**
- ✓ Understands EUR-Lex legal documents
- ✓ Provides accurate citations (Article + CELEX)
- ✓ Handles multilingual queries (5 languages)
- ✓ Maintains general knowledge
- ✓ Follows instruction format
- ✓ Conservative responses (says "I don't know" when uncertain)

**Model Size:**
- ✓ Parameters: 70 billion
- ✓ Checkpoint size: ~140GB (BF16 format, standard)
- ✓ FP8 training memory: ~70GB (CPT) / ~40GB (SFT)
- ✓ NVFP4 training memory: ~35GB (CPT) / ~20GB (SFT)
- ✓ Converted for inference: ~140GB (BF16) or ~70GB (FP8) or ~35GB (INT4/NVFP4)

**Inference Performance:**
- ✓ Context length: Up to 4096 tokens
- ✓ Response time: ~2-5 seconds (on B200)
- ✓ Throughput: ~50-100 tokens/sec (inference)
- ✓ Memory requirement: ~70GB (BF16 inference)

### Benchmark Comparisons

**vs. Base LLaMA 3.3 70B:**
- ✓ Legal perplexity: 40% improvement (25 → 15)
- ✓ Citation accuracy: From 20% → 85%+
- ✓ EUR-Lex knowledge: Significant improvement
- ✓ General capabilities: Maintained (>90%)

**vs. Other Legal Models:**
- ✓ Multilingual: 5 languages (vs. typically EN-only)
- ✓ Citation quality: Superior structured citations
- ✓ EUR-Lex specific: Optimized for EU law
- ✓ Model size: 70B (larger than most legal models)

### Quality Assurance Checklist

**Data Quality:**
- [ ] All languages properly represented
- [ ] Citations verified against EUR-Lex
- [ ] No duplicate documents
- [ ] Text properly cleaned and normalized
- [ ] Statistics match expected distributions

**Training Quality:**
- [ ] Loss curves show steady convergence
- [ ] No training instability or spikes
- [ ] Validation metrics improve over time
- [ ] Checkpoints saved successfully
- [ ] Logs show no errors or warnings

**Model Quality:**
- [ ] Citation accuracy >85% on test set
- [ ] ROUGE-L >0.6 on test set
- [ ] Hallucination rate <5%
- [ ] Multilingual performance balanced
- [ ] Manual inspection of 100 samples passes
- [ ] General benchmarks >90% of base model (MMLU, HellaSwag)
- [ ] No catastrophic forgetting detected (see `docs/PREVENTING_CATASTROPHIC_FORGETTING.md`)

**System Quality:**
- [ ] Memory usage stable and predictable
- [ ] No OOM errors during training
- [ ] FP4 quantization working correctly
- [ ] Checkpoints can be loaded and resumed
- [ ] Model can be converted for inference

## Monitoring

Training metrics are logged to Weights & Biases:
- Loss, perplexity, learning rate
- GPU memory/utilization
- Tokens/second throughput
- FP4 scaling factors

Set your W&B credentials:
```bash
wandb login
```

## Citation Format

The model uses structured citations:
```
Article [X], [Document Type] (CELEX: [NUMBER])

Example:
"According to Article 5, GDPR (CELEX: 32016R0679), personal data must be..."
```

## Troubleshooting

### OOM Errors

**On RTX Pro 6000 (96GB GDDR7) — Phases 2 & 3:**
- **First try**: NVFP4 for 50% memory savings (~35GB CPT / ~20GB SFT):
  ```bash
  PRECISION=nvfp4 ./scripts/run_training.sh cpt
  ```
- Reduce `per_device_train_batch_size` in `configs/cpt_config.yaml`
- Enable CPU offload in FSDP config (`cpu_offload.offload_params: true`)
- Reduce `max_seq_length` (e.g., 2048 for CPT)

**On B200 (192GB HBM3e) — Phase 5:**
- OOM is unlikely with NVFP4 (~50GB CPT / ~30GB SFT in 192GB)
- If it occurs, check for memory leaks or reduce `per_device_train_batch_size` in `configs/cpt_config_b200.yaml`

### Training Instability
- Check gradient norms (should be < 10)
- Reduce learning rate
- Increase warmup steps
- **If using NVFP4**: Switch to FP8 for better numerical stability
  ```bash
  # Remove PRECISION override, use default FP8
  ./scripts/run_training.sh cpt
  ```

### Precision Mode Issues

**FP8 Issues**:
- Verify Transformer Engine installation (`pip list | grep transformer-engine`)
- Check GPU driver version (requires CUDA 12.1+ for Blackwell)
- Monitor FP8 scaling factors in logs (should be stable)
- Verify environment variables: `NVTE_FP8_DPA_BWD=1` and `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1`

**NVFP4 Issues** (Phase 5 — B200 only):
- **B200/B100 only**: Requires Blackwell data center GPU (GB100 die). **Not available on RTX Pro 6000** (GB202 die) — use `run_training.sh` with FP8 for Phases 2 & 3 instead.
- **Loss spikes**: Normal, check if magnitude is reasonable (<10% increase)
- **Quality degradation**: Expected 2-5%, if higher switch to FP8
- **NaN in loss**: Reduce learning rate or switch to FP8
- **Slow convergence**: Try FP8 for more stable training

**Switching Precision Modes**:
```bash
# Check current mode in logs
grep "Precision mode" logs/cpt_training/training.log

# Switch from NVFP4 to FP8
# (remove PRECISION env var or set explicitly)
PRECISION=fp8 ./scripts/run_training.sh cpt

# Switch from FP8 to NVFP4
PRECISION=nvfp4 ./scripts/run_training.sh cpt
```

### FSDP vs DeepSpeed
- **Default**: FSDP2 (recommended for both RTX Pro 6000 and B200 GPUs)
- **Legacy**: Use `USE_FSDP=false` to revert to DeepSpeed ZeRO-3
- **Conversion**: Use `python -m src.utils.checkpoint_utils convert-ds-to-fsdp` to convert existing DeepSpeed checkpoints

### Performance Monitoring
```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Monitor training progress
tail -f logs/cpt_training/training.log | grep -E "loss|perplexity|tok/s"

# Check precision mode and memory
grep -E "Precision|Memory|Backend" logs/cpt_training/training.log | head -20
```

## Testing & Verification

### Testing Individual Components

Before running the full pipeline, you can test individual components:

```bash
# Test FORMEX parser
python data_processing/parsers/formex_parser.py --help

# Test CPT corpus builder
python data_processing/dataset_builders/cpt_corpus_builder.py --help

# Test SFT dataset builder
python data_processing/dataset_builders/sft_dataset_builder.py --help

# Test data collators
python src/training/data_collators.py

# Test evaluation metrics
python src/evaluation/metrics.py

# Test FP8/NVFP4 utilities
python src/utils/fp8_utils.py

# Test checkpoint utilities
python src/utils/checkpoint_utils.py --help
```

### Quick Verification Commands

**After Data Processing** (Mac Studio):
```bash
# Check CPT corpus statistics
cat data/cpt/corpus_statistics.json

# Verify training shards were created
ls -lh data/cpt/train/*.parquet | wc -l  # Should be 32

# Check validation data
ls -lh data/cpt/validation/

# Verify SFT dataset
ls -lh data/sft/train/*.jsonl | wc -l  # Should be 8
```

**After Phase 2 CPT Training** (RTX Pro 6000):
```bash
ls -lht checkpoints/cpt/ | head -5
ls -lh checkpoints/cpt/final/
du -sh checkpoints/cpt/final
tail -100 logs/cpt_training/training.log | grep -E "Training completed|✓"
```

**After Phase 3 SFT Training** (RTX Pro 6000):
```bash
ls -lh checkpoints/sft/final/
tail -100 logs/sft_training/training.log | grep -E "eval_loss|citation_accuracy"
python scripts/test_inference.py \
  --model_path checkpoints/sft/final \
  --prompt "What is GDPR Article 5?"
```

**After Phase 5 Advanced Training** (B200):
```bash
ls -lh checkpoints/cpt_b200/final/
ls -lh checkpoints/sft_b200/final/
tail -100 logs/cpt_b200_training/training.log | grep -E "Training completed|✓"
tail -100 logs/sft_b200_training/training.log | grep -E "eval_loss|citation_accuracy"
```

**Data Transfer Verification**:
```bash
# Check data integrity after transfer
md5sum data/cpt/train/cpt_train_shard_00.parquet  # On Mac
md5sum data/cpt/train/cpt_train_shard_00.parquet  # On GPU cluster (should match)

# Verify all shards transferred
ls data/cpt/train/*.parquet | wc -l  # Should be 32 on both machines
```

## License

This project uses LLaMA 3.3 70B which requires Meta's license agreement.

## Support

For issues and guidance, please refer to:

**Documentation**:
- **`docs/PRECISION_MODES.md`** - **Comprehensive FP8 vs NVFP4 comparison guide**
- `docs/PREVENTING_CATASTROPHIC_FORGETTING.md` - Training best practices and parameter tuning
- `docs/COMPARISON_GUIDE.md` - Model evaluation and comparison methodology
- `README.md` (this file) - Project overview and setup

**Configuration**:
- `configs/` - Training configuration files
- `.claude/plans/hidden-toasting-comet.md` - Project plan document

**Logs & Debugging**:
- `logs/` - Training logs and metrics
- W&B dashboard - Real-time training monitoring

## Acknowledgments

- Meta AI for LLaMA 3.3
- NVIDIA for Transformer Engine and B200 GPU architecture
- PyTorch team for FSDP2 implementation
- EUR-Lex for legal document corpus
