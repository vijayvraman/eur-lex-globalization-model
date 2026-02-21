# Quick Start Guide

## Two-Machine Workflow

### Machine 1: Mac Studio (Data Processing)

#### Step 1: Setup Environment
```bash
cd eur-lex-globalization-model
./setup_mac.sh
source venv/bin/activate
```

#### Step 2: Prepare Data
Place your 25GB FORMEX XML files in `data/raw/`:
```
data/raw/
├── en/  # English documents
├── fr/  # French documents
├── de/  # German documents
├── es/  # Spanish documents
└── pt/  # Portuguese documents
```

#### Step 3: Parse FORMEX XML (4-6 hours)
```bash
python data_processing/parsers/formex_parser.py \
  --input data/raw \
  --output data/parsed \
  --workers 24
```

**Output**: Parsed JSON documents in `data/parsed/`

#### Step 4: Build CPT Corpus (2-3 hours)
```bash
python data_processing/dataset_builders/cpt_corpus_builder.py \
  --input_dir data/parsed \
  --output_dir data/cpt \
  --max_seq_length 4096 \
  --num_shards 32
```

**Output**:
- `data/cpt/train/cpt_train_shard_00.parquet` through `cpt_train_shard_31.parquet`
- `data/cpt/validation/cpt_val.parquet`
- `data/cpt/corpus_statistics.json`

#### Step 5: Transfer Data to GPU Cluster
```bash
# Compress data
tar -czf cpt_data.tar.gz data/cpt/

# Transfer to GPU cluster
scp cpt_data.tar.gz user@gpu-cluster:/path/to/project/

# On GPU cluster, extract:
# tar -xzf cpt_data.tar.gz
```

---

### Machine 2: GPU Cluster (4x B200 - Training)

#### Step 1: Setup Environment
```bash
cd eur-lex-globalization-model
./setup_gpu.sh
source venv/bin/activate
```

#### Step 2: Verify GPU Setup
```bash
nvidia-smi  # Should show 4x B200 GPUs
python3 -c "import torch; print(torch.cuda.device_count())"  # Should output: 4
```

#### Step 3: Configure Weights & Biases (Optional but Recommended)
```bash
wandb login
# Enter your W&B API key
```

#### Step 4: Start CPT Training (~20-24 hours)
```bash
# Set environment variables for Transformer Engine FP4
export NVTE_FP8_DPA_BWD=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Launch CPT training
deepspeed --num_gpus=4 scripts/train_cpt.py \
  --config configs/cpt_config.yaml \
  --deepspeed configs/ds_config_zero3.json \
  --use_fp8
```

**Monitor**:
- W&B dashboard: https://wandb.ai/your-username/llama33-70b-eurlex
- Local logs: `logs/cpt_training/`

**Checkpoints**: Saved every 1000 steps in `checkpoints/cpt/`

#### Step 5: Start SFT Training (~6-8 hours)
```bash
# Same environment variables
export NVTE_FP8_DPA_BWD=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Launch SFT training
deepspeed --num_gpus=4 scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --deepspeed configs/ds_config_zero3_sft.json \
  --use_fp8
```

**Checkpoints**: Saved every 500 steps in `checkpoints/sft/`

#### Step 6: Evaluate Model
```bash
python scripts/evaluate_model.py \
  --model_path models/llama33-70b-eurlex-sft-final \
  --eval_dataset data/sft/validation/sft_test.jsonl \
  --output_file results/evaluation_report.json
```

---

## Timeline Summary

| Phase | Machine | Duration | Output |
|-------|---------|----------|--------|
| XML Parsing | Mac Studio | 4-6 hrs | Parsed JSON |
| CPT Corpus | Mac Studio | 2-3 hrs | 32 shards, ~20GB |
| Data Transfer | Both | 1-2 hrs | Data on GPU cluster |
| CPT Training | GPU Cluster | 20-24 hrs | CPT checkpoint |
| SFT Training | GPU Cluster | 6-8 hrs | Final model |
| **Total** | - | **~3-4 days** | Ready model |

---

## Monitoring Training

### GPU Utilization
```bash
# Watch GPU memory and utilization
watch -n 1 nvidia-smi
```

### Training Logs
```bash
# Tail training logs
tail -f logs/cpt_training/train.log
```

### Weights & Biases
- Real-time loss curves
- GPU memory usage
- Tokens/second throughput
- FP4 scaling factors

---

## Troubleshooting

### Mac Studio: OOM during data processing
```bash
# Reduce number of workers
python data_processing/parsers/formex_parser.py \
  --workers 12  # Instead of 24
```

### GPU Cluster: OOM during training
```bash
# Reduce batch size in config
# Edit configs/cpt_config.yaml:
# per_device_train_batch_size: 1  # Instead of 2
```

### GPU Cluster: Slow training
```bash
# Verify FP4 is enabled
python -c "import transformer_engine; print('TE available')"

# Check environment variables
echo $NVTE_FP8_DPA_BWD  # Should be 1
```

### Data Transfer Issues
```bash
# Use compression for faster transfer
tar -czf - data/cpt | ssh user@gpu-cluster "tar -xzf - -C /path/to/project/"

# Or use rsync with compression
rsync -avz --compress-level=9 data/cpt/ user@gpu-cluster:/path/to/project/data/cpt/
```

---

## Quick Verification

### After Data Processing (Mac Studio)
```bash
# Check CPT corpus statistics
cat data/cpt/corpus_statistics.json

# Verify shards were created
ls -lh data/cpt/train/*.parquet | wc -l  # Should be 32
```

### After CPT Training (GPU Cluster)
```bash
# Check latest checkpoint
ls -lht checkpoints/cpt/ | head -5

# Verify model size
du -sh models/llama33-70b-eurlex-cpt-final
```

### After SFT Training (GPU Cluster)
```bash
# Check final model
ls -lh models/llama33-70b-eurlex-sft-final/

# Test inference (quick check)
python scripts/test_inference.py \
  --model_path models/llama33-70b-eurlex-sft-final \
  --prompt "What is GDPR Article 5?"
```

---

## Next Steps

After training completes:
1. ✅ Run comprehensive evaluation
2. ✅ Test citation accuracy on held-out set
3. ✅ Deploy for inference
4. ✅ Create API endpoint
5. ✅ Write deployment documentation

For detailed information, see [README.md](README.md)
