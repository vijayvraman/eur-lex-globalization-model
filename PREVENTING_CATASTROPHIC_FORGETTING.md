# Preventing Catastrophic Forgetting in Continued Pretraining

This guide explains how to prevent catastrophic forgetting when adapting LLaMA 3.3 70B to the EUR-Lex legal domain through continued pretraining (CPT).

## What is Catastrophic Forgetting?

**Catastrophic forgetting** occurs when a pretrained model loses its general capabilities while being fine-tuned on domain-specific data. For example:
- Model becomes excellent at legal terminology but forgets basic math
- Improves citation accuracy but loses common sense reasoning
- Adapts to EUR-Lex format but can't answer simple questions

## Why It Matters

LLaMA 3.3 70B has been pretrained on trillions of tokens across diverse domains. During CPT on legal text, we want to:
- ✅ **Gain**: Legal domain expertise, EUR-Lex knowledge, citation accuracy
- ❌ **Preserve**: General knowledge, reasoning, multilingual capabilities, instruction following

## Our Balanced Approach

### Training Configuration

We use **5 epochs** (4,260 steps) over our 446M token corpus:

```yaml
max_steps: 4260              # 5 epochs through 446M tokens
learning_rate: 2.0e-5        # Conservative LR (6.7% of pretraining LR)
weight_decay: 0.01           # Moderate regularization
warmup_steps: 426            # 10% warmup
max_grad_norm: 1.0           # Gradient clipping
```

**Why 5 epochs?**
- 2-5 epochs is the sweet spot for domain adaptation without forgetting
- Less than 2 epochs: Insufficient adaptation
- More than 7 epochs: High risk of catastrophic forgetting
- Original config (47 epochs) would cause severe forgetting

### Comparison

| Configuration | Epochs | Risk Level | Use Case |
|--------------|--------|------------|----------|
| Fast CPT | 2 | Very Low | Rapid prototyping |
| **Full CPT (Recommended)** | **5** | **Low** | **Production quality** |
| Extended CPT | 6-7 | Medium | Maximum adaptation |
| Original (40K steps) | 47 | Very High | ❌ Not recommended |

---

## Key Parameters to Prevent Forgetting

### 🔴 High Impact Parameters

#### 1. Learning Rate

**Current**: `2.0e-5`

**Effect**: Lower learning rate = smaller weight updates = less forgetting

**Recommendations**:
- **Conservative (minimal forgetting)**: `1.0e-5` to `1.5e-5`
  - Best for: Critical applications where preserving general capabilities is paramount
  - Trade-off: Slower domain adaptation

- **Balanced (current)**: `2.0e-5` ✓
  - Best for: Production use with good balance
  - Sweet spot: 6.7% of LLaMA's pretraining LR (~3e-4)

- **Aggressive (faster learning)**: `3.0e-5` to `5.0e-5`
  - Best for: Maximum domain adaptation, willing to accept some forgetting
  - Trade-off: Higher risk of catastrophic forgetting

**Rule of thumb**: For CPT, use 5-10% of the base model's original pretraining learning rate.

---

#### 2. Weight Decay

**Current**: `0.01`

**Effect**: Regularization that prevents weights from drifting too far from their pretrained values

**Recommendations**:
- **Stronger regularization**: `0.05` to `0.1`
  - Prevents forgetting but may slow domain adaptation
  - Use if you see general capability degradation on benchmarks

- **Balanced (current)**: `0.01` ✓
  - Standard for continued pretraining

- **Weaker regularization**: `0.001` to `0.005`
  - Faster adaptation but higher forgetting risk
  - Only use if domain adaptation is critically slow

**When to increase**: If validation on general benchmarks (MMLU, HellaSwag) drops below 90% of base model performance.

---

#### 3. Max Gradient Norm

**Current**: `1.0`

**Effect**: Gradient clipping prevents large, destabilizing updates that can cause forgetting

**Recommendations**:
- **Conservative**: `0.5` to `0.8`
  - Smaller maximum updates = more stability
  - Use if you see validation loss spikes

- **Balanced (current)**: `1.0` ✓
  - Standard for large language models

- **Aggressive**: `1.5` to `2.0`
  - Allows larger updates for faster adaptation
  - Only if training is too slow and loss is very stable

**Monitor**: Track gradient norms in logs. If consistently hitting the limit, consider adjusting.

---

### 🟡 Medium Impact Parameters

#### 4. Warmup Steps

**Current**: `426` (10% of max_steps)

**Effect**: Gradual learning rate increase prevents catastrophic early updates

**Recommendations**:
- **Conservative**: 15-20% of max_steps (`640-850 steps`)
  - Longer warmup = safer start
  - Prevents early-training forgetting

- **Balanced (current)**: 10% ✓
  - Standard practice in LLM research

- **Aggressive**: 5% (`213 steps`)
  - Faster to reach peak LR
  - Only if very confident in stability

**Best practice**: When in doubt, use longer warmup. It's a cheap safety measure.

---

#### 5. LR Scheduler Type

**Current**: `cosine`

**Effect**: Determines how learning rate decays during training

**Options**:
- **`cosine` (current)**: Smooth decay with gradual tail ✓
  - Best for preventing forgetting
  - Gold standard for LLM training

- **`linear`**: Linear decay to zero
  - Slightly more aggressive than cosine
  - Can work well but less commonly used

- **`constant_with_warmup`**: No decay after warmup
  - Very conservative approach
  - May underfit if corpus is small

- **`polynomial`**: Customizable decay rate
  - Flexible but requires tuning

**Recommendation**: Keep `cosine` - it's battle-tested and performs well.

---

#### 6. Optimizer Betas (AdamW)

**Current**: `[0.9, 0.95]` (beta1, beta2)

**Effect**:
- **beta1**: First moment (gradient momentum)
- **beta2**: Second moment (variance momentum)

**Recommendations**:
- **Conservative**: `[0.9, 0.99]`
  - Higher beta2 = smoother, more stable updates
  - Better for preventing forgetting

- **Balanced (current)**: `[0.9, 0.95]` ✓
  - Standard for pretraining

- **Aggressive**: `[0.9, 0.9]`
  - Less smoothing, more responsive to recent gradients
  - Rarely needed

**When to adjust**: Increase beta2 to 0.98-0.99 if training is unstable or you see forgetting.

---

### 🟢 Lower Impact Parameters

#### 7. Effective Batch Size

**Current**: `128` (2 per device × 16 grad accum × 4 GPUs)

**Effect**: Larger batches = more stable gradients = less noisy updates = less forgetting

**Trade-offs**:
- Very large batches (>256) can hurt generalization
- Very small batches (<64) increase noise and forgetting risk
- 128-256 is the sweet spot for 70B models

**Current setting**: Well-balanced ✓

---

#### 8. Sequence Length

**Current**: `4096` (Full CPT), `3072` (Fast CPT)

**Effect**: Shorter sequences mean more frequent parameter updates per token

**Considerations**:
- Longer sequences (4096): Better context learning, fewer updates
- Shorter sequences (3072): More updates, slightly more conservative
- Fast CPT uses 3072 for this reason

---

## Configuration Profiles

### Profile 1: Conservative (Minimal Forgetting)

**When to use**: Critical applications, high-stakes deployments, must maintain general capabilities

```yaml
# configs/cpt_config_conservative.yaml
training:
  learning_rate: 1.5e-5        # ↓ 25% lower
  max_steps: 3400              # ↓ ~4 epochs
  weight_decay: 0.05           # ↑ 5x higher
  max_grad_norm: 0.8           # ↓ 20% lower
  warmup_steps: 510            # ↑ 15% of steps (vs 10%)
```

```json
// DeepSpeed config adjustments
"betas": [0.9, 0.98]           // ↑ Higher beta2 for smoothing
"warmup_num_steps": 510,
"total_num_steps": 3400
```

**Expected results**:
- Slower domain adaptation
- Excellent general capability preservation (>95% of base model)
- Lower legal perplexity improvement (~20% vs ~40%)
- Training time: ~2 hours
- Cost: ~$65

---

### Profile 2: Balanced (Current - Recommended)

**When to use**: Production deployments, standard applications

```yaml
# configs/cpt_config.yaml (current)
training:
  learning_rate: 2.0e-5        # ✓ Balanced
  max_steps: 4260              # ✓ 5 epochs
  weight_decay: 0.01           # ✓ Standard
  max_grad_norm: 1.0           # ✓ Standard
  warmup_steps: 426            # ✓ 10% of steps
```

```json
// DeepSpeed config
"betas": [0.9, 0.95]           // ✓ Standard
"warmup_num_steps": 426,
"total_num_steps": 4260
```

**Expected results**:
- Good domain adaptation
- Strong general capability preservation (>90% of base model)
- Significant legal perplexity improvement (~40%)
- Training time: ~2.5-3 hours
- Cost: ~$85

---

### Profile 3: Aggressive (Maximum Adaptation)

**When to use**: Domain-specific applications, less concern about general capabilities

```yaml
# configs/cpt_config_aggressive.yaml
training:
  learning_rate: 3.0e-5        # ↑ 50% higher
  max_steps: 5100              # ↑ ~6 epochs
  weight_decay: 0.005          # ↓ 50% lower
  max_grad_norm: 1.5           # ↑ 50% higher
  warmup_steps: 255            # ↓ 5% of steps
```

```json
// DeepSpeed config adjustments
"betas": [0.9, 0.9]            // ↓ Less smoothing
"warmup_num_steps": 255,
"total_num_steps": 5100
```

**Expected results**:
- Strong domain adaptation
- Moderate general capability preservation (~85% of base model)
- Maximum legal perplexity improvement (~50%)
- Training time: ~3.5-4 hours
- Cost: ~$100

**Warning**: Monitor general benchmarks closely. Be prepared to roll back if forgetting is too severe.

---

## Advanced Anti-Forgetting Strategies

### 1. Data Mixing (Not Currently Implemented)

Mix general domain data with legal data during training:

```python
# Pseudocode
legal_data_ratio = 0.80      # 80% legal documents
general_data_ratio = 0.20    # 20% general text (Wikipedia, books, etc.)
```

**Benefits**:
- Keeps model exposed to general knowledge
- Proven effective in research (e.g., LIMA, Orca)
- Minimal impact on legal adaptation

**Trade-offs**:
- Requires additional data preparation
- Slightly slower legal domain convergence
- More complex data pipeline

**Implementation**: Would require modifying `cpt_corpus_builder.py` to include general data sources.

---

### 2. Layer-wise Learning Rates (Not Currently Implemented)

Apply different learning rates to different transformer layers:

```python
# Pseudocode
early_layers_lr = 1.0e-5     # Layers 0-20 (general features)
middle_layers_lr = 2.0e-5    # Layers 21-50 (intermediate features)
late_layers_lr = 3.0e-5      # Layers 51-80 (task-specific features)
```

**Rationale**:
- Early layers learn general features (preserve them with lower LR)
- Late layers learn task-specific features (adapt them with higher LR)
- Middle layers bridge the two (moderate LR)

**Benefits**:
- Fine-grained control over adaptation
- Better preservation of general knowledge
- Used by GPT-3, PaLM, and other large models

**Trade-offs**:
- More complex setup
- Harder to tune
- Not supported out-of-box by HuggingFace Trainer

**Implementation**: Would require custom optimizer configuration in DeepSpeed.

---

### 3. LoRA/Adapter Approach (Alternative Architecture)

Instead of full fine-tuning, use Low-Rank Adaptation (LoRA):

```yaml
# Alternative approach using PEFT
use_peft: true
peft_config:
  peft_type: "LORA"
  r: 16                      # Rank
  lora_alpha: 32             # Scaling factor
  lora_dropout: 0.05
  target_modules:            # Which layers to adapt
    - q_proj
    - v_proj
    - k_proj
    - o_proj
```

**Benefits**:
- **Zero catastrophic forgetting** (base model is frozen)
- Much faster training (only ~0.1% of parameters updated)
- Tiny checkpoint size (~100MB vs 140GB)
- Can train multiple adapters for different domains

**Trade-offs**:
- Slightly lower domain adaptation quality (~90% of full fine-tuning)
- Requires PEFT library
- Different deployment workflow

**When to use**:
- Extreme risk aversion to forgetting
- Need to maintain multiple domain adaptations
- Limited compute/storage budget

---

### 4. Replay-Based Methods (Research Frontier)

Interleave batches from original pretraining data:

```python
# Pseudocode
for batch in training:
    if random() < 0.1:  # 10% of time
        batch = sample_from_original_pretraining_data()
    else:
        batch = sample_from_legal_corpus()
```

**Benefits**:
- Directly prevents forgetting by replaying old data
- Proven effective in continual learning research

**Challenges**:
- Requires access to original pretraining data (not available for LLaMA)
- Increased training time
- Complex data pipeline

---

## Monitoring for Catastrophic Forgetting

### Essential Metrics

Track these during training to detect forgetting early:

#### 1. General Benchmarks
Run every 1,000 steps on:
- **MMLU** (Massive Multitask Language Understanding): General knowledge
- **HellaSwag**: Common sense reasoning
- **ARC** (AI2 Reasoning Challenge): Scientific reasoning
- **TruthfulQA**: Factual accuracy

**Threshold**: Performance should stay >90% of base model scores

#### 2. Domain-Specific Metrics
- **Legal perplexity**: Should decrease steadily
- **Citation accuracy**: Should increase
- **EUR-Lex question answering**: Custom validation set

#### 3. Training Stability
- **Gradient norms**: Should stay <5.0 (spike = potential forgetting)
- **Validation loss**: Should decrease smoothly (spike = instability)
- **Learning rate**: Track current LR value

#### 4. Sample Outputs
Every 500 steps, test on standard prompts:
```
Prompt: "What is 2+2?"
Expected: "4" (not legal jargon)

Prompt: "Explain gravity to a 5-year-old."
Expected: Simple, accessible explanation (not legal speak)

Prompt: "What does GDPR stand for?"
Expected: "General Data Protection Regulation" (domain knowledge)
```

### Example W&B Dashboard

```python
# Log to Weights & Biases
wandb.log({
    # Training metrics
    "train/loss": loss,
    "train/perplexity": perplexity,
    "train/grad_norm": grad_norm,

    # Domain metrics
    "eval/legal_perplexity": legal_perplexity,
    "eval/citation_accuracy": citation_acc,

    # General capability metrics
    "eval/mmlu_accuracy": mmlu_acc,
    "eval/hellaswag_accuracy": hellaswag_acc,
    "eval/common_sense_qa": cs_qa_acc,
})
```

---

## Warning Signs of Catastrophic Forgetting

### Red Flags

1. **Severe forgetting** (stop training immediately):
   - MMLU drops >15% from base model
   - Model gives legal answers to simple math questions
   - Common sense reasoning fails completely
   - Instruction following degraded

2. **Moderate forgetting** (adjust hyperparameters):
   - MMLU drops 10-15% from base model
   - Slight degradation in general conversation
   - Some cross-domain confusion

3. **Acceptable trade-off**:
   - MMLU drops 5-10% from base model
   - Strong legal domain improvement
   - General capabilities mostly intact

### Recovery Actions

If you detect catastrophic forgetting:

1. **Stop training** at last checkpoint before degradation
2. **Roll back** to checkpoint with best general/legal balance
3. **Adjust hyperparameters**:
   - Reduce learning rate by 30-50%
   - Increase weight decay by 2-5x
   - Reduce max steps (fewer epochs)
4. **Resume training** from rolled-back checkpoint
5. **Monitor more frequently** (every 250 steps instead of 500)

---

## Evaluation Protocol

### After CPT Training

Run comprehensive evaluation before deploying:

#### 1. General Capability Benchmarks
```bash
# Use lm-evaluation-harness
lm_eval --model hf \
  --model_args pretrained=./checkpoints/cpt/final \
  --tasks mmlu,hellaswag,arc_easy,arc_challenge,truthfulqa \
  --batch_size 8
```

**Success criteria**:
- MMLU: >90% of base model score (base: ~70%, target: >63%)
- HellaSwag: >90% of base model score
- ARC: >90% of base model score

#### 2. Domain-Specific Evaluation
```bash
# Custom legal evaluation
python scripts/evaluate_legal_qa.py \
  --model_path ./checkpoints/cpt/final \
  --test_set data/test/legal_qa_test.jsonl
```

**Success criteria**:
- Legal perplexity: <15 (vs base model ~25)
- Citation accuracy: >85%
- Legal terminology: Significant improvement

#### 3. Manual Inspection
Test on diverse prompts:
- General knowledge questions
- Math problems
- Creative writing
- Legal questions
- Multilingual queries

---

## Comparison Table: All Profiles

| Metric | Conservative | Balanced (Current) | Aggressive |
|--------|-------------|-------------------|------------|
| **Learning Rate** | 1.5e-5 | 2.0e-5 | 3.0e-5 |
| **Epochs** | 4 | 5 | 6 |
| **Steps** | 3,400 | 4,260 | 5,100 |
| **Weight Decay** | 0.05 | 0.01 | 0.005 |
| **Warmup Steps** | 510 (15%) | 426 (10%) | 255 (5%) |
| **Max Grad Norm** | 0.8 | 1.0 | 1.5 |
| **Beta2** | 0.98 | 0.95 | 0.9 |
| **Training Time** | ~2h | ~2.5-3h | ~3.5-4h |
| **Cost** | ~$65 | ~$85 | ~$100 |
| **General Capability** | >95% | >90% | ~85% |
| **Legal Improvement** | +20% | +40% | +50% |
| **Forgetting Risk** | Very Low | Low | Medium |

---

## Best Practices Summary

1. **Start Conservative**: Use balanced config, monitor closely
2. **Monitor Continuously**: Track both domain and general metrics
3. **Validate Early**: Test on general benchmarks at step 500, 1000, etc.
4. **Keep Checkpoints**: Save every 500-1000 steps to enable rollback
5. **Test Diverse Prompts**: Don't just test legal questions
6. **Compare to Base**: Always benchmark against base LLaMA 3.3 70B
7. **Document Decisions**: Log why you chose specific hyperparameters
8. **Iterate Carefully**: Small adjustments, one parameter at a time

---

## Quick Reference: Parameter Impact

**To reduce forgetting (more conservative)**:
- ↓ Lower learning rate
- ↓ Fewer epochs/steps
- ↑ Increase weight decay
- ↑ Increase warmup steps
- ↑ Increase beta2
- ↓ Lower max gradient norm

**To increase adaptation (more aggressive)**:
- ↑ Higher learning rate
- ↑ More epochs/steps
- ↓ Decrease weight decay
- ↓ Decrease warmup steps
- ↓ Decrease beta2
- ↑ Higher max gradient norm

---

## Further Reading

- **Research Papers**:
  - "Don't Stop Pretraining" (Gururangan et al., 2020)
  - "LIMA: Less Is More for Alignment" (Zhou et al., 2023)
  - "Continual Learning in Neural Networks" (Parisi et al., 2019)

- **LLaMA Documentation**:
  - Meta LLaMA 3.3 Model Card
  - Hugging Face Transformers Trainer API
  - DeepSpeed ZeRO Optimization

- **Related Documentation**:
  - `README.md`: Main project documentation
  - `COMPARISON_GUIDE.md`: Model comparison methodology
  - `configs/`: Training configuration files

---

## Support

For questions about preventing catastrophic forgetting:
1. Check training logs in `logs/cpt_training/`
2. Review W&B dashboard for metric trends
3. Compare checkpoints using evaluation scripts
4. Consult this guide for parameter adjustments

**Remember**: The goal is not zero forgetting (impossible), but maintaining >90% of general capabilities while achieving strong domain adaptation. The balanced profile (current config) is designed to hit this sweet spot.
