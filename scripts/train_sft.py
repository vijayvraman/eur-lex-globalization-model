"""
SFT Training Script with FP8/NVFP4 Quantization

Trains LLaMA 3.3 70B for instruction-following with:
- Supervised Fine-Tuning on EUR-Lex Q&A pairs
- Input masking (loss only on assistant responses)
- FSDP2 (default) or DeepSpeed ZeRO-3 for distributed training
- FP8 (default) or NVFP4 (experimental) quantization via Transformer Engine

NVIDIA's recommended stack for Blackwell hardware (B200):
- PyTorch FSDP2 + Transformer Engine
- FP8: 8-bit (E4M3/E5M2) - Default, ~70GB memory
- NVFP4: 4-bit (E2M1) - Experimental, ~35GB memory
- As used in NeMo 2.0 and Megatron-Core
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.data_collators import DataCollatorForSFT
from src.training.fsdp_config import apply_fsdp_wrapping, setup_transformer_engine

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict, use_fp8: bool = False, use_fsdp: bool = False):
    """
    Setup model with FP4/FP8 quantization if enabled

    Args:
        config: Model configuration
        use_fp8: Whether to enable FP8/FP4 quantization
        use_fsdp: Whether FSDP is being used (affects gradient checkpointing setup)

    Returns:
        model, tokenizer
    """
    model_path = config['model']['name']
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    attn_implementation = config['model']['attn_implementation']

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Torch dtype: {torch_dtype}")
    logger.info(f"Attention: {attn_implementation}")
    logger.info(f"FP8 enabled: {use_fp8}")
    logger.info(f"FSDP enabled: {use_fsdp}")

    # Load tokenizer
    # Try loading from CPT checkpoint first, fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        logger.warning(f"Could not load tokenizer from {model_path}, using base model")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        use_cache=config['model'].get('use_cache', False)
    )

    # Enable gradient checkpointing
    # NOTE: With FSDP, gradient checkpointing is applied AFTER wrapping
    # With DeepSpeed, we enable it here
    if config['training'].get('gradient_checkpointing', False) and not use_fsdp:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    elif config['training'].get('gradient_checkpointing', False) and use_fsdp:
        logger.info("Gradient checkpointing will be applied after FSDP wrapping")

    return model, tokenizer


def load_dataset_from_config(config: dict):
    """Load training and validation datasets"""
    data_config = config['data']

    logger.info(f"Loading training data from: {data_config['train_file']}")

    # Load datasets
    dataset = load_dataset(
        'json',
        data_files={
            'train': data_config['train_file'],
            'validation': data_config['validation_file']
        }
    )

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='SFT Training with FP4')
    parser.add_argument('--config', type=str, default='configs/sft_config.yaml',
                       help='Path to config file')
    parser.add_argument('--deepspeed', type=str, default='configs/ds_config_zero3_sft.json',
                       help='Path to DeepSpeed config')
    parser.add_argument('--fsdp', action='store_true',
                       help='Enable FSDP2 training (replaces DeepSpeed)')
    parser.add_argument('--fsdp_config', type=str, default='sft',
                       help='FSDP config type: cpt, sft, or fast_cpt')
    parser.add_argument('--use_fp8', action='store_true',
                       help='Enable quantization via Transformer Engine')
    parser.add_argument('--precision', type=str, default='fp8', choices=['fp8', 'nvfp4'],
                       help='Precision mode: fp8 (8-bit, default) or nvfp4 (4-bit, experimental)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed
    set_seed(42)

    # Setup quantization environment if enabled
    if args.use_fp8:
        precision_name = args.precision.upper()
        if args.fsdp:
            logger.info(f"Setting up {precision_name} quantization for FSDP2 + Transformer Engine")
            logger.info("This is NVIDIA's recommended stack for Blackwell hardware")
            if args.precision == "nvfp4":
                logger.info("⚠️  NVFP4 is experimental - expect ~50% memory savings vs FP8")
                logger.info("⚠️  NVFP4 uses 4-bit (E2M1) with block scaling")
        else:
            logger.info(f"Setting up {precision_name} quantization for DeepSpeed + Transformer Engine")
        os.environ['NVTE_FP8_DPA_BWD'] = '1'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '1'

    # Setup model and tokenizer
    model, tokenizer = setup_model(config, use_fp8=args.use_fp8, use_fsdp=args.fsdp)

    # Load datasets
    dataset = load_dataset_from_config(config)

    # Setup data collator with input masking
    data_collator = DataCollatorForSFT(
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )

    # Setup training arguments
    training_config = config['training']

    # Conditional FSDP vs DeepSpeed configuration
    if args.fsdp:
        # FSDP2 configuration
        distributed_kwargs = {
            "fsdp": "full_shard",  # FULL_SHARD strategy (ZeRO-3 equivalent)
            "fsdp_config": {
                "backward_prefetch": "backward_pre",  # Overlap communication
                "forward_prefetch": False,
                "limit_all_gathers": True,  # Reduce memory overhead
                "use_orig_params": True,  # CRITICAL for optimizer state management
                "sync_module_states": True,  # Sync model states across ranks
                "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",  # Auto-wrap policy
            }
        }
    else:
        # DeepSpeed configuration
        distributed_kwargs = {
            "deepspeed": args.deepspeed
        }

    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        num_train_epochs=training_config.get('num_train_epochs', 3),
        max_steps=training_config.get('max_steps', -1),
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_ratio=training_config.get('warmup_ratio', 0.05),
        weight_decay=training_config['weight_decay'],
        max_grad_norm=training_config['max_grad_norm'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        eval_strategy='steps',
        eval_steps=training_config['eval_steps'],
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
        bf16=training_config['bf16'],
        # Gradient checkpointing handled by FSDP wrapping, not Trainer
        gradient_checkpointing=False if args.fsdp else training_config['gradient_checkpointing'],
        **distributed_kwargs,  # FSDP or DeepSpeed configuration
        report_to=training_config.get('report_to', 'wandb'),
        logging_dir=training_config.get('logging_dir', 'logs/sft_training'),
        run_name=config.get('wandb', {}).get('name', 'sft-training'),
        ddp_find_unused_parameters=False,
        dataloader_num_workers=config['data'].get('preprocessing_num_workers', 4),
        dataloader_pin_memory=True,
    )

    # Apply FSDP wrapping if enabled
    if args.fsdp:
        precision_name = args.precision.upper() if args.use_fp8 else "BF16"
        logger.info(f"Applying FSDP2 wrapping with Transformer Engine ({precision_name})...")
        model = apply_fsdp_wrapping(
            model,
            fsdp_config_type=args.fsdp_config,
            use_quantization=args.use_fp8,
            precision_mode=args.precision
        )
        logger.info("FSDP2 wrapping complete")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Log configuration
    logger.info("=" * 80)
    logger.info("SFT Training Configuration:")
    logger.info(f"  Model: {config['model']['name']}")
    logger.info(f"  Backend: {'FSDP2 + Transformer Engine' if args.fsdp else 'DeepSpeed ZeRO-3 + Transformer Engine'}")
    logger.info(f"  Output dir: {training_config['output_dir']}")
    logger.info(f"  Batch size per device: {training_config['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps'] * torch.cuda.device_count()}")
    logger.info(f"  Learning rate: {training_config['learning_rate']}")
    logger.info(f"  Epochs: {training_config.get('num_train_epochs', 'N/A')}")
    logger.info(f"  Input masking: Enabled (loss only on assistant responses)")
    logger.info(f"  Quantization enabled: {args.use_fp8}")
    if args.use_fp8:
        logger.info(f"  Precision mode: {args.precision.upper()}")
        if args.precision == "nvfp4":
            logger.info(f"    → 4-bit NVFP4 (E2M1) - Experimental")
        else:
            logger.info(f"    → 8-bit FP8 (E4M3/E5M2) - Default")
    if args.fsdp:
        logger.info(f"  FSDP config type: {args.fsdp_config}")
    else:
        logger.info(f"  DeepSpeed config: {args.deepspeed}")
    logger.info("=" * 80)

    # Start training
    logger.info("Starting SFT training...")

    try:
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()

        logger.info("Training completed successfully!")

        # Save final model
        output_dir = Path(training_config['output_dir']) / 'final'
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        logger.info("Model saved successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
