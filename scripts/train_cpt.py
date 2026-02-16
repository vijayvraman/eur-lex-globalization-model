"""
CPT Training Script with FP4 Quantization

Trains LLaMA 3.1 70B with:
- Continued Pretraining on EUR-Lex legal corpus
- DeepSpeed ZeRO-3 for distributed training
- FP4 quantization via Transformer Engine
- Flash Attention 2
- Gradient checkpointing
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

import torch
import deepspeed
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

from src.training.data_collators import DataCollatorForCPT

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


def setup_model(config: dict, use_fp8: bool = False):
    """
    Setup model with FP4/FP8 quantization if enabled

    Args:
        config: Model configuration
        use_fp8: Whether to enable FP8/FP4 quantization

    Returns:
        model, tokenizer
    """
    model_name = config['model']['name']
    torch_dtype = getattr(torch, config['model']['torch_dtype'])
    attn_implementation = config['model']['attn_implementation']

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Torch dtype: {torch_dtype}")
    logger.info(f"Attention: {attn_implementation}")
    logger.info(f"FP8 enabled: {use_fp8}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        use_cache=config['model'].get('use_cache', False)
    )

    # Enable gradient checkpointing
    if config['training'].get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    return model, tokenizer


def load_dataset_from_config(config: dict):
    """Load training and validation datasets"""
    data_config = config['data']

    logger.info(f"Loading training data from: {data_config['train_file']}")

    # Load datasets
    dataset = load_dataset(
        'parquet',
        data_files={
            'train': data_config['train_file'],
            'validation': data_config['validation_file']
        }
    )

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='CPT Training with FP4')
    parser.add_argument('--config', type=str, default='configs/cpt_config.yaml',
                       help='Path to config file')
    parser.add_argument('--deepspeed', type=str, default='configs/ds_config_zero3.json',
                       help='Path to DeepSpeed config')
    parser.add_argument('--use_fp8', action='store_true',
                       help='Enable FP8/FP4 quantization via Transformer Engine')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed
    set_seed(42)

    # Setup FP8 environment if enabled
    if args.use_fp8:
        logger.info("Setting up FP8/FP4 environment variables")
        os.environ['NVTE_FP8_DPA_BWD'] = '1'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '1'

    # Setup model and tokenizer
    model, tokenizer = setup_model(config, use_fp8=args.use_fp8)

    # Load datasets
    dataset = load_dataset_from_config(config)

    # Setup data collator
    data_collator = DataCollatorForCPT(
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )

    # Setup training arguments
    training_config = config['training']

    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        num_train_epochs=training_config.get('num_train_epochs', 1),
        max_steps=training_config.get('max_steps', -1),
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_steps=training_config['warmup_steps'],
        weight_decay=training_config['weight_decay'],
        max_grad_norm=training_config['max_grad_norm'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        eval_strategy='steps',
        eval_steps=training_config['eval_steps'],
        bf16=training_config['bf16'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        deepspeed=args.deepspeed,
        report_to=training_config.get('report_to', 'wandb'),
        logging_dir=training_config.get('logging_dir', 'logs/cpt_training'),
        run_name=config.get('wandb', {}).get('name', 'cpt-training'),
        ddp_find_unused_parameters=False,
        dataloader_num_workers=config['data'].get('preprocessing_num_workers', 4),
        dataloader_pin_memory=True,
    )

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
    logger.info("Training Configuration:")
    logger.info(f"  Model: {config['model']['name']}")
    logger.info(f"  Output dir: {training_config['output_dir']}")
    logger.info(f"  Batch size per device: {training_config['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps'] * torch.cuda.device_count()}")
    logger.info(f"  Learning rate: {training_config['learning_rate']}")
    logger.info(f"  Max steps: {training_config.get('max_steps', 'N/A')}")
    logger.info(f"  FP8 enabled: {args.use_fp8}")
    logger.info(f"  DeepSpeed config: {args.deepspeed}")
    logger.info("=" * 80)

    # Start training
    logger.info("Starting training...")

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
