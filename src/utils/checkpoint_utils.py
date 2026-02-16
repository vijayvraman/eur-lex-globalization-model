"""
Checkpoint Utilities

Utilities for:
- Converting DeepSpeed checkpoints to HuggingFace format
- Converting FP4/FP8 checkpoints to BF16 for inference
- Checkpoint consolidation and management
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_deepspeed_to_hf(checkpoint_dir: str, output_dir: str,
                            zero_stage: int = 3):
    """
    Convert DeepSpeed checkpoint to HuggingFace format

    Args:
        checkpoint_dir: Path to DeepSpeed checkpoint directory
        output_dir: Path to save HuggingFace checkpoint
        zero_stage: DeepSpeed ZeRO stage (3 for ZeRO-3)
    """
    logger.info(f"Converting DeepSpeed checkpoint: {checkpoint_dir}")
    logger.info(f"Output directory: {output_dir}")

    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if zero_stage == 3:
        logger.info("Converting ZeRO-3 checkpoint...")

        # For ZeRO-3, we need to consolidate sharded weights
        # DeepSpeed provides utilities for this
        try:
            import deepspeed

            # Load DeepSpeed checkpoint
            logger.info("Loading sharded weights...")

            # Use DeepSpeed's checkpoint consolidation
            from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

            state_dict = convert_zero_checkpoint_to_fp32_state_dict(
                str(checkpoint_path),
                str(output_path / "pytorch_model.bin")
            )

            logger.info("Checkpoint converted successfully")

        except ImportError:
            logger.error("DeepSpeed not available. Cannot convert checkpoint.")
            return
        except Exception as e:
            logger.error(f"Error converting checkpoint: {e}")
            return

    else:
        logger.info(f"Converting ZeRO-{zero_stage} checkpoint...")
        # For ZeRO-1/2, weights are not sharded across devices
        # Can load directly
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        model.save_pretrained(output_dir)

    # Copy tokenizer files if they exist
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json',
                       'special_tokens_map.json', 'vocab.json', 'merges.txt']

    for filename in tokenizer_files:
        src = checkpoint_path / filename
        if src.exists():
            dst = output_path / filename
            shutil.copy(src, dst)
            logger.info(f"Copied {filename}")

    logger.info("Conversion complete!")


def convert_fp4_to_bf16(model_path: str, output_path: str):
    """
    Convert FP4/FP8 checkpoint to BF16 for inference

    This loads the quantized checkpoint and saves it in BF16 format
    which is more widely compatible for inference.

    Args:
        model_path: Path to FP4/FP8 checkpoint
        output_path: Path to save BF16 checkpoint
    """
    logger.info(f"Converting FP4/FP8 checkpoint to BF16")
    logger.info(f"Input: {model_path}")
    logger.info(f"Output: {output_path}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model in BF16 (automatically converts from FP4/FP8)
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        # Save in BF16
        logger.info("Saving model in BF16 format...")
        model.save_pretrained(output_path, max_shard_size="5GB")

        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(output_path)

        logger.info("Conversion complete!")

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise


def consolidate_checkpoints(checkpoint_dir: str, output_file: str):
    """
    Consolidate multiple checkpoint shards into single file

    Args:
        checkpoint_dir: Directory containing checkpoint shards
        output_file: Output path for consolidated checkpoint
    """
    logger.info(f"Consolidating checkpoints from {checkpoint_dir}")

    checkpoint_path = Path(checkpoint_dir)

    # Find all shard files
    shard_files = sorted(checkpoint_path.glob("pytorch_model-*.bin"))

    if not shard_files:
        logger.error("No checkpoint shards found")
        return

    logger.info(f"Found {len(shard_files)} shard files")

    # Load and merge shards
    consolidated_state_dict = {}

    for shard_file in shard_files:
        logger.info(f"Loading {shard_file.name}")
        shard = torch.load(shard_file, map_location='cpu')
        consolidated_state_dict.update(shard)

    # Save consolidated checkpoint
    logger.info(f"Saving consolidated checkpoint to {output_file}")
    torch.save(consolidated_state_dict, output_file)

    logger.info("Consolidation complete!")


def verify_checkpoint(checkpoint_path: str) -> bool:
    """
    Verify that a checkpoint can be loaded

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        True if checkpoint is valid
    """
    logger.info(f"Verifying checkpoint: {checkpoint_path}")

    try:
        # Try loading model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map='cpu'
        )

        # Try loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        logger.info("✓ Checkpoint is valid")
        logger.info(f"  Model parameters: {model.num_parameters():,}")
        logger.info(f"  Tokenizer vocab size: {tokenizer.vocab_size:,}")

        return True

    except Exception as e:
        logger.error(f"✗ Checkpoint verification failed: {e}")
        return False


def get_checkpoint_info(checkpoint_path: str) -> dict:
    """
    Get information about a checkpoint

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint_dir = Path(checkpoint_path)

    info = {
        'path': str(checkpoint_path),
        'exists': checkpoint_dir.exists(),
        'files': []
    }

    if checkpoint_dir.exists():
        # List all files
        info['files'] = [str(f.name) for f in checkpoint_dir.iterdir()]

        # Check for specific files
        info['has_model'] = any('pytorch_model' in f for f in info['files'])
        info['has_tokenizer'] = any('tokenizer' in f for f in info['files'])
        info['has_config'] = 'config.json' in info['files']

        # Get size
        total_size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
        info['total_size_gb'] = total_size / (1024**3)

    return info


def main():
    """Command-line interface for checkpoint utilities"""
    import argparse

    parser = argparse.ArgumentParser(description='Checkpoint Utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Convert DeepSpeed to HF
    ds_parser = subparsers.add_parser('convert-ds', help='Convert DeepSpeed checkpoint to HuggingFace')
    ds_parser.add_argument('--checkpoint_dir', required=True, help='DeepSpeed checkpoint directory')
    ds_parser.add_argument('--output_dir', required=True, help='Output directory')
    ds_parser.add_argument('--zero_stage', type=int, default=3, help='ZeRO stage')

    # Convert FP4 to BF16
    fp_parser = subparsers.add_parser('convert-fp4', help='Convert FP4/FP8 to BF16')
    fp_parser.add_argument('--model_path', required=True, help='Model path')
    fp_parser.add_argument('--output_path', required=True, help='Output path')

    # Verify checkpoint
    verify_parser = subparsers.add_parser('verify', help='Verify checkpoint')
    verify_parser.add_argument('--checkpoint_path', required=True, help='Checkpoint path')

    # Get info
    info_parser = subparsers.add_parser('info', help='Get checkpoint info')
    info_parser.add_argument('--checkpoint_path', required=True, help='Checkpoint path')

    args = parser.parse_args()

    if args.command == 'convert-ds':
        convert_deepspeed_to_hf(args.checkpoint_dir, args.output_dir, args.zero_stage)
    elif args.command == 'convert-fp4':
        convert_fp4_to_bf16(args.model_path, args.output_path)
    elif args.command == 'verify':
        verify_checkpoint(args.checkpoint_path)
    elif args.command == 'info':
        info = get_checkpoint_info(args.checkpoint_path)
        import json
        print(json.dumps(info, indent=2))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
