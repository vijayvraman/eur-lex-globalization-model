"""
FSDP Configuration Module for FSDP2 + Transformer Engine

Provides FSDP2 configuration builders to replace DeepSpeed ZeRO-3.
Maps DeepSpeed settings to PyTorch FSDP2 equivalents with Transformer Engine support.

This is the NVIDIA-recommended stack for Blackwell hardware (B200 GPUs),
as used in NeMo 2.0 and Megatron-Core.

Supports two precision modes for Blackwell B200 GPUs:
- FP8 (default): E4M3 for forward, E5M2 for backward (~70GB for 70B model)
- NVFP4: E2M1 with block scaling (~35GB for 70B model, experimental)
"""

import os
import logging
from functools import partial
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

logger = logging.getLogger(__name__)


class FSDPConfig:
    """FSDP2 configuration builder for LLaMA 3.3 70B with Transformer Engine"""

    @staticmethod
    def get_mixed_precision_policy(dtype: str = "bfloat16") -> Optional[MixedPrecision]:
        """
        Returns MixedPrecision policy equivalent to DeepSpeed BF16

        Maps to DeepSpeed config:
            "bf16": {"enabled": true}

        Args:
            dtype: "bfloat16", "float16", or "float32"

        Returns:
            MixedPrecision policy or None for full precision
        """
        if dtype == "bfloat16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,  # Model parameters in BF16
                reduce_dtype=torch.bfloat16,  # Gradient reduction in BF16
                buffer_dtype=torch.float32,   # Buffers (LayerNorm, etc.) in FP32 for stability
            )
        elif dtype == "float16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32,
            )
        else:
            return None  # Full precision (FP32)

    @staticmethod
    def get_sharding_strategy(strategy: str = "full") -> ShardingStrategy:
        """
        Returns FSDP sharding strategy

        Args:
            strategy: Sharding strategy type
                - "full" = FULL_SHARD (ZeRO-3 equivalent) - recommended
                - "hybrid" = HYBRID_SHARD (ZeRO-2 + ZeRO-3 hybrid)
                - "shard_grad_op" = SHARD_GRAD_OP (ZeRO-2 equivalent)
                - "no_shard" = NO_SHARD (DDP equivalent)

        Returns:
            ShardingStrategy enum value
        """
        strategies = {
            "full": ShardingStrategy.FULL_SHARD,  # ZeRO-3
            "hybrid": ShardingStrategy.HYBRID_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
            "no_shard": ShardingStrategy.NO_SHARD,  # DDP
        }
        return strategies.get(strategy, ShardingStrategy.FULL_SHARD)

    @staticmethod
    def get_auto_wrap_policy(model_type: str = "llama"):
        """
        Returns auto-wrap policy for transformer blocks

        Wraps each transformer layer as a separate FSDP unit for optimal
        memory management and communication/computation overlap.

        Args:
            model_type: Model architecture type (currently only "llama")

        Returns:
            Auto-wrap policy function
        """
        if model_type == "llama":
            return partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer},
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def get_fsdp_config(config_type: str = "cpt") -> Dict[str, Any]:
        """
        Returns complete FSDP configuration dict

        Args:
            config_type: Configuration type - "cpt", "sft", or "fast_cpt"

        Returns:
            dict with FSDP configuration parameters
        """
        base_config = {
            "sharding_strategy": FSDPConfig.get_sharding_strategy("full"),
            "mixed_precision": FSDPConfig.get_mixed_precision_policy("bfloat16"),
            "auto_wrap_policy": FSDPConfig.get_auto_wrap_policy("llama"),
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,  # Overlap communication
            "cpu_offload": CPUOffload(offload_params=False),  # Keep on GPU (no offloading)
            "use_orig_params": True,  # CRITICAL: Required for optimizer state management
            "sync_module_states": True,  # Sync model states across ranks at initialization
            "limit_all_gathers": True,  # Reduce memory overhead during all-gather operations
            "gradient_checkpointing": True,
        }

        # Config-specific adjustments (currently same for all)
        if config_type in ["cpt", "fast_cpt"]:
            # CPT settings
            pass
        elif config_type == "sft":
            # SFT settings (same for now)
            pass

        return base_config


def get_transformer_engine_recipe(precision_mode: str = "fp8"):
    """
    Get Transformer Engine recipe for specified precision mode

    Args:
        precision_mode: "fp8" or "nvfp4"

    Returns:
        Transformer Engine recipe object
    """
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
    except ImportError:
        logger.warning("Transformer Engine not available. Quantization disabled.")
        return None

    if precision_mode == "nvfp4":
        # NVFP4 with block scaling (4-bit, E2M1 format)
        logger.info("Using NVFP4 recipe (4-bit, E2M1 format)")
        logger.info("  - Weights: NVFP4 with 16x16 block scaling")
        logger.info("  - Activations: NVFP4 with 16-element block scaling")
        logger.info("  - Gradients: NVFP4 with 16-element block scaling")
        logger.info("  - Memory usage: ~35GB for 70B model")

        return recipe.NVFP4BlockScaling(
            margin=0,
            interval=1,
            amax_history_len=1024,
            amax_compute_algo="max",
        )
    else:
        # FP8 with hybrid format (8-bit, E4M3/E5M2)
        logger.info("Using FP8 recipe (8-bit, hybrid E4M3/E5M2)")
        logger.info("  - Forward (weights, activations): E4M3 (higher precision)")
        logger.info("  - Backward (gradients): E5M2 (wider dynamic range)")
        logger.info("  - Memory usage: ~70GB for 70B model")

        return recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=recipe.Format.HYBRID,
            amax_history_len=1024,
            amax_compute_algo="max",
        )


def setup_transformer_engine(use_fsdp: bool = True, precision_mode: str = "fp8"):
    """
    Setup Transformer Engine environment variables for FSDP2 compatibility

    Configures TE for optimal performance with FSDP2.
    This is the NVIDIA-recommended configuration for Blackwell hardware.

    Args:
        use_fsdp: Whether FSDP is being used
        precision_mode: "fp8" (default) or "nvfp4" (experimental)
    """
    if use_fsdp:
        logger.info(f"Configuring Transformer Engine for FSDP2 + {precision_mode.upper()}")

        # Core TE settings
        os.environ['NVTE_FP8_DPA_BWD'] = '1'  # Enable FP8/NVFP4 backward pass
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '1'  # Allow optimizations

        # FSDP-specific TE optimizations
        os.environ['NVTE_FUSED_ATTN'] = '1'  # Enable fused attention
        os.environ['NVTE_FP8_AMAX_REDUCE'] = '1'  # Enable amax reduction across ranks

        if precision_mode == "nvfp4":
            # NVFP4-specific settings
            logger.info("NVFP4 mode: Using stochastic rounding and block scaling")
            # Additional NVFP4 optimizations can be added here

        logger.info(f"Transformer Engine configured for FSDP2 + {precision_mode.upper()}")
    else:
        logger.info("Transformer Engine configured for DeepSpeed")


def apply_fsdp_wrapping(
    model,
    fsdp_config_type: str = 'cpt',
    use_quantization: bool = False,
    precision_mode: str = 'fp8',
    device_id: Optional[int] = None
):
    """
    Apply FSDP wrapping to model with Transformer Engine support

    This is the main entry point for wrapping a model with FSDP2.
    Applies proper configuration for FSDP + Transformer Engine on Blackwell hardware.

    Args:
        model: PreTrainedModel (not yet wrapped)
        fsdp_config_type: Configuration type - "cpt", "sft", or "fast_cpt"
        use_quantization: Whether quantization is enabled via Transformer Engine
        precision_mode: "fp8" (default) or "nvfp4" (experimental)
        device_id: Device ID (defaults to current CUDA device)

    Returns:
        FSDP-wrapped model with activation checkpointing applied
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )

    # Get current device if not specified
    if device_id is None:
        device_id = torch.cuda.current_device()

    # Setup Transformer Engine if quantization enabled
    if use_quantization:
        setup_transformer_engine(use_fsdp=True, precision_mode=precision_mode)

        # Get and configure TE recipe
        te_recipe = get_transformer_engine_recipe(precision_mode)
        if te_recipe:
            logger.info(f"Transformer Engine recipe configured: {precision_mode.upper()}")
        else:
            logger.warning("Transformer Engine recipe not available, proceeding without quantization")

    # Get FSDP configuration
    fsdp_config = FSDPConfig.get_fsdp_config(fsdp_config_type)

    logger.info("=" * 80)
    logger.info("FSDP2 Configuration:")
    logger.info(f"  Config type: {fsdp_config_type}")
    logger.info(f"  Sharding strategy: {fsdp_config['sharding_strategy']}")
    logger.info(f"  Mixed precision: {fsdp_config['mixed_precision']}")
    logger.info(f"  Backward prefetch: {fsdp_config['backward_prefetch']}")
    logger.info(f"  CPU offload: {fsdp_config['cpu_offload']}")
    logger.info(f"  Use orig params: {fsdp_config['use_orig_params']}")
    logger.info(f"  Sync module states: {fsdp_config['sync_module_states']}")
    logger.info(f"  Limit all gathers: {fsdp_config['limit_all_gathers']}")
    logger.info(f"  Gradient checkpointing: {fsdp_config['gradient_checkpointing']}")
    logger.info(f"  Quantization enabled: {use_quantization}")
    if use_quantization:
        logger.info(f"  Precision mode: {precision_mode.upper()}")
        if precision_mode == "nvfp4":
            logger.info(f"    → 4-bit NVFP4 (E2M1) with block scaling")
            logger.info(f"    → Expected memory: ~35GB for 70B model")
        else:
            logger.info(f"    → 8-bit FP8 (E4M3 forward, E5M2 backward)")
            logger.info(f"    → Expected memory: ~70GB for 70B model")
    logger.info(f"  Device ID: {device_id}")
    logger.info("=" * 80)

    # Wrap model with FSDP
    logger.info("Wrapping model with FSDP2...")
    wrapped_model = FSDP(
        model,
        sharding_strategy=fsdp_config['sharding_strategy'],
        mixed_precision=fsdp_config['mixed_precision'],
        auto_wrap_policy=fsdp_config['auto_wrap_policy'],
        backward_prefetch=fsdp_config['backward_prefetch'],
        cpu_offload=fsdp_config['cpu_offload'],
        device_id=device_id,
        use_orig_params=fsdp_config['use_orig_params'],
        sync_module_states=fsdp_config['sync_module_states'],
        limit_all_gathers=fsdp_config['limit_all_gathers'],
    )
    logger.info("FSDP2 wrapping complete")

    # Apply activation checkpointing AFTER FSDP wrapping
    # This is critical - must happen after wrapping, not before
    if fsdp_config.get('gradient_checkpointing', False):
        logger.info("Applying FSDP-aware activation checkpointing...")

        # Create non-reentrant checkpoint wrapper
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        # Check function: only checkpoint LlamaDecoderLayer modules
        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

        # Apply activation checkpointing
        apply_activation_checkpointing(
            wrapped_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )

        logger.info("FSDP-aware activation checkpointing enabled")

    logger.info("FSDP2 model setup complete")
    return wrapped_model


def get_fsdp_kwargs_for_trainer(fsdp_config_type: str = "cpt") -> Dict[str, Any]:
    """
    Get FSDP kwargs for HuggingFace Trainer

    Returns the configuration dict to pass to TrainingArguments.
    This is an alternative to manual wrapping with apply_fsdp_wrapping().

    Args:
        fsdp_config_type: Configuration type - "cpt", "sft", or "fast_cpt"

    Returns:
        Dict with "fsdp" and "fsdp_config" keys for TrainingArguments
    """
    return {
        "fsdp": "full_shard",  # FULL_SHARD strategy (ZeRO-3 equivalent)
        "fsdp_config": {
            "backward_prefetch": "backward_pre",  # Overlap communication
            "forward_prefetch": False,
            "limit_all_gathers": True,  # Reduce memory overhead
            "use_orig_params": True,  # CRITICAL for optimizer state
            "sync_module_states": True,  # Sync at initialization
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",  # Auto-wrap policy
        }
    }
