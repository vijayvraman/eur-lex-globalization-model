"""
FP8/FP4 Utilities for Transformer Engine

Utilities for:
- Monitoring FP8 scaling factors
- Calibration
- Performance tracking
- FP8 health checks
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FP8Monitor:
    """Monitor FP8/FP4 quantization during training"""

    def __init__(self):
        self.scaling_history = []
        self.overflow_count = 0
        self.underflow_count = 0

    def log_scaling_factors(self, step: int, amax_history: Dict[str, float]):
        """
        Log scaling factors for monitoring

        Args:
            step: Training step
            amax_history: Dictionary of amax values per layer/tensor
        """
        self.scaling_history.append({
            'step': step,
            'amax_values': amax_history
        })

    def check_for_anomalies(self, threshold_overflow: float = 1e10,
                           threshold_underflow: float = 1e-10) -> Dict:
        """
        Check for numerical anomalies in scaling factors

        Args:
            threshold_overflow: Threshold for detecting overflow
            threshold_underflow: Threshold for detecting underflow

        Returns:
            Dictionary with anomaly statistics
        """
        if not self.scaling_history:
            return {}

        latest = self.scaling_history[-1]
        amax_values = list(latest['amax_values'].values())

        overflow = sum(1 for v in amax_values if v > threshold_overflow)
        underflow = sum(1 for v in amax_values if v < threshold_underflow)

        if overflow > 0:
            self.overflow_count += overflow
            logger.warning(f"Detected {overflow} overflow conditions in FP8 scaling")

        if underflow > 0:
            self.underflow_count += underflow
            logger.warning(f"Detected {underflow} underflow conditions in FP8 scaling")

        return {
            'overflow_detected': overflow,
            'underflow_detected': underflow,
            'total_overflow': self.overflow_count,
            'total_underflow': self.underflow_count,
            'amax_mean': np.mean(amax_values),
            'amax_std': np.std(amax_values),
            'amax_min': np.min(amax_values),
            'amax_max': np.max(amax_values)
        }

    def get_statistics(self) -> Dict:
        """Get statistics on FP8 scaling over training"""
        if not self.scaling_history:
            return {}

        all_amax = []
        for entry in self.scaling_history:
            all_amax.extend(entry['amax_values'].values())

        return {
            'num_steps_monitored': len(self.scaling_history),
            'amax_overall_mean': np.mean(all_amax),
            'amax_overall_std': np.std(all_amax),
            'total_overflows': self.overflow_count,
            'total_underflows': self.underflow_count
        }


def check_fp8_compatibility():
    """
    Check if FP8/FP4 is supported

    Returns:
        Boolean indicating support
    """
    try:
        import transformer_engine
        logger.info("Transformer Engine available")
        return True
    except ImportError:
        logger.warning("Transformer Engine not available")
        return False


def get_fp8_recipe(fp8_format: str = 'hybrid',
                   amax_history_len: int = 1024,
                   amax_compute_algo: str = 'max'):
    """
    Get FP8 recipe for Transformer Engine

    Args:
        fp8_format: FP8 format ('E4M3', 'E5M2', or 'hybrid')
        amax_history_len: Length of amax history for scaling
        amax_compute_algo: Algorithm for computing amax ('max' or 'most_recent')

    Returns:
        FP8 recipe object
    """
    try:
        from transformer_engine.common import recipe

        fp8_recipe = recipe.DelayedScaling(
            fp8_format=getattr(recipe.Format, fp8_format.upper()),
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo
        )

        logger.info(f"FP8 recipe created: format={fp8_format}, "
                   f"history_len={amax_history_len}, algo={amax_compute_algo}")

        return fp8_recipe

    except ImportError:
        logger.error("Transformer Engine not available")
        return None


def enable_fp8_autocast(enabled: bool = True, fp8_recipe=None):
    """
    Context manager for FP8 autocasting

    Args:
        enabled: Whether to enable FP8
        fp8_recipe: FP8 recipe (if None, uses default)

    Returns:
        Context manager for FP8 autocasting
    """
    try:
        import transformer_engine.pytorch as te

        if fp8_recipe is None:
            fp8_recipe = get_fp8_recipe()

        return te.fp8_autocast(enabled=enabled, fp8_recipe=fp8_recipe)

    except ImportError:
        logger.error("Transformer Engine not available")
        # Return dummy context manager
        from contextlib import nullcontext
        return nullcontext()


class FP8PerformanceTracker:
    """Track performance metrics with FP8 enabled"""

    def __init__(self):
        self.throughput_history = []
        self.memory_usage_history = []

    def log_throughput(self, step: int, tokens_per_second: float):
        """Log throughput at step"""
        self.throughput_history.append({
            'step': step,
            'tokens_per_second': tokens_per_second
        })

    def log_memory_usage(self, step: int, memory_gb: float):
        """Log memory usage at step"""
        self.memory_usage_history.append({
            'step': step,
            'memory_gb': memory_gb
        })

    def get_summary(self) -> Dict:
        """Get performance summary"""
        if not self.throughput_history:
            return {}

        throughputs = [entry['tokens_per_second'] for entry in self.throughput_history]
        memories = [entry['memory_gb'] for entry in self.memory_usage_history]

        summary = {
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'max_throughput': np.max(throughputs),
            'min_throughput': np.min(throughputs),
        }

        if memories:
            summary.update({
                'avg_memory_gb': np.mean(memories),
                'max_memory_gb': np.max(memories),
            })

        return summary


def test_fp8_availability():
    """Test FP8 availability and print info"""
    print("=" * 60)
    print("FP8/FP4 Availability Check")
    print("=" * 60)

    # Check Transformer Engine
    te_available = check_fp8_compatibility()
    print(f"Transformer Engine: {'✓ Available' if te_available else '✗ Not available'}")

    if te_available:
        import transformer_engine
        print(f"Version: {transformer_engine.__version__}")

        # Try to create recipe
        recipe = get_fp8_recipe()
        if recipe:
            print("✓ FP8 recipe created successfully")
        else:
            print("✗ Failed to create FP8 recipe")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("✗ CUDA not available")
    except:
        print("✗ PyTorch not available")

    print("=" * 60)


if __name__ == '__main__':
    test_fp8_availability()
