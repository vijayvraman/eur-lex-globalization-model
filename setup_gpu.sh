#!/bin/bash
# Setup script for GPU Cluster (4x B200 GPUs - Training)

set -e

echo "================================"
echo "EUR-Lex Model - GPU Cluster Setup"
echo "Training Environment (4x B200)"
echo "================================"

if [ -z "$TMUX" ]; then
    echo ""
    echo "WARNING: Not running inside a tmux session."
    echo "HIGHLY RECOMMENDED: run this script inside tmux for protecting"
    echo "flash-attn build (20-30 min) from terminal disconnects:"
    echo "  tmux new -s setup && bash setup_gpu.sh"
    echo ""
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPUs."
    exit 1
fi

echo "Detected GPUs:"
nvidia-smi --list-gpus

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.8+ support (required for Blackwell/B200)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing NVIDIA Transformer Engine (FP4/FP8 support)..."
pip3 install "transformer-engine[pytorch]>=1.5.0"

echo "Installing build tools..."
pip3 install wheel ninja packaging

echo "Installing core dependencies..."
pip3 install -r requirements.txt

echo "Installing Flash Attention from source (required for ABI compatibility with installed torch)..."
echo "This may take 20-30 minutes..."
mkdir -p /workspace/tmp
MAX_JOBS=4 TMPDIR=/workspace/tmp TORCH_CUDA_ARCH_LIST="10.0" pip3 install flash-attn --no-build-isolation --no-binary flash-attn --no-cache-dir

echo "Installing DeepSpeed with FP8 quantizer support..."
DS_BUILD_OPS=1 DS_BUILD_FP_QUANTIZER=1 pip3 install deepspeed

echo ""
echo "Verifying B200 GPU compatibility..."

# Check CUDA version >= 12.8 (required for Blackwell/B200)
python3 -c "
import torch
cuda_ver = torch.version.cuda
if cuda_ver is None:
    print('✗ CUDA not available')
    exit(1)
major, minor = map(int, cuda_ver.split('.')[:2])
if (major, minor) >= (12, 8):
    print(f'✓ CUDA {cuda_ver} meets B200 requirement (>=12.8)')
else:
    print(f'✗ CUDA {cuda_ver} is below B200 requirement (>=12.8)')
    exit(1)
"

# Check Transformer Engine version >= 1.5.0 (FP4 support for B200)
python3 -c "
import transformer_engine
from packaging.version import Version
te_ver = transformer_engine.__version__
if Version(te_ver) >= Version('1.5.0'):
    print(f'✓ Transformer Engine {te_ver} meets B200 FP4 requirement (>=1.5.0)')
else:
    print(f'✗ Transformer Engine {te_ver} is below B200 FP4 requirement (>=1.5.0)')
    exit(1)
"

# Check DeepSpeed version >= 0.14.0 (FP8 quantizer support)
python3 -c "
import deepspeed
from packaging.version import Version
ds_ver = deepspeed.__version__
if Version(ds_ver) >= Version('0.14.0'):
    print(f'✓ DeepSpeed {ds_ver} meets B200 requirement (>=0.14.0)')
else:
    print(f'✗ DeepSpeed {ds_ver} is below B200 requirement (>=0.14.0)')
    exit(1)
"

# Check Flash Attention version >= 2.6.0 (Blackwell/compute capability 10.0 support)
python3 -c "
import flash_attn
from packaging.version import Version
fa_ver = flash_attn.__version__
if Version(fa_ver) >= Version('2.6.0'):
    print(f'✓ Flash Attention {fa_ver} meets B200 requirement (>=2.6.0)')
else:
    print(f'✗ Flash Attention {fa_ver} is below B200 requirement (>=2.6.0)')
    exit(1)
"

echo ""
echo "================================"
echo "GPU Cluster Setup Complete!"
echo "================================"
echo ""
echo "Environment configured for:"
echo "  ✓ 4x B200 GPU training"
echo "  ✓ FP4 quantization via Transformer Engine"
echo "  ✓ DeepSpeed ZeRO-3 distributed training"
echo "  ✓ Flash Attention 2"
echo ""
echo "Next steps:"
echo "1. Transfer processed data from Mac Studio to this cluster:"
echo "   rsync -avz data/cpt/ user@gpu-cluster:/path/to/project/data/cpt/"
echo "2. Start CPT training:"
echo "   export NVTE_FP8_DPA_BWD=1"
echo "   export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1"
echo "   deepspeed --num_gpus=4 scripts/train_cpt.py --config configs/cpt_config.yaml"
echo "3. Monitor training:"
echo "   wandb login  # Configure W&B for monitoring"
