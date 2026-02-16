#!/bin/bash
# Setup script for GPU Cluster (4x B200 GPUs - Training)

set -e

echo "================================"
echo "EUR-Lex Model - GPU Cluster Setup"
echo "Training Environment (4x B200)"
echo "================================"

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

echo "Installing PyTorch with CUDA 12.1+ support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing NVIDIA Transformer Engine (FP4/FP8 support)..."
pip3 install transformer-engine[pytorch]>=1.5.0

echo "Installing core dependencies..."
pip3 install -r requirements.txt

echo "Installing Flash Attention (this may take a few minutes)..."
pip3 install flash-attn --no-build-isolation

echo "Installing DeepSpeed with FP8 quantizer support..."
DS_BUILD_OPS=1 DS_BUILD_FP_QUANTIZER=1 pip3 install deepspeed

echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✓ CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'✓ CUDA Version: {torch.version.cuda}')"
python3 -c "import torch; print(f'✓ Number of GPUs: {torch.cuda.device_count()}')"
python3 -c "import transformer_engine; print(f'✓ Transformer Engine: {transformer_engine.__version__}')"
python3 -c "import deepspeed; print(f'✓ DeepSpeed: {deepspeed.__version__}')"
python3 -c "import flash_attn; print(f'✓ Flash Attention: {flash_attn.__version__}')"

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
