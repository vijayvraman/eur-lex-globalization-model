#!/bin/bash
# Setup script for Mac Studio (Data Processing Only)

set -e

echo "================================"
echo "EUR-Lex Model - Mac Studio Setup"
echo "Data Processing Environment"
echo "================================"

# Check if on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Warning: This script is intended for macOS"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Use .venv if it exists (PyCharm default), otherwise create venv
if [ -d ".venv" ]; then
    echo "Using existing .venv virtual environment..."
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    echo "Using existing venv virtual environment..."
    VENV_DIR="venv"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    VENV_DIR="venv"
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch for macOS (CPU/MPS)..."
pip3 install torch torchvision torchaudio

echo "Installing data processing dependencies..."
# Install only what's needed for data processing
pip3 install \
    "transformers>=4.36.0" \
    "datasets>=2.16.0" \
    "pyarrow>=14.0.0" \
    "pandas>=2.0.0" \
    "lxml>=4.9.0" \
    "beautifulsoup4>=4.12.0" \
    "tokenizers>=0.15.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.66.0" \
    "numpy>=1.24.0" \
    "pytest>=7.4.0"

echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python3 -c "import datasets; print(f'✓ Datasets: {datasets.__version__}')"
python3 -c "import lxml; print(f'✓ LXML: {lxml.__version__}')"

echo ""
echo "================================"
echo "Mac Studio Setup Complete!"
echo "================================"
echo ""
echo "This environment is configured for:"
echo "  ✓ FORMEX XML parsing"
echo "  ✓ CPT corpus building"
echo "  ✓ SFT dataset generation"
echo ""
echo "Next steps:"
echo "1. Place your FORMEX XML files in data/raw/"
echo "2. Parse XML files:"
echo "   python data_processing/parsers/formex_parser.py --input data/raw --output data/parsed --workers 24"
echo "3. Build CPT corpus:"
echo "   python data_processing/dataset_builders/cpt_corpus_builder.py --input_dir data/parsed --output_dir data/cpt"
echo ""
echo "For GPU training, transfer the processed data to your GPU cluster"
echo "and run setup_gpu.sh there."
