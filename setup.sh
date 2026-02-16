#!/bin/bash
# Auto-detecting setup script for EUR-Lex Legal Q&A Model Training

set -e

echo "================================"
echo "EUR-Lex Model Training Setup"
echo "Auto-detecting environment..."
echo "================================"
echo ""

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected: macOS (Mac Studio)"
    echo "This will install data processing environment only."
    echo ""
    read -p "Continue with Mac setup? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        chmod +x setup_mac.sh
        ./setup_mac.sh
    fi
elif command -v nvidia-smi &> /dev/null; then
    echo "Detected: Linux with NVIDIA GPU"
    echo "This will install full training environment with CUDA support."
    echo ""
    read -p "Continue with GPU cluster setup? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        chmod +x setup_gpu.sh
        ./setup_gpu.sh
    fi
else
    echo "Detected: Linux without NVIDIA GPU"
    echo "For data processing only, use: ./setup_mac.sh"
    echo "For GPU training, run this on a machine with NVIDIA GPUs."
    exit 1
fi
