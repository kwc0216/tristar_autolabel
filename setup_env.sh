#!/bin/bash
# ============================================================================
# Environment Setup Script for Tristar on Digital Alliance (Fir)
# Run this once to set up the environment before training
# ============================================================================

echo "=========================================="
echo "Setting up Tristar environment..."
echo "=========================================="

# Load necessary modules
echo "Loading modules..."
module load python/3.8
module load cuda/11.8
module load cudnn/8.6

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    virtualenv --no-download venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --no-index --upgrade pip

# Install PyTorch with CUDA support
echo "=========================================="
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install Lightning and metrics
echo "=========================================="
echo "Installing Lightning and dependencies..."
pip install lightning==2.0.0
pip install torchmetrics

# Install other dependencies
echo "Installing other dependencies..."
pip install click
pip install scikit-learn
pip install numpy
pip install pandas
pip install pillow
pip install matplotlib

# Verify installation
echo "=========================================="
echo "Verifying installation..."
python -c "
import torch
import lightning
import torchmetrics
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'Lightning version: {lightning.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
print('=' * 50)
print('✅ Environment setup complete!')
"

echo "=========================================="
echo "Setup complete!"
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo "=========================================="
