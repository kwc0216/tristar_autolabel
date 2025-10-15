#!/bin/bash
#SBATCH --job-name=tristar_train
#SBATCH --account=def-your_account    # 替换为你的账户名
#SBATCH --time=24:00:00               # 最大运行时间 24小时
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4             # CPU核心数
#SBATCH --mem=32G                     # 内存
#SBATCH --gres=gpu:1                  # 请求1个GPU
#SBATCH --output=logs/slurm-%j.out    # 标准输出日志
#SBATCH --error=logs/slurm-%j.err     # 错误输出日志

# ============================================================================
# Tristar Training Setup and Execution Script for Digital Alliance (Fir)
# Version 2 - With flexible data path support
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="

# Load necessary modules
module load python/3.8
module load cuda/11.8  # 根据可用的CUDA版本调整
module load cudnn/8.6

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Data Path Configuration
# ============================================================================
# Option 1: Use SCRATCH for better I/O performance (recommended)
DATA_DIR="${SCRATCH}/tristar_data"

# Option 2: Use PROJECT for shared data
# DATA_DIR="${PROJECT}/tristar_data"

# Option 3: Use local directory (only for small datasets)
# DATA_DIR="$(pwd)/data"

echo "Using data directory: $DATA_DIR"

# Check if data exists
if [ ! -d "$DATA_DIR/tristar/train" ]; then
    echo "=========================================="
    echo "ERROR: Training data not found!"
    echo "Expected location: $DATA_DIR/tristar/train"
    echo ""
    echo "Please upload your data using one of these methods:"
    echo "  1. scp -r data/tristar your_username@fir.alliancecan.ca:\$SCRATCH/tristar_data/"
    echo "  2. rsync -avzP data/tristar/ your_username@fir.alliancecan.ca:\$SCRATCH/tristar_data/tristar/"
    echo "=========================================="
    exit 1
fi

# Create symbolic link to data
if [ ! -L "data" ]; then
    ln -s $DATA_DIR data
    echo "Created symbolic link: data -> $DATA_DIR"
fi

# ============================================================================
# Environment Setup
# ============================================================================
# Set up Python virtual environment (first time setup)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    virtualenv --no-download venv
    source venv/bin/activate

    echo "Installing dependencies..."
    pip install --no-index --upgrade pip

    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA 11.8..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

    # Install other requirements
    echo "Installing other dependencies..."
    pip install lightning torchmetrics click scikit-learn numpy pandas

    echo "Virtual environment setup complete!"
else
    echo "Using existing virtual environment..."
    source venv/bin/activate
fi

# Verify CUDA availability
echo "=========================================="
echo "Verifying CUDA setup..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "=========================================="

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================================
# Training Configuration
# ============================================================================
TASK="action_classification"
ARCHITECTURE="dependent"
EPOCHS=10
BATCH_SIZE=8  # 根据GPU显存调整
LEARNING_RATE=0.0001

echo "=========================================="
echo "Training Configuration:"
echo "  Task: $TASK"
echo "  Architecture: $ARCHITECTURE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Data Path: $DATA_DIR/tristar"
echo "=========================================="

# Run training
echo "Starting training..."
python -m tristar.train \
    --task $TASK \
    --architecture $ARCHITECTURE \
    --rgb \
    --no-depth \
    --no-thermal \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE

echo "=========================================="
echo "Training completed!"
echo "Job finished at: $(date)"
echo "=========================================="
