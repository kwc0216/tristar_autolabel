#!/bin/bash
#SBATCH --job-name=tristar_all
#SBATCH --account=def-your_account    # 替换为你的账户名
#SBATCH --time=72:00:00               # 最大运行时间 72小时（批量训练需要更长时间）
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # CPU核心数
#SBATCH --mem=64G                     # 内存
#SBATCH --gres=gpu:1                  # 请求1个GPU
#SBATCH --output=logs/slurm_all-%j.out
#SBATCH --error=logs/slurm_all-%j.err

# ============================================================================
# Tristar Batch Training Script for Digital Alliance (Fir)
# Runs all training configurations using train_all.sh
# ============================================================================

echo "=========================================="
echo "Batch training job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="

# Load necessary modules
module load python/3.8
module load cuda/11.8
module load cudnn/8.6

# Create logs directory
mkdir -p logs

# Set up Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    virtualenv --no-download venv
    source venv/bin/activate

    echo "Installing dependencies..."
    pip install --no-index --upgrade pip

    # Install PyTorch with CUDA support
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

    # Install other requirements
    pip install lightning torchmetrics click scikit-learn numpy pandas
else
    source venv/bin/activate
fi

# Verify CUDA
echo "=========================================="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "=========================================="

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Number of epochs for each configuration
EPOCHS=${1:-10}

echo "Running all training configurations with $EPOCHS epochs..."
echo "=========================================="

# Run train_all.sh
bash train_all.sh $EPOCHS

echo "=========================================="
echo "All training completed at: $(date)"
echo "=========================================="
