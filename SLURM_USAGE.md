# Tristar Training on Digital Alliance (Fir) - Usage Guide

## 📋 前置准备

### 1. 在 Fir 上克隆项目
```bash
cd $HOME
git clone https://github.com/kwc0216/tristar_autolabel.git
cd tristar_autolabel
```

### 2. 上传数据集
将你的数据上传到 `data/tristar/` 目录：
```bash
# 在本地电脑上，使用 scp 上传
scp -r data/tristar username@fir.alliancecan.ca:~/tristar_autolabel/data/

# 或者使用 Globus 等工具传输大文件
```

确保数据结构如下：
```
data/tristar/
├── train/
├── val/
└── test/
```

---

## 🚀 快速开始

### 方法 1: 单个训练任务（推荐用于测试）

**步骤 1: 修改账户信息**
```bash
nano setup_and_train.sh
# 将 #SBATCH --account=def-your_account 改为你的实际账户
```

**步骤 2: 提交任务**
```bash
sbatch setup_and_train.sh
```

**步骤 3: 查看任务状态**
```bash
squeue -u $USER
```

**步骤 4: 查看日志**
```bash
# 查看标准输出
tail -f logs/slurm-<job_id>.out

# 查看错误输出
tail -f logs/slurm-<job_id>.err
```

---

### 方法 2: 批量训练所有配置

**步骤 1: 修改账户信息**
```bash
nano sbatch_train_all.sh
# 修改 #SBATCH --account
```

**步骤 2: 提交批量任务**
```bash
# 运行 10 个 epochs
sbatch sbatch_train_all.sh 10

# 或指定其他 epoch 数
sbatch sbatch_train_all.sh 50
```

---

### 方法 3: 手动设置环境（高级用户）

**步骤 1: 设置环境**
```bash
bash setup_env.sh
```

**步骤 2: 交互式调试**
```bash
# 申请交互式 GPU 节点
salloc --time=2:00:00 --mem=32G --gres=gpu:1 --account=def-your_account

# 激活环境
source venv/bin/activate

# 运行训练
python -m tristar.train --task action_classification --architecture dependent --rgb --epochs 5 --batch_size 8
```

**步骤 3: 提交自定义任务**
创建自己的 sbatch 脚本或直接提交：
```bash
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=my_train
#SBATCH --account=def-your_account
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/my_train-%j.out

module load python/3.8 cuda/11.8 cudnn/8.6
source venv/bin/activate

python -m tristar.train \
    --task action_classification \
    --architecture dependent \
    --rgb --depth --thermal \
    --epochs 20 \
    --batch_size 16
EOF
```

---

## 📊 监控和管理任务

### 查看任务状态
```bash
# 查看所有任务
squeue -u $USER

# 查看特定任务详情
scontrol show job <job_id>
```

### 取消任务
```bash
scancel <job_id>

# 取消所有你的任务
scancel -u $USER
```

### 查看任务历史
```bash
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
```

### 实时监控日志
```bash
# 监控标准输出
tail -f logs/slurm-<job_id>.out

# 监控训练日志
tail -f logs/action_classification-*/*.log
```

---

## ⚙️ 配置调整

### 调整 SLURM 参数

编辑 `setup_and_train.sh` 或 `sbatch_train_all.sh`：

```bash
#SBATCH --time=24:00:00       # 时间限制（HH:MM:SS）
#SBATCH --mem=32G             # 内存需求
#SBATCH --gres=gpu:1          # GPU数量（v100, a100等）
#SBATCH --cpus-per-task=4     # CPU核心数
```

### 调整训练参数

修改脚本中的参数：
```bash
EPOCHS=10
BATCH_SIZE=8         # 根据GPU显存调整
LEARNING_RATE=0.0001
```

### 选择不同的模态组合

```bash
# 只使用 RGB
python -m tristar.train --rgb --no-depth --no-thermal

# 使用 RGB + Depth
python -m tristar.train --rgb --depth --no-thermal

# 使用所有模态
python -m tristar.train --rgb --depth --thermal
```

---

## 📁 输出文件位置

训练完成后，你会在以下位置找到输出：

```
logs/                                    # SLURM 日志
├── slurm-<job_id>.out                  # 标准输出
├── slurm-<job_id>.err                  # 错误输出
└── <config_name>_<timestamp>/          # 训练日志
    ├── training.log                     # 详细训练日志（含指标）
    ├── test_metrics.json                # 测试指标
    └── metrics/                         # CSV格式指标
        └── version_0/
            └── metrics.csv

data/checkpoints/                        # 模型检查点
└── <config_name>/
    ├── epoch=00-val_loss=2.35.ckpt
    ├── epoch=05-val_loss=1.87.ckpt
    └── epoch=09-val_loss=1.65.ckpt
```

---

## 🔧 故障排查

### 问题 1: CUDA Out of Memory
**解决方案**: 减小 batch_size
```bash
--batch_size 4  # 或 2
```

### 问题 2: 数据路径错误
**解决方案**: 检查数据是否上传到正确位置
```bash
ls -la data/tristar/train
ls -la data/tristar/val
```

### 问题 3: 模块加载失败
**解决方案**: 检查可用的模块版本
```bash
module avail cuda
module avail cudnn
module avail python
```

### 问题 4: 依赖安装失败
**解决方案**: 手动安装依赖
```bash
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning torchmetrics
```

---

## 💡 最佳实践

1. **先用小规模测试**
   ```bash
   # 使用 1 个 epoch 测试
   python -m tristar.train --epochs 1 --batch_size 4
   ```

2. **使用 tmux 或 screen**
   ```bash
   tmux new -s tristar
   # 运行命令
   # Ctrl+B, D 分离会话
   tmux attach -t tristar  # 重新连接
   ```

3. **定期备份结果**
   ```bash
   # 将结果下载到本地
   scp -r username@fir.alliancecan.ca:~/tristar_autolabel/logs ./
   scp -r username@fir.alliancecan.ca:~/tristar_autolabel/data/checkpoints ./
   ```

4. **查看 GPU 使用情况**
   ```bash
   # 在交互式会话中
   nvidia-smi
   watch -n 1 nvidia-smi  # 实时监控
   ```

---

## 📚 更多资源

- [Digital Alliance Documentation](https://docs.alliancecan.ca/)
- [SLURM User Guide](https://slurm.schedmd.com/documentation.html)
- [Project GitHub](https://github.com/kwc0216/tristar_autolabel)

---

## 需要帮助？

如果遇到问题：
1. 检查日志文件 `logs/slurm-*.err`
2. 查看训练日志 `logs/*/training.log`
3. 联系 Digital Alliance 支持：support@tech.alliancecan.ca
