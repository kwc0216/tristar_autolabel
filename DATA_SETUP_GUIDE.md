# 数据设置指南 - Digital Alliance (Fir) 服务器

## 📁 数据目录结构

你的数据应该具有以下结构：

```
tristar/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── ...
├── val/
│   ├── 0/
│   ├── 1/
│   └── ...
└── test/
    ├── 0/
    ├── 1/
    └── ...
```

---

## 🎯 推荐的数据存放位置

### 在 Fir 服务器上的三个存储选项：

| 位置 | 路径 | 适用场景 | 容量 | 速度 |
|------|------|----------|------|------|
| **SCRATCH** | `$SCRATCH/tristar_data` | 训练时使用（推荐） | 大（TB级） | 最快 |
| **PROJECT** | `$PROJECT/tristar_data` | 长期存储和共享 | 大（TB级） | 快 |
| **HOME** | `~/tristar_autolabel/data` | 小数据集 | 小（几GB） | 一般 |

**推荐使用 SCRATCH** 因为：
- I/O 性能最好，训练速度更快
- 容量大，适合大型数据集
- 专为计算任务设计

---

## 🚀 数据上传步骤

### 步骤 1: 在 Fir 上创建数据目录

```bash
# SSH 登录到 Fir
ssh your_username@fir.alliancecan.ca

# 创建数据目录
mkdir -p $SCRATCH/tristar_data
```

### 步骤 2: 从本地上传数据

#### 方法 A: 使用 SCP（简单，适合小数据集 < 10GB）

**在 Windows 上使用 Git Bash 或 PowerShell:**
```bash
# 上传整个 tristar 文件夹
scp -r f:/tristar/data/tristar your_username@fir.alliancecan.ca:$SCRATCH/tristar_data/

# 或者使用绝对路径
scp -r "f:\tristar\data\tristar" your_username@fir.alliancecan.ca:/scratch/your_username/tristar_data/
```

#### 方法 B: 使用 Rsync（推荐，支持断点续传）

```bash
# 在 Git Bash 中运行
rsync -avzP --info=progress2 \
    f:/tristar/data/tristar/ \
    your_username@fir.alliancecan.ca:$SCRATCH/tristar_data/tristar/

# 参数说明：
# -a: 归档模式，保持文件属性
# -v: 详细输出
# -z: 压缩传输
# -P: 显示进度并支持断点续传
```

**优点：**
- 可以断点续传（网络中断后可继续）
- 只传输变化的文件
- 传输前压缩，节省带宽

#### 方法 C: 使用 Globus（最快，适合大数据集 > 100GB）

1. **设置 Globus:**
   - 访问 https://globus.computecanada.ca/
   - 登录你的账户

2. **配置本地端点:**
   - 下载并安装 Globus Connect Personal
   - 创建个人端点
   - 设置访问路径到 `f:\tristar\data`

3. **配置远程端点:**
   - 搜索 "computecanada#fir-dtn"
   - 选择目标路径：`/scratch/your_username/tristar_data/`

4. **开始传输:**
   - 选择源文件夹：`tristar`
   - 点击 "Start" 开始传输
   - 可以关闭浏览器，传输在后台继续

**优点：**
- 速度最快（可达 10Gbps+）
- 自动重试和恢复
- 可以关闭电脑，传输继续进行
- 适合 TB 级数据

---

## 🔍 验证数据上传

### 在 Fir 上检查数据：

```bash
# SSH 到 Fir
ssh your_username@fir.alliancecan.ca

# 检查数据目录结构
ls -lh $SCRATCH/tristar_data/tristar/

# 应该看到：
# drwxr-xr-x train/
# drwxr-xr-x val/
# drwxr-xr-x test/

# 检查每个子目录
ls $SCRATCH/tristar_data/tristar/train/ | head
ls $SCRATCH/tristar_data/tristar/val/ | head
ls $SCRATCH/tristar_data/tristar/test/ | head

# 统计文件数量
find $SCRATCH/tristar_data/tristar/train -type f | wc -l
find $SCRATCH/tristar_data/tristar/val -type f | wc -l
find $SCRATCH/tristar_data/tristar/test -type f | wc -l

# 检查总大小
du -sh $SCRATCH/tristar_data/tristar/
```

---

## ⚙️ 配置训练脚本使用数据

### 选项 1: 使用 setup_and_train_v2.sh（推荐）

这个脚本会自动处理数据路径：

```bash
# 1. 修改脚本中的账户名
nano setup_and_train_v2.sh
# 修改: #SBATCH --account=def-your_account

# 2. 脚本默认使用 $SCRATCH/tristar_data
#    如果数据在其他位置，修改 DATA_DIR 变量

# 3. 提交任务
sbatch setup_and_train_v2.sh
```

### 选项 2: 手动创建符号链接

```bash
# 在项目目录中
cd ~/tristar_autolabel

# 创建指向数据的符号链接
ln -s $SCRATCH/tristar_data data

# 验证链接
ls -l data
# 应该显示: data -> /scratch/your_username/tristar_data

# 现在可以正常运行训练
sbatch setup_and_train.sh
```

---

## 📊 数据管理最佳实践

### 1. 定期备份

```bash
# 从 SCRATCH 备份到 PROJECT（长期存储）
rsync -av $SCRATCH/tristar_data/ $PROJECT/tristar_data_backup/
```

### 2. 清理临时数据

SCRATCH 上的数据可能会被自动清理（60天未访问），重要数据要备份到 PROJECT：

```bash
# 检查文件的最后访问时间
find $SCRATCH/tristar_data -atime +30 -ls

# 移动到 PROJECT
mv $SCRATCH/tristar_data $PROJECT/
```

### 3. 压缩存档

```bash
# 压缩数据以节省空间
tar -czf tristar_data.tar.gz -C $SCRATCH tristar_data/

# 解压
tar -xzf tristar_data.tar.gz -C $SCRATCH/
```

---

## 🔧 故障排查

### 问题 1: "Permission denied"
```bash
# 检查权限
ls -la $SCRATCH/tristar_data/

# 修复权限
chmod -R 755 $SCRATCH/tristar_data/
```

### 问题 2: "No space left on device"
```bash
# 检查配额使用情况
diskusage_report

# 清理旧文件
find $SCRATCH -type f -atime +60 -delete
```

### 问题 3: 训练脚本找不到数据
```bash
# 检查符号链接
ls -l ~/tristar_autolabel/data

# 重新创建链接
cd ~/tristar_autolabel
rm data  # 删除旧链接
ln -s $SCRATCH/tristar_data data

# 验证数据路径
ls data/tristar/train/
```

### 问题 4: 上传速度慢
```bash
# 使用 rsync 的限速选项避免网络拥塞
rsync -avzP --bwlimit=10000 \  # 限制到 10MB/s
    f:/tristar/data/tristar/ \
    your_username@fir.alliancecan.ca:$SCRATCH/tristar_data/tristar/
```

---

## 📝 快速参考

### 常用路径

```bash
# SCRATCH (训练用)
$SCRATCH/tristar_data/tristar/

# PROJECT (长期存储)
$PROJECT/tristar_data/tristar/

# HOME (项目代码)
~/tristar_autolabel/
```

### 常用命令

```bash
# 查看磁盘使用
diskusage_report

# 检查数据大小
du -sh $SCRATCH/tristar_data

# 统计文件数
find $SCRATCH/tristar_data -type f | wc -l

# 传输进度
rsync -avzP --info=progress2 source/ destination/
```

---

## 需要帮助？

- [Digital Alliance 存储文档](https://docs.alliancecan.ca/wiki/Storage_and_file_management)
- [Globus 使用指南](https://docs.alliancecan.ca/wiki/Globus)
- [技术支持](mailto:support@tech.alliancecan.ca)
