#!/bin/bash
# JPGPT 一键启动脚本
# 自动完成：安装依赖 → 下载数据 → Tokenize → 训练（后台）→ 启动服务器

set -e
cd "$(dirname "$0")"

echo "========================================"
echo "  JPGPT — 0.1B GPT 从零训练"
echo "========================================"

# ── 1. 虚拟环境 ──────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[1/5] 创建虚拟环境..."
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "[1/5] 虚拟环境已激活"

# ── 2. 安装依赖 ──────────────────────────────────────────────
echo "[2/5] 安装依赖..."
pip install -q -r requirements.txt

# ── 3. 下载数据 ──────────────────────────────────────────────
echo "[3/5] 下载 TinyStories 数据集..."
python data/download.py

# ── 4. Tokenize ──────────────────────────────────────────────
echo "[4/5] Tokenize 数据..."
python data/prepare.py

# ── 5. 后台训练 ──────────────────────────────────────────────
echo "[5/5] 启动训练（后台运行）..."
nohup python train.py > checkpoints/train_stdout.log 2>&1 &
TRAIN_PID=$!
echo "训练进程 PID：$TRAIN_PID"
echo $TRAIN_PID > checkpoints/train.pid
echo "训练日志：checkpoints/train_stdout.log"
echo "查看实时日志：tail -f checkpoints/train_stdout.log"

# ── 6. 启动服务器 ─────────────────────────────────────────────
echo ""
echo "========================================"
echo "  启动推理服务器..."
echo "  访问：http://localhost:8080"
echo "========================================"
python server.py
