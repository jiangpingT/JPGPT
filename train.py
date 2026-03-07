"""
JPGPT 训练主脚本

核心训练流程：
  for each step:
    1. 取一批 (x, y) — x 是输入序列，y 是右移一位的目标
    2. forward：模型预测每个位置的下一个 token
    3. loss = 交叉熵（预测分布 vs 真实 token）
    4. backward：计算每个参数对 loss 的梯度
    5. step：沿梯度方向更新参数

关键超参数说明：
  learning_rate   — 每步更新多大步伐，太大发散，太小收敛慢
  weight_decay    — L2 正则化，防止参数过大
  grad_clip       — 梯度裁剪，防止梯度爆炸
  grad_accum_steps— 梯度累积，等效于更大的 batch size，但不增加内存
"""

import os
import sys
import time
import math
import json

import numpy as np
import torch

from model import GPT, GPTConfig

# ─── 超参数 ────────────────────────────────────────────────────────────────────

# 目标训练量（tokens）
TARGET_TOKENS = 463_170_388  # 1 epoch（TinyStories train.bin 实际 token 数）

# batch 配置
BATCH_SIZE = 8          # 每次实际前向的样本数
SEQ_LEN = 1024          # 每个样本的序列长度
GRAD_ACCUM_STEPS = 8    # 梯度累积步数（有效 batch size = 8×1024×8 = 65536 tokens）

# 学习率配置（cosine 衰减）
MAX_LR = 6e-4           # 峰值学习率
MIN_LR = 6e-5           # 最低学习率（= MAX_LR / 10）
WARMUP_STEPS = 200      # 前 N 步线性热身

# 优化器
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0         # 梯度裁剪阈值

# 日志与保存
EVAL_INTERVAL = 500     # 每 N 步评估一次 val loss
SAVE_INTERVAL = 1000    # 每 N 步保存一次 checkpoint
LOG_INTERVAL = 10       # 每 N 步打印一次训练状态

# 路径
TRAIN_BIN = "data/train.bin"
VAL_BIN   = "data/val.bin"
CKPT_DIR  = "checkpoints"

# ─── 设备 ──────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("使用设备：Apple MPS（M 系列 GPU）")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("使用设备：CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("使用设备：CPU（训练会很慢）")

# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_bin(path: str) -> np.ndarray:
    if not os.path.exists(path):
        print(f"找不到 {path}，请先运行：python data/download.py && python data/tokenize.py")
        sys.exit(1)
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: torch.device):
    """随机从数据中取一批 (x, y)"""
    n = len(data)
    # 随机选起始位置
    idx = torch.randint(n - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + seq_len].astype(np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(data[i + 1: i + seq_len + 1].astype(np.int64)) for i in idx])
    return x.to(device), y.to(device)

# ─── 学习率调度 ────────────────────────────────────────────────────────────────

def get_lr(step: int, total_steps: int) -> float:
    """
    Cosine 学习率衰减 + 线性热身
    热身阶段：0 → MAX_LR（线性增长，让训练开始时更稳定）
    衰减阶段：MAX_LR → MIN_LR（cosine 曲线，平滑下降）
    """
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    if step > total_steps:
        return MIN_LR
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

# ─── 评估 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model: GPT, val_data: np.ndarray, n_batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

# ─── 主训练循环 ────────────────────────────────────────────────────────────────

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 加载数据
    print("加载数据...")
    train_data = load_bin(TRAIN_BIN)
    val_data   = load_bin(VAL_BIN)
    train_tokens = len(train_data)
    print(f"  train: {train_tokens:,} tokens")
    print(f"  val:   {len(val_data):,} tokens")

    # 计算总步数
    tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
    total_steps = TARGET_TOKENS // tokens_per_step
    print(f"\n训练配置：")
    print(f"  目标训练量：{TARGET_TOKENS / 1e9:.1f}B tokens")
    print(f"  有效 batch size：{tokens_per_step:,} tokens/step")
    print(f"  总步数：{total_steps:,} steps")

    # 构建模型
    config = GPTConfig()
    model = GPT(config).to(DEVICE)
    n_params = model.num_params()
    print(f"\n模型参数量：{n_params / 1e6:.1f}M（{n_params:,}）")

    # 优化器
    optimizer = model.configure_optimizer(MAX_LR, WEIGHT_DECAY, BETAS)

    # 恢复 checkpoint（如果存在）
    start_step = 0
    best_val_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(ckpt_path):
        print(f"\n发现 checkpoint，恢复训练：{ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  从 step {start_step} 继续，最佳 val loss = {best_val_loss:.4f}")

    model.train()
    print(f"\n开始训练（device={DEVICE}）...\n")

    # 日志文件
    log_file = open(os.path.join(CKPT_DIR, "train_log.jsonl"), "a")

    t0 = time.time()
    tokens_seen = start_step * tokens_per_step

    for step in range(start_step, total_steps + 1):
        # 动态调整学习率
        lr = get_lr(step, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 梯度累积：分多个 micro-step，累积后再更新参数
        optimizer.zero_grad()
        total_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM_STEPS  # 归一化（等效于更大 batch 的平均 loss）
            loss.backward()
            total_loss += loss.item()

        # 梯度裁剪：把梯度的 L2 范数限制在 GRAD_CLIP 以内，防止梯度爆炸
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        tokens_seen += tokens_per_step

        # ── 打印训练状态 ────────────────────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            t1 = time.time()
            elapsed = t1 - t0
            tok_per_sec = (LOG_INTERVAL * tokens_per_step) / elapsed if step > start_step else 0
            t0 = t1

            # 预计完成时间
            remaining_steps = total_steps - step
            eta_sec = remaining_steps * tokens_per_step / tok_per_sec if tok_per_sec > 0 else 0
            eta_h = eta_sec / 3600

            pct = tokens_seen / TARGET_TOKENS * 100
            print(
                f"step {step:6d}/{total_steps} | "
                f"loss {total_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"{tok_per_sec/1000:.1f}K tok/s | "
                f"进度 {pct:.1f}% | "
                f"预计完成 {eta_h:.1f}h"
            )

            log_file.write(json.dumps({
                "step": step,
                "loss": total_loss,
                "lr": lr,
                "tokens_seen": tokens_seen,
                "tok_per_sec": tok_per_sec,
            }) + "\n")
            log_file.flush()

        # ── 评估 val loss ───────────────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0 and step > 0:
            val_loss = estimate_loss(model, val_data)
            print(f"\n>>> step {step} | val loss = {val_loss:.4f} | train loss = {total_loss:.4f}\n")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": config, "step": step, "val_loss": val_loss},
                    os.path.join(CKPT_DIR, "best.pt"),
                )
                print(f"    ✓ 保存最佳模型（val loss {val_loss:.4f}）")

        # ── 保存 checkpoint ────────────────────────────────────────────────────
        if step % SAVE_INTERVAL == 0 and step > 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )

    log_file.close()
    print(f"\n训练完成！最佳 val loss = {best_val_loss:.4f}")
    print(f"最佳模型保存在：{CKPT_DIR}/best.pt")


if __name__ == "__main__":
    train()
