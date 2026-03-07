# JPGPT — 从零训练 GPT 项目记录

> 姜哥亲手实现的第一个 GPT，从零到能讲英文故事。
> 完成时间：2026-03-06

---

## 项目定位

**学习工具，不是生产工具。**
目的：理解 Transformer 从零到训练的完整流程，为后续 LoRA 微调打基础。

---

## 里程碑

| 时间 | 事件 |
|------|------|
| 2026-03-04 | 从零搭建工程，下载 TinyStories，开始训练 |
| 2026-03-04 | step 3240，loss 1.50，故事"有模有样" |
| 2026-03-05 | 因 PyTorch 2.6 checkpoint 加载 bug 崩溃，修复重启 |
| 2026-03-06 | 1 epoch 完成，step 7060，loss 1.24，训练总用时约 38 小时 |

---

## 模型配置（GPT-2 Small）

```python
vocab_size  = 50257   # tiktoken GPT-2 词表
n_layer     = 12      # Transformer 层数
n_head      = 12      # 注意力头数
n_embd      = 768     # 嵌入维度
block_size  = 1024    # 上下文长度
参数量       = 124.4M  # ≈ 0.1B
```

---

## 训练配置

```python
数据集         = TinyStories（roneneldan/TinyStories）
训练 tokens    = 463M（1 epoch）
BATCH_SIZE     = 8
SEQ_LEN        = 1024
GRAD_ACCUM     = 8     # 有效 batch = 65536 tokens/step
MAX_LR         = 6e-4
MIN_LR         = 6e-5
WARMUP_STEPS   = 200
WEIGHT_DECAY   = 0.1
GRAD_CLIP      = 1.0
总步数          = 7067
实测速度        = ~3700–4000 tok/sec（M4 Pro MPS）
实际用时        = 约 38 小时
```

---

## Loss 曲线

```
step 0     loss 10.95  ← 随机初始化
step 500   loss ~3.5   ← 学会了基本词频
step 1000  loss ~2.2   ← 学会了简单语法
step 3000  loss ~1.35  ← 能写出连贯句子
step 7060  loss  1.24  ← 1 epoch 完成，能讲儿童故事
```

---

## 模型评价（Playwright 实测，2026-03-06）

**总评：B+**（对于 0.1B + 1 epoch 来说非常合格）

**优点：**
- 语法几乎完美
- 角色和道具在故事内保持一致
- 语气符合儿童故事风格

**局限：**
- 故事短（30-40 tokens 就结束，TinyStories 原本就是短故事）
- 情节偏简单，高频套话多（"so happy / so proud"）
- 只会英文，不懂中文

---

## 踩过的坑

### 1. `data/tokenize.py` 和标准库 `tokenize` 模块同名
- 现象：`import datasets` 时触发循环导入崩溃
- 解决：改名为 `data/prepare.py`

### 2. PyTorch 2.6 torch.load 默认行为变更
- 现象：`weights_only` 默认改为 `True`，自定义 dataclass（GPTConfig）不在 allowlist，加载 checkpoint 时 `UnpicklingError`
- 解决：所有 `torch.load` 调用加 `weights_only=False`

### 3. EOT token 不停止 → 模型生成多个连续故事
- 现象：推理时模型遇到 `<|endoftext|>` 不停止，把下一个故事也生成出来
- 解决：`server.py` 里 decode 后截断 EOT 之前的部分

### 4. EOT 截断后只生成 10 tokens 就结束
- 现象：修了 EOT 截断后，每次只生成极少 token
- 原因：训练格式是 `EOT + story`，推理时没有 EOT 前缀，模型不知道"故事开始了"，很快生成 EOT
- 解决：`server.py` 编码 prompt 前先 prepend `enc.eot_token`

---

## 核心概念收获

### AdamW 优化器
- Adam：自适应学习率，每个参数自己调步长
- W（Weight Decay）：每步把参数轻推向 0，防止参数膨胀
- 直觉：Adam 加了刹车，LLM 训练标配

### Cosine LR 调度
- 热身阶段（前 200 步）：0 → MAX_LR 线性升，防止初期乱跑
- 下降阶段：MAX_LR → MIN_LR 沿 cosine 曲线平滑降
- 目的：训练后期小步精调，避免震荡

### 参数 vs Token 的区别
- 参数 = 模型的脑子大小（设计时固定）
- Token = 训练时读过多少文字（训练中累积）
- Chinchilla 法则：最优比例 ≈ 20 × 参数数（100M 参数 → 2B tokens 最优）

### Weight Tying（权重绑定）
- token 嵌入矩阵和 lm_head 输出矩阵共享同一份权重
- 直觉：输入"认识"token 和输出"预测"token 用同一个字典，减少参数且效果更好

---

## 文件结构

```
JPGPT/
├── model.py          # GPT 模型（CausalSelfAttention + MLP + TransformerBlock + GPT）
├── train.py          # 训练主循环（AdamW + cosine LR + grad clip + checkpoint）
├── server.py         # Flask 推理服务器（含 EOT 前缀 + EOT 截断修复）
├── chat.html         # 故事续写 + 训练监控 Dashboard（Loss 曲线可视化）
├── data/
│   ├── download.py   # 下载 TinyStories（HuggingFace datasets）
│   └── prepare.py    # Tokenize → train.bin / val.bin（uint16 memmap）
└── checkpoints/
    ├── best.pt       # 最佳模型（522MB，val loss 最低时保存）
    ├── latest.pt     # 最新 checkpoint（含 optimizer state，1.4GB）
    └── train_log.jsonl  # 每 10 步一条，供 Dashboard 绘图
```

---

## 下一步：LoRA 微调

**JPGPT 价值已实现**，下一阶段切换到 Qwen。

| | JPGPT | Qwen2.5 + LoRA |
|--|--|--|
| 基础能力 | 只会英文短故事 | 中英双语、海量知识 |
| LoRA 后天花板 | 低 | 高 |
| 适合 Demo | ❌ | ✅ |

建议：Qwen2.5-7B（48GB 跑 LoRA 完全够），用明略/核聚变/机器人领域数据微调。
