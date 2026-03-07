"""
JPGPT — 从零实现的 GPT 语言模型
架构：Embedding + N × TransformerBlock + LM Head
参考：GPT-2 Small（124M 参数）
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    # 词表大小（tiktoken GPT-2 编码，50257 个不同 token）
    vocab_size: int = 50257
    # 上下文窗口长度：模型每次最多看多少个 token
    block_size: int = 1024
    # Transformer 层数：堆叠多少个 Block
    n_layer: int = 12
    # 注意力头数：每个头负责关注不同的信息
    n_head: int = 12
    # 嵌入维度：每个 token 用多少维的向量表示
    n_embd: int = 768
    # Dropout 概率：训练时随机丢弃神经元，防止过拟合
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）
    "因果"= 每个位置只能看到它左边的 token，不能看未来
    这是语言模型的核心：预测下一个词时，不能偷看答案
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # 每个头的维度

        # 把输入 x 同时投影成 Q（Query）、K（Key）、V（Value）三份
        # 比分开三个 Linear 效率更高
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # 把多头结果拼接后映射回 n_embd 维度
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码（下三角矩阵）：位置 i 只能看到 0..i 的 token
        # register_buffer: 不是参数，但跟着模型走（保存/加载/移动设备）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch大小, 序列长度, 嵌入维度

        # 计算 Q K V，并拆分成多头
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # 各 (B, T, C)

        # 变形为 (B, n_head, T, head_dim) 以并行计算每个头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力分数 = Q @ K^T / sqrt(head_dim)
        # sqrt(head_dim) 是缩放因子，防止点积过大导致 softmax 梯度消失
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        # 把未来位置的分数设为 -inf，softmax 后变为 0（无法看到未来）
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 用注意力权重加权求和 V
        out = att @ v  # (B, n_head, T, head_dim)

        # 拼接所有头，恢复 (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    """
    前馈网络（Feed-Forward Network）
    先扩展到 4 倍维度，再压回来，中间用 GELU 激活
    直觉：Attention 负责"信息汇聚"，MLP 负责"信息加工"
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 层 = Attention + MLP，各带残差连接和 LayerNorm
    残差连接（x = x + sublayer(x)）：让梯度能直接流回浅层，解决深层网络难训练问题
    Pre-LN：先 LayerNorm 再进子层，比 Post-LN 训练更稳定
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # 残差 + Attention
        x = x + self.mlp(self.ln2(x))   # 残差 + MLP
        return x


class GPT(nn.Module):
    """
    完整 GPT 模型
    输入：token id 序列
    输出：每个位置预测下一个 token 的概率分布（logits）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # token 嵌入
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # 位置嵌入
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # 最后的 LayerNorm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定（Weight Tying）：token 嵌入矩阵和输出层共享同一份权重
        # 直觉：输入时把 token 映射成向量，输出时把向量映射回 token，用同一个"字典"
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # 残差连接输出层用更小的初始化，防止深层网络初始输出过大
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列长度 {T} 超过上下文窗口 {self.config.block_size}"

        positions = torch.arange(T, device=idx.device)

        # token 嵌入 + 位置嵌入
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        # 逐层通过 Transformer Block
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if targets is not None:
            # 训练阶段：计算所有位置的交叉熵 loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # 推理阶段：只需要最后一个位置的 logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        自回归生成：每次预测一个 token，然后追加到序列末尾，循环
        temperature：越高越随机，越低越保守
        top_k：只从概率最高的 k 个 token 中采样
        """
        for _ in range(max_new_tokens):
            # 超过上下文窗口时，截断最早的 token
            idx_crop = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-K 过滤：把 top-k 以外的 token 设为 -inf
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple):
        """
        AdamW 优化器：对 2D 以上的参数（权重矩阵）做 weight decay，
        对 1D 参数（LayerNorm、bias）不做 weight decay
        """
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
