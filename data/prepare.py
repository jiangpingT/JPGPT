"""
把原始文本切成 token，存成二进制文件供训练使用

Token：文字被切碎后的最小单元（大约 0.5 个英文词 = 1 token）
这里用 tiktoken 的 GPT-2 编码（词表 50257 个 token）

输出：
  data/train.bin  — uint16 numpy 数组，直接 memmap 读取
  data/val.bin    — 同上（验证集）
"""

import os
import sys
import numpy as np
from tqdm import tqdm


def tokenize():
    try:
        import tiktoken
    except ImportError:
        print("请先安装依赖：pip install tiktoken")
        sys.exit(1)

    enc = tiktoken.get_encoding("gpt2")
    # EOT（End of Text）token：故事与故事之间插入分隔符
    eot = enc.eot_token  # 50256

    for split in ["train", "validation"]:
        in_file = f"data/raw/{split}.txt"
        # validation → val（文件名统一）
        out_name = "val" if split == "validation" else split
        out_file = f"data/{out_name}.bin"

        if not os.path.exists(in_file):
            print(f"找不到 {in_file}，请先运行 python data/download.py")
            sys.exit(1)

        if os.path.exists(out_file):
            size_mb = os.path.getsize(out_file) / 1024 / 1024
            n_tokens = os.path.getsize(out_file) // 2  # uint16 = 2 字节
            print(f"[跳过] {out_file} 已存在（{n_tokens:,} tokens，{size_mb:.0f} MB）")
            continue

        print(f"正在 tokenize {in_file}...")

        with open(in_file, "r", encoding="utf-8") as f:
            text = f.read()

        # 按故事切分（用空行分隔）
        stories = [s.strip() for s in text.split("\n\n") if s.strip()]
        print(f"  共 {len(stories):,} 个故事，开始编码...")

        all_tokens = []
        for story in tqdm(stories, desc=f"编码 {split}"):
            tokens = enc.encode_ordinary(story)
            all_tokens.append(eot)       # 故事开头加 EOT
            all_tokens.extend(tokens)

        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(out_file)

        size_mb = arr.nbytes / 1024 / 1024
        print(f"完成：{out_file}（{len(arr):,} tokens，{size_mb:.0f} MB）")

    print("\nTokenize 完毕，请运行 python train.py")


if __name__ == "__main__":
    tokenize()
