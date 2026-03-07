"""
下载 TinyStories 数据集（HuggingFace）
TinyStories：由 GPT-4 生成的简单英文儿童故事，约 500M tokens
适合训练小参数量语言模型，验证方式直观（能讲故事就算成功）
"""

import os
import sys

def download():
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装依赖：pip install datasets")
        sys.exit(1)

    os.makedirs("data/raw", exist_ok=True)

    for split in ["train", "validation"]:
        out_file = f"data/raw/{split}.txt"
        if os.path.exists(out_file):
            size_mb = os.path.getsize(out_file) / 1024 / 1024
            print(f"[跳过] {out_file} 已存在（{size_mb:.0f} MB）")
            continue

        print(f"正在下载 TinyStories {split} split...")
        ds = load_dataset("roneneldan/TinyStories", split=split)

        print(f"正在写入 {out_file}（共 {len(ds):,} 条故事）...")
        with open(out_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(ds):
                f.write(item["text"].strip())
                f.write("\n\n")  # 故事之间用空行分隔
                if (i + 1) % 50000 == 0:
                    print(f"  已写入 {i+1:,} / {len(ds):,} 条")

        size_mb = os.path.getsize(out_file) / 1024 / 1024
        print(f"完成：{out_file}（{size_mb:.0f} MB）")

    print("\n数据下载完毕，请运行 python data/tokenize.py")


if __name__ == "__main__":
    download()
