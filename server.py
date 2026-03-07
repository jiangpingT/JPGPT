"""
JPGPT 推理服务器
提供 HTTP API，供 chat.html 调用

端点：
  GET  /            — 返回 chat.html
  GET  /status      — 模型状态（是否加载、参数量、训练进度）
  POST /generate    — 续写文本
  GET  /train_log   — 训练曲线数据（供可视化）
"""

import os
import json
import glob
import time

import torch
from flask import Flask, request, jsonify, send_from_directory

from model import GPT, GPTConfig

app = Flask(__name__)

CKPT_DIR = "checkpoints"
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# 全局模型（懒加载）
model = None
enc = None


def load_model():
    global model, enc
    import tiktoken

    best_pt = os.path.join(CKPT_DIR, "best.pt")
    latest_pt = os.path.join(CKPT_DIR, "latest.pt")

    ckpt_path = best_pt if os.path.exists(best_pt) else latest_pt
    if not os.path.exists(ckpt_path):
        return False

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", GPTConfig())
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"模型加载完成：step={step}，val_loss={val_loss}")
    return True


def try_reload():
    """尝试加载最新 checkpoint（训练中途也可调用推理）"""
    global model
    loaded = load_model()
    return loaded


@app.route("/")
def index():
    return send_from_directory(".", "chat.html")


@app.route("/status")
def status():
    best_pt = os.path.join(CKPT_DIR, "best.pt")
    latest_pt = os.path.join(CKPT_DIR, "latest.pt")

    # 读取最新训练日志
    log_path = os.path.join(CKPT_DIR, "train_log.jsonl")
    last_log = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            if lines:
                last_log = json.loads(lines[-1])
        except Exception:
            pass

    ckpt_exists = os.path.exists(best_pt) or os.path.exists(latest_pt)

    return jsonify({
        "model_loaded": model is not None,
        "checkpoint_exists": ckpt_exists,
        "device": str(DEVICE),
        "last_step": last_log.get("step", 0),
        "last_loss": last_log.get("loss", None),
        "tokens_seen": last_log.get("tokens_seen", 0),
        "tok_per_sec": last_log.get("tok_per_sec", 0),
    })


@app.route("/train_log")
def train_log():
    """返回完整训练日志，用于前端绘制 loss 曲线"""
    log_path = os.path.join(CKPT_DIR, "train_log.jsonl")
    if not os.path.exists(log_path):
        return jsonify([])
    records = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(records)


@app.route("/generate", methods=["POST"])
def generate():
    global model

    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    max_new_tokens = int(data.get("max_new_tokens", 200))
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 50))

    if not prompt:
        return jsonify({"error": "prompt 不能为空"}), 400

    # 尝试加载模型（如果还未加载）
    if model is None:
        loaded = try_reload()
        if not loaded:
            return jsonify({
                "error": "模型尚未训练完成，请等待 checkpoints/best.pt 生成",
                "tip": "训练开始后每隔 500 步会保存一次，届时可开始推理"
            }), 503

    # 编码 prompt（开头加 EOT，与训练格式一致：训练时每个故事以 EOT 开头）
    tokens = [enc.eot_token] + enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    elapsed_ms = (time.time() - t0) * 1000

    # 只返回新生成的部分，遇到 EOT token（故事结束符）截断
    generated_tokens = y[0][len(tokens):].tolist()
    eot = enc.eot_token  # 50256
    if eot in generated_tokens:
        generated_tokens = generated_tokens[:generated_tokens.index(eot)]
    generated_text = enc.decode(generated_tokens)

    return jsonify({
        "prompt": prompt,
        "generated": generated_text,
        "full_text": prompt + generated_text,
        "new_tokens": len(generated_tokens),
        "elapsed_ms": round(elapsed_ms),
    })


if __name__ == "__main__":
    print(f"JPGPT 推理服务器启动中（device={DEVICE}）...")
    try_reload()
    if model is None:
        print(">>> 模型未就绪，训练完成后刷新页面即可自动加载")
    print(">>> 访问：http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
