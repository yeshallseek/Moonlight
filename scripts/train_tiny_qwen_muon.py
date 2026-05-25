#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def zeropower_via_newtonschulz5(g, steps):
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if g.device.type == "cuda" else torch.float32
    x = g.to(work_dtype)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.float()


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        lr,
        wd,
        muon_params,
        adamw_params,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        scale_mode="paper",
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            scale_mode=scale_mode,
        )
        params = list(muon_params) + list(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @staticmethod
    def adjusted_lr(lr, shape, scale_mode):
        if scale_mode == "paper":
            return lr * 0.2 * math.sqrt(max(shape[:2]))
        if scale_mode == "none":
            return lr
        raise ValueError(f"unknown scale_mode={scale_mode}")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            muon_params = [p for p in group["params"] if self.state[p]["use_muon"]]
            for p in muon_params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                adjusted_lr = self.adjusted_lr(lr, p.shape, group["scale_mode"])
                p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-adjusted_lr)

            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                update = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr / scale)

        return loss


class TokenBlockDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        usable = (len(self.tokens) // seq_len) * seq_len
        self.tokens = self.tokens[:usable]
        if len(self.tokens) < seq_len:
            raise ValueError("not enough tokens for one sequence")

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_texts(args):
    if args.dataset == "synthetic":
        rng = random.Random(args.seed)
        words = [
            "matrix",
            "optimizer",
            "gradient",
            "language",
            "token",
            "orthogonal",
            "scale",
            "decay",
            "model",
            "loss",
        ]
        return [" ".join(rng.choice(words) for _ in range(80)) for _ in range(args.max_texts)]

    ds = load_dataset(args.dataset, split=args.dataset_split, trust_remote_code=True)
    if args.max_texts:
        ds = ds.select(range(min(args.max_texts, len(ds))))
    texts = []
    for row in ds:
        value = row.get(args.text_field)
        if value:
            texts.append(str(value))
    if not texts:
        raise ValueError(f"no non-empty text found in field {args.text_field}")
    return texts


def token_cache_path(args, tokenizer_name):
    raw = json.dumps(
        {
            "dataset": args.dataset,
            "split": args.dataset_split,
            "text_field": args.text_field,
            "max_texts": args.max_texts,
            "tokenizer": tokenizer_name,
            "seq_len": args.seq_len,
            "seed": args.seed if args.dataset == "synthetic" else None,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    safe_dataset = args.dataset.replace("/", "_")
    return Path(args.cache_dir) / f"tokens_{safe_dataset}_{digest}.pt"


def load_or_tokenize(args, tokenizer):
    cache_path = token_cache_path(args, args.tokenizer)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.retokenize:
        payload = torch.load(cache_path, map_location="cpu")
        return payload["tokens"], cache_path, payload

    texts = load_texts(args)
    tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        encoded = tokenizer.encode(text, add_special_tokens=True)
        tokens.extend(encoded)
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)
    if len(tokens) < args.seq_len * 8:
        raise ValueError(f"only got {len(tokens)} tokens; increase --max-texts or reduce --seq-len")
    payload = {
        "tokens": tokens,
        "num_tokens": len(tokens),
        "num_texts": len(texts),
        "dataset": args.dataset,
        "split": args.dataset_split,
        "tokenizer": args.tokenizer,
    }
    torch.save(payload, cache_path)
    return tokens, cache_path, payload


def build_model(args, tokenizer):
    vocab_size = len(tokenizer)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size or args.hidden_size * 4,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads or args.num_heads,
        max_position_embeddings=args.seq_len + 1,
        tie_word_embeddings=True,
        use_cache=False,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    return Qwen2ForCausalLM(config)


def build_optimizer(args, model):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        use_muon = p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name
        if use_muon:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    scale_mode = "paper" if args.optimizer == "muon" else "none"
    return Muon(
        lr=args.lr,
        wd=args.wd,
        muon_params=muon_params,
        adamw_params=adamw_params,
        ns_steps=args.ns_steps,
        scale_mode=scale_mode,
    )


def lr_factor(step, max_steps, warmup_steps, min_lr_ratio):
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denom = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_lr(optimizer, base_lr, factor):
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor


def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        value = p.grad.detach().float().norm(2).item()
        total += value * value
    return math.sqrt(total)


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= args.eval_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with autocast_context(device, args.bf16):
            loss = model(input_ids=batch, labels=batch).loss
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def append_jsonl(path, payload):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["adamw", "muon", "muon-noscale"], required=True)
    parser.add_argument("--dataset", default="Elriggs/openwebtext-100k")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--cache-dir", default="artifacts/datasets")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runs-jsonl", default="experiments/runs.jsonl")
    parser.add_argument("--max-texts", type=int, default=512)
    parser.add_argument("--retokenize", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokens, cache_path, token_payload = load_or_tokenize(args, tokenizer)
    split_idx = int(len(tokens) * 0.9)
    split_idx = max(args.seq_len * args.batch_size, split_idx)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    if len(val_tokens) < args.seq_len:
        val_tokens = tokens[-max(args.seq_len * args.eval_batches, args.seq_len):]

    train_data = TokenBlockDataset(train_tokens, args.seq_len)
    val_data = TokenBlockDataset(val_tokens, args.seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, tokenizer).to(device)
    optimizer = build_optimizer(args, model)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run_header = {
        "kind": "tiny_qwen_training",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "args": vars(args),
        "device": str(device),
        "param_count": param_count,
        "trainable_count": trainable_count,
        "token_cache": str(cache_path),
        "num_tokens": len(tokens),
        "num_train_blocks": len(train_data),
        "num_val_blocks": len(val_data),
        "torch": torch.__version__,
    }
    append_jsonl(metrics_path, {"event": "start", **run_header})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    last_log_time = start
    tokens_seen = 0
    initial_val = evaluate(model, val_loader, device, args)
    append_jsonl(metrics_path, {"event": "eval", "step": 0, "val_loss": initial_val})

    train_iter = itertools.cycle(train_loader)
    final_train_loss = None
    final_val_loss = initial_val

    for step in range(1, args.max_steps + 1):
        batch = next(train_iter).to(device, non_blocking=True)
        set_lr(optimizer, args.lr, lr_factor(step - 1, args.max_steps, args.warmup_steps, args.min_lr_ratio))
        with autocast_context(device, args.bf16):
            loss = model(input_ids=batch, labels=batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        current_grad_norm = grad_norm(model.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        final_train_loss = float(loss.detach().cpu())
        tokens_seen += int(batch.numel())

        now = time.time()
        if step % args.log_interval == 0 or step == 1:
            elapsed = now - start
            interval_tps = int(args.log_interval * args.batch_size * args.seq_len / max(1e-9, now - last_log_time))
            last_log_time = now
            append_jsonl(
                metrics_path,
                {
                    "event": "train",
                    "step": step,
                    "train_loss": final_train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": current_grad_norm,
                    "tokens_seen": tokens_seen,
                    "tokens_per_s_avg": tokens_seen / max(1e-9, elapsed),
                    "tokens_per_s_interval": interval_tps,
                },
            )

        if step % args.eval_interval == 0 or step == args.max_steps:
            final_val_loss = evaluate(model, val_loader, device, args)
            append_jsonl(metrics_path, {"event": "eval", "step": step, "val_loss": final_val_loss})

    elapsed_s = time.time() - start
    summary = {
        **run_header,
        "elapsed_s": elapsed_s,
        "final_train_loss": final_train_loss,
        "initial_val_loss": initial_val,
        "final_val_loss": final_val_loss,
        "tokens_seen": tokens_seen,
        "tokens_per_s_avg": tokens_seen / max(1e-9, elapsed_s),
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    append_jsonl(args.runs_jsonl, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
