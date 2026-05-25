#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_RUNS = [
    ("muon_paper_640l12_b16", "Muon paper"),
    ("adamw_640l12_b16", "AdamW"),
    ("muon_nowd_640l12_b16", "Muon no wd"),
    ("muon_noscale_640l12_b16", "Muon no scale"),
]


def load_jsonl(path):
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def latest(records, event):
    matches = [record for record in records if record.get("event") == event]
    return matches[-1] if matches else None


def scale_label(args):
    optimizer = args.get("optimizer", "")
    if optimizer == "adamw":
        return "n/a"
    if optimizer == "muon-noscale":
        return "none"
    if optimizer == "muon":
        return "paper"
    return ""


def fmt_float(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_int(value):
    if value is None:
        return ""
    return f"{int(value):,}"


def collect_run(root, name, label):
    run_dir = root / name
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    samples = load_jsonl(run_dir / "samples.jsonl")
    summary = load_json(run_dir / "summary.json")
    start = latest(metrics, "start")
    train = latest(metrics, "train")
    eval_record = latest(metrics, "eval")
    weight_rms = latest(metrics, "weight_rms")
    args = {}
    if summary and summary.get("args"):
        args = summary["args"]
    elif start and start.get("args"):
        args = start["args"]

    status = "missing"
    if metrics and not summary:
        status = "partial"
    if summary:
        status = summary.get("stop_reason", "complete")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_count = len(list(checkpoint_dir.glob("step_*.pt"))) if checkpoint_dir.exists() else 0

    return {
        "name": name,
        "label": label,
        "run_dir": run_dir,
        "metrics": metrics,
        "samples": samples,
        "summary": summary,
        "start": start,
        "train": train,
        "eval": eval_record,
        "weight_rms": weight_rms,
        "args": args,
        "status": status,
        "checkpoint_count": checkpoint_count,
    }


def plot_event(rows, event, value_key, ylabel, title, out_path):
    plotted = False
    plt.figure(figsize=(8.8, 5.2))
    for row in rows:
        xs = []
        ys = []
        for record in row["metrics"]:
            if record.get("event") == event and value_key in record:
                xs.append(record["step"])
                ys.append(record[value_key])
        if xs:
            marker = "o" if event == "eval" else None
            plt.plot(xs, ys, linewidth=1.8, marker=marker, markersize=3, label=row["label"])
            plotted = True
    if not plotted:
        plt.close()
        return None
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def markdown_table(rows):
    lines = [
        "| Run | Status | Optimizer | WD | Scale | Last step | Latest val loss | Latest train loss | Tokens seen | Tokens/s | Samples | Checkpoints |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        args = row["args"]
        train = row["train"] or {}
        eval_record = row["eval"] or {}
        summary = row["summary"] or {}
        last_step = summary.get("last_step") or train.get("step") or eval_record.get("step")
        tokens_seen = summary.get("tokens_seen") or train.get("tokens_seen")
        tokens_per_s = summary.get("tokens_per_s_avg") or train.get("tokens_per_s_avg")
        val_loss = summary.get("final_val_loss") or eval_record.get("val_loss")
        train_loss = summary.get("final_train_loss") or train.get("train_loss")
        lines.append(
            "| {run} | {status} | {opt} | {wd} | {scale} | {step} | {val} | {train_loss} | {tokens} | {tps} | {samples} | {checkpoints} |".format(
                run=row["name"],
                status=row["status"],
                opt=args.get("optimizer", ""),
                wd=args.get("wd", ""),
                scale=scale_label(args),
                step=fmt_int(last_step),
                val=fmt_float(val_loss),
                train_loss=fmt_float(train_loss),
                tokens=fmt_int(tokens_seen),
                tps=fmt_int(tokens_per_s),
                samples=len(row["samples"]),
                checkpoints=row["checkpoint_count"],
            )
        )
    return "\n".join(lines)


def model_config(rows):
    for row in rows:
        args = row["args"]
        start = row["start"] or {}
        if args:
            return {
                "param_count": start.get("param_count"),
                "dataset": args.get("dataset"),
                "max_texts": args.get("max_texts"),
                "seq_len": args.get("seq_len"),
                "batch_size": args.get("batch_size"),
                "hidden_size": args.get("hidden_size"),
                "intermediate_size": args.get("intermediate_size"),
                "num_layers": args.get("num_layers"),
                "num_heads": args.get("num_heads"),
                "num_kv_heads": args.get("num_kv_heads"),
                "lr": args.get("lr"),
                "warmup_steps": args.get("warmup_steps"),
                "max_steps": args.get("max_steps"),
                "seed": args.get("seed"),
            }
    return {}


def write_markdown(path, rows, figure_paths):
    config = model_config(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    with path.open("w") as f:
        f.write("# Overnight Large Results\n\n")
        f.write(f"Generated: `{now}`\n\n")
        if config:
            f.write("## Configuration\n\n")
            f.write(
                "- Model: {params} parameters, hidden size {hidden}, {layers} layers, {heads} attention heads, {kv_heads} KV heads\n".format(
                    params=fmt_int(config.get("param_count")),
                    hidden=config.get("hidden_size"),
                    layers=config.get("num_layers"),
                    heads=config.get("num_heads"),
                    kv_heads=config.get("num_kv_heads"),
                )
            )
            f.write(
                "- Data: `{dataset}`, max texts {max_texts}, seq len {seq_len}, batch size {batch_size}\n".format(
                    dataset=config.get("dataset"),
                    max_texts=fmt_int(config.get("max_texts")),
                    seq_len=config.get("seq_len"),
                    batch_size=config.get("batch_size"),
                )
            )
            f.write(
                "- Schedule: LR {lr}, warmup {warmup}, max steps {max_steps}, seed {seed}\n\n".format(
                    lr=config.get("lr"),
                    warmup=fmt_int(config.get("warmup_steps")),
                    max_steps=fmt_int(config.get("max_steps")),
                    seed=config.get("seed"),
                )
            )
        f.write("## Runs\n\n")
        f.write(markdown_table(rows))
        f.write("\n\n")
        if figure_paths:
            f.write("## Figures\n\n")
            for figure_path in figure_paths:
                f.write(f"- `{figure_path}`\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/overnight_large")
    parser.add_argument("--out", default="reports/overnight-large-summary.md")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [collect_run(root, name, label) for name, label in EXPECTED_RUNS]
    figures_dir = Path(args.figures_dir)
    figure_paths = []
    for path in [
        plot_event(
            rows,
            "eval",
            "val_loss",
            "Validation loss",
            "Overnight 175.7M Qwen-like LM validation loss",
            figures_dir / "overnight_large_val_loss.png",
        ),
        plot_event(
            rows,
            "train",
            "train_loss",
            "Training loss",
            "Overnight 175.7M Qwen-like LM training loss",
            figures_dir / "overnight_large_train_loss.png",
        ),
    ]:
        if path is not None:
            figure_paths.append(path)

    out = Path(args.out)
    write_markdown(out, rows, figure_paths)
    print(out)


if __name__ == "__main__":
    main()
