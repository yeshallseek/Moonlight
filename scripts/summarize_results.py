#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


RUN_GROUPS = {
    "300": [
        "artifacts/runs/adamw_openwebtext_300",
        "artifacts/runs/muon_paper_openwebtext_300",
        "artifacts/runs/muon_nowd_openwebtext_300",
        "artifacts/runs/muon_noscale_openwebtext_300",
    ],
    "3000": [
        "artifacts/runs/adamw_openwebtext_3000",
        "artifacts/runs/muon_paper_openwebtext_3000",
        "artifacts/runs/muon_nowd_openwebtext_3000",
        "artifacts/runs/muon_noscale_openwebtext_3000",
    ],
}


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def load_metrics(path):
    records = []
    with Path(path).open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def label(summary):
    opt = summary["args"]["optimizer"]
    wd = summary["args"]["wd"]
    if opt == "adamw":
        return "AdamW"
    if opt == "muon" and wd == 0.1:
        return "Muon paper"
    if opt == "muon" and wd == 0.0:
        return "Muon no wd"
    if opt == "muon-noscale":
        return "Muon no scale"
    return f"{opt} wd={wd}"


def collect_group(paths):
    rows = []
    for run_dir in paths:
        run_dir = Path(run_dir)
        summary = load_json(run_dir / "summary.json")
        metrics = load_metrics(run_dir / "metrics.jsonl")
        rows.append({"run_dir": run_dir, "summary": summary, "metrics": metrics, "label": label(summary)})
    return rows


def plot_val(group_name, rows, out_dir):
    plt.figure(figsize=(8, 5))
    for row in rows:
        xs = []
        ys = []
        for record in row["metrics"]:
            if record.get("event") == "eval":
                xs.append(record["step"])
                ys.append(record["val_loss"])
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3, label=row["label"])
    plt.xlabel("Training step")
    plt.ylabel("Validation loss")
    plt.title(f"Tiny Qwen-like LM validation loss ({group_name} steps)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"val_loss_{group_name}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_probe(out_dir):
    probe = load_json("artifacts/probes/muon_mechanics_summary.json")
    rows = probe["shape_summary"]
    shapes = [r["shape"] for r in rows]
    exact_scaled = [r["exact_scaled_rms_mean"] for r in rows]
    ns_scaled = [r["observed_scaled_rms_mean"] for r in rows]
    x = list(range(len(rows)))
    width = 0.35
    plt.figure(figsize=(9, 4.8))
    plt.bar([i - width / 2 for i in x], exact_scaled, width=width, label="Exact SVD scaled")
    plt.bar([i + width / 2 for i in x], ns_scaled, width=width, label="5-step NS scaled")
    plt.axhline(0.2, color="black", linewidth=1, linestyle="--", label="0.2 target")
    plt.xticks(x, shapes, rotation=35, ha="right")
    plt.ylabel("Update RMS after paper scale")
    plt.title("Muon update RMS scaling probe")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = out_dir / "muon_rms_probe.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def markdown_table(rows):
    lines = [
        "| Run | Optimizer | WD | Scale | Steps | Final val loss | Final train loss | Tokens/s | Max CUDA MiB |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        s = row["summary"]
        scale = "paper" if s["args"]["optimizer"] == "muon" else ("none" if s["args"]["optimizer"] == "muon-noscale" else "n/a")
        lines.append(
            "| {run} | {opt} | {wd:.1f} | {scale} | {steps} | {val:.4f} | {train:.4f} | {tps:.0f} | {mem:.0f} |".format(
                run=row["run_dir"].name,
                opt=s["args"]["optimizer"],
                wd=s["args"]["wd"],
                scale=scale,
                steps=s["args"]["max_steps"],
                val=s["final_val_loss"],
                train=s["final_train_loss"],
                tps=s["tokens_per_s_avg"],
                mem=(s["max_cuda_memory_allocated"] or 0) / (1024 * 1024),
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/results-summary.md")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    figure_paths = []
    for group_name, paths in RUN_GROUPS.items():
        rows = collect_group(paths)
        all_rows[group_name] = rows
        figure_paths.append(plot_val(group_name, rows, figures_dir))
    figure_paths.append(plot_probe(figures_dir))

    with out.open("w") as f:
        f.write("# Results Summary\n\n")
        for group_name, rows in all_rows.items():
            f.write(f"## {group_name}-step OpenWebText suite\n\n")
            f.write(markdown_table(rows))
            f.write("\n\n")
        f.write("## Figures\n\n")
        for path in figure_paths:
            f.write(f"- `{path}`\n")
    print(out)


if __name__ == "__main__":
    main()
