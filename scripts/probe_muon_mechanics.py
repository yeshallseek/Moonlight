#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch


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


def rms(x):
    return x.float().square().mean().sqrt().item()


def exact_orthogonal_update(g):
    u, _, vh = torch.linalg.svd(g.float().cpu(), full_matrices=False)
    return u @ vh


def parse_shapes(raw):
    shapes = []
    for item in raw.split(","):
        left, right = item.lower().split("x", 1)
        shapes.append((int(left), int(right)))
    return shapes


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_probe(args):
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    records = []

    for a_dim, b_dim in parse_shapes(args.shapes):
        for trial in range(args.trials):
            g = torch.randn(a_dim, b_dim, device=device)
            update = zeropower_via_newtonschulz5(g, args.ns_steps)
            exact = exact_orthogonal_update(g)
            singular = torch.linalg.svdvals(update.float().cpu())
            theoretical_rms = 1.0 / math.sqrt(max(a_dim, b_dim))
            unscaled_rms = rms(update)
            exact_rms = rms(exact)
            paper_scale = 0.2 * math.sqrt(max(a_dim, b_dim))
            scaled_rms = rms(update * paper_scale)
            exact_scaled_rms = rms(exact * paper_scale)
            records.append(
                {
                    "shape": f"{a_dim}x{b_dim}",
                    "rows": a_dim,
                    "cols": b_dim,
                    "trial": trial,
                    "ns_steps": args.ns_steps,
                    "theoretical_unscaled_rms": theoretical_rms,
                    "exact_unscaled_rms": exact_rms,
                    "exact_scaled_rms": exact_scaled_rms,
                    "observed_unscaled_rms": unscaled_rms,
                    "observed_scaled_rms": scaled_rms,
                    "scaled_target_rms": 0.2,
                    "exact_unscaled_rms_ratio": exact_rms / theoretical_rms,
                    "unscaled_rms_ratio": unscaled_rms / theoretical_rms,
                    "scaled_rms_error": scaled_rms - 0.2,
                    "singular_min": singular.min().item(),
                    "singular_mean": singular.mean().item(),
                    "singular_max": singular.max().item(),
                }
            )

    return records


def write_outputs(args, records, elapsed_s):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "muon_mechanics_records.csv"
    json_path = out_dir / "muon_mechanics_summary.json"
    md_path = out_dir / "muon_mechanics_summary.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    by_shape = {}
    for record in records:
        by_shape.setdefault(record["shape"], []).append(record)

    shape_summary = []
    for shape, rows in by_shape.items():
        shape_summary.append(
            {
                "shape": shape,
                "theoretical_unscaled_rms": rows[0]["theoretical_unscaled_rms"],
                "exact_unscaled_rms_mean": sum(r["exact_unscaled_rms"] for r in rows) / len(rows),
                "exact_scaled_rms_mean": sum(r["exact_scaled_rms"] for r in rows) / len(rows),
                "observed_unscaled_rms_mean": sum(r["observed_unscaled_rms"] for r in rows) / len(rows),
                "observed_scaled_rms_mean": sum(r["observed_scaled_rms"] for r in rows) / len(rows),
                "singular_mean_mean": sum(r["singular_mean"] for r in rows) / len(rows),
                "trials": len(rows),
            }
        )

    summary = {
        "kind": "muon_mechanics_probe",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": elapsed_s,
        "git_commit": git_commit(),
        "device": args.device,
        "seed": args.seed,
        "ns_steps": args.ns_steps,
        "records_path": str(csv_path),
        "shape_summary": shape_summary,
    }

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    with md_path.open("w") as f:
        f.write("# Muon Mechanics Probe\n\n")
        f.write(f"- Device: `{args.device}`\n")
        f.write(f"- Newton-Schulz steps: `{args.ns_steps}`\n")
        f.write(f"- Trials per shape: `{args.trials}`\n")
        f.write(f"- Elapsed: `{elapsed_s:.3f}` seconds\n\n")
        f.write("| Shape | Theory RMS | Exact RMS | NS RMS | Exact scaled RMS | NS scaled RMS | Mean NS singular value |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in shape_summary:
            f.write(
                "| {shape} | {theoretical_unscaled_rms:.6f} | {exact_unscaled_rms_mean:.6f} | "
                "{observed_unscaled_rms_mean:.6f} | {exact_scaled_rms_mean:.6f} | "
                "{observed_scaled_rms_mean:.6f} | {singular_mean_mean:.6f} |\n".format(**row)
            )

    if args.runs_jsonl:
        with Path(args.runs_jsonl).open("a") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default="artifacts/probes")
    parser.add_argument("--runs-jsonl", default="experiments/runs.jsonl")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument(
        "--shapes",
        default="16x16,16x64,64x16,64x256,256x64,128x512,512x128,256x256",
    )
    args = parser.parse_args()

    start = time.time()
    records = run_probe(args)
    summary = write_outputs(args, records, time.time() - start)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
