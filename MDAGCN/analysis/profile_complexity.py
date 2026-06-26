import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from thop import profile

from fine_grained_ablation import build_model
from model.Utils import ReadConfig


DEFAULT_VARIANTS = [
    "full",
    "no_temporal_attention",
    "no_spatial_attention",
    "no_attention",
    "no_graph_loss",
    "fixed_identity_graph",
    "fixed_uniform_graph",
]


DISPLAY_NAMES = {
    "full": "Full MDAGCN",
    "cnn_gru_no_gcn": "CNN-GRU / w/o GCN",
    "no_temporal_attention": "w/o temporal attention",
    "no_spatial_attention": "w/o spatial attention",
    "no_attention": "w/o temporal and spatial attention",
    "no_graph_loss": "w/o graph-learning regularization",
    "fixed_identity_graph": "fixed identity graph",
    "fixed_uniform_graph": "fixed uniform graph",
}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_full_metrics(path):
    if path is None or not Path(path).exists():
        return {}
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    patterns = {
        "acc": r"Accuracy\s+([0-9.]+)",
        "macro_f1": r"F1-Score\s+([0-9.]+)",
        "kappa": r"Cohen Kappa\s+([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def load_ablation_metrics(summary_path):
    if summary_path is None or not Path(summary_path).exists():
        return {}
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    out = {}
    for variant, summary in data.items():
        per_fold = summary.get("per_fold", {})
        out[variant] = {
            "acc": per_fold.get("acc", {}).get("mean"),
            "acc_std": per_fold.get("acc", {}).get("std"),
            "macro_f1": per_fold.get("macro_f1", {}).get("mean"),
            "macro_f1_std": per_fold.get("macro_f1", {}).get("std"),
            "kappa": per_fold.get("kappa", {}).get("mean"),
            "kappa_std": per_fold.get("kappa", {}).get("std"),
        }
    return out


def load_coarse_ablation_metrics(path):
    if path is None or not Path(path).exists():
        return {}
    data = np.load(path, allow_pickle=True).item()
    mapping = {
        "Ablation c: CNN Only (No GCN)": "cnn_gru_no_gcn",
        "Complete Model (Our Proposed)": "full_coarse",
    }
    out = {}
    for source_name, variant in mapping.items():
        if source_name not in data:
            continue
        item = data[source_name]
        out[variant] = {
            "acc": item.get("mean_accuracy", np.nan) / 100.0,
            "acc_std": item.get("std_accuracy", np.nan) / 100.0,
            "macro_f1": item.get("mean_f1_macro"),
            "macro_f1_std": item.get("std_f1_macro"),
            "kappa": item.get("mean_kappa"),
            "kappa_std": item.get("std_kappa"),
        }
    return out


def build_profile_model(variant, context, channels, cheb_k, cheb_filters, time_filters, time_stride, time_kernel, gl_alpha, dropout):
    if variant == "cnn_gru_no_gcn":
        from evaluate_ablation import MDAGCN_Ablation_c

        return MDAGCN_Ablation_c(
            context,
            channels,
            128,
            cheb_k,
            cheb_filters,
            time_filters,
            time_stride,
            time_kernel,
            gl_alpha,
            dropout,
            num_classes=5,
        )
    return build_model(
        variant,
        context,
        channels,
        cheb_k,
        cheb_filters,
        time_filters,
        time_stride,
        time_kernel,
        gl_alpha,
        dropout,
    )


def measure_inference(model, x, warmup, repeats, device):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_memory_mb = None
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return elapsed / repeats, peak_memory_mb


def safe_profile_macs(model, x):
    try:
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs), None
    except Exception as exc:
        return None, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Profile MDAGCN complexity without retraining.")
    parser.add_argument("-c", "--config", default="config/ISRUC_S3.config")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ablation-summary", default="output_ISRUC/fine_grained_ablation_adamw_s242526/all_summary.json")
    parser.add_argument("--coarse-ablation", default="output_ISRUC/proper_ablation_results.npy")
    parser.add_argument("--full-eval", default="output_ISRUC_0/Result_MDAGCN_Evaluation.txt")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    path_cfg, _, cfg_train, cfg_model = ReadConfig(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else Path(path_cfg["save"]) / "complexity_profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    context = int(cfg_train["context"])
    channels = int(cfg_train["channels"])
    cheb_k = int(cfg_model["cheb_k"])
    cheb_filters = int(cfg_model["cheb_filters"])
    time_filters = int(cfg_model["time_filters"])
    time_stride = int(cfg_model["time_conv_strides"])
    time_kernel = int(cfg_model["time_conv_kernel"])
    gl_alpha = float(cfg_model["GLalpha"])
    dropout = float(cfg_model["dropout"])

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    ablation_metrics = load_ablation_metrics(args.ablation_summary)
    coarse_metrics = load_coarse_ablation_metrics(args.coarse_ablation)
    full_metrics = parse_full_metrics(args.full_eval)
    performance = {"full": full_metrics, **ablation_metrics, **coarse_metrics}

    rows = []
    for variant in variants:
        model = build_profile_model(
            variant,
            context,
            channels,
            cheb_k,
            cheb_filters,
            time_filters,
            time_stride,
            time_kernel,
            gl_alpha,
            dropout,
        ).to(device)
        x = torch.randn(args.batch_size, context, channels, args.feature_dim, device=device)

        total_params, trainable_params = count_params(model)
        macs, macs_error = safe_profile_macs(model, x)
        avg_batch_sec, peak_memory_mb = measure_inference(model, x, args.warmup, args.repeats, device)
        metrics = performance.get(variant, {})

        row = {
            "variant": variant,
            "method": DISPLAY_NAMES.get(variant, variant),
            "input_shape": f"{args.batch_size}x{context}x{channels}x{args.feature_dim}",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "params_m": total_params / 1e6,
            "macs": macs,
            "macs_m": None if macs is None else macs / 1e6,
            "macs_per_sample_m": None if macs is None else macs / args.batch_size / 1e6,
            "macs_error": macs_error,
            "avg_inference_ms_per_batch": avg_batch_sec * 1000.0,
            "avg_inference_ms_per_sample": avg_batch_sec * 1000.0 / args.batch_size,
            "peak_cuda_memory_mb": peak_memory_mb,
            "acc": metrics.get("acc"),
            "acc_std": metrics.get("acc_std"),
            "macro_f1": metrics.get("macro_f1"),
            "macro_f1_std": metrics.get("macro_f1_std"),
            "kappa": metrics.get("kappa"),
            "kappa_std": metrics.get("kappa_std"),
        }
        rows.append(row)
        print(
            f"{row['method']}: params={row['params_m']:.3f}M, "
            f"MACs={(row['macs_m'] if row['macs_m'] is not None else float('nan')):.3f}M, "
            f"time={row['avg_inference_ms_per_batch']:.3f} ms/batch"
        )
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out_dir / "complexity_profile.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": args.config,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "ablation_summary": args.ablation_summary,
        "full_eval": args.full_eval,
        "rows": rows,
    }
    with (out_dir / "complexity_profile_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print(f"Saved complexity profile to: {csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
