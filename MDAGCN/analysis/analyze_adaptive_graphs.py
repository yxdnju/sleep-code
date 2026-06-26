import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MDAGCN import MDAGCN
from model.Utils import AddContext_SingleSub, ReadConfig


CLASSES = ["W", "N1", "N2", "N3", "REM"]


def to_labels(targets):
    if targets.ndim > 1:
        return np.argmax(targets, axis=1).astype(np.int64)
    return targets.astype(np.int64)


def entropy_from_graph(graph):
    p = graph.reshape(graph.shape[0], -1)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def offdiag_values(graph):
    v = graph.shape[-1]
    mask = ~np.eye(v, dtype=bool)
    return graph[:, mask]


def plot_heatmap(matrix, title, out_path, node_labels):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(float(matrix.max()), 1e-8))
    ax.set_title(title)
    ax.set_xticks(np.arange(len(node_labels)))
    ax.set_yticks(np.arange(len(node_labels)))
    ax.set_xticklabels(node_labels, rotation=45, ha="right")
    ax.set_yticklabels(node_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return q_values
    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    valid_q = np.empty_like(valid_p)
    valid_q[order] = adjusted
    q_values[valid] = valid_q
    return q_values


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative interpretability analysis for MDAGCN adaptive graphs."
    )
    parser.add_argument("-c", "--config", default="config/SleepEDF.config")
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--folds", default=None, help="Comma-separated fold ids. Default: all folds in config.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--slice", choices=["center", "mean"], default="center")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    path_cfg, _, cfg_train, cfg_model = ReadConfig(args.config)
    save_root = Path(path_cfg["save"])
    feature_root = Path(args.feature_root) if args.feature_root else save_root
    model_root = Path(args.model_root) if args.model_root else save_root
    out_dir = Path(args.out_dir) if args.out_dir else save_root / "adaptive_graph_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = int(cfg_train["channels"])
    fold_count = int(cfg_train["fold"])
    context = int(cfg_train["context"])
    num_of_chev_filters = int(cfg_model["cheb_filters"])
    num_of_time_filters = int(cfg_model["time_filters"])
    time_conv_strides = int(cfg_model["time_conv_strides"])
    time_conv_kernel = int(cfg_model["time_conv_kernel"])
    cheb_k = int(cfg_model["cheb_k"])
    gl_alpha = float(cfg_model["GLalpha"])
    dropout = float(cfg_model["dropout"])

    if args.folds:
        folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    else:
        folds = list(range(fold_count))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    graphs = []
    labels = []
    fold_ids = []

    for fold in folds:
        feature_path = feature_root / f"Feature_{fold}.npz"
        model_path = model_root / f"MDAGCN_Best_{fold}.pth"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        features = np.load(feature_path, allow_pickle=True)
        train_feature = np.float32(features["train_feature"])
        val_feature = np.float32(features["val_feature"])
        val_labels = to_labels(features["val_targets"])

        f_mean = np.mean(train_feature, axis=(0, 1), keepdims=True)
        f_std = np.std(train_feature, axis=(0, 1), keepdims=True) + 1e-8
        val_feature = (val_feature - f_mean) / f_std
        val_feature, val_labels = AddContext_SingleSub(val_feature, val_labels, context)

        model = MDAGCN(
            context,
            channels,
            128,
            cheb_k,
            num_of_chev_filters,
            num_of_time_filters,
            time_conv_strides,
            time_conv_kernel,
            gl_alpha,
            dropout,
            num_classes=5,
        ).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        fold_graphs = []
        with torch.no_grad():
            for start in range(0, len(val_feature), args.batch_size):
                batch = torch.tensor(val_feature[start:start + args.batch_size], dtype=torch.float32, device=device)
                s, _, _ = model.gcn.graph_learn(batch)
                if args.slice == "center":
                    g = s[:, context // 2].detach().cpu().numpy()
                else:
                    g = s.mean(dim=1).detach().cpu().numpy()
                fold_graphs.append(g)

        fold_graphs = np.concatenate(fold_graphs, axis=0)
        graphs.append(fold_graphs)
        labels.append(val_labels)
        fold_ids.append(np.full(len(val_labels), fold, dtype=np.int64))
        print(f"Fold {fold}: graphs={fold_graphs.shape}, labels={val_labels.shape}")

    graphs = np.concatenate(graphs, axis=0)
    labels = np.concatenate(labels, axis=0)
    fold_ids = np.concatenate(fold_ids, axis=0)

    np.savez(out_dir / "adaptive_graph_samples.npz", graphs=graphs, labels=labels, folds=fold_ids)

    node_labels = [f"N{i}" for i in range(channels)]
    if channels == 10:
        node_labels = ["C3", "C4", "F3", "F4", "O1", "O2", "EOG-L", "EOG-R", "EMG", "ECG"]
    elif channels == 4:
        node_labels = ["EEG-1", "EEG-2", "EOG", "EMG"]

    stage_summary = {}
    rows = []
    for class_id, class_name in enumerate(CLASSES):
        mask = labels == class_id
        class_graphs = graphs[mask]
        mean_graph = class_graphs.mean(axis=0)
        std_graph = class_graphs.std(axis=0)
        ent = entropy_from_graph(class_graphs)
        offdiag = offdiag_values(class_graphs)
        stage_summary[class_name] = {
            "n": int(mask.sum()),
            "mean_entropy": float(ent.mean()),
            "std_entropy": float(ent.std(ddof=1)) if len(ent) > 1 else 0.0,
            "mean_offdiag_weight": float(offdiag.mean()),
            "std_offdiag_weight": float(offdiag.std(ddof=1)) if offdiag.size > 1 else 0.0,
        }
        np.savetxt(out_dir / f"mean_graph_{class_name}.csv", mean_graph, delimiter=",")
        np.savetxt(out_dir / f"std_graph_{class_name}.csv", std_graph, delimiter=",")
        plot_heatmap(mean_graph, f"Mean adaptive graph: {class_name}", out_dir / f"mean_graph_{class_name}.png", node_labels)
        rows.append([class_name, int(mask.sum()), ent.mean(), ent.std(ddof=1), offdiag.mean(), offdiag.std(ddof=1)])

    with (out_dir / "stage_graph_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "graph_entropy_mean", "graph_entropy_std", "offdiag_weight_mean", "offdiag_weight_std"])
        writer.writerows(rows)

    edge_rows = []
    v = graphs.shape[-1]
    for i in range(v):
        for j in range(v):
            if i == j:
                continue
            groups = [graphs[labels == class_id, i, j] for class_id in range(len(CLASSES))]
            if all(len(g) > 0 for g in groups):
                h_stat, p_value = stats.kruskal(*groups)
            else:
                h_stat, p_value = np.nan, np.nan
            n_total = sum(len(g) for g in groups)
            k_groups = len(groups)
            epsilon_sq = (h_stat - k_groups + 1) / (n_total - k_groups) if np.isfinite(h_stat) and n_total > k_groups else np.nan
            means = [float(g.mean()) if len(g) else np.nan for g in groups]
            max_stage = CLASSES[int(np.nanargmax(means))]
            min_stage = CLASSES[int(np.nanargmin(means))]
            edge_rows.append({
                "source": node_labels[i],
                "target": node_labels[j],
                "kruskal_h": float(h_stat),
                "p_value": float(p_value),
                "q_value_fdr": np.nan,
                "epsilon_squared": float(epsilon_sq),
                "max_stage": max_stage,
                "min_stage": min_stage,
                **{f"mean_{CLASSES[k]}": means[k] for k in range(len(CLASSES))},
            })

    q_values = benjamini_hochberg([row["p_value"] for row in edge_rows])
    for row, q_value in zip(edge_rows, q_values):
        row["q_value_fdr"] = float(q_value)

    edge_rows.sort(key=lambda r: (np.nan_to_num(r["p_value"], nan=1.0), -np.nan_to_num(r["kruskal_h"], nan=0.0)))
    with (out_dir / "edge_stage_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source", "target", "kruskal_h", "p_value", "q_value_fdr", "epsilon_squared", "max_stage", "min_stage"] + [f"mean_{c}" for c in CLASSES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edge_rows)

    summary = {
        "config": args.config,
        "feature_root": str(feature_root),
        "model_root": str(model_root),
        "folds": folds,
        "slice": args.slice,
        "num_samples": int(len(labels)),
        "channels": channels,
        "node_labels": node_labels,
        "stage_summary": stage_summary,
        "top_stage_dependent_edges": edge_rows[:20],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"Saved adaptive graph analysis to: {out_dir}")
    print("Stage summary:")
    for stage, values in stage_summary.items():
        print(
            f"{stage}: n={values['n']}, entropy={values['mean_entropy']:.4f} +/- {values['std_entropy']:.4f}, "
            f"offdiag={values['mean_offdiag_weight']:.4f} +/- {values['std_offdiag_weight']:.4f}"
        )
    print("Top 5 stage-dependent edges:")
    for row in edge_rows[:5]:
        print(f"{row['source']}->{row['target']}: H={row['kruskal_h']:.3f}, p={row['p_value']:.3e}, max={row['max_stage']}")


if __name__ == "__main__":
    main()
