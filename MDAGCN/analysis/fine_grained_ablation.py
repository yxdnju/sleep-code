import argparse
import csv
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.Dataset import SimpleDataset
from model.MDAGCN import (
    MDAGCN,
    Graph_Learn,
    SpatialAttention,
    TemporalAttention,
    cheb_conv_with_Att_GL,
)
from model.Utils import AddContext_MultiSub, ReadConfig
from strict_protocol_diagnostic import (
    build_optimizer,
    build_loss,
    score_predictions,
    selected_score,
    set_seed,
    summarize,
    to_labels,
    train_one_epoch,
    evaluate,
)


class IdentityTemporalAttention(nn.Module):
    def __init__(self, num_of_timesteps):
        super().__init__()
        self.num_of_timesteps = num_of_timesteps

    def forward(self, x):
        n = x.shape[0]
        eye = torch.eye(self.num_of_timesteps, device=x.device)
        return eye.unsqueeze(0).expand(n, -1, -1)


class IdentitySpatialAttention(nn.Module):
    def __init__(self, num_of_vertices):
        super().__init__()
        self.num_of_vertices = num_of_vertices

    def forward(self, x):
        n = x.shape[0]
        eye = torch.eye(self.num_of_vertices, device=x.device)
        return eye.unsqueeze(0).expand(n, -1, -1)


class FixedGraphLearn(nn.Module):
    def __init__(self, num_of_vertices, mode):
        super().__init__()
        if mode not in {"identity", "uniform"}:
            raise ValueError(f"Unknown fixed graph mode: {mode}")
        self.num_of_vertices = num_of_vertices
        self.mode = mode

    def forward(self, x):
        n, t, v, _ = x.shape
        if self.mode == "identity":
            graph = torch.eye(v, device=x.device).unsqueeze(0).unsqueeze(0)
        else:
            graph = torch.ones((1, 1, v, v), device=x.device) / float(v)
        graph = graph.expand(n, t, v, v)
        zero = x.new_tensor(0.0)
        return graph, zero, zero


class NoGraphLossLearn(Graph_Learn):
    def forward(self, x):
        graph, _, _ = super().forward(x)
        zero = x.new_tensor(0.0)
        return graph, zero, zero


class FineAblationBlock(nn.Module):
    def __init__(
        self,
        num_of_timesteps,
        num_of_vertices,
        num_of_features,
        k,
        num_of_chev_filters,
        num_of_time_filters,
        time_conv_strides,
        time_conv_kernel,
        gl_alpha,
        variant,
    ):
        super().__init__()
        self.variant = variant

        if variant in {"no_temporal_attention", "no_attention"}:
            self.temporal_Att = IdentityTemporalAttention(num_of_timesteps)
        else:
            self.temporal_Att = TemporalAttention(num_of_timesteps, num_of_vertices, num_of_features)

        if variant in {"no_spatial_attention", "no_attention"}:
            self.spatial_Att = IdentitySpatialAttention(num_of_vertices)
        else:
            self.spatial_Att = SpatialAttention(num_of_timesteps, num_of_vertices, num_of_features)

        if variant == "fixed_identity_graph":
            self.graph_learn = FixedGraphLearn(num_of_vertices, "identity")
        elif variant == "fixed_uniform_graph":
            self.graph_learn = FixedGraphLearn(num_of_vertices, "uniform")
        elif variant == "no_graph_loss":
            self.graph_learn = NoGraphLossLearn(num_of_features, gl_alpha)
        else:
            self.graph_learn = Graph_Learn(num_of_features, gl_alpha)

        self.graph_drop = nn.Dropout(0.3)
        self.gcn_GL1 = cheb_conv_with_Att_GL(num_of_features, 256, k)
        self.gcn_GL2 = cheb_conv_with_Att_GL(256, num_of_chev_filters, k)
        self.cnn_GL = nn.Conv2d(
            in_channels=num_of_chev_filters,
            out_channels=num_of_time_filters,
            kernel_size=(time_conv_kernel, 1),
            stride=(time_conv_strides, 1),
            padding="same",
        )
        self.norm_GL = nn.LayerNorm([num_of_time_filters], elementwise_affine=False)

    def forward(self, x):
        n, t, v, f = x.shape
        t_att = self.temporal_Att(x)
        x_t_att = torch.bmm(x.permute([0, 2, 3, 1]).reshape(n, v * f, t), t_att)
        x_t_att = x_t_att.reshape(n, v, f, t).permute([0, 3, 1, 2])
        s_att = self.spatial_Att(x_t_att)

        graph, loss1, loss2 = self.graph_learn(x)
        graph = self.graph_drop(graph)
        x_gl = self.gcn_GL1(x, s_att, graph)
        x_gl = self.gcn_GL2(x_gl, s_att, graph)
        x_gl = self.cnn_GL(x_gl.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        x_gl = self.norm_GL(torch.relu(x_gl))
        return x_gl, loss1, loss2


class FineAblationMDAGCN(nn.Module):
    def __init__(
        self,
        num_of_timesteps,
        num_of_vertices,
        num_of_features,
        k,
        num_of_chev_filters,
        num_of_time_filters,
        time_conv_strides,
        time_conv_kernel,
        gl_alpha,
        dropout,
        variant,
        num_classes=5,
    ):
        super().__init__()
        self.gcn = FineAblationBlock(
            num_of_timesteps,
            num_of_vertices,
            num_of_features,
            k,
            num_of_chev_filters,
            num_of_time_filters,
            time_conv_strides,
            time_conv_kernel,
            gl_alpha,
            variant,
        )
        self.dropout = dropout
        self.drop = nn.Dropout(dropout) if dropout != 0 else nn.Identity()
        self.gru = nn.GRU(num_of_time_filters * num_of_vertices, num_of_time_filters * num_of_vertices, 2)
        self.dense_class = nn.Sequential(
            nn.Linear(num_of_time_filters * num_of_timesteps * num_of_vertices, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        gcn_out, loss1, loss2 = self.gcn(x)
        gcn_out = torch.flatten(gcn_out, start_dim=2)
        gru_out, _ = self.gru(gcn_out)
        gru_out = torch.flatten(gru_out, start_dim=1)
        gru_out = self.drop(gru_out)
        return self.dense_class(gru_out), loss1, loss2


def build_model(variant, context, channels, cheb_k, cheb_filters, time_filters, time_stride, time_kernel, gl_alpha, dropout):
    if variant == "full":
        return MDAGCN(context, channels, 128, cheb_k, cheb_filters, time_filters, time_stride, time_kernel, gl_alpha, dropout, num_classes=5)
    return FineAblationMDAGCN(
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
        variant,
        num_classes=5,
    )


def load_fold_data(path_cfg, feature_root, fold_num, context, fold_idx):
    features = np.load(os.path.join(feature_root, f"Feature_{fold_idx}.npz"), allow_pickle=True)
    train_feature = np.float32(features["train_feature"])
    train_labels = to_labels(features["train_targets"])
    val_feature = np.float32(features["val_feature"])
    val_labels = to_labels(features["val_targets"])

    f_mean = np.mean(train_feature, axis=(0, 1), keepdims=True)
    f_std = np.std(train_feature, axis=(0, 1), keepdims=True) + 1e-8
    train_feature = (train_feature - f_mean) / f_std
    val_feature = (val_feature - f_mean) / f_std

    train_fold_num = np.delete(fold_num.copy(), [fold_idx, (fold_idx + 9) % 10])
    train_feature, train_labels = AddContext_MultiSub(train_feature, train_labels, train_fold_num, context, fold_idx)
    val_feature, val_labels = AddContext_MultiSub(val_feature, val_labels, np.array([len(val_labels)]), context, 0)
    return train_feature, train_labels, val_feature, val_labels


def run_variant(args, variant, cfg_train, cfg_model, path_cfg, fold_num, feature_root, save_root, device):
    channels = int(cfg_train["channels"])
    fold = int(cfg_train["fold"])
    context = int(cfg_train["context"])
    epochs = args.epochs if args.epochs is not None else int(cfg_train["epoch"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg_train["batch_size"])
    optimizer_name = args.optimizer if args.optimizer is not None else cfg_train["optimizer"]
    learn_rate = args.lr if args.lr is not None else float(cfg_train["learn_rate"])

    gl_alpha = args.gl_alpha if args.gl_alpha is not None else float(cfg_model["GLalpha"])
    cheb_filters = int(cfg_model["cheb_filters"])
    time_filters = int(cfg_model["time_filters"])
    time_stride = int(cfg_model["time_conv_strides"])
    time_kernel = int(cfg_model["time_conv_kernel"])
    cheb_k = args.cheb_k if args.cheb_k is not None else int(cfg_model["cheb_k"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(cfg_model["l2"])
    dropout = args.dropout if args.dropout is not None else float(cfg_model["dropout"])

    folds = list(range(fold))
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    rows = []
    aggregated = {}
    variant_root = save_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        set_seed(seed, deterministic=args.deterministic)
        seed_true = []
        seed_pred = []

        for fold_idx in folds:
            start = time.time()
            train_feature, train_labels, val_feature, val_labels = load_fold_data(
                path_cfg,
                feature_root,
                fold_num,
                context,
                fold_idx,
            )

            train_dataset = SimpleDataset(np.float32(train_feature), train_labels)
            val_dataset = SimpleDataset(np.float32(val_feature), val_labels)
            generator = torch.Generator()
            generator.manual_seed(seed * 1000 + fold_idx)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                generator=generator,
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

            model = build_model(
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
            loss_func = build_loss(
                args.loss,
                train_labels,
                device,
                args.weight_power,
                args.focal_gamma,
                args.label_smoothing,
            )
            optimizer = build_optimizer(optimizer_name, learn_rate, model, weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

            best_metric = -float("inf")
            best_epoch = -1
            best_result = None
            best_true = None
            best_pred = None
            stale = 0

            print("-" * 100)
            print(f"Variant {variant}, seed {seed}, fold {fold_idx}: train={len(train_dataset)}, val={len(val_dataset)}")

            for epoch in range(epochs):
                train_result = train_one_epoch(
                    model,
                    train_loader,
                    loss_func,
                    optimizer,
                    device,
                    accumulation_steps=args.accumulation_steps,
                    grad_clip=args.grad_clip,
                )
                val_result, y_true, y_pred = evaluate(model, val_loader, loss_func, device)
                scheduler.step(val_result["loss"])
                current = selected_score(val_result, args.selection_metric)

                if current > best_metric:
                    best_metric = current
                    best_epoch = epoch + 1
                    best_result = dict(val_result)
                    best_true = y_true.copy()
                    best_pred = y_pred.copy()
                    stale = 0
                else:
                    stale += 1

                print(
                    "Epoch {:03d} | tr_acc {:.4f} | val_acc {:.4f} | val_f1 {:.4f} | val_kappa {:.4f} | best {:.4f} @ {:03d}".format(
                        epoch + 1,
                        train_result["acc"],
                        val_result["acc"],
                        val_result["macro_f1"],
                        val_result["kappa"],
                        best_metric,
                        best_epoch,
                    )
                )

                if stale >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            row = {
                "variant": variant,
                "seed": seed,
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "elapsed_sec": round(time.time() - start, 3),
            }
            row.update(best_result)
            rows.append(row)
            seed_true.append(best_true)
            seed_pred.append(best_pred)

            fold_dir = variant_root / f"seed_{seed}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.savez(fold_dir / f"fold_{fold_idx}_predictions.npz", y_true=best_true, y_pred=best_pred)

            del model, train_dataset, val_dataset
            torch.cuda.empty_cache()

        seed_true = np.concatenate(seed_true)
        seed_pred = np.concatenate(seed_pred)
        aggregated[str(seed)] = score_predictions(seed_true, seed_pred)
        np.savez(variant_root / f"seed_{seed}_all_predictions.npz", y_true=seed_true, y_pred=seed_pred)

    summary = {
        "variant": variant,
        "seeds": args.seeds,
        "selection_metric": args.selection_metric,
        "per_fold": {},
        "per_seed_aggregated_predictions": aggregated,
    }
    keys = ["acc", "macro_f1", "kappa", "precision_macro", "recall_macro", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    for key in keys:
        summary["per_fold"][key] = summarize([row[key] for row in rows])

    with (variant_root / "fold_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (variant_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-grained structural ablation for MDAGCN.")
    parser.add_argument("-c", "--config", default="./config/ISRUC_S3.config")
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--variants", default="full,no_temporal_attention,no_spatial_attention,no_attention,no_graph_loss,fixed_identity_graph,fixed_uniform_graph")
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--selection-metric", choices=["macro_f1", "kappa", "acc", "hybrid", "acc_f1"], default="hybrid")
    parser.add_argument("--loss", choices=["weighted_ce", "ce", "focal", "balanced_softmax"], default="weighted_ce")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "RMSprop", "SGD"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--cheb-k", type=int, default=None)
    parser.add_argument("--gl-alpha", type=float, default=None)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    args.seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_cfg, _, cfg_train, cfg_model = ReadConfig(args.config)

    feature_root = args.feature_root if args.feature_root is not None else path_cfg["save"]
    if not feature_root.endswith(("/", "\\")):
        feature_root = feature_root + os.sep
    save_root = Path(args.save_root or os.path.join(feature_root, "fine_grained_ablation"))
    save_root.mkdir(parents=True, exist_ok=True)

    read_list = np.load(path_cfg["data"], allow_pickle=True)
    fold_num = read_list["Fold_len"]

    all_summaries = {}
    for variant in variants:
        print("=" * 100)
        print(f"Running variant: {variant}")
        print("=" * 100)
        all_summaries[variant] = run_variant(args, variant, cfg_train, cfg_model, path_cfg, fold_num, feature_root, save_root, device)

    with (save_root / "all_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 100)
    print("Fine-grained structural ablation summary")
    for variant, summary in all_summaries.items():
        acc = summary["per_fold"]["acc"]
        f1 = summary["per_fold"]["macro_f1"]
        kappa = summary["per_fold"]["kappa"]
        n1 = summary["per_fold"]["f1_N1"]
        print(
            f"{variant}: Acc {acc['mean']:.4f}+/-{acc['std']:.4f}, "
            f"F1 {f1['mean']:.4f}+/-{f1['std']:.4f}, "
            f"Kappa {kappa['mean']:.4f}+/-{kappa['std']:.4f}, "
            f"N1-F1 {n1['mean']:.4f}+/-{n1['std']:.4f}"
        )
    print(f"Saved to: {save_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()
