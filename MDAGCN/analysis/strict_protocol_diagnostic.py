import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.Dataset import SimpleDataset
from model.MDAGCN import MDAGCN
from model.Utils import AddContext_MultiSub, AddContext_MultiSub_EDF, AddContext_SingleSub
from model.Utils import Instantiation_optim, ReadConfig


CLASSES = ["W", "N1", "N2", "N3", "REM"]


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts, label_smoothing=0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        self.register_buffer("log_counts", torch.log(counts + 1e-8))
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        return nn.functional.cross_entropy(
            logits + self.log_counts.to(logits.device),
            target,
            label_smoothing=self.label_smoothing,
        )


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_labels(targets):
    if targets.ndim > 1:
        return np.argmax(targets, axis=1)
    return targets.astype(np.int64)


def compute_class_weights(labels, power=0.5):
    class_counts = np.bincount(labels, minlength=5)
    total_samples = len(labels)
    weights = (total_samples / (class_counts + 1e-8)) ** power
    weights = weights / np.sum(weights) * 5.0
    return weights.astype(np.float32)


def build_loss(loss_name, labels, device, weight_power, focal_gamma, label_smoothing):
    class_counts = np.bincount(labels, minlength=5).astype(np.float32)
    if loss_name in {"weighted_ce", "focal"}:
        weights = torch.tensor(compute_class_weights(labels, power=weight_power), device=device)
    else:
        weights = None

    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(weight=weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    if loss_name == "balanced_softmax":
        return BalancedSoftmaxLoss(class_counts, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss: {loss_name}")


def build_optimizer(name, lr, model, weight_decay):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return Instantiation_optim(name, lr, model, weight_decay)


def selected_score(result, metric):
    if metric == "hybrid":
        return result["acc"] + result["macro_f1"] + result["kappa"]
    if metric == "acc_f1":
        return result["acc"] + result["macro_f1"]
    return result[metric]


def build_context(train_feature, train_labels, val_feature, val_labels, fold_num, context, fold_idx, mode):
    if mode == "edf_pad":
        train_feature, train_labels = AddContext_MultiSub_EDF(
            train_feature,
            train_labels,
            np.delete(fold_num.copy(), [fold_idx, (fold_idx + 9) % 10]),
            context,
            fold_idx,
        )
    elif mode == "existing":
        train_feature, train_labels = AddContext_MultiSub(
            train_feature,
            train_labels,
            np.delete(fold_num.copy(), [fold_idx, (fold_idx + 9) % 10]),
            context,
            fold_idx,
        )
    else:
        raise ValueError(f"Unknown context mode: {mode}")

    val_feature, val_labels = AddContext_SingleSub(val_feature, val_labels, context)
    return train_feature, train_labels, val_feature, val_labels


def score_predictions(y_true, y_pred):
    per_class_f1 = metrics.f1_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    return {
        "acc": float(metrics.accuracy_score(y_true, y_pred)),
        "macro_f1": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(metrics.cohen_kappa_score(y_true, y_pred)),
        "precision_macro": float(metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_W": float(per_class_f1[0]),
        "f1_N1": float(per_class_f1[1]),
        "f1_N2": float(per_class_f1[2]),
        "f1_N3": float(per_class_f1[3]),
        "f1_REM": float(per_class_f1[4]),
    }


def train_one_epoch(model, loader, loss_func, optimizer, device, accumulation_steps, grad_clip):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    total_main_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        class_out, loss1, loss2 = model(x)
        main_loss = loss_func(class_out, y)
        loss = main_loss + loss1 + loss2
        (loss / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        pred = torch.argmax(class_out.detach(), dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        total_loss += float(loss.item()) * int(y.numel())
        total_main_loss += float(main_loss.item()) * int(y.numel())

    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": total_loss / max(total, 1),
        "main_loss": total_main_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def evaluate(model, loader, loss_func, device):
    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    total = 0
    preds = []
    trues = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            class_out, loss1, loss2 = model(x)
            main_loss = loss_func(class_out, y)
            loss = main_loss + loss1 + loss2
            pred = torch.argmax(class_out, dim=1)

            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            total += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())
            total_main_loss += float(main_loss.item()) * int(y.numel())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    result = score_predictions(y_true, y_pred)
    result["loss"] = total_loss / max(total, 1)
    result["main_loss"] = total_main_loss / max(total, 1)
    return result, y_true, y_pred


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def main():
    parser = argparse.ArgumentParser(
        description="Strict protocol diagnostic for MDAGCN: fixed seeds, one run per fold, no best-of-repeats."
    )
    parser.add_argument("-c", "--config", default="./config/ISRUC_S3.config")
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--feature-root", default=None, help="Directory containing Feature_*.npz. Defaults to config [path] save.")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--selection-metric", choices=["macro_f1", "kappa", "acc", "hybrid", "acc_f1"], default="macro_f1")
    parser.add_argument("--context-mode", choices=["existing", "edf_pad"], default="existing")
    parser.add_argument("--loss", choices=["weighted_ce", "ce", "focal", "balanced_softmax"], default="weighted_ce")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "RMSprop", "SGD"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--sampler", choices=["none", "balanced"], default="none")
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_cfg, _, cfg_train, cfg_model = ReadConfig(args.config)
    data_path = path_cfg["data"]
    feature_root = args.feature_root if args.feature_root is not None else path_cfg["save"]
    if not feature_root.endswith(("/", "\\")):
        feature_root = feature_root + os.sep
    save_root = args.save_root
    if save_root is None:
        save_root = os.path.join(feature_root, "strict_protocol")
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    channels = int(cfg_train["channels"])
    fold = int(cfg_train["fold"])
    context = int(cfg_train["context"])
    num_epochs = args.epochs if args.epochs is not None else int(cfg_train["epoch"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg_train["batch_size"])
    optimizer_name = args.optimizer if args.optimizer is not None else cfg_train["optimizer"]
    learn_rate = args.lr if args.lr is not None else float(cfg_train["learn_rate"])

    gl_alpha = float(cfg_model["GLalpha"])
    num_of_chev_filters = int(cfg_model["cheb_filters"])
    num_of_time_filters = int(cfg_model["time_filters"])
    time_conv_strides = int(cfg_model["time_conv_strides"])
    time_conv_kernel = int(cfg_model["time_conv_kernel"])
    cheb_k = int(cfg_model["cheb_k"])
    l2 = args.weight_decay if args.weight_decay is not None else float(cfg_model["l2"])
    dropout = args.dropout if args.dropout is not None else float(cfg_model["dropout"])

    read_list = np.load(data_path, allow_pickle=True)
    fold_num = read_list["Fold_len"]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    folds = list(range(fold))
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    rows = []
    all_predictions = {}

    print("=" * 100)
    print("Strict protocol diagnostic")
    print(f"Config: {args.config}")
    print(f"Feature root: {feature_root}")
    print(f"Save root: {save_root}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Folds: {folds}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Context mode: {args.context_mode}")
    print(f"Loss: {args.loss}")
    print(f"Optimizer: {optimizer_name}, lr={learn_rate}, weight_decay={l2}, dropout={dropout}")
    print(f"Weight power: {args.weight_power}, focal_gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}")
    print(f"Sampler: {args.sampler}, accumulation_steps={args.accumulation_steps}, grad_clip={args.grad_clip}")
    print("=" * 100)

    for seed in seeds:
        set_seed(seed, deterministic=args.deterministic)
        seed_true = []
        seed_pred = []

        for fold_idx in folds:
            start_time = time.time()
            feature_path = os.path.join(feature_root, f"Feature_{fold_idx}.npz")
            features = np.load(feature_path, allow_pickle=True)
            train_feature = np.float32(features["train_feature"])
            train_labels = to_labels(features["train_targets"])
            val_feature = np.float32(features["val_feature"])
            val_labels = to_labels(features["val_targets"])

            f_mean = np.mean(train_feature, axis=(0, 1), keepdims=True)
            f_std = np.std(train_feature, axis=(0, 1), keepdims=True) + 1e-8
            train_feature = (train_feature - f_mean) / f_std
            val_feature = (val_feature - f_mean) / f_std

            train_feature, train_labels, val_feature, val_labels = build_context(
                train_feature,
                train_labels,
                val_feature,
                val_labels,
                fold_num,
                context,
                fold_idx,
                args.context_mode,
            )

            train_dataset = SimpleDataset(np.float32(train_feature), train_labels)
            val_dataset = SimpleDataset(np.float32(val_feature), val_labels)
            generator = torch.Generator()
            generator.manual_seed(seed * 1000 + fold_idx)
            sampler = None
            shuffle = True
            if args.sampler == "balanced":
                counts = np.bincount(train_labels, minlength=5)
                sample_weights = 1.0 / (counts[train_labels] + 1e-8)
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=generator,
                )
                shuffle = False

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=args.num_workers,
                generator=None if sampler is not None else generator,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

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

            loss_func = build_loss(
                args.loss,
                train_labels,
                device,
                args.weight_power,
                args.focal_gamma,
                args.label_smoothing,
            )
            optimizer = build_optimizer(optimizer_name, learn_rate, model, l2)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

            best_metric = -float("inf")
            best_epoch = -1
            best_result = None
            best_true = None
            best_pred = None
            best_state = None
            stale_epochs = 0

            print("-" * 100)
            print(f"Seed {seed}, fold {fold_idx}: train={len(train_dataset)}, val={len(val_dataset)}")

            for epoch in range(num_epochs):
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

                selected = selected_score(val_result, args.selection_metric)
                improved = selected > best_metric
                if improved:
                    best_metric = selected
                    best_epoch = epoch + 1
                    best_result = dict(val_result)
                    best_true = y_true.copy()
                    best_pred = y_pred.copy()
                    if not args.no_save_models:
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                print(
                    "Epoch {:03d} | tr_acc {:.4f} | val_acc {:.4f} | val_f1 {:.4f} | "
                    "val_kappa {:.4f} | best_{} {:.4f} @ {:03d}".format(
                        epoch + 1,
                        train_result["acc"],
                        val_result["acc"],
                        val_result["macro_f1"],
                        val_result["kappa"],
                        args.selection_metric,
                        best_metric,
                        best_epoch,
                    )
                )

                if stale_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            elapsed = time.time() - start_time
            row = {
                "seed": seed,
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "selection_metric": args.selection_metric,
                "elapsed_sec": round(elapsed, 3),
            }
            row.update(best_result)
            rows.append(row)
            seed_true.append(best_true)
            seed_pred.append(best_pred)

            fold_dir = save_root / f"seed_{seed}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                fold_dir / f"fold_{fold_idx}_predictions.npz",
                y_true=best_true,
                y_pred=best_pred,
            )
            if best_state is not None:
                torch.save(best_state, fold_dir / f"MDAGCN_seed_{seed}_fold_{fold_idx}.pth")

            print(
                "Best seed {}, fold {}: epoch {}, acc {:.4f}, macro_f1 {:.4f}, kappa {:.4f}, N1-F1 {:.4f}".format(
                    seed,
                    fold_idx,
                    best_epoch,
                    best_result["acc"],
                    best_result["macro_f1"],
                    best_result["kappa"],
                    best_result["f1_N1"],
                )
            )

        seed_true = np.concatenate(seed_true)
        seed_pred = np.concatenate(seed_pred)
        all_predictions[seed] = score_predictions(seed_true, seed_pred)
        np.savez(save_root / f"seed_{seed}_all_predictions.npz", y_true=seed_true, y_pred=seed_pred)

    csv_path = save_root / "fold_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": args.config,
        "feature_root": feature_root,
        "selection_metric": args.selection_metric,
        "context_mode": args.context_mode,
        "seeds": seeds,
        "folds": folds,
        "per_fold": {},
        "per_seed_aggregated_predictions": all_predictions,
    }

    for key in ["acc", "macro_f1", "kappa", "precision_macro", "recall_macro", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]:
        summary["per_fold"][key] = summarize([row[key] for row in rows])

    summary_path = save_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print(f"Saved fold results to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print("Per-fold summary:")
    for key, value in summary["per_fold"].items():
        print(f"{key}: {value['mean']:.4f} +/- {value['std']:.4f} (95% CI +/- {value['ci95']:.4f})")
    print("Per-seed aggregated predictions:")
    for seed, value in all_predictions.items():
        print(f"seed {seed}: acc {value['acc']:.4f}, macro_f1 {value['macro_f1']:.4f}, kappa {value['kappa']:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
