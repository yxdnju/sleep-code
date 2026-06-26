import os
import numpy as np
import argparse
import shutil
import gc
import time
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import collections
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.Dataset import SimpleDataset
from model.MDAGCN import MDAGCN, TemporalAttention, SpatialAttention, Graph_Learn, cheb_conv_with_Att_GL, MDAGCN_block
from model.Utils import *

print(128 * '#')
print('Start proper ablation study for MDAGCN (training each variant from scratch).')


# Ablation model variants (重新设计，确保结构合理)
class MDAGCN_Ablation_a(nn.Module):
    """Ablation a: Removing Multi-Branch Layer (single branch only)"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_Ablation_a, self).__init__()

        # Simplified single branch (remove multi-branch complexity)
        self.temporal_Att = TemporalAttention(num_of_timesteps, num_of_vertices, num_of_features)
        self.spatial_Att = SpatialAttention(num_of_timesteps, num_of_vertices, num_of_features)
        self.graph_learn = Graph_Learn(num_of_features, GLalpha)

        # Single GCN path
        self.gcn_single = cheb_conv_with_Att_GL(num_of_features, num_of_chev_filters, k)

        # Temporal convolution
        self.cnn_temporal = nn.Conv2d(num_of_chev_filters, num_of_time_filters,
                                      kernel_size=(time_conv_kernel, 1), stride=(time_conv_strides, 1), padding='same')
        self.norm = nn.LayerNorm([num_of_time_filters], elementwise_affine=False)

        # Classification head (same as original)
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(num_of_time_filters * num_of_vertices,
                          num_of_time_filters * num_of_vertices, 2)
        self.dense_class = nn.Sequential(
            nn.Linear(num_of_time_filters * num_of_timesteps * num_of_vertices, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        N, T, V, F = x.shape

        # Attention
        TAtt = self.temporal_Att(x)
        x_TAtt = torch.bmm(x.permute([0, 2, 3, 1]).reshape(N, V * F, T), TAtt)
        x_TAtt = x_TAtt.reshape(N, V, F, T).permute([0, 3, 1, 2])
        SAtt = self.spatial_Att(x_TAtt)

        # Single GCN branch
        S, loss1, loss2 = self.graph_learn(x)
        x_gcn = self.gcn_single(x, SAtt, S)

        # Temporal convolution
        x_temp = self.cnn_temporal(x_gcn.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        x_temp = self.norm(torch.relu(x_temp))

        # GRU and classification
        x_flat = torch.flatten(x_temp, start_dim=2)
        gru_out, _ = self.gru(x_flat)
        gru_out = torch.flatten(gru_out, start_dim=1)

        if self.dropout != 0:
            gru_out = self.drop(gru_out)

        class_out = self.dense_class(gru_out)
        return class_out, loss1, loss2


class MDAGCN_Ablation_b(nn.Module):
    """Ablation b: Removing Multi-Scale Convolution (single scale only)"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_Ablation_b, self).__init__()

        # Use single-scale MDAGCN block (remove multi-scale)
        self.gcn = MDAGCN_block(num_of_timesteps, num_of_vertices, num_of_features,
                                k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                                time_conv_kernel, GLalpha, 1)

        # Classification head
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(num_of_time_filters * num_of_vertices,
                          num_of_time_filters * num_of_vertices, 2)
        self.dense_class = nn.Sequential(
            nn.Linear(num_of_time_filters * num_of_timesteps * num_of_vertices, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        gcn_out, loss1, loss2 = self.gcn(x)
        gcn_out = torch.flatten(gcn_out, start_dim=2)

        gru_out, _ = self.gru(gcn_out)
        gru_out = torch.flatten(gru_out, start_dim=1)

        if self.dropout != 0:
            gru_out = self.drop(gru_out)

        class_out = self.dense_class(gru_out)
        return class_out, loss1, loss2


class MDAGCN_Ablation_c(nn.Module):
    """Ablation c: Removing GCN Layer (CNN only)"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_Ablation_c, self).__init__()

        # Remove GCN components, use CNN only
        self.temporal_Att = TemporalAttention(num_of_timesteps, num_of_vertices, num_of_features)
        self.spatial_Att = SpatialAttention(num_of_timesteps, num_of_vertices, num_of_features)

        # CNN only (no GCN)
        self.cnn_spatial = nn.Conv2d(num_of_features, num_of_chev_filters,
                                     kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.cnn_temporal = nn.Conv2d(num_of_chev_filters, num_of_time_filters,
                                      kernel_size=(time_conv_kernel, 1), stride=(time_conv_strides, 1), padding='same')
        self.norm = nn.LayerNorm([num_of_time_filters], elementwise_affine=False)

        # Classification head
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(num_of_time_filters * num_of_vertices,
                          num_of_time_filters * num_of_vertices, 2)
        self.dense_class = nn.Sequential(
            nn.Linear(num_of_time_filters * num_of_timesteps * num_of_vertices, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        N, T, V, F = x.shape
        
        # Attention only (no GCN)
        TAtt = self.temporal_Att(x)
        x_TAtt = torch.bmm(x.permute([0, 2, 3, 1]).reshape(N, V * F, T), TAtt)
        x_TAtt = x_TAtt.reshape(N, V, F, T).permute([0, 3, 1, 2])
        SAtt = self.spatial_Att(x_TAtt)

        # Apply attention as feature weighting - fix dimension mismatch
        # SAtt shape: (N, V, V) -> we need to extract diagonal elements for feature weighting
        # Use the diagonal of SAtt as attention weights for each vertex
        SAtt_diag = torch.diagonal(SAtt, dim1=1, dim2=2)  # Shape: (N, V)
        SAtt_diag = SAtt_diag.unsqueeze(1).unsqueeze(-1)  # Shape: (N, 1, V, 1)
        x_weighted = x * SAtt_diag

        # CNN only
        x_cnn = self.cnn_spatial(x_weighted.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        x_temp = self.cnn_temporal(x_cnn.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        x_temp = self.norm(torch.relu(x_temp))

        # GRU and classification
        x_flat = torch.flatten(x_temp, start_dim=2)
        gru_out, _ = self.gru(x_flat)
        gru_out = torch.flatten(gru_out, start_dim=1)

        if self.dropout != 0:
            gru_out = self.drop(gru_out)

        class_out = self.dense_class(gru_out)
        return class_out, torch.tensor(0.0), torch.tensor(0.0)


class MDAGCN_Ablation_d(nn.Module):
    """Ablation d: Removing GRU Layers (no temporal modeling)"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_Ablation_d, self).__init__()

        # Keep GCN but remove GRU
        self.gcn = MDAGCN_block(num_of_timesteps, num_of_vertices, num_of_features,
                                k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                                time_conv_kernel, GLalpha, 1)

        # Direct classification (no GRU)
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout(dropout)

        self.dense_class = nn.Sequential(
            nn.Linear(num_of_time_filters * num_of_timesteps * num_of_vertices, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        gcn_out, loss1, loss2 = self.gcn(x)

        # Flatten and classify directly (no GRU)
        x_flat = torch.flatten(gcn_out, start_dim=1)

        if self.dropout != 0:
            x_flat = self.drop(x_flat)

        class_out = self.dense_class(x_flat)
        return class_out, loss1, loss2


def train_ablation_model(model, train_loader, val_loader, config, fold_idx, model_name):
    """Train ablation model from scratch with proper validation"""
    print(f"Training {model_name} for fold {fold_idx}")

    # Training parameters
    epochs = int(config["epoch"])
    learning_rate = float(config["learn_rate"])

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            # Handle model output
            output, loss1, loss2 = model(data)
            loss_main = loss_func(output, target)
            loss = loss_main + loss1 + loss2

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_acc, val_loss = evaluate_model(model, val_loader, loss_func)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping and model saving
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best validation accuracy for fold {fold_idx}: {best_acc:.2f}%")
    return best_model_state, best_acc


def evaluate_model(model, data_loader, loss_func):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            output, loss1, loss2 = model(data)

            loss = loss_func(output, target) + loss1 + loss2
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    accuracy = 100. * correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


def compute_comprehensive_metrics(true_labels, pred_labels, classes):
    """Compute comprehensive evaluation metrics with standard deviation"""
    metrics_dict = {}

    # Overall metrics
    metrics_dict['accuracy'] = accuracy_score(true_labels, pred_labels)
    metrics_dict['precision_macro'] = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    metrics_dict['recall_macro'] = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    metrics_dict['f1_macro'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    metrics_dict['kappa'] = cohen_kappa_score(true_labels, pred_labels)

    # Per-class metrics
    precision_per_class = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, pred_labels, average=None, zero_division=0)

    for i, class_name in enumerate(classes):
        metrics_dict[f'precision_{class_name}'] = precision_per_class[i]
        metrics_dict[f'recall_{class_name}'] = recall_per_class[i]
        metrics_dict[f'f1_{class_name}'] = f1_per_class[i]

    return metrics_dict


def run_proper_ablation_experiment(model_class, model_name, config_path, num_runs=3):
    """Run proper ablation experiment with training each variant from scratch"""
    print(f"\n{'=' * 80}")
    print(f"Running {model_name} ablation experiment (training from scratch)")
    print(f"{'=' * 80}")

    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default=config_path)
    parser.add_argument("-g", type=str, help="GPU number to use", default='0')
    args, _ = parser.parse_known_args()
    Path, _, cfgTrain, cfgModel = ReadConfig(args.c)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    # Parse parameters
    channels = int(cfgTrain["channels"])
    fold = int(cfgTrain["fold"])
    context = int(cfgTrain["context"])
    batch_size = int(cfgTrain["batch_size"])

    GLalpha = float(cfgModel["GLalpha"])
    num_of_chev_filters = int(cfgModel["cheb_filters"])
    num_of_time_filters = int(cfgModel["time_filters"])
    time_conv_strides = int(cfgModel["time_conv_strides"])
    time_conv_kernel = int(cfgModel["time_conv_kernel"])
    cheb_k = int(cfgModel["cheb_k"])
    dropout = float(cfgModel["dropout"])

    # Read data
    print("Loading data...")
    ReadList = np.load(Path['data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len']

    print("Data loaded successfully")

    # Store results for all runs
    all_results = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        # k-fold cross-validation with training
        fold_accuracies = []
        AllPred = None
        AllTrue = None

        for i in range(fold):
            print(f'Fold #{i}')

            # Load features
            Features = np.load(Path['save'] + 'Feature_' + str(i) + '.npz', allow_pickle=True)

            # Training data
            train_feature = np.float32(Features['train_feature'])
            train_targets = Features['train_targets']

            # Validation data
            val_feature = np.float32(Features['val_feature'])
            val_targets = Features['val_targets']

            # Process targets
            if train_targets.ndim == 1:
                train_labels = train_targets
            else:
                train_labels = np.argmax(train_targets, axis=1)

            if val_targets.ndim == 1:
                val_labels = val_targets
            else:
                val_labels = np.argmax(val_targets, axis=1)

            # Normalize features
            f_mean = np.mean(train_feature, axis=(0, 1), keepdims=True)
            f_std = np.std(train_feature, axis=(0, 1), keepdims=True) + 1e-8
            train_feature = (train_feature - f_mean) / f_std
            val_feature = (val_feature - f_mean) / f_std

            # Add context
            train_feature, train_labels = AddContext_SingleSub(train_feature, train_labels, context)
            val_feature, val_labels = AddContext_SingleSub(val_feature, val_labels, context)

            # Create datasets
            trainDataset = SimpleDataset(train_feature, train_labels)
            valDataset = SimpleDataset(val_feature, val_labels)

            train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(valDataset, batch_size=batch_size, shuffle=False, num_workers=0)

            # Create and train model from scratch
            model = model_class(context, channels, 128, cheb_k, num_of_chev_filters,
                                num_of_time_filters, time_conv_strides, time_conv_kernel,
                                GLalpha, dropout, num_classes=5)
            model = model.cuda()

            # Train the ablation model
            best_state, best_acc = train_ablation_model(model, train_loader, val_loader, cfgTrain, i, model_name)

            # Final evaluation with best model
            if best_state is not None:
                model.load_state_dict(best_state)

            # Get predictions for comprehensive metrics
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.cuda(), target.cuda()
                    output, _, _ = model(data)
                    pred = output.data.max(1, keepdim=True)[1]
                    all_preds.extend(pred.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy().flatten())

            fold_acc = accuracy_score(all_targets, all_preds) * 100
            fold_accuracies.append(fold_acc)
            print(f'Fold {i} Final Accuracy: {fold_acc:.2f}%')

            if AllPred is None:
                AllPred = all_preds
                AllTrue = all_targets
            else:
                AllPred = np.concatenate((AllPred, all_preds))
                AllTrue = np.concatenate((AllTrue, all_targets))

            del model
            gc.collect()

        # Compute comprehensive metrics for this run
        metrics_run = compute_comprehensive_metrics(AllTrue, AllPred, ['W', 'N1', 'N2', 'N3', 'REM'])
        metrics_run['fold_accuracies'] = fold_accuracies
        metrics_run['mean_accuracy'] = np.mean(fold_accuracies)
        metrics_run['std_accuracy'] = np.std(fold_accuracies)

        all_results.append(metrics_run)
        print(
            f"Run {run + 1} completed - Mean Acc: {metrics_run['mean_accuracy']:.2f}% ± {metrics_run['std_accuracy']:.2f}%")

    # Aggregate results across runs
    final_results = {
        'model_name': model_name,
        'num_runs': num_runs,
        'mean_accuracy': np.mean([r['mean_accuracy'] for r in all_results]),
        'std_accuracy': np.std([r['mean_accuracy'] for r in all_results]),
        'mean_precision_macro': np.mean([r['precision_macro'] for r in all_results]),
        'std_precision_macro': np.std([r['precision_macro'] for r in all_results]),
        'mean_recall_macro': np.mean([r['recall_macro'] for r in all_results]),
        'std_recall_macro': np.std([r['recall_macro'] for r in all_results]),
        'mean_f1_macro': np.mean([r['f1_macro'] for r in all_results]),
        'std_f1_macro': np.std([r['f1_macro'] for r in all_results]),
        'mean_kappa': np.mean([r['kappa'] for r in all_results]),
        'std_kappa': np.std([r['kappa'] for r in all_results]),
        'all_runs': all_results
    }

    return final_results


def main():
    """Main function to run all ablation experiments on ISRUC-S3 dataset"""
    config_path = "config/ISRUC_S3.config"  # Use correct filename with underscore

    # Read configuration first to get Path
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default=config_path)
    parser.add_argument("-g", type=str, help="GPU number to use", default='0')
    args, _ = parser.parse_known_args()

    # Read config to get Path
    Path, _, cfgTrain, cfgModel = ReadConfig(args.c)
    ablation_experiments = [
        (MDAGCN_Ablation_a, "Ablation a: Single Branch (No Multi-Branch)"),
        (MDAGCN_Ablation_b, "Ablation b: Single Scale (No Multi-Scale)"),
        (MDAGCN_Ablation_c, "Ablation c: CNN Only (No GCN)"),
        (MDAGCN_Ablation_d, "Ablation d: No GRU (No Temporal Modeling)"),
        (MDAGCN, "Complete Model (Our Proposed)")
    ]

    all_results = {}

    for model_class, model_name in ablation_experiments:
        print(f"\n{'=' * 100}")
        print(f"STARTING: {model_name}")
        print(f"{'=' * 100}")

        results = run_proper_ablation_experiment(model_class, model_name, config_path, num_runs=3)
        all_results[model_name] = results

        # Print detailed results
        print(f"\nRESULTS: {model_name}")
        print(f"Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
        print(f"Precision: {results['mean_precision_macro']:.4f} ± {results['std_precision_macro']:.4f}")
        print(f"Recall: {results['mean_recall_macro']:.4f} ± {results['std_recall_macro']:.4f}")
        print(f"F1-score: {results['mean_f1_macro']:.4f} ± {results['std_f1_macro']:.4f}")
        print(f"Kappa: {results['mean_kappa']:.4f} ± {results['std_kappa']:.4f}")

    # Save results
    output_file = Path['save'] + 'proper_ablation_results.npy'
    np.save(output_file, all_results)
    print(f"\nResults saved to: {output_file}")

    # Print final comparison
    print_final_comparison(all_results)

    print("\nProper ablation study completed successfully!")
    print(128 * '#')


def print_final_comparison(results):
    """Print final comparison table for all ablation experiments with per-class F1 scores"""
    print(f"\n{'='*150}")
    print("FINAL COMPARISON TABLE: Ablation Study Results on ISRUC-S3 Dataset")
    print("(Mean ± Standard Deviation over 3 runs)")
    print(f"{'='*150}")
    
    # Main table header (matching your LaTeX format)
    print("Method                          | Overall Results              | F1-score for Each Class")
    print("-" * 150)
    print("                                | Acc      F1       Kappa     | W        N1       N2       N3       REM")
    print("-" * 150)
    
    for model_name, result in results.items():
        # Extract short name for display
        short_name = model_name.split(':')[1].strip() if ':' in model_name else model_name
        if len(short_name) > 25:
            short_name = short_name[:22] + "..."
        else:
            short_name = short_name.ljust(25)
        
        # Calculate per-class F1 means and stds across runs
        classes = ['W', 'N1', 'N2', 'N3', 'REM']
        f1_per_class_means = []
        f1_per_class_stds = []
        
        for class_name in classes:
            f1_vals = [run[f'f1_{class_name}'] for run in result['all_runs']]
            f1_per_class_means.append(np.mean(f1_vals))
            f1_per_class_stds.append(np.std(f1_vals))
        
        # Print main results (matching your LaTeX format)
        print(f"{short_name} | "
              f"{result['mean_accuracy']:.3f}±{result['std_accuracy']:.3f}  "
              f"{result['mean_f1_macro']:.3f}±{result['std_f1_macro']:.3f}  "
              f"{result['mean_kappa']:.3f}±{result['std_kappa']:.3f} | "
              f"{f1_per_class_means[0]:.3f}±{f1_per_class_stds[0]:.3f}  "
              f"{f1_per_class_means[1]:.3f}±{f1_per_class_stds[1]:.3f}  "
              f"{f1_per_class_means[2]:.3f}±{f1_per_class_stds[2]:.3f}  "
              f"{f1_per_class_means[3]:.3f}±{f1_per_class_stds[3]:.3f}  "
              f"{f1_per_class_means[4]:.3f}±{f1_per_class_stds[4]:.3f}")
    
    print("-" * 150)


if __name__ == '__main__':
    main()
