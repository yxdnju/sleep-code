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
from model.MDAGCN import MDAGCN
from model.Utils import *

print(128 * '#')
print('Start multimodal data ablation study for MDAGCN.')


# 多模态数据消融模型变体
class MDAGCN_EEG_Only(nn.Module):
    """EEG Only: 仅使用EEG数据"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_EEG_Only, self).__init__()

        # 使用原始MDAGCN结构，但输入特征维度调整为仅EEG
        # EEG特征占前6个通道
        self.eeg_channels = 6  # EEG通道数
        self.gcn = MDAGCN(num_of_timesteps, num_of_vertices, self.eeg_channels,
                          k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                          time_conv_kernel, GLalpha, dropout, num_classes)

    def forward(self, x):
        # 仅使用EEG通道（前6个通道）
        x_eeg = x[:, :, :, :self.eeg_channels]
        return self.gcn(x_eeg)


class MDAGCN_EEG_EOG(nn.Module):
    """EEG + EOG: 使用EEG和EOG数据"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_EEG_EOG, self).__init__()

        # EEG + EOG通道
        self.eeg_eog_channels = 8  # EEG(6) + EOG(2)
        self.gcn = MDAGCN(num_of_timesteps, num_of_vertices, self.eeg_eog_channels,
                          k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                          time_conv_kernel, GLalpha, dropout, num_classes)

    def forward(self, x):
        # 使用EEG和EOG通道（前8个通道）
        x_eeg_eog = x[:, :, :, :self.eeg_eog_channels]
        return self.gcn(x_eeg_eog)


class MDAGCN_EEG_EMG(nn.Module):
    """EEG + EMG: 使用EEG和EMG数据"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_EEG_EMG, self).__init__()

        # EEG + 特定EMG通道（需要根据实际数据调整）
        self.eeg_emg_indices = [0, 1, 2, 3, 4, 5, 8]  # EEG(0-5) + EMG(8)
        self.gcn = MDAGCN(num_of_timesteps, num_of_vertices, len(self.eeg_emg_indices),
                          k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                          time_conv_kernel, GLalpha, dropout, num_classes)

    def forward(self, x):
        # 使用EEG和EMG通道
        x_eeg_emg = x[:, :, :, self.eeg_emg_indices]
        return self.gcn(x_eeg_emg)


class MDAGCN_EEG_ECG(nn.Module):
    """EEG + ECG: 使用EEG和ECG数据"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_EEG_ECG, self).__init__()

        # EEG + ECG通道（需要根据实际数据调整）
        self.eeg_ecg_indices = [0, 1, 2, 3, 4, 5, 9]  # EEG(0-5) + ECG(9)
        self.gcn = MDAGCN(num_of_timesteps, num_of_vertices, len(self.eeg_ecg_indices),
                          k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                          time_conv_kernel, GLalpha, dropout, num_classes)

    def forward(self, x):
        # 使用EEG和ECG通道
        x_eeg_ecg = x[:, :, :, self.eeg_ecg_indices]
        return self.gcn(x_eeg_ecg)


class MDAGCN_All_Modalities(nn.Module):
    """EEG + EOG + EMG + ECG: 使用所有模态数据"""

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features,
                 k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 time_conv_kernel, GLalpha, dropout, num_classes=5):
        super(MDAGCN_All_Modalities, self).__init__()

        # 使用所有10个通道
        self.gcn = MDAGCN(num_of_timesteps, num_of_vertices, num_of_features,
                          k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                          time_conv_kernel, GLalpha, dropout, num_classes)

    def forward(self, x):
        # 使用所有通道
        return self.gcn(x)


def train_multimodal_model(model, train_loader, val_loader, config, fold_idx, model_name):
    """训练多模态消融模型"""
    print(f"Training {model_name} for fold {fold_idx}")

    epochs = int(config["epoch"])
    learning_rate = float(config["learn_rate"])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output, loss1, loss2 = model(data)
            loss_main = loss_func(output, target)
            loss = loss_main + loss1 + loss2

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_acc, val_loss = evaluate_multimodal_model(model, val_loader, loss_func)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # 早停和模型保存
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


def evaluate_multimodal_model(model, data_loader, loss_func):
    """评估多模态模型性能"""
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


def compute_multimodal_metrics(true_labels, pred_labels, classes):
    """计算多模态消融实验的评估指标"""
    metrics_dict = {}

    # 整体指标
    metrics_dict['accuracy'] = accuracy_score(true_labels, pred_labels)
    metrics_dict['precision_macro'] = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    metrics_dict['recall_macro'] = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    metrics_dict['f1_macro'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    metrics_dict['kappa'] = cohen_kappa_score(true_labels, pred_labels)

    # 每个类别的指标
    precision_per_class = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, pred_labels, average=None, zero_division=0)

    for i, class_name in enumerate(classes):
        metrics_dict[f'precision_{class_name}'] = precision_per_class[i]
        metrics_dict[f'recall_{class_name}'] = recall_per_class[i]
        metrics_dict[f'f1_{class_name}'] = f1_per_class[i]

    return metrics_dict


def run_multimodal_ablation_experiment(model_class, model_name, config_path, num_runs=3):
    """运行多模态数据消融实验"""
    print(f"\n{'=' * 80}")
    print(f"Running {model_name} multimodal ablation experiment")
    print(f"{'=' * 80}")

    # 获取配置
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default=config_path)
    parser.add_argument("-g", type=str, help="GPU number to use", default='0')
    args, _ = parser.parse_known_args()
    Path, _, cfgTrain, cfgModel = ReadConfig(args.c)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    # 解析参数
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

    # 读取数据
    print("Loading data...")
    ReadList = np.load(Path['data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len']

    print("Data loaded successfully")

    # 存储所有运行结果
    all_results = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        # k折交叉验证
        fold_accuracies = []
        AllPred = None
        AllTrue = None

        for i in range(fold):
            print(f'Fold #{i}')

            # 加载特征
            Features = np.load(Path['save'] + 'Feature_' + str(i) + '.npz', allow_pickle=True)

            # 训练数据
            train_feature = np.float32(Features['train_feature'])
            train_targets = Features['train_targets']

            # 验证数据
            val_feature = np.float32(Features['val_feature'])
            val_targets = Features['val_targets']

            # 处理标签
            if train_targets.ndim == 1:
                train_labels = train_targets
            else:
                train_labels = np.argmax(train_targets, axis=1)

            if val_targets.ndim == 1:
                val_labels = val_targets
            else:
                val_labels = np.argmax(val_targets, axis=1)

            # 归一化特征
            f_mean = np.mean(train_feature, axis=(0, 1), keepdims=True)
            f_std = np.std(train_feature, axis=(0, 1), keepdims=True) + 1e-8
            train_feature = (train_feature - f_mean) / f_std
            val_feature = (val_feature - f_mean) / f_std

            # 添加上下文
            train_feature, train_labels = AddContext_SingleSub(train_feature, train_labels, context)
            val_feature, val_labels = AddContext_SingleSub(val_feature, val_labels, context)

            # 创建数据集
            trainDataset = SimpleDataset(train_feature, train_labels)
            valDataset = SimpleDataset(val_feature, val_labels)

            train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(valDataset, batch_size=batch_size, shuffle=False, num_workers=0)

            # 创建并训练模型
            model = model_class(context, channels, 128, cheb_k, num_of_chev_filters,
                                num_of_time_filters, time_conv_strides, time_conv_kernel,
                                GLalpha, dropout, num_classes=5)
            model = model.cuda()

            # 训练多模态模型
            best_state, best_acc = train_multimodal_model(model, train_loader, val_loader, cfgTrain, i, model_name)

            # 使用最佳模型进行最终评估
            if best_state is not None:
                model.load_state_dict(best_state)

            # 获取预测结果
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

        # 计算该运行的指标
        metrics_run = compute_multimodal_metrics(AllTrue, AllPred, ['W', 'N1', 'N2', 'N3', 'REM'])
        metrics_run['fold_accuracies'] = fold_accuracies
        metrics_run['mean_accuracy'] = np.mean(fold_accuracies)
        metrics_run['std_accuracy'] = np.std(fold_accuracies)

        all_results.append(metrics_run)
        print(
            f"Run {run + 1} completed - Mean Acc: {metrics_run['mean_accuracy']:.2f}% ± {metrics_run['std_accuracy']:.2f}%")

    # 聚合所有运行结果
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
    """多模态数据消融实验主函数"""
    config_path = "config/ISRUC_S3.config"

    # Read configuration first to get Path
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default=config_path)
    parser.add_argument("-g", type=str, help="GPU number to use", default='0')
    args, _ = parser.parse_known_args()

    # Read config to get Path
    Path, _, cfgTrain, cfgModel = ReadConfig(args.c)

    multimodal_experiments = [
        (MDAGCN_EEG_Only, "EEG only"),
        (MDAGCN_EEG_EOG, "EEG + EOG"),
        (MDAGCN_EEG_EMG, "EEG + EMG"),
        (MDAGCN_EEG_ECG, "EEG + ECG"),
        (MDAGCN_All_Modalities, "EEG + EOG + EMG + ECG (ALL)")
    ]

    all_results = {}

    for model_class, model_name in multimodal_experiments:
        print(f"\n{'=' * 100}")
        print(f"STARTING: {model_name}")
        print(f"{'=' * 100}")

        results = run_multimodal_ablation_experiment(model_class, model_name, config_path, num_runs=3)
        all_results[model_name] = results

        # 打印详细结果
        print(f"\nRESULTS: {model_name}")
        print(f"Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        print(f"F1-score: {results['mean_f1_macro']:.3f} ± {results['std_f1_macro']:.3f}")
        print(f"Kappa: {results['mean_kappa']:.3f} ± {results['std_kappa']:.3f}")

    # 保存结果
    output_file = Path['save'] + 'multimodal_ablation_results.npy'
    np.save(output_file, all_results)
    print(f"\nResults saved to: {output_file}")

    print("\nMultimodal ablation study completed successfully!")
    print(128 * '#')

    all_results = np.load(output_file, allow_pickle=True).item()
    print_final_comparison(all_results)

def print_final_comparison(results):
    """Print final comparison table for all ablation experiments with per-class F1 scores"""
    print(f"\n{'=' * 150}")
    print("FINAL COMPARISON TABLE: Ablation Study Results on ISRUC-S3 Dataset")
    print("(Mean ± Standard Deviation over 3 runs)")
    print(f"{'=' * 150}")

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
