import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE
import seaborn as sns
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.Utils import ReadConfig


def load_and_plot_tsne():
    """从保存的npy文件中加载多模态消融实验结果并绘制t-SNE图"""

    # 读取配置获取路径
    config_path = "config/ISRUC_S3.config"
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default=config_path)
    args, _ = parser.parse_known_args()
    Path, _, _, _ = ReadConfig(args.c)

    # 加载保存的结果
    results_file = Path['save'] + 'multimodal_ablation_results.npy'

    if not os.path.exists(results_file):
        print(f"错误：结果文件 {results_file} 不存在")
        return

    # 加载结果
    all_results = np.load(results_file, allow_pickle=True).item()

    print("=" * 80)
    print("从保存文件加载多模态消融实验结果并绘制t-SNE图")
    print("=" * 80)

    # 绘制t-SNE图
    plot_multimodal_tsne(all_results, Path['save'])

    print("\nt-SNE图绘制完成！")


def plot_multimodal_tsne(results, save_path):
    """为多模态消融实验绘制t-SNE图"""

    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 定义颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '>']

    # 为每个多模态组合绘制t-SNE图
    for idx, (model_name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # 提取每个fold的准确率作为特征
        fold_accuracies = []
        for run in result['all_runs']:
            fold_accuracies.append(run['fold_accuracies'])

        # 转换为numpy数组
        fold_accuracies = np.array(fold_accuracies)

        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(fold_accuracies) - 1))
        tsne_results = tsne.fit_transform(fold_accuracies)

        # 绘制散点图
        for i in range(len(fold_accuracies)):
            ax.scatter(tsne_results[i, 0], tsne_results[i, 1],
                       c=colors[i], marker=markers[i], s=100,
                       label=f'Run {i + 1}', alpha=0.7)

        # 设置标题和标签
        short_name = model_name
        if len(short_name) > 30:
            short_name = short_name[:27] + "..."

        ax.set_title(f'{short_name}\nAcc: {result["mean_accuracy"]:.3f}±{result["std_accuracy"]:.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # 保存图像
    tsne_file = save_path + 'multimodal_ablation_tsne.png'
    plt.savefig(tsne_file, dpi=300, bbox_inches='tight')
    print(f"t-SNE图已保存到: {tsne_file}")

    # 显示图像
    plt.show()


def plot_combined_tsne(results, save_path):
    """绘制所有多模态组合的合并t-SNE图"""

    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 定义颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '>']

    # 收集所有数据
    all_features = []
    all_labels = []

    for model_idx, (model_name, result) in enumerate(results.items()):
        # 提取每个fold的准确率作为特征
        for run_idx, run in enumerate(result['all_runs']):
            features = run['fold_accuracies']
            all_features.append(features)
            all_labels.append(f"{model_name[:15]}...")

    # 转换为numpy数组
    all_features = np.array(all_features)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(all_features) - 1))
    tsne_results = tsne.fit_transform(all_features)

    # 绘制散点图
    unique_labels = list(set(all_labels))
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(all_labels) if l == label]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    c=[colors[i % len(colors)]], marker=markers[i % len(markers)],
                    s=100, label=label, alpha=0.7)

    plt.title('Combined t-SNE Visualization of Multimodal Ablation Experiments\n'
              '(Colored by Modality Combination)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    combined_file = save_path + 'multimodal_ablation_combined_tsne.png'
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"合并t-SNE图已保存到: {combined_file}")

    # 显示图像
    plt.show()


def plot_performance_comparison(results, save_path):
    """绘制性能对比图"""

    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 指标列表
    metrics = ['mean_accuracy', 'mean_f1_macro', 'mean_kappa', 'mean_precision_macro']
    metric_names = ['Accuracy', 'F1-Score', 'Kappa', 'Precision']
    metric_std = ['std_accuracy', 'std_f1_macro', 'std_kappa', 'std_precision_macro']

    # 提取模型名称和性能指标
    model_names = []
    metric_values = []
    std_values = []

    for model_name, result in results.items():
        short_name = model_name
        if len(short_name) > 20:
            short_name = short_name[:17] + "..."
        model_names.append(short_name)

        values = [result[metric] for metric in metrics]
        stds = [result[std] for std in metric_std]
        metric_values.append(values)
        std_values.append(stds)

    metric_values = np.array(metric_values)
    std_values = np.array(std_values)

    # 绘制每个指标的柱状图
    x = np.arange(len(model_names))
    width = 0.2

    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]

        # 绘制柱状图
        bars = ax.bar(x + i * width, metric_values[:, i], width,
                      yerr=std_values[:, i], capsize=5,
                      label=metric_name, alpha=0.8)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Modality Combination')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    performance_file = save_path + 'multimodal_ablation_performance.png'
    plt.savefig(performance_file, dpi=300, bbox_inches='tight')
    print(f"性能对比图已保存到: {performance_file}")

    # 显示图像
    plt.show()


if __name__ == '__main__':
    # 加载结果并绘制所有图表
    load_and_plot_tsne()

    # 重新加载结果用于其他图表
    config_path = "config/ISRUC_S3.config"
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default=config_path)
    args, _ = parser.parse_known_args()
    Path, _, _, _ = ReadConfig(args.c)

    results_file = Path['save'] + 'multimodal_ablation_results.npy'
    if os.path.exists(results_file):
        all_results = np.load(results_file, allow_pickle=True).item()

        # 绘制合并t-SNE图
        plot_combined_tsne(all_results, Path['save'])

        # 绘制性能对比图
        plot_performance_comparison(all_results, Path['save'])

        print("\n所有可视化图表生成完成！")
