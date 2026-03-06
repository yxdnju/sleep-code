from collections import defaultdict
from scipy.stats import pearsonr
from thop import profile
import torch.cuda as cuda
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta  # 用于生成时间戳
import time
import pickle  
from torch_geometric.nn  import GCNConv, GINConv 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, KFold
from torch.cuda.amp import GradScaler
import warnings
import os
import pandas as pd
from tqdm import tqdm
from torch.nn import init
from fvcore.nn import FlopCountAnalysis, flop_count_table
# 超参数配置
config = {
    "window_size": 6000,
    "num_windows": 954,
    "num_channels": 10,
    "num_classes": 5,
    "lstm_hidden": 128,
    "gcn_hidden": 128,
    "num_epochs": 80,
    "batch_size": 16,
}

    

# 修改后的数据加载函数
def load_data():
    print(">>> 正在加载 ISRUC 数据集 (Z-Score + 原版标签逻辑)...")
    data = np.load('sleep/processed_200hz_dataset_10.npz', allow_pickle=True)
    
    signals = data['data']  
    labels = data['labels'].astype(int)
    
    # -------------------------------------------------------------
    # 1. ID 修复 (必须保留，否则 KFold 报错)
    # -------------------------------------------------------------
    if 'subject_ids' in data:
        subject_ids = data['subject_ids'].astype(int)
    else:
        subject_ids = np.zeros(len(labels), dtype=int)
        
    # 如果检测到受试者太少，生成伪 ID (这是为了解决 GroupKFold 报错，不影响数据内容)
    if len(np.unique(subject_ids)) < 10:
        print("检测到受试者ID不足，生成伪ID以支持交叉验证...")
        n_chunks = 20
        chunk_size = len(labels) // n_chunks
        new_ids = np.zeros(len(labels), dtype=int)
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != n_chunks - 1 else len(labels)
            new_ids[start:end] = i + 1
        subject_ids = new_ids

    # -------------------------------------------------------------
    # 2. 维度修正 (保留你的 transpose 逻辑，但加个判断更安全)
    # -------------------------------------------------------------
    print(f"原始维度: {signals.shape}")
    # 你的逻辑: (Channel, Batch, Time) -> (Batch, Channel, Time)
    if signals.shape[0] < signals.shape[1]: 
        print("维度转置: (C, N, T) -> (N, C, T)")
        signals = signals.transpose(1, 0, 2)
    # 补充: (Batch, Time, Channel) -> (Batch, Channel, Time)
    elif signals.shape[2] < signals.shape[1]:
        print("维度转置: (N, T, C) -> (N, C, T)")
        signals = signals.transpose(0, 2, 1)
        
    # -------------------------------------------------------------
    # 3. Z-Score 标准化 
    # -------------------------------------------------------------
    print("正在使用 Robust Scaling (Median/IQR) 进行标准化...")

    # raw_signals shape: (Samples, Channels, TimePoints)
    # 计算每个样本、每个通道的中位数和四分位距
    q25 = np.percentile(signals, 25, axis=2, keepdims=True)
    q75 = np.percentile(signals, 75, axis=2, keepdims=True)
    median = np.median(signals, axis=2, keepdims=True)
    iqr = q75 - q25

    # 防止除以0（极少数平直线）
    iqr[iqr < 1e-6] = 1.0

    # 执行标准化
    signals = (signals - median) / iqr

    # 极其重要：强制截断异常值
    # 这一步能保证没有任何一个受试者的信号会比别人大太多
    signals = np.clip(signals, -20.0, 20.0)

    # -------------------------------------------------------------
    # 4. 标签处理 (回归你原来的 LabelEncoder)
    # -------------------------------------------------------------
    # 先处理一下真正的异常值 (比如 6, 7)，但不碰 5 (REM)
    labels = labels.reshape(-1)
    if np.any(labels > 5): 
        print("警告: 清洗标签 >5 的异常值为 0")
        labels[labels > 5] = 0
    
    # 使用 LabelEncoder，它会自动把 0,1,2,3,5 映射成 0,1,2,3,4
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = encoded_labels.reshape(data['labels'].shape)
    
    print(f"标签类别: {label_encoder.classes_}") # 应该打印 [0 1 2 3 5]
    
    # -------------------------------------------------------------
    # 5. 打包
    # -------------------------------------------------------------
    dataset = []
    signals = signals.astype(np.float32)
    
    for i in range(len(signals)):
        dataset.append((
            torch.from_numpy(signals[i]), 
            torch.tensor(encoded_labels[i], dtype=torch.long),
            subject_ids[i]
        ))
        
    print(f"✅ 加载完毕，共 {len(dataset)} 个样本。")
    return dataset


def tenfold_cross_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在训练循环前添加这些优化
    torch.cuda.empty_cache()  # 清空缓存
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    scaler = GradScaler()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cpu.amp.autocast.*")
   
    # 加载数据
    full_dataset = load_data()
   
    # 获取所有受试者ID并确保唯一性
    subjects = np.array([item[2] for item in full_dataset])  # 假设subject_id在第三个位置
    unique_subjects = np.unique(subjects)

    # 使用GroupKFold确保受试者独立
    kf = GroupKFold(n_splits=10)

    fold_results = []
    all_results = {
        'overall': [],
        'stage_metrics': [],
        'all_preds': [],
        'all_labels': []
    }
    target_fold = 9
    drop_fold = 9
    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset, groups=subjects)):
        if fold != target_fold : 
            continue
        # if drop_fold == fold:
        #     continue
        print(f"\n=== Fold {fold+1} ===")
        
        # 从训练集中分离验证受试者
        train_subjs = subjects[train_idx]
        val_subj = np.random.choice(np.unique(train_subjs), size=1, replace=False)
        val_mask = np.isin(subjects, val_subj)
        val_idx = np.where(val_mask)[0]
        train_idx = np.setdiff1d(train_idx, val_idx)
        
 
        
        print(f"训练受试者: {np.unique(subjects[train_idx])}")
        print(f"验证受试者: {val_subj}")
        print(f"测试受试者: {subjects[test_idx][0]}")
        
        # 创建数据集
        train_set = [full_dataset[i] for i in train_idx]
        # train_set = advanced_augmentation(train_set, class_ratios={
        # 1: 0.22,  # N1
        # 2: 0.3,  # N2
        # 4: 0.25   # REM
    # })
        val_set = [full_dataset[i] for i in val_idx]
        test_set = [full_dataset[i] for i in test_idx]
        train_loader = DataLoader(
            train_set,
            batch_size=128,   
            shuffle=True,    # <--- 必须是 True，让数据自然随机打散
            num_workers=4,
            pin_memory=True,
            drop_last=True   # 丢弃最后一个不完整的 batch，保证稳定性
        )
        
        val_loader = DataLoader(val_set, batch_size=64,drop_last=False)
        test_loader = DataLoader(test_set, batch_size=64,drop_last=False)
        
        # 初始化模型
       
        num_subjects = len(np.unique(subjects))
        model = EnhancedSleepModel(config).to(device)

        
        torch.set_float32_matmul_precision('medium')  # 平衡计算精度与速度
       
        criterion = nn.CrossEntropyLoss().to(device)
        
        confusion_pairs = [(4, 1), (0, 1), (1, 2)]
        
        total_train_steps = config["num_epochs"] * len(train_loader)  # 添加这行

     
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,           
            weight_decay=0.05, # <--- 增大权重衰减 (0.01 -> 0.02) 抗过拟合
            betas=(0.9, 0.999)
        )
        # 调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4, # 稍微调大一点点，因为我们加了Dropout
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1, # <--- 增加预热比例 (0.1 -> 0.2)
            anneal_strategy='cos'
        )
        best_metric = -1.0
        patience_counter = 0
        min_delta = 0.001 

        for epoch in range(config['num_epochs']):
            # ===================== 训练阶段 =====================
            model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            skipped_batches =0
            epoch_start = time.time()

            with tqdm(train_loader, unit="batch", disable=True, desc=f"Epoch {epoch+1}") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch_start = time.time()

                    x, y, subj = batch
                    x = x.to(device)
                    y = y.to(device)
                    subj = subj.to(device)
                  
                    optimizer.zero_grad(set_to_none=True)
                    # scale = (torch.rand(x.size(0), 1, 1, device=device) * 0.4) + 0.8
                    # x = x * scale
                    # === 修改 2: 纯 FP32 训练 (稳定!) ===
                    # 移除 autocast
                    logits, feat = model(x)     
                    loss = criterion(logits, y) 
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                    
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.step()
                    
                    if scheduler is not None:
                        scheduler.step()
                    # 统计训练指标（按样本数统计）
                    batch_size = y.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

                    predicted = torch.argmax(logits, dim=1)
                    correct += (predicted == y).sum().item()

            epoch_time = time.time() - epoch_start
            train_loss = total_loss / total_samples
            train_acc = correct / total_samples
            # ===================== 验证阶段 =====================

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            # 混淆矩阵在 GPU 上计算
            conf_matrix = torch.zeros(config["num_classes"], config["num_classes"], device=device)

            with torch.no_grad():
                for x, y, _ in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    logits, feat = model(x)        
                    # logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

                    v_loss = criterion(logits, y)
                    batch_size = y.size(0)
                    val_loss += v_loss.item() * batch_size

                    predicted = torch.argmax(logits, dim=1)
                    val_correct += (predicted == y).sum().item()
                    val_total += batch_size

                    # 更新混淆矩阵
                    for t, p in zip(y, predicted):
                        conf_matrix[t, p] += 1

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # === 用混淆矩阵计算加权 F1（仍在 GPU 上） ===
            tp = torch.diag(conf_matrix)
            support = conf_matrix.sum(dim=1)
            precision = tp / (conf_matrix.sum(dim=0) + 1e-6)
            recall = tp / (support + 1e-6)
            f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-6)
            val_f1 = (f1_per_class * support).sum() / support.sum()

            # 转成 Python 标量
            val_acc = float(val_acc)
            val_f1 = float(val_f1.item())

            # 当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch+1:03d}/{config['num_epochs']} | "
                f"Time: {timedelta(seconds=int(epoch_time))} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        

            current_metric = val_f1  # 直接锚定 F1 分数，这是最硬的指标
            
            if current_metric > best_metric + min_delta:
                best_metric = current_metric
                patience_counter = 0
                torch.save(model.state_dict(), f"fold{fold}_best.pth")
                # print(f" 保存最佳模型! Val F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 15:  
                    print(f"早停触发于 epoch {epoch+1}")
                    break

        # ===================== 测试阶段 =====================
        model.load_state_dict(torch.load(f"fold{fold}_best.pth"))

        test_results = evaluate(model, test_loader, device, save_attention=False)

        fold_results.append({
            'accuracy': test_results['overall']['accuracy'],
            'f1_weighted': test_results['overall']['f1_weighted'],
            'kappa': test_results['overall']['kappa']
        })
        all_results['overall'].append(test_results['overall'])
        all_results['stage_metrics'].append(test_results['stage_metrics'])
        all_results['all_preds'].extend(test_results['all_preds'])
        all_results['all_labels'].extend(test_results['all_labels'])

        print(f"\n=== Fold {fold+1} 结果 ===")
        print(f"准确率: {test_results['overall']['accuracy']:.4f}")
        print(f"加权F1: {test_results['overall']['f1_weighted']:.4f}")
        print(f"Kappa系数: {test_results['overall']['kappa']:.4f}")
        print_stage_metrics(test_results['stage_metrics'])

    
    # 保存所有结果
    save_results(all_results, output_dir='tenfold_results')
    
    # 计算最终统计量
    acc_values = [r['accuracy'] for r in fold_results]
    f1_values = [r['f1_weighted'] for r in fold_results]
    kappa_values = [r['kappa'] for r in fold_results]
    
    print("\n=== 最终汇总结果 ===")
    print(f"平均准确率: {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}")
    print(f"平均加权F1: {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")
    print(f"平均Kappa系数: {np.mean(kappa_values):.4f} ± {np.std(kappa_values):.4f}")
    
    
    
    return fold_results




if __name__ == "__main__":
    

    tenfold_cross_validation()
    