def evaluate(model, loader, device, save_attention=False):
    model.eval()
    all_preds = []
    all_labels = []
       # 新增：保存原始信号和注意力数据的结构
    attention_records = []
    max_save_samples = 50
    saved_count = 0
    total_time = 0.0  # 添加初始化
    sample_count = 0  # 添加初始化
    
    # 获取所有交叉注意力模块
    attn_modules = []
    if save_attention:
        for module in model.modules():
            if isinstance(module, CrossWindowSparseAttention):
                module.eval_save_mode = True
                attn_modules.append(module)
    # 初始化混淆矩阵
    conf_matrix = torch.zeros(config["num_classes"], config["num_classes"], device=device)

    all_subjects = []
    with torch.no_grad():
        for x, y, subj in loader:  # 加载时包含subject_id
            x, y = x.to(device), y.to(device)
            # 添加推理时间测量
            start_time = time.time()
            logits,_ = model(x)
            torch.cuda.synchronize()  # 确保准确计时
            total_time += time.time() - start_time
            sample_count += x.size(0)
            
            preds = logits.argmax(dim=1)
            
            # 保存当前批次的原始信号和预测结果
            if save_attention and saved_count < max_save_samples:
                batch_data = {
                    'raw_signal': x.cpu().numpy(),
                    'predictions': preds.cpu().numpy(),
                    'true_labels': y.cpu().numpy()
                }
                attention_records.append(batch_data)
                saved_count += x.size(0)
            all_subjects.extend(subj.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            # 更新混淆矩阵
            for t, p in zip(y, preds):
                conf_matrix[t, p] += 1
    # 添加实时推理速度统计
    avg_infer_time = total_time / sample_count
    print(f"\n平均单样本推理时间: {avg_infer_time*1e3:.2f}ms")
    # 打印混淆矩阵（文本形式）
    conf_matrix_np = conf_matrix.int().cpu().numpy()
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    print("\n混淆矩阵（行：真实标签，列：预测标签）")
    header = " " * 12 + "".join([f"{name:>6}" for name in stage_names])
    print(header)
    for i, row in enumerate(conf_matrix_np):
        row_str = "".join([f"{num:6d}" for num in row])
        print(f"{stage_names[i]:<12}{row_str}")
    # 计算每个阶段的准确率、召回率和 F1 分数
    tp = torch.diag(conf_matrix)
    precision = tp / (conf_matrix.sum(dim=0) + 1e-6)
    recall = tp / (conf_matrix.sum(dim=1) + 1e-6)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-6)

    # 分类报告
    class_report = classification_report(
        all_labels, all_preds,
        target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
        output_dict=True,
        zero_division=0
    )
    
    # 合并保存的注意力数据
    if save_attention and attn_modules:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"attention_{timestamp}.pkl"
        
        # 从所有注意力模块收集数据
        all_attn = [m.attention_archive for m in attn_modules]
        
        # 截取前N个样本的注意力数据
        max_samples = min(max_save_samples, len(attention_records))
        final_data = {
            'attention': [attn[:max_samples] for attn in all_attn],
            'signals': [rec['raw_signal'] for rec in attention_records[:max_samples]],
            'predictions': [rec['predictions'] for rec in attention_records[:max_samples]],
            'true_labels': [rec['true_labels'] for rec in attention_records[:max_samples]],
            'metadata': {
                'save_time': timestamp,
                'batch_size': loader.batch_size,
                'sample_indices': np.random.choice(len(loader.dataset), max_samples)
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(final_data, f)
        
        # 重置存储
        for m in attn_modules:
            m.attention_archive = []
            m.eval_save_mode = False

    # 直接使用整体分类报告
    avg_stage_metrics = {}
    for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        avg_stage_metrics[stage] = {
            'precision': class_report[stage]['precision'],
            'recall': class_report[stage]['recall'],
            'f1': class_report[stage]['f1-score'],
            'support': class_report[stage]['support']
        }
    
    return {
        'overall': {
            'accuracy': class_report['accuracy'],
            'f1_weighted': class_report['weighted avg']['f1-score'],
            'kappa': cohen_kappa_score(all_labels, all_preds)
        },
        'stage_metrics': avg_stage_metrics,
        'all_preds': np.array(all_preds),
        'all_labels': np.array(all_labels),
        'subjects': np.array(all_subjects)  # 保存受试者信息用于后续分析
    
    }

def calculate_stage_metrics(results, subject):
    mask = np.array(results['subjects']) == subject
    subj_labels = np.array(results['all_labels'])[mask]
    subj_preds = np.array(results['all_preds'])[mask]
    
    class_report = classification_report(
        subj_labels, subj_preds,
        target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
        output_dict=True,
        zero_division=0
    )
    
    stage_metrics = {}
    for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        stage_metrics[stage] = {
            'precision': class_report[stage]['precision'],
            'recall': class_report[stage]['recall'],
            'f1': class_report[stage]['f1-score'],
            'support': class_report[stage]['support']
        }
    return stage_metrics

def save_results(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存整体指标
    pd.DataFrame([results['overall']]).to_csv(
        os.path.join(output_dir, 'overall_metrics.csv'),
        index=False
    )
    
    # 2. 保存阶段指标
    stage_data = []
    for stage_metrics in results['stage_metrics']:  # 遍历列表
        for stage, metrics in stage_metrics.items():  # 遍历每个字典的键值对
            stage_data.append({
                'stage': stage,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'support': metrics['support']
            })
    pd.DataFrame(stage_data).to_csv(
        os.path.join(output_dir, 'stage_metrics.csv'),
        index=False
    )
    
    # 3. 保存预测结果
    pd.DataFrame({
        'true_label': results['all_labels'],
        'pred_label': results['all_preds']
    }).to_csv(
        os.path.join(output_dir, 'predictions.csv'),
        index=False
    )
    
    print(f"结果已保存至 {output_dir} 目录")


def print_stage_metrics(stage_metrics):
    print("\n各睡眠阶段平均表现：")
    print(f"{'阶段':<6} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'样本数':<8}")
    for stage, metrics in stage_metrics.items():
        print(f"{stage:<6} {metrics['precision']:.4f}   {metrics['recall']:.4f}   {metrics['f1']:.4f}   {metrics['support']:>8}")
