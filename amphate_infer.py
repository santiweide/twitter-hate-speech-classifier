import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorWithPadding,
    pipeline,
    set_seed,
)
from datasets import Dataset
import kagglehub
from tqdm import tqdm

import matplotlib.pyplot as plt

# 假设 'amphate_model' 脚本在同一目录下
from amphate_model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

# ---
# 辅助函数区域
# ---

def expected_calibration_error(probs, labels, n_bins=10):
    """
    计算预期校准误差 (ECE) 和绘图所需的数据。
    
    参数:
    probs (np.array): 模型的预测概率 (N_samples, N_classes)
    labels (np.array): 真实标签 (N_samples,)
    n_bins (int): 分箱数量
    
    返回:
    ece (float): ECE 分数
    tuple: (bins, bin_accuracies, bin_confidences, bin_sizes)
    """
    # np.linspace 创建 n_bins+1 个边界，从而定义 n_bins 个区间
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    # 获取每个样本的最高置信度（概率）及其对应的预测类别
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    
    # 检查预测是否正确
    accuracies = (preds == labels).astype(float)
    
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    
    # 遍历每个分箱
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        
        # 找出置信度落在此区间的所有样本
        # 注意：使用 (lo, hi] 左开右闭区间（除了第一个箱）
        if i == 0:
             mask = (confidences >= lo) & (confidences <= hi)
        else:
             mask = (confidences > lo) & (confidences <= hi)

        if mask.any():
            # 计算这个箱的平均准确率
            bin_acc = accuracies[mask].mean()
            # 计算这个箱的平均置信度
            bin_conf = confidences[mask].mean()
            # 计算这个箱中的样本数占总样本数的比例
            bin_w = mask.mean() # 等价于 mask.sum() / len(probs)
            
            # ECE 是 |Acc - Conf| 的加权平均值
            ece += np.abs(bin_acc - bin_conf) * bin_w
            
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_sizes.append(mask.sum())
        else:
            # 如果箱子为空，则添加nan，以便绘图时跳过
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_sizes.append(0)
            
    return ece, (bins, np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes))


#
# 切片分析 (Slice Analysis) 辅助函数
#
def has_target_entity(ner_list):
    """检查NER结果列表是否包含任何目标实体"""
    # 检查是否可迭代。这可以正确处理列表、数组、None 和 np.nan
    if not hasattr(ner_list, '__iter__'):
        return False
    # 使用从 amphate_model 导入的 TARGET_NER_LABELS
    # 添加 isinstance(e, dict) 以安全处理可迭代对象中可能的非字典项
    return any(isinstance(e, dict) and e.get("entity_group") in TARGET_NER_LABELS for e in ner_list)

def slice_metrics(df, slice_col):
    """
    计算数据切片 (slice) 的性能指标。
    
    参数:
    df (pd.DataFrame): 包含 'correct', 'true_label', 'predicted_label' 的分析数据
    slice_col (str): 用于分组的列名 (例如 'has_target_entity')
    
    返回:
    pd.DataFrame: 包含 'n', 'acc', 'FPR', 'FNR' 的指标
    """
    
    def get_metrics(sub_df):
        if len(sub_df) == 0:
            return pd.Series({
                "n": 0, "acc": np.nan, "FPR": np.nan, "FNR": np.nan
            })
        
        n = len(sub_df)
        acc = sub_df["correct"].mean()
        
        # FPR (False Positive Rate) - 在此切片中，(真=0, 预=1) 的比例
        fpr_slice = ((sub_df["true_label"] == 0) & (sub_df["predicted_label"] == 1)).sum() / n
        
        # FNR (False Negative Rate) - 在此切片中，(真=1, 预=0) 的比例
        fnr_slice = ((sub_df["true_label"] == 1) & (sub_df["predicted_label"] == 0)).sum() / n
        
        return pd.Series({
            "n": n, "acc": acc, "FPR": fpr_slice, "FNR": fnr_slice
        })

    # 修复 DeprecationWarning
    # 按切片列分组并应用指标函数
    grouped = df.groupby(slice_col).apply(get_metrics, include_groups=False)
    
    return grouped.reset_index()


def run_inference_and_analysis():

    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

    # 创建一个目录来存放所有的分析结果
    analysis_dir = Path("error_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis artifacts will be saved to: {analysis_dir.resolve()}")

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    print(f"Using Tokenizer: {BASE_MODEL_NAME}")

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AmpleHateModel.from_pretrained(CHECKPOINT_PATH)
    data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded onto {device} and set to eval() mode.")
    
    print("Loading and preparing test data...")
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    ner_pipeline = pipeline(
        "ner",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple"
    )

    path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")
    # 尝试两个常见的文件名
    data_file_path = Path(path) / "train_E6oV3lV.csv"
    if not data_file_path.exists():
        data_file_path = Path(path) / "train_E6oV3LgV.csv"
        if not data_file_path.exists():
            print(f"Error: Could not find 'train_E6oV3lV.csv' or 'train_E6oV3LgV.csv' in {path}")
            return
            
    df = pd.read_csv(data_file_path)

    df = df[['label', 'tweet']]
    df = df.rename(columns={"tweet": "text"})
    df = df.dropna(subset=['text'])
    
    dataset = Dataset.from_pandas(df)
    dataset_split = dataset.shuffle(seed=TRAINING_SEED).train_test_split(test_size=0.1)
    test_dataset_raw = dataset_split["test"]

    print("Pre-computing NER results for TEST set...")
    test_texts_list = list(test_dataset_raw["text"])
    test_ner_results = ner_pipeline(test_texts_list, batch_size=64)
    test_dataset_raw = test_dataset_raw.add_column("ner_results", test_ner_results)

    print("Tokenizing TEST set...")
    preprocess_fn = create_preprocessing_function(base_tokenizer)    
    tokenized_test_dataset = test_dataset_raw.map(preprocess_fn, batched=True)
    
    true_labels = tokenized_test_dataset["labels"] 
    # 转换为 NumPy 数组，以便 ECE 函数使用
    y_true = np.array(true_labels) 
    
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text", "label", "ner_results"])
    tokenized_test_dataset.set_format("torch")
    
    print("\nStarting manual forward pass on test set...")
    
    test_dataloader = DataLoader(
        tokenized_test_dataset,
        batch_size=16,
        collate_fn=data_collator
    )
    
    all_logits = []

    with torch.no_grad(): # Disable gradient calculation
        for batch in tqdm(test_dataloader, desc="Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            
            logits = outputs.logits.cpu()
            all_logits.append(logits)

    final_logits = torch.cat(all_logits, dim=0)
    print("Inference complete.")

    # --- 4. Error Analysis ---
    print("\n--- Error Analysis Results ---")

    # --- 4.1. 基础 DataFrame 设置 ---
    # 1. 计算概率 (Probabilities) 和置信度 (Confidences)
    probabilities = torch.softmax(final_logits, dim=1)
    max_probs_tensor, pred_labels_tensor = torch.max(probabilities, axis=1)
    
    # 转换为 NumPy 数组
    pred_labels = pred_labels_tensor.numpy()
    max_probs = max_probs_tensor.numpy()
    y_prob = probabilities.numpy() # (N_samples, N_classes) 

    original_texts = test_dataset_raw["text"]
    
    # 2. 将置信度添加到分析 DataFrame 中
    df_analysis = pd.DataFrame({
        "text": original_texts,
        "true_label": y_true,       # 使用 np 数组
        "predicted_label": pred_labels,
        "confidence": max_probs   # 添加置信度列
    })
    
    # 保存完整的测试集预测结果，以便将来分析
    full_analysis_file = analysis_dir / "full_test_predictions.csv"
    df_analysis.to_csv(full_analysis_file, index=False)
    print(f"\nFull prediction results (with confidence) saved to: {full_analysis_file}")
    
    # 筛选出错误样本
    df_errors = df_analysis[df_analysis["predicted_label"] != df_analysis["true_label"]].copy()

    # 按置信度降序排列错误 —— 优先查看模型“非常自信但错了”的样本
    df_errors_sorted = df_errors.sort_values(by="confidence", ascending=False)

    total_samples = len(df_analysis)
    total_errors = len(df_errors)
    accuracy = 1 - (total_errors / total_samples)
    
    print(f"\nTotal test samples: {total_samples}")
    print(f"Prediction errors: {total_errors}")
    print(f"Test Set Accuracy: {accuracy * 100 :.2f}%")

    # --- 4.2. 校准分析 (Calibration Analysis) ---
    print("\n--- Calibration Analysis (Reliability) ---")
    n_bins = 15
    ece, (bins, bacc, bconf, bsize) = expected_calibration_error(y_prob, y_true, n_bins=n_bins)
    print(f"ECE (Expected Calibration Error) @ {n_bins} bins: {ece:.4f} (lower is better)")

    # 开始绘图
    fig = plt.figure(figsize=(8, 6)) # 设置图像大小
    plt.plot([0,1],[0,1], linestyle="--", color="gray", label="Perfect Calibration")
    centers = (bins[:-1] + bins[1:]) / 2
    mask = ~np.isnan(bacc)
    
    if mask.any():
        plt.bar(centers[mask], bacc[mask], width=(bins[1]-bins[0]), alpha=0.6, 
                edgecolor="black", label="Bin Accuracy")
        plt.plot(centers[mask], bconf[mask], marker="o", linestyle="-", 
                 color="red", label="Bin Avg. Confidence")
    
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.xlabel("Confidence (Predicted Probability)")
    plt.ylabel("Accuracy (Fraction of Positives)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout() 
    
    plot_file = analysis_dir / "reliability_diagram.png"
    fig.savefig(plot_file, dpi=180)
    print(f"Reliability diagram saved to: {plot_file}")
    
    
    # --- 4.3. 切片分析 (Slice-based Analysis) ---
    print("\n--- Slice-based Analysis (NER) ---")
    
    # 我们需要原始的 test_dataset_raw 来访问 'ner_results'
    # 将其转换为 pandas DF
    test_df = test_dataset_raw.to_pandas()
    
    # 为 df_analysis 添加 'correct' 列，以便 slice_metrics 函数使用
    df_analysis["correct"] = (df_analysis["true_label"] == df_analysis["predicted_label"])
    
    # --- 4.3a. 'has_target_entity' 切片 ---
    
    # 从 test_df (包含 ner_results) 重建 'has_target_entity' 标志
    # df_analysis 和 test_df 的顺序是一致的
    df_analysis["has_target_entity"] = test_df["ner_results"].map(has_target_entity).astype(int)
    
    # 调用 slice_metrics 辅助函数
    target_slice_metrics = slice_metrics(df_analysis, "has_target_entity")
    target_slice_file = analysis_dir / "slice_has_target_entity.csv"
    target_slice_metrics.to_csv(target_slice_file, index=False)
    
    print("Metrics for 'has_target_entity' slice (1 = True):")
    print(target_slice_metrics.to_string(index=False))
    print(f"Saved to: {target_slice_file}")

    # --- 4.3b. 'per_entity_type' 切片 ---
    
    rows = []
    # 使用从 amphate_model 导入的 TARGET_NER_LABELS
    for ent in sorted(TARGET_NER_LABELS):
        mask = test_df["ner_results"].map(
            lambda lst: any(isinstance(e, dict) and e.get("entity_group") == ent 
                            for e in (lst if hasattr(lst, '__iter__') else []))
        )
        
        # 将掩码应用于 df_analysis
        sub = df_analysis[mask]
        
        if len(sub) > 0:
            # 计算指标
            rows.append({
                "entity_type": ent,
                "n": len(sub),
                "acc": sub["correct"].mean(),
                # 使用 df_analysis 的列名
                "FPR": ((sub["true_label"] == 0) & (sub["predicted_label"] == 1)).sum() / len(sub),
                "FNR": ((sub["true_label"] == 1) & (sub["predicted_label"] == 0)).sum() / len(sub),
            })
    
    if rows:
        per_ent_metrics = pd.DataFrame(rows).sort_values("acc")
        per_ent_file = analysis_dir / "slice_per_entity.csv"
        per_ent_metrics.to_csv(per_ent_file, index=False)
        print(f"\nSaved per-entity slice metrics to: {per_ent_file}")
        print("Per-entity slice metrics:")
        print(per_ent_metrics.to_string(index=False))
    else:
        print("\nNo samples found for any target entity types.")
    
    # --- 4.4. 错误样本定性分析 (Qualitative Analysis) ---
    if total_errors > 0:
        print("\n--- Most Confident Error Samples ---")
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', 100)
        
        # 打印按置信度排序的错误
        print(df_errors_sorted.head(20).to_string(index=False)) 
        
        # 保存排序后的错误文件
        error_file = analysis_dir / "error_analysis_sorted.csv"
        df_errors_sorted.to_csv(error_file, index=False)
        print(f"\nError samples (sorted by confidence) saved to: {error_file}")
    else:
        print("\n--- No errors found! ---")


if __name__ == "__main__":
    run_inference_and_analysis()