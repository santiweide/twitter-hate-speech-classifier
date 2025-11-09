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

# --- [NEW] --- 
# 添加 Matplotlib 和辅助函数所需的导入
import matplotlib.pyplot as plt

# 假设 'amphate_model' 脚本在同一目录下
from amphate_model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

# --- [NEW] --- 
# 
# ECE (Expected Calibration Error) 辅助函数
#
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
# --- [END NEW] ---


def run_inference_and_analysis():

    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

    # --- [NEW] --- 
    # 创建一个目录来存放所有的分析结果
    analysis_dir = Path("error_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis artifacts will be saved to: {analysis_dir.resolve()}")
    # --- [END NEW] ---

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
    data_file_path = Path(path) / "train_E6oV3LgV.csv"
    
    if not data_file_path.exists():
        # 修正了原始代码中的拼写错误 (V3lV -> V3LgV)
        data_file_path = Path(path) / "train_E6oV3lV.csv"
        if not data_file_path.exists():
            print(f"Error: Could not find 'train_E6oV3lV.csv' in {path}")
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
    # --- [NEW] --- 
    # 转换为 NumPy 数组，以便 ECE 函数使用
    y_true = np.array(true_labels) 
    # --- [END NEW] ---
    
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

    # --- [NEW] --- 
    # 1. 计算概率 (Probabilities) 和置信度 (Confidences)
    #    从 logits 转换为概率分布
    probabilities = torch.softmax(final_logits, dim=1)
    
    # 获取最高概率（置信度）和对应的预测标签
    max_probs_tensor, pred_labels_tensor = torch.max(probabilities, axis=1)
    
    # 转换为 NumPy 数组
    pred_labels = pred_labels_tensor.numpy()
    max_probs = max_probs_tensor.numpy()
    y_prob = probabilities.numpy() # (N_samples, N_classes) 
    # --- [END NEW] ---

    original_texts = test_dataset_raw["text"]
    
    # --- [NEW] --- 
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
    # --- [END NEW] ---

    total_samples = len(df_analysis)
    total_errors = len(df_errors)
    accuracy = 1 - (total_errors / total_samples)
    
    print(f"Total test samples: {total_samples}")
    print(f"Prediction errors: {total_errors}")
    print(f"Test Set Accuracy: {accuracy * 100 :.2f}%")

    # --- [NEW] --- 
    # 3. 计算和绘制 ECE 及可靠性图
    print("\n--- Calibration Analysis (Reliability) ---")
    n_bins = 15
    ece, (bins, bacc, bconf, bsize) = expected_calibration_error(y_prob, y_true, n_bins=n_bins)
    print(f"ECE (Expected Calibration Error) @ {n_bins} bins: {ece:.4f} (lower is better)")

    # 开始绘图
    fig = plt.figure(figsize=(8, 6)) # 设置图像大小
    
    # 绘制理想的“完美校准”对角线
    plt.plot([0,1],[0,1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # 计算每个 bin 的中心点
    centers = (bins[:-1] + bins[1:]) / 2
    # 创建一个 mask，只选择那些有数据（非nan）的 bin
    mask = ~np.isnan(bacc)
    
    if mask.any():
        # 绘制柱状图 (Bin Accuracy)
        # width 是每个 bin 的宽度
        plt.bar(centers[mask], bacc[mask], width=(bins[1]-bins[0]), alpha=0.6, 
                edgecolor="black", label="Bin Accuracy")
        
        # 绘制折线图 (Bin Avg. Confidence)
        # 连接 bin 中心的平均置信度点
        plt.plot(centers[mask], bconf[mask], marker="o", linestyle="-", 
                 color="red", label="Bin Avg. Confidence")
    
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.xlabel("Confidence (Predicted Probability)")
    plt.ylabel("Accuracy (Fraction of Positives)")
    plt.legend()       # 显示图例
    plt.grid(alpha=0.3) # 添加网格
    plt.tight_layout() # 自动调整布局
    
    # 保存图像
    plot_file = analysis_dir / "reliability_diagram.png"
    fig.savefig(plot_file, dpi=180)
    print(f"Reliability diagram saved to: {plot_file}")
    # --- [END NEW] ---

    if total_errors > 0:
        print("\n--- Most Confident Error Samples ---")
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', 100)
        
        # 打印按置信度排序的错误
        print(df_errors_sorted.head(20).to_string()) 
        
        # 保存排序后的错误文件
        error_file = analysis_dir / "error_analysis_sorted.csv"
        df_errors_sorted.to_csv(error_file, index=False)
        print(f"\nError samples (sorted by confidence) saved to: {error_file}")
    else:
        print("\n--- No errors found! ---")


if __name__ == "__main__":
    run_inference_and_analysis()