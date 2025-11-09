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

from amphate_model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    
    accuracies = (preds == labels).astype(float)
    
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        
        if i == 0:
             mask = (confidences >= lo) & (confidences <= hi)
        else:
             mask = (confidences > lo) & (confidences <= hi)

        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_w = mask.mean()
            
            ece += np.abs(bin_acc - bin_conf) * bin_w
            
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_sizes.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_sizes.append(0)
            
    return ece, (bins, np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes))

### For Slice Analysis
def has_target_entity(ner_list):
    if not hasattr(ner_list, '__iter__'):
        return False
    return any(isinstance(e, dict) and e.get("entity_group") in TARGET_NER_LABELS for e in ner_list)

def slice_metrics(df, slice_col):

    def get_metrics(sub_df):
        if len(sub_df) == 0:
            return pd.Series({
                "n": 0, "acc": np.nan, "FPR": np.nan, "FNR": np.nan
            })
        
        n = len(sub_df)
        acc = sub_df["correct"].mean()
        
        # FPR (False Positive Rate) 
        fpr_slice = ((sub_df["true_label"] == 0) & (sub_df["predicted_label"] == 1)).sum() / n
        
        # FNR (False Negative Rate) 
        fnr_slice = ((sub_df["true_label"] == 1) & (sub_df["predicted_label"] == 0)).sum() / n
        
        return pd.Series({
            "n": n, "acc": acc, "FPR": fpr_slice, "FNR": fnr_slice
        })

    grouped = df.groupby(slice_col).apply(get_metrics, include_groups=False)
    
    return grouped.reset_index()


def run_inference_and_analysis():

    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

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
    data_file_path = Path(path) / "train_E6oV3lV.csv"
    if not data_file_path.exists():
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

    probabilities = torch.softmax(final_logits, dim=1)
    max_probs_tensor, pred_labels_tensor = torch.max(probabilities, axis=1)
    
    pred_labels = pred_labels_tensor.numpy()
    max_probs = max_probs_tensor.numpy()
    y_prob = probabilities.numpy() # (N_samples, N_classes) 

    original_texts = test_dataset_raw["text"]
    
    df_analysis = pd.DataFrame({
        "text": original_texts,
        "true_label": y_true,
        "predicted_label": pred_labels,
        "confidence": max_probs 
    })
    
    full_analysis_file = analysis_dir / "full_test_predictions.csv"
    df_analysis.to_csv(full_analysis_file, index=False)
    print(f"\nFull prediction results (with confidence) saved to: {full_analysis_file}")
    
    df_errors = df_analysis[df_analysis["predicted_label"] != df_analysis["true_label"]].copy()

    df_errors_sorted = df_errors.sort_values(by="confidence", ascending=False)

    total_samples = len(df_analysis)
    total_errors = len(df_errors)
    accuracy = 1 - (total_errors / total_samples)
    
    print(f"\nTotal test samples: {total_samples}")
    print(f"Prediction errors: {total_errors}")
    print(f"Test Set Accuracy: {accuracy * 100 :.2f}%")

    print("\n--- Calibration Analysis (Reliability) ---")
    n_bins = 15
    ece, (bins, bacc, bconf, bsize) = expected_calibration_error(y_prob, y_true, n_bins=n_bins)
    print(f"ECE (Expected Calibration Error) @ {n_bins} bins: {ece:.4f} (lower is better)")

    fig = plt.figure(figsize=(8, 6))
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
    
    print("\n--- Slice-based Analysis (NER) ---")
    
    test_df = test_dataset_raw.to_pandas()
    
    df_analysis["correct"] = (df_analysis["true_label"] == df_analysis["predicted_label"])
    df_analysis["has_target_entity"] = test_df["ner_results"].map(has_target_entity).astype(int)
    
    target_slice_metrics = slice_metrics(df_analysis, "has_target_entity")
    target_slice_file = analysis_dir / "slice_has_target_entity.csv"
    target_slice_metrics.to_csv(target_slice_file, index=False)
    
    print("Metrics for 'has_target_entity' slice (1 = True):")
    print(target_slice_metrics.to_string(index=False))
    print(f"Saved to: {target_slice_file}")

    rows = []
    for ent in sorted(TARGET_NER_LABELS):
        mask = test_df["ner_results"].map(
            lambda lst: any(isinstance(e, dict) and e.get("entity_group") == ent 
                            for e in (lst if hasattr(lst, '__iter__') else []))
        )
        
        sub = df_analysis[mask]
        
        if len(sub) > 0:
            rows.append({
                "entity_type": ent,
                "n": len(sub),
                "acc": sub["correct"].mean(),
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
    
    if total_errors > 0:
        print("\n--- Most Confident Error Samples ---")
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', 100)
        
        print(df_errors_sorted.head(20).to_string(index=False)) 
        
        error_file = analysis_dir / "error_analysis_sorted.csv"
        df_errors_sorted.to_csv(error_file, index=False)
        print(f"\nError samples (sorted by confidence) saved to: {error_file}")
    else:
        print("\n--- No errors found! ---")


if __name__ == "__main__":
    run_inference_and_analysis()