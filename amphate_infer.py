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
from datasets import Dataset, load_dataset # NEW: Added load_dataset
import kagglehub
from tqdm import tqdm
import matplotlib.pyplot as plt
import random # NEW: For typo injection

# Ensure these are available in your environment
from amphate_model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS
from my_metrics import expected_calibration_error

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

# ==========================================================
# --- 🛠️ HELPER FUNCTIONS (Original & New) ---
# ==========================================================

### For Slice Analysis
def has_target_entity(ner_list):
    if not hasattr(ner_list, '__iter__'):
        return False
    return any(isinstance(e, dict) and e.get("entity_group") in TARGET_NER_LABELS for e in ner_list)

def slice_metrics(df, slice_col):
    def get_metrics(sub_df):
        if len(sub_df) == 0:
            return pd.Series({"n": 0, "acc": np.nan, "FPR": np.nan, "FNR": np.nan})
        n = len(sub_df)
        acc = sub_df["correct"].mean()
        fpr_slice = ((sub_df["true_label"] == 0) & (sub_df["predicted_label"] == 1)).sum() / n
        fnr_slice = ((sub_df["true_label"] == 1) & (sub_df["predicted_label"] == 0)).sum() / n
        return pd.Series({"n": n, "acc": acc, "FPR": fpr_slice, "FNR": fnr_slice})
    grouped = df.groupby(slice_col).apply(get_metrics, include_groups=False)
    return grouped.reset_index()

# --- NEW: Function to add typos ---
def add_typos(example, p=0.05):
    """Adds typos to the 'text' field of an example."""
    text = example['text']
    if not isinstance(text, str): return example
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p and chars[i].isalpha(): # Only corrupt letters
            chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    example['text'] = "".join(chars)
    return example

# --- NEW: Function to remap Davidson labels ---
def remap_davidson_labels(example):
    """
    Maps Davidson dataset labels (0, 1, 2) to our binary labels (1, 0, 0).
    Class 0 (hate) -> 1
    Class 1 (offensive) -> 0
    Class 2 (neither) -> 0
    """
    example['labels'] = 1 if example['class'] == 0 else 0
    example['text'] = example['tweet'] # Rename column
    return example

# --- NEW: Reusable function to compute and print metrics ---
def compute_and_print_metrics(y_true, y_prob, dataset_name):
    """Calculates and prints standard metrics for a given experiment."""
    if y_true is None or y_prob is None:
        print(f"Skipping metrics for {dataset_name} due to evaluation error.")
        return

    from sklearn.metrics import accuracy_score, f1_score, recall_score
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    ece, _ = expected_calibration_error(y_prob, y_true, n_bins=10)
    
    print("\n" + "="*30)
    print(f"📊 METRICS FOR: {dataset_name}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"F1 (Macro):   {f1:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"ECE (10 bins):  {ece:.4f}")
    print("="*30 + "\n")

# --- NEW: Refactored main evaluation pipeline ---
def run_evaluation_pipeline(
    raw_dataset, 
    ner_pipeline, 
    preprocess_fn, 
    model, 
    data_collator, 
    device, 
    ner_results_override=None
):
    """
    Runs the full NER -> Preprocessing -> Inference pipeline on a raw dataset.
    """
    print("Pre-computing NER results...")
    if ner_results_override is not None:
        print("Using NER results override.")
        test_ner_results = ner_results_override
    else:
        test_texts_list = list(raw_dataset["text"])
        test_ner_results = ner_pipeline(test_texts_list, batch_size=64)
    
    # Add NER results column to dataset
    try:
        eval_dataset_with_ner = raw_dataset.remove_columns("ner_results")
    except:
        eval_dataset_with_ner = raw_dataset
    eval_dataset_with_ner = eval_dataset_with_ner.add_column("ner_results", test_ner_results)

    print("Tokenizing TEST set...")
    tokenized_eval_dataset = eval_dataset_with_ner.map(preprocess_fn, batched=True)
    
    y_true = np.array(tokenized_eval_dataset["labels"]) 
    
    # Remove non-model columns
    columns_to_remove = [col for col in tokenized_eval_dataset.column_names if col not in ["input_ids", "attention_mask"]]
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(columns_to_remove)
    tokenized_eval_dataset.set_format("torch")
    
    print("Starting manual forward pass...")
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        batch_size=16,
        collate_fn=data_collator
    )
    
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu()
            all_logits.append(logits)

    final_logits = torch.cat(all_logits, dim=0)
    print("Inference complete.")
    
    probabilities = torch.softmax(final_logits, dim=1).numpy()
    
    return y_true, probabilities, eval_dataset_with_ner.to_pandas()


# --- This is your original function, now for baseline analysis ---
def run_baseline_analysis(y_true, y_prob, test_df, analysis_dir):
    """Runs the detailed slice and error analysis on baseline results."""
    
    print("\n--- 4. Baseline Error Analysis ---")
    pred_labels = y_prob.argmax(axis=1)
    confidences = y_prob.max(axis=1)

    df_analysis = pd.DataFrame({
        "text": test_df["text"],
        "true_label": y_true,
        "predicted_label": pred_labels,
        "confidence": confidences 
    })
    
    full_analysis_file = analysis_dir / "full_test_predictions.csv"
    df_analysis.to_csv(full_analysis_file, index=False)
    print(f"\nFull prediction results (with confidence) saved to: {full_analysis_file}")
    
    total_samples = len(df_analysis)
    total_errors = (df_analysis["predicted_label"] != df_analysis["true_label"]).sum()
    print(f"\nTotal test samples: {total_samples}")
    print(f"Prediction errors: {total_errors}")
    print(f"Test Set Accuracy: {(1 - (total_errors / total_samples)) * 100 :.2f}%")

    print("\n--- Calibration Analysis (Reliability) ---")
    n_bins = 15
    ece, (bins, bacc, bconf, bsize) = expected_calibration_error(y_prob, y_true, n_bins=n_bins)
    print(f"ECE (Expected Calibration Error) @ {n_bins} bins: {ece:.4f} (lower is better)")
    
    # (Plotting code omitted for brevity, but it's identical to your original)
    fig = plt.figure(figsize=(8, 6))
    plt.plot([0,1],[0,1], linestyle="--", color="gray", label="Perfect Calibration")
    centers = (bins[:-1] + bins[1:]) / 2; mask = ~np.isnan(bacc)
    if mask.any():
        plt.bar(centers[mask], bacc[mask], width=(bins[1]-bins[0]), alpha=0.6, edgecolor="black", label="Bin Accuracy")
        plt.plot(centers[mask], bconf[mask], marker="o", linestyle="-", color="red", label="Bin Avg. Confidence")
    plt.title(f"Reliability Diagram (ECE={ece:.3f})"); plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout() 
    plot_file = analysis_dir / "reliability_diagram.png"
    fig.savefig(plot_file, dpi=180)
    print(f"Reliability diagram saved to: {plot_file}")
    
    
    print("\n--- Slice-based Analysis (NER) ---")
    df_analysis["correct"] = (df_analysis["true_label"] == df_analysis["predicted_label"])
    df_analysis["has_target_entity"] = test_df["ner_results"].map(has_target_entity).astype(int)
    
    target_slice_metrics = slice_metrics(df_analysis, "has_target_entity")
    target_slice_file = analysis_dir / "slice_has_target_entity.csv"
    target_slice_metrics.to_csv(target_slice_file, index=False)
    print("Metrics for 'has_target_entity' slice (1 = True):")
    print(target_slice_metrics.to_string(index=False))
    
    # (Per-entity slice analysis code omitted for brevity, identical to original)
    # ...


# ==========================================================
# --- 🚀 MAIN EXPERIMENT SCRIPT ---
# ==========================================================
def main():
    
    # --- 1. Setup Models, Tokenizers, and Pipelines ---
    print("--- 1. Loading Models and Pipelines ---")
    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

    analysis_dir = Path("error_analysis_experiments")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis artifacts will be saved to: {analysis_dir.resolve()}")

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AmpleHateModel.from_pretrained(CHECKPOINT_PATH)
    data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    ner_pipeline = pipeline(
        "ner", model=ner_model, tokenizer=ner_tokenizer, 
        aggregation_strategy="simple", device=device
    )
    preprocess_fn = create_preprocessing_function(base_tokenizer)
    print("All models loaded.")

    # --- 2. Load Baseline Data ---
    print("\n--- 2. Loading Baseline Data ---")
    path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")
    data_file_path = Path(path) / "train_E6oV3lV.csv"
    if not data_file_path.exists():
        print("Error: Baseline data not found."); return
            
    df = pd.read_csv(data_file_path)
    df = df[['label', 'tweet']].rename(columns={"tweet": "text", "label": "labels"})
    df = df.dropna(subset=['text'])
    
    dataset = Dataset.from_pandas(df)
    dataset_split = dataset.shuffle(seed=TRAINING_SEED).train_test_split(test_size=0.1)
    test_dataset_raw = dataset_split["test"] # This is our raw baseline test set
    
    # --- 3. Run Baseline Evaluation & Analysis ---
    print("\n--- 3. Running Baseline Evaluation ---")
    y_true_base, y_prob_base, test_df_with_ner = run_evaluation_pipeline(
        test_dataset_raw, ner_pipeline, preprocess_fn, model, data_collator, device
    )
    # Print baseline metrics
    compute_and_print_metrics(y_true_base, y_prob_base, "Baseline (In-Domain)")
    
    # Run the deep-dive analysis *only* on the baseline
    run_baseline_analysis(y_true_base, y_prob_base, test_df_with_ner, analysis_dir)

    # --- 4. NEW: Run Generalizability (OOD) Experiment ---
    print("\n--- 4. Running Generalizability (OOD) Experiment ---")
    
    # Test 4.1: Ethos
    try:
        print("\nLoading OOD Dataset 1: 'ethos' (binary)")
        ethos_raw = load_dataset("ethos", "binary", split="train") # Use train as test
        ethos_raw = ethos_raw.rename_column("label", "labels")
        
        y_true_ethos, y_prob_ethos, _ = run_evaluation_pipeline(
            ethos_raw, ner_pipeline, preprocess_fn, model, data_collator, device
        )
        compute_and_print_metrics(y_true_ethos, y_prob_ethos, "OOD - Ethos")
    except Exception as e:
        print(f"Failed to run 'ethos' evaluation: {e}")

    # Test 4.2: Davidson
    try:
        print("\nLoading OOD Dataset 2: 'davidson/hate_speech_and_offensive_language'")
        davidson_raw = load_dataset("davidson/hate_speech_and_offensive_language", split="train")
        davidson_raw = davidson_raw.map(remap_davidson_labels, remove_columns=["class", "count", "tweet"])
        
        y_true_davidson, y_prob_davidson, _ = run_evaluation_pipeline(
            davidson_raw, ner_pipeline, preprocess_fn, model, data_collator, device
        )
        compute_and_print_metrics(y_true_davidson, y_prob_davidson, "OOD - Davidson (Remapped)")
    except Exception as e:
        print(f"Failed to run 'davidson' evaluation: {e}")

    # --- 5. NEW: Run Robustness (Perturbation) Experiment ---
    print("\n--- 5. Running Robustness (Perturbation) Experiment ---")
    
    # Test 5.1: Text Perturbation (Typos)
    print("\nCreating Perturbation 1: Typo Injection (p=0.05)")
    test_data_typos = test_dataset_raw.map(add_typos)
    y_true_typo, y_prob_typo, _ = run_evaluation_pipeline(
        test_data_typos, ner_pipeline, preprocess_fn, model, data_collator, device
    )
    compute_and_print_metrics(y_true_typo, y_prob_typo, "Robustness - Typos (p=0.05)")

    # Test 5.2: Feature Perturbation (NER Ablation)
    print("\nCreating Perturbation 2: NER Ablation (No Entities)")
    # Create a list of empty lists, one for each sample in the test set
    empty_ner_results = [[] for _ in range(len(test_dataset_raw))]
    
    y_true_ner, y_prob_ner, _ = run_evaluation_pipeline(
        test_dataset_raw, ner_pipeline, preprocess_fn, model, data_collator, device,
        ner_results_override=empty_ner_results # Pass in the empty results
    )
    compute_and_print_metrics(y_true_ner, y_prob_ner, "Robustness - NER Ablation (No Entities)")
    
    print("\n--- All Experiments Complete ---")


if __name__ == "__main__":
    main()