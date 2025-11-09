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

# --- ADDED FOR PROFILER ---
from torch.profiler import profile, record_function, ProfilerActivity
# --------------------------

from amphate_model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS
from my_metrics import expected_calibration_error

from datasets import load_dataset
import random 
from sklearn.metrics import accuracy_score, f1_score, recall_score

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

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
        
        # FPR (False PositiveRate) 
        fpr_slice = ((sub_df["true_label"] == 0) & (sub_df["predicted_label"] == 1)).sum() / n
        
        # FNR (False Negative Rate) 
        fnr_slice = ((sub_df["true_label"] == 1) & (sub_df["predicted_label"] == 0)).sum() / n
        
        return pd.Series({
            "n": n, "acc": acc, "FPR": fpr_slice, "FNR": fnr_slice
        })

    grouped = df.groupby(slice_col).apply(get_metrics, include_groups=False)
    
    return grouped.reset_index()

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

def remap_davidson_labels(example):
    """
    Maps Davidson dataset labels (0, 1, 2) to our binary labels (1, 0, 0).
    Class 0 (hate) -> 1
    Class 1 (offensive) -> 0
    Class 2 (neither) -> 0
    """
    example['label'] = 1 if example['class'] == 0 else 0
    example['text'] = example['tweet']
    return example

def compute_and_print_metrics(y_true, y_prob, dataset_name):
    """Calculates and prints standard metrics for a given experiment."""
    if y_true is None or y_prob is None:
        print(f"Skipping metrics for {dataset_name} due to evaluation error.")
        return

    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    try:
        ece, _ = expected_calibration_error(y_prob, y_true, n_bins=10)
    except Exception as e:
        print(f"Could not calculate ECE: {e}")
        ece = np.nan
    
    print("\n" + "="*30)
    print(f"📊 METRICS FOR: {dataset_name}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"F1 (Macro):   {f1:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"ECE (10 bins):  {ece:.4f}")
    print("="*30 + "\n")

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
    print(f"\n--- Running evaluation for: {raw_dataset.builder_name or 'custom'} ---")
    print("Pre-computing NER results...")
    if ner_results_override is not None:
        print("Using NER results override.")
        test_ner_results = ner_results_override
    else:
        test_texts_list = list(raw_dataset["text"])
        test_ner_results = ner_pipeline(test_texts_list, batch_size=64)
    
    try:
        eval_dataset_with_ner = raw_dataset.remove_columns("ner_results")
    except:
        eval_dataset_with_ner = raw_dataset
    eval_dataset_with_ner = eval_dataset_with_ner.add_column("ner_results", test_ner_results)

    print("Tokenizing TEST set...")
    tokenized_eval_dataset = eval_dataset_with_ner.map(preprocess_fn, batched=True, load_from_cache_file=False)
    
    y_true = np.array(tokenized_eval_dataset["labels"]) 
    
    model_input_names = ["input_ids", "attention_mask", "token_type_ids", "explicit_target_mask"]
    columns_to_remove = [
        col for col in tokenized_eval_dataset.column_names 
        if col not in model_input_names
    ]
    
    if columns_to_remove:
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
    
    return y_true, probabilities

def run_generalizability_and_robustness_experiments():
    """
    Runs the new experiment suite.
    This function re-loads the models to avoid modifying the
    original run_inference_and_analysis() function.
    """
    
    print("\n" + "#"*60)
    print("### STARTING GENERALIZABILITY & ROBUSTNESS EXPERIMENTS ###")
    print("#"*60 + "\n")

    print("--- 1. Loading Models and Pipelines for Experiments ---")
    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

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
    print("All models loaded for experiments.")

    print("\n--- 2. Loading Baseline Data for Experiments ---")
    try:
        path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")
        data_file_path = Path(path) / "train_E6oV3lV.csv"
        df = pd.read_csv(data_file_path)
        df = df[['label', 'tweet']].rename(columns={"tweet": "text"}) # <-- FIX: Do not rename 'label'
        df = df.dropna(subset=['text'])
        dataset = Dataset.from_pandas(df)
        dataset_split = dataset.shuffle(seed=TRAINING_SEED).train_test_split(test_size=0.1)
        test_dataset_raw = dataset_split["test"]
    except Exception as e:
        print(f"Failed to load baseline dataset: {e}")
        return

    print("\n--- 3. Running Generalizability (OOD) Experiment ---")
    
    try:
        print("\nLoading OOD Dataset 1: 'ethos' (binary)")
        ethos_raw = load_dataset("ethos", "binary", split="train") # Use train as test
        
        y_true_ethos, y_prob_ethos = run_evaluation_pipeline(
            ethos_raw, ner_pipeline, preprocess_fn, model, data_collator, device
        )
        compute_and_print_metrics(y_true_ethos, y_prob_ethos, "OOD - Ethos")
    except Exception as e:
        print(f"Failed to run 'ethos' evaluation: {e}")

    try:
        print("\nLoading OOD Dataset 2: 'davidson/hate_speech_and_offensive_language'")
        davidson_raw = load_dataset("davidson/hate_speech_and_offensive_language", split="train")
        davidson_raw = davidson_raw.map(remap_davidson_labels, remove_columns=["class", "count", "tweet"])
        
        y_true_davidson, y_prob_davidson = run_evaluation_pipeline(
            davidson_raw, ner_pipeline, preprocess_fn, model, data_collator, device
        )
        compute_and_print_metrics(y_true_davidson, y_prob_davidson, "OOD - Davidson (Remapped)")
    except Exception as e:
        print(f"Failed to run 'davidson' evaluation: {e}")

    # --- 4. Run Robustness (Perturbation) Experiment ---
    print("\n--- 4. Running Robustness (Perturbation) Experiment ---")
    
    print("\nCreating Perturbation 1: Typo Injection (p=0.05)")
    test_data_typos = test_dataset_raw.map(add_typos, load_from_cache_file=False)
    y_true_typo, y_prob_typo = run_evaluation_pipeline(
        test_data_typos, ner_pipeline, preprocess_fn, model, data_collator, device
    )
    compute_and_print_metrics(y_true_typo, y_prob_typo, "Robustness - Typos (p=0.05)")

    print("\nCreating Perturbation 2: NER Ablation (No Entities)")
    empty_ner_results = [[] for _ in range(len(test_dataset_raw))]
    
    y_true_ner, y_prob_ner = run_evaluation_pipeline(
        test_dataset_raw, ner_pipeline, preprocess_fn, model, data_collator, device,
        ner_results_override=empty_ner_results # Pass in the empty results
    )
    compute_and_print_metrics(y_true_ner, y_prob_ner, "Robustness - NER Ablation (No Entities)")
    
    print("\n--- All Experiments Complete ---")

def run_inference_and_analysis():

    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

    analysis_dir = Path("error_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis artifacts will be saved to: {analysis_dir.resolve()}")

    # --- ADDED: Create directory for profiler logs ---
    profile_log_dir = Path("profile_logs")
    profile_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Profiler logs will be saved to: {profile_log_dir.resolve()}")
    # --------------------------------------------------

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
    
    # --- PROFILER ADDED for NER ---
    print("\nProfiling NER pipeline...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(profile_log_dir / "ner_pipeline")
        )
    ) as prof_ner:
        with record_function("ner_pipeline_inference"):
            test_ner_results = ner_pipeline(test_texts_list, batch_size=64)
            
    print("\n--- NER Profiler Summary ---")
    print(prof_ner.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"NER profile trace saved. View in TensorBoard:\n tensorboard --logdir {profile_log_dir}")
    # --- END PROFILER ---

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

    # --- PROFILER ADDED for main model inference ---
    print("\nProfiling Main Model inference loop...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(profile_log_dir / "main_model_inference")
        )
    ) as prof_main:
        with torch.no_grad(): # Disable gradient calculation
            for batch in tqdm(test_dataloader, desc="Inference"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with record_function("model_forward_pass"):
                    outputs = model(**batch)
                
                logits = outputs.logits.cpu()
                all_logits.append(logits)

    print("\n--- Main Model Profiler Summary ---")
    print(prof_main.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(f"Main Model profile trace saved. View in TensorBoard:\n tensorboard --logdir {profile_log_dir}")
    # --- END PROFILER ---

    final_logits = torch.cat(all_logits, dim=0)
    print("Inference complete.")

    print("\n--- Error Analysis Results ---")

    probabilities = torch.softmax(final_logits, dim=1)
    max_probs_tensor, pred_labels_tensor = torch.max(probabilities, axis=1)
    
    pred_labels = pred_labels_tensor.numpy()
    max_probs = max_probs_tensor.numpy()
    y_prob = probabilities.numpy() 

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

    print("\n\nRunning Generalizability and Robustness Suite...")
    run_generalizability_and_robustness_experiments()