import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, load_dataset # NEW: Added load_dataset
from pathlib import Path
from scipy.special import softmax
import urllib.request
import csv
import kagglehub 
import sys 
from sklearn.metrics import accuracy_score, f1_score, recall_score 
from transformers import set_seed

# --- NEW: Added missing imports ---
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random # NEW: For typo injection

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

# --- NEW: Define a top-level output directory ---
OUTPUT_DIR = Path("./roberta_results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

# ==========================================================
# --- 🛠️ HELPER FUNCTIONS (Refactored & New) ---
# ==========================================================

def preprocess(text):
    # This is your original preprocessing function
    if not isinstance(text, str):
        return "" # Handle potential np.nan or other types
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def create_tokenizer_function(tokenizer):
    # Wraps your tokenize function to pass the tokenizer
    def tokenize_function(examples):
        processed_texts = [preprocess(t) for t in examples['text']]
        return tokenizer(processed_texts, padding="max_length", truncation=True, max_length=128)
    return tokenize_function

def expected_calibration_error(probs, labels, n_bins=10):
    # Your original ECE function (assuming it exists in my_metrics)
    # This is a placeholder; replace with your actual function if it's different.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accuracies = (preds == labels).astype(float)
    
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
             mask = (confidences >= lo) & (confidences <= hi) # Include 0

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

def run_evaluation(dataset, model, data_collator, device):
    """
    NEW: Reusable function to run inference on any given dataset.
    """
    # --- FIX: Set format *only* for this run and save original ---
    original_format = dataset.format
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    eval_dataloader = DataLoader(
        dataset,
        batch_size=16, 
        collate_fn=data_collator
    )
    
    all_logits = []
    all_labels = [] # <-- NEW: Get labels from the batch
    
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Running evaluation"):
            all_labels.append(batch['labels'].cpu()) # <-- Store labels
            model_inputs = {
                k: v.to(device) for k, v in batch.items() 
                if k in ["input_ids", "attention_mask"] 
            }
            if not model_inputs:
                print("Warning: Batch contains no model inputs.")
                continue
                
            outputs = model(**model_inputs)
            logits = outputs.logits.cpu()
            all_logits.append(logits)
    
    if not all_logits:
        print("Error: No logits were generated. Check dataset and inputs.")
        dataset.set_format(**original_format) # <-- Reset format
        return None, None

    final_logits = torch.cat(all_logits, dim=0)
    y_true = np.array(torch.cat(all_labels, dim=0)) # <-- Use stored labels
    
    probabilities = softmax(final_logits.numpy(), axis=1)
    
    # --- FIX: Reset format to what it was before ---
    dataset.set_format(**original_format)
    return y_true, probabilities

def compute_and_print_metrics(y_true, probabilities, dataset_name):
    """
    NEW: Reusable function to compute and print all relevant metrics.
    """
    if y_true is None or probabilities is None:
        print(f"Skipping metrics for {dataset_name} due to evaluation error.")
        return

    y_pred = np.argmax(probabilities, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    ece, _ = expected_calibration_error(probabilities, y_true, n_bins=10)
    
    print("\n--- 📊 Metrics ---")
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"F1 (Macro):   {f1:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"ECE (10 bins):  {ece:.4f}")
    print("--------------------")

def add_typos(text, p=0.05):
    """
    NEW: Robustness function to add typos to text.
    """
    if not isinstance(text, str): return ""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p:
            # Simple typo: replace with a random letter
            chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return "".join(chars)

def remap_davidson_labels(example):
    """
    NEW: Generalizability helper to map Davidson dataset labels (0, 1, 2)
    to our model's binary labels (1, 0, 0).
    Class 0 (hate) -> 1
    Class 1 (offensive) -> 0
    Class 2 (neither) -> 0
    """
    example['labels'] = 1 if example['class'] == 0 else 0
    return example


# ==========================================================
# --- 1. MODEL AND BASELINE DATA LOADING ---
# ==========================================================

print("--- 1. Loading Model and Baseline Data ---")
CHECKPOINT_PATH = "./roberta_results/checkpoint-9591"
MODEL = "cardiffnlp/twitter-roberta-base-hate" # Base model for tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH, num_labels=2)
tokenize_function = create_tokenizer_function(tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded from {CHECKPOINT_PATH} onto {device}.")

# Load original dataset
try:
    path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")
    data_file_path = Path(path) / "train_E6oV3lV.csv"
    if not data_file_path.exists():
        print(f"Error: Could not find 'train_E6oV3lV.csv' in {path}")
        sys.exit(1)
    df = pd.read_csv(data_file_path)
except Exception as e:
    print(f"Failed to load Kaggle dataset: {e}")
    sys.exit(1)
    
df = df[['label', 'tweet']]
df = df.rename(columns={"tweet": "text"})
df = df.dropna(subset=['text'])
dataset = Dataset.from_pandas(df)

print("Tokenizing baseline dataset...")
# --- FIX: Do NOT remove the 'text' column here ---
tokenized_dataset = dataset.map(tokenize_function, batched=True) 
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=TRAINING_SEED)
eval_dataset = split_dataset['test'] # <-- This dataset NOW contains the 'text' column

# --- FIX: Get text and labels directly from eval_dataset. ---
raw_eval_texts = eval_dataset['text']
raw_eval_labels = eval_dataset['labels'] # <-- THIS IS THE FIX
# ------------------ END OF FIX ------------------

print(f"Baseline Eval dataset size: {len(eval_dataset)}")

# ==========================================================
# --- 2. BASELINE EVALUATION (IN-DOMAIN) ---
# ==========================================================

print("\n--- 2. Running Baseline In-Domain Evaluation ---")
y_true_base, probs_base = run_evaluation(eval_dataset, model, data_collator, device)
compute_and_print_metrics(y_true_base, probs_base, "Baseline (In-Domain)")

# Run the detailed error analysis ONLY for the baseline
print("\n--- Baseline Error Analysis ---")
probabilities = probs_base
y_true = y_true_base
y_pred = np.argmax(probabilities, axis=1)
confidences = np.max(probabilities, axis=1)

df_analysis = pd.DataFrame({
    "text": raw_eval_texts, # Use raw text
    "true_label": y_true,
    "predicted_label": y_pred,
    "confidence": confidences,
    "prob_0 (non-hate)": probabilities[:, 0],
    "prob_1 (hate)": probabilities[:, 1],
})

analysis_file = OUTPUT_DIR / "full_evaluation_predictions.csv"
df_analysis.to_csv(analysis_file, index=False)
print(f"Full prediction results saved to: {analysis_file}")

df_errors = df_analysis[df_analysis["true_label"] != df_analysis["predicted_label"]]
df_errors_sorted = df_errors.sort_values(by="confidence", ascending=False)

print("\n--- Top 10 Most Confident Errors (Baseline) ---")
pd.set_option('display.max_colwidth', 200)
print(df_errors_sorted.head(10).to_string(index=False))

print("\n--- Calibration Analysis (Baseline) ---")
ece, (bins, bin_accs, bin_confs, bin_sizes) = expected_calibration_error(probabilities, y_true, n_bins=10)
print(f"Expected Calibration Error (ECE) @ 10 bins: {ece:.4f}")

plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
bin_width = 1.0 / 10
bin_centers = bins[:-1] + (bin_width / 2.0)
plt.bar(bin_centers, bin_accs, width=bin_width, edgecolor="black", alpha=0.7, label="Bin Accuracy")
plt.plot(bin_centers, bin_confs, "o-", color="red", label="Bin Avg. Confidence")
plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"Baseline Reliability Diagram (ECE = {ece:.4f})")
plt.legend(); plt.grid(alpha=0.4); plt.xlim(0, 1); plt.ylim(0, 1)
plot_file = OUTPUT_DIR / "reliability_diagram_baseline.png"
plt.savefig(plot_file)
print(f"Baseline reliability diagram saved to: {plot_file}")

# ==========================================================
# --- 3. GENERALIZABILITY EXPERIMENT (OOD) ---
# ==========================================================

print("\n--- 3. Running Generalizability (OOD) Experiment ---")

# --- Test 1: Ethos Dataset ---
print("\nLoading OOD Dataset 1: 'ethos' (binary)")
try:
    ethos_dataset = load_dataset("ethos", "binary", split="train") # Using train split as test set
    # Rename 'label' to 'labels' to match our pipeline
    ethos_dataset = ethos_dataset.rename_column("label", "labels")
    
    print("Tokenizing 'ethos' dataset...")
    tokenized_ethos = ethos_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    y_true_ethos, probs_ethos = run_evaluation(tokenized_ethos, model, data_collator, device)
    compute_and_print_metrics(y_true_ethos, probs_ethos, "OOD - Ethos (binary)")
except Exception as e:
    print(f"Failed to run 'ethos' evaluation: {e}")

# --- Test 2: Davidson Dataset ---
print("\nLoading OOD Dataset 2: 'davidson/hate_speech_and_offensive_language'")
try:
    davidson_dataset = load_dataset("davidson/hate_speech_and_offensive_language", split="train")
    # Map their 3 labels (0, 1, 2) to our binary format (1, 0, 0)
    davidson_dataset = davidson_dataset.map(remap_davidson_labels, remove_columns=["class", "count"])
    
    print("Tokenizing 'davidson' dataset...")
    tokenized_davidson = davidson_dataset.map(tokenize_function, batched=True, remove_columns=["text", "tweet"])
    
    y_true_davidson, probs_davidson = run_evaluation(tokenized_davidson, model, data_collator, device)
    compute_and_print_metrics(y_true_davidson, probs_davidson, "OOD - Davidson (remapped)")
except Exception as e:
    print(f"Failed to run 'davidson' evaluation: {e}")


# ==========================================================
# --- 4. ROBUSTNESS EXPERIMENT (PERTURBATION) ---
# ==========================================================

print("\n--- 4. Running Robustness (Perturbation) Experiment ---")
# We use the raw_eval_texts and raw_eval_labels saved earlier

# --- Test 1: Lowercase ---
print("\nCreating Perturbation 1: Lowercase")
lower_texts = [t.lower() for t in raw_eval_texts]
lower_df = pd.DataFrame({'text': lower_texts, 'labels': raw_eval_labels})
lower_dataset = Dataset.from_pandas(lower_df)

print("Tokenizing 'lowercase' dataset...")
tokenized_lower = lower_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
y_true_lower, probs_lower = run_evaluation(tokenized_lower, model, data_collator, device)
compute_and_print_metrics(y_true_lower, probs_lower, "Robustness - Lowercase")


# --- Test 2: Typos ---
print("\nCreating Perturbation 2: Typo Injection (p=0.05)")
typo_texts = [add_typos(t, p=0.05) for t in raw_eval_texts]
typo_df = pd.DataFrame({'text': typo_texts, 'labels': raw_eval_labels})
typo_dataset = Dataset.from_pandas(typo_df)

print("Tokenizing 'typo' dataset...")
tokenized_typo = typo_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
y_true_typo, probs_typo = run_evaluation(tokenized_typo, model, data_collator, device)
compute_and_print_metrics(y_true_typo, probs_typo, "Robustness - Typos (p=0.05)")

print("\n--- All Experiments Complete ---")