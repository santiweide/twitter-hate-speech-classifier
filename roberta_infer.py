import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformers import Trainer, TrainingArguments # REMOVED
from datasets import Dataset
from pathlib import Path
from scipy.special import softmax
import urllib.request
import csv
import kagglehub 
import sys 
from sklearn.metrics import accuracy_score, f1_score, recall_score 
from transformers import set_seed
from my_metrics import compute_metrics, expected_calibration_error # NEW: Added expected_calibration_error

# --- NEW: Added missing imports for manual inference and plotting ---
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

# --- Evaluation Results ---
# Accuracy: 0.9840
# F1 (Macro): 0.9374
# Recall (Macro): 0.9172
# --------------------------

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

CHECKPOINT_PATH = "./roberta_results/checkpoint-9591"
# --- NEW: Define a top-level output directory ---
OUTPUT_DIR = Path("./roberta_results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

MODEL = f"cardiffnlp/twitter-roberta-base-hate"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# --- NEW: Define the data collator for the DataLoader ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH, num_labels=2)

print("Downloading dataset from Kaggle Hub...")
path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")
print(f"Dataset downloaded to: {path}")

data_file_path = Path(path) / "train_E6oV3lV.csv"

if not data_file_path.exists():
    print(f"Error: Could not find 'train_E6oV3lV.csv' in {path}")
    sys.exit(1)

print(f"Loading data from {data_file_path}...")
df = pd.read_csv(data_file_path)

df = df[['label', 'tweet']]
df = df.rename(columns={"tweet": "text"})
df = df.dropna(subset=['text'])

dataset = Dataset.from_pandas(df)

def tokenize_function(examples):
    processed_texts = [preprocess(t) for t in examples['text']]
    return tokenizer(processed_texts, padding="max_length", truncation=True, max_length=128)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=TRAINING_SEED)
eval_dataset = split_dataset['test']

print(f"Eval dataset size: {len(eval_dataset)}")

test_dataloader = DataLoader(
    eval_dataset,
    batch_size=16, 
    collate_fn=data_collator
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded onto {device} and set to eval() mode.")

all_logits = []

with torch.no_grad(): # Disable gradient calculation
    for batch in tqdm(test_dataloader, desc="Inference"):
        batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        
        outputs = model(**batch)
        
        logits = outputs.logits.cpu()
        all_logits.append(logits)

final_logits = torch.cat(all_logits, dim=0)
print("Inference complete.")

print("\n--- Starting Error Analysis ---")

print("Processing results from manual inference...")
logits = final_logits.numpy()
y_true = np.array(eval_dataset['labels']) # Get true labels from the dataset

probabilities = softmax(logits, axis=1)
y_pred = np.argmax(probabilities, axis=1)
confidences = np.max(probabilities, axis=1) # Confidence is the max probability

print("\n--- Full Prediction Results ---")

original_texts = eval_dataset['text']

df_analysis = pd.DataFrame({
    "text": original_texts,
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

print("\n--- Top 10 Most Confident Errors ---")
pd.set_option('display.max_colwidth', 200) # To see the full text
print(df_errors_sorted.head(10).to_string(index=False))

print("\n--- Calibration Analysis (Reliability) ---")
n_bins = 10
ece, bin_accs, bin_confs, bin_counts, bin_lowers = expected_calibration_error(
    y_true, y_pred, n_bins, confidences
)

print(f"Expected Calibration Error (ECE) @ {n_bins} bins: {ece:.4f}")

plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

bin_width = 1.0 / n_bins
bin_centers = bin_lowers + (bin_width / 2.0)

plt.bar(bin_centers, bin_accs, width=bin_width, edgecolor="black", 
        alpha=0.7, label="Bin Accuracy")

plt.plot(bin_centers, bin_confs, "o-", color="red", 
         label="Bin Avg. Confidence")

plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
plt.legend()
plt.grid(alpha=0.4)
plt.xlim(0, 1)
plt.ylim(0, 1)

plot_file = OUTPUT_DIR / "reliability_diagram.png"
plt.savefig(plot_file)
print(f"Reliability diagram saved to: {plot_file}")
print("--- Error Analysis Complete ---")