import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path
from scipy.special import softmax
import urllib.request
import csv
import kagglehub 
import sys 
from sklearn.metrics import accuracy_score, f1_score, recall_score # NEW: Import metrics
from transformers import set_seed
from my_metrics import compute_metrics

TRAINING_SEED = 42
set_seed(TRAINING_SEED)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task = 'hate'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

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

print("Calculating class weights...")
class_counts = df['label'].value_counts()
num_samples = len(df)

num_classes = len(class_counts)
weights = num_samples / (num_classes * class_counts)

weights = weights.sort_index()
class_weights = torch.tensor(weights.values, dtype=torch.float32)
print(f"Class Weights (for labels {weights.index.values}): {class_weights}")

dataset = Dataset.from_pandas(df)

def tokenize_function(examples):
    processed_texts = [preprocess(t) for t in examples['text']]
    return tokenizer(processed_texts, padding="max_length", truncation=True, max_length=128)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./roberta_results",
    num_train_epochs=3, 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=50, 
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False, 
    metric_for_best_model="f1",
    seed=TRAINING_SEED,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    class_weights=class_weights,
    compute_metrics=compute_metrics 
)

print("Starting training...")
trainer.train()
print("Training complete.")
print("\nEvaluating on test set...")
eval_results = trainer.evaluate()

print("\n--- Evaluation Results ---")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 (Macro): {eval_results['eval_f1']:.4f}")
print(f"Recall (Macro): {eval_results['eval_recall']:.4f}")
print("--------------------------")