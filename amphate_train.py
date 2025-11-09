
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertModel,
    BertPreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score
import kagglehub
import pandas as pd
from pathlib import Path
from transformers import set_seed

from amphate_model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS

TRAINING_SEED = 42
set_seed(TRAINING_SEED)

# --- Evaluation Results ---
# Accuracy: 0.9634
# F1 (Macro): 0.8664
# Recall (Macro): 0.8760
# --------------------------

def compute_metrics(p):
    """Compute F1, Accuracy, and Recall."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')

    return {
        "accuracy": acc,
        "f1": f1,
        "recall": recall,
    }

def main():
    print("Setting up models and tokenizers...")

    BASE_MODEL_NAME = "bert-base-cased"

    # NER Tagger from the paper 
    NER_MODEL_NAME = "dslim/bert-base-NER"

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)

    ner_pipeline = pipeline(
        "ner",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple"
    )

    path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")
    print(f"Dataset downloaded to: {path}")

    data_file_path = Path(path) / "train_E6oV3lV.csv"
    
    if not data_file_path.exists():
        print(f"Error: Could not find 'train.csv' in {path}")
        return

    print(f"Loading data from {data_file_path}...")
    df = pd.read_csv(data_file_path)

    df = df[['label', 'tweet']]
    df = df.rename(columns={"tweet": "text"})
    df = df.dropna(subset=['text'])
    
    class_counts = df['label'].value_counts()
    num_samples = len(df)
    
    num_classes = len(class_counts)
    weights = num_samples / (num_classes * class_counts)
    
    weights = weights.sort_index()
    class_weights = torch.tensor(weights.values, dtype=torch.float32)
    print(f"Class Weights: {class_weights}")

    dataset = Dataset.from_pandas(df)
    
    dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
    print(f"Dataset created: {dataset}")

    print("Pre-computing NER results for TRAIN set... (This will be cached by 'datasets')")

    train_texts = dataset["train"]["text"]

    train_ner_results = ner_pipeline(train_texts, batch_size=64) 

    dataset["train"] = dataset["train"].add_column("ner_results", train_ner_results)
    
    print("Pre-computing NER results for TEST set... (This will be cached by 'datasets')")
    test_texts = dataset["test"]["text"]
    test_ner_results = ner_pipeline(test_texts, batch_size=64)
    dataset["test"] = dataset["test"].add_column("ner_results", test_ner_results)
    
    print("NER results pre-computed and added to dataset.")

    print("Pre-processing data (running alignment and tokenization)...")
    preprocess_fn = create_preprocessing_function(base_tokenizer)    
    tokenized_datasets = dataset.map(preprocess_fn, batched=True)
    tokenized_datasets.set_format("torch", columns=[
        "input_ids", "attention_mask", "explicit_target_mask", "labels"
    ])

    print(f"Loading AmpleHate model with {BASE_MODEL_NAME} backbone...")
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    model = AmpleHateModel.from_pretrained(
        BASE_MODEL_NAME,
        config=config,
        lambda_val=0.05,
        class_weights=class_weights,
    )
    data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)

    training_args = TrainingArguments(
        output_dir="./amplehate_results_test",
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=2e-6,
        max_grad_norm=1.0,
        fp16=False,
        warmup_ratio=0.1,
        seed=TRAINING_SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=base_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )



    print("Starting training...")
    trainer.train()

    print("\nTraining complete.")
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate()

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 (Macro): {eval_results['eval_f1']:.4f}")
    print(f"Recall (Macro): {eval_results['eval_recall']:.4f}")
    print("--------------------------")


if __name__ == "__main__":
    main()