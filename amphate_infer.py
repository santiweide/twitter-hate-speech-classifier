import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader # ⬅️ Import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    # Trainer, ⬅️ No longer needed
    DataCollatorWithPadding,
    pipeline,
    set_seed,
)
from datasets import Dataset
import kagglehub
from tqdm import tqdm # ⬅️ For our own progress bar

# --- Import your custom model and function---
from amphate_model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS

# --- Use the same seed as training to get the same test set ---
TRAINING_SEED = 42
set_seed(TRAINING_SEED)

def run_inference_and_analysis():

    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    print(f"Using Tokenizer: {BASE_MODEL_NAME}")

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AmpleHateModel.from_pretrained(CHECKPOINT_PATH)
    data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Move model to GPU
    model.eval()     # Set model to evaluation mode (disables dropout, etc.)
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
    
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text", "label", "ner_results"])
    tokenized_test_dataset.set_format("torch")
    
    print("\nStarting manual forward pass on test set...")
    
    test_dataloader = DataLoader(
        tokenized_test_dataset,
        batch_size=16, # You can adjust this batch size
        collate_fn=data_collator
    )
    
    all_logits = [] # List to store logits from all batches

    with torch.no_grad(): # Disable gradient calculation
        for batch in tqdm(test_dataloader, desc="Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            
            logits = outputs.logits.cpu()
            all_logits.append(logits)

    # Combine all logits from all batches into one big tensor
    final_logits = torch.cat(all_logits, dim=0)
    print("Inference complete.")

    # --- 4. Error Analysis ---
    # Get the predicted labels (0 or 1)
    pred_labels = torch.argmax(final_logits, axis=1).numpy()
    # Get the original texts
    original_texts = test_dataset_raw["text"] # We kept this from earlier

    print("\n--- 📊 Error Analysis Results ---")
    
    df_analysis = pd.DataFrame({
        "text": original_texts,
        "true_label": true_labels, # We saved this earlier
        "predicted_label": pred_labels
    })

    df_errors = df_analysis[df_analysis["predicted_label"] != df_analysis["true_label"]].copy()

    total_samples = len(df_analysis)
    total_errors = len(df_errors)
    accuracy = 1 - (total_errors / total_samples)
    
    print(f"Total test samples: {total_samples}")
    print(f"Prediction errors: {total_errors}")
    print(f"Test Set Accuracy: {accuracy * 100 :.2f}%")

    if total_errors > 0:
        print("\n--- Error Samples ---")
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', 100)
        print(df_errors.to_string())
        error_file = "error_analysis.csv"
        df_errors.to_csv(error_file, index=False)
        print(f"\nError samples saved to: {error_file}")
    else:
        print("\n--- 🎉 No errors found! ---")


if __name__ == "__main__":
    run_inference_and_analysis()