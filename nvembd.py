import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset # <-- Added Dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
import numpy as np
from tqdm import tqdm
import kagglehub # <-- Added
import pandas as pd # <-- Added
from pathlib import Path # <-- Added

# --- Configuration ---
# Use the official pre-trained model as requested
MODEL_NAME = "nvidia/NV-Embed-v2"
# DATASET_NAME = "davidson/hate_speech_offensive" # <-- Replaced
DATASET_NAME = "vkrahul/twitter-hate-speech"
DATASET_FILE = "train_E6oV3lV.csv"
# Instruction for MTEB-style classification
# INSTRUCTION = "Classify the given tweet as hate speech, offensive, or neither." # <-- Replaced
INSTRUCTION = "Classify the given tweet as hate speech or not."

# Hyperparameters
MAX_LENGTH = 512 # Paper evaluates at 512, but can be longer
EVAL_BATCH_SIZE = 32 # Batch size for generating embeddings

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    # Load the official model from Hugging Face
    print(f"Loading model: {MODEL_NAME}")
    # trust_remote_code is required for NV-Embed
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    # The .encode() method comes with the model, no tokenizer needed separately
    
    print(f"Downloading dataset: {DATASET_NAME}") # <-- Your new code starts here
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset downloaded to: {path}")

    data_file_path = Path(path) / DATASET_FILE
    
    if not data_file_path.exists():
        print(f"Error: Could not find '{DATASET_FILE}' in {path}")
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

    dataset_full = Dataset.from_pandas(df)
    
    # User wants 0.1 test split
    dataset_split = dataset_full.shuffle(seed=42).train_test_split(test_size=0.1)
    print(f"Dataset created: {dataset_split}")

    # Prepare texts and labels from the new dataset split object
    train_texts = dataset_split['train']['text']
    train_labels = np.array(dataset_split['train']['label'])
    
    test_texts = dataset_split['test']['text']
    test_labels = np.array(dataset_split['test']['label'])

    print(f"Total samples: {len(dataset_full)}, Train: {len(train_texts)}, Test: {len(test_texts)}")
    # <-- Your new code ends here

    # --- 3. Generate Embeddings ---
    # Add the required instruction prefix for classification
    query_prefix = f"Instruct: {INSTRUCTION}\nQuery: "
    
    train_texts_with_instruction = [query_prefix + text for text in train_texts]
    test_texts_with_instruction = [query_prefix + text for text in test_texts]

    print("Generating embeddings for training data...")
    # Use the model's .encode() method
    train_embeddings = model.encode(
        train_texts_with_instruction, 
        max_length=MAX_LENGTH, 
        batch_size=EVAL_BATCH_SIZE
    )
    
    print("Generating embeddings for test data...")
    test_embeddings = model.encode(
        test_texts_with_instruction, 
        max_length=MAX_LENGTH, 
        batch_size=EVAL_BATCH_SIZE
    )
    
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # --- 4. Normalize Embeddings (as shown in your example) ---
    train_embeddings = F.normalize(torch.tensor(train_embeddings), p=2, dim=1).numpy()
    test_embeddings = F.normalize(torch.tensor(test_embeddings), p=2, dim=1).numpy()

    # --- 5. Train k-NN Classifier (MTEB-style eval) ---
    print("Training k-NN classifier on embeddings...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_embeddings, train_labels)
    
    # --- 6. Predict and Evaluate ---
    print("Predicting with k-NN...")
    preds = knn.predict(test_embeddings)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, preds)
    f1_macro = f1_score(test_labels, preds, average='macro')
    recall_macro = recall_score(test_labels, preds, average='macro')
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"F1 (Macro):     {f1_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()