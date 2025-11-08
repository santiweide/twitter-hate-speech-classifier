import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Use the official pre-trained model as requested
MODEL_NAME = "nvidia/NV-Embed-v2"
DATASET_NAME = "davidson/hate_speech_offensive"
# Instruction for MTEB-style classification
INSTRUCTION = "Classify the given tweet as hate speech, offensive, or neither."

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
    
    # --- 2. Load and Split Dataset ---
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)['train']
    # Re-map labels to be 0, 1, 2
    data_list = [{"tweet": item['tweet'], "class": item['class']} for item in dataset]
    
    # Split data
    train_data, test_data = train_test_split(
        data_list, 
        test_size=0.2, 
        random_state=42, 
        stratify=[d['class'] for d in data_list]
    )
    
    print(f"Total samples: {len(data_list)}, Train: {len(train_data)}, Test: {len(test_data)}")

    # Prepare texts and labels
    train_texts = [d['tweet'] for d in train_data]
    train_labels = np.array([d['class'] for d in train_data])
    
    test_texts = [d['tweet'] for d in test_data]
    test_labels = np.array([d['class'] for d in test_data])

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