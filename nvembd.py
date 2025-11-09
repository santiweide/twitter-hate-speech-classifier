import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm
import kagglehub
import pandas as pd
from pathlib import Path

# --- Configuration ---
MODEL_NAME = "nvidia/NV-Embed-v2"
DATASET_NAME = "vkrahul/twitter-hate-speech"
DATASET_FILE = "train_E6oV3lV.csv"
INSTRUCTION = "Classify the given tweet as hate speech or not."

# Hyperparameters
MAX_LENGTH = 512
# --- MODIFIED: 这现在是我们唯一的批处理大小 ---
# 从 8 开始。如果 OOM，请尝试 4、2，最后是 1。
EFFECTIVE_BATCH_SIZE = 8
# -----------------------------------------------

# --- MODIFIED HELPER FUNCTION ---
def generate_embeddings_in_batches(model, texts, max_length, batch_size):
    """
    通过在显式的小批量中处理数据来生成嵌入，以避免 OOM。
    """
    all_embeddings_list = []
    print(f"Generating embeddings in batches of {batch_size}...")
    
    # 以小批量处理文本列表
    for i in tqdm(range(0, len(texts), batch_size)):
        # 这现在是一个小批量，例如 8 个文本
        text_batch = texts[i : i + batch_size]
        
        # model.encode() 现在只接收小批量。
        # 我们不需要传递 'batch_size' 参数，
        # 因为列表本身已经是我们想要的批处理大小。
        batch_embeddings = model.encode(
            text_batch,
            max_length=max_length
        )
        
        # 附加 numpy 数组到我们的列表
        all_embeddings_list.append(batch_embeddings.detach().cpu().numpy())
        
        # 以防万一，清除 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 在 CPU 上连接所有 numpy 数组
    print("Concatenating all embedding batches...")
    return np.concatenate(all_embeddings_list, axis=0)
# --- END MODIFIED HELPER FUNCTION ---


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading model: {MODEL_NAME}")
    # 以 bfloat16 加载模型以节省内存，并移动到 device
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    print(f"Downloading dataset: {DATASET_NAME}")
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
    
    dataset_split = dataset_full.shuffle(seed=42).train_test_split(test_size=0.1)
    print(f"Dataset created: {dataset_split}")

    train_texts = dataset_split['train']['text']
    train_labels = np.array(dataset_split['train']['label'])
    
    test_texts = dataset_split['test']['text']
    test_labels = np.array(dataset_split['test']['label'])

    print(f"Total samples: {len(dataset_full)}, Train: {len(train_texts)}, Test: {len(test_texts)}")

    # --- 3. Generate Embeddings ---
    query_prefix = f"Instruct: {INSTRUCTION}\nQuery: "
    
    train_texts_with_instruction = [query_prefix + text for text in train_texts]
    test_texts_with_instruction = [query_prefix + text for text in test_texts]

    print("Generating embeddings for training data...")
    # --- MODIFIED: 使用新的辅助函数和新的批处理大小 ---
    train_embeddings = generate_embeddings_in_batches(
        model,
        train_texts_with_instruction,
        max_length=MAX_LENGTH,
        batch_size=EFFECTIVE_BATCH_SIZE
    )
    
    print("Generating embeddings for test data...")
    # --- MODIFIED: 使用新的辅助函数并修复拼写错误 ---
    test_embeddings = generate_embeddings_in_batches(
        model,
        test_texts_with_instruction,
        max_length=MAX_LENGTH,
        batch_size=EFFECTIVE_BATCH_SIZE # <-- 修复了拼写错误
    )
    
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # --- 4. Normalize Embeddings (在 CPU 上) ---
    print("Normalizing embeddings on CPU...")
    train_embeddings = normalize(train_embeddings, norm='l2', axis=1)
    test_embeddings = normalize(test_embeddings, norm='l2', axis=1)
    # --------------------------------------------------------------------

    # --- 5. Train k-NN Classifier ---
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