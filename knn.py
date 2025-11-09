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
import kagglehub  # 导入 kagglehub
import sys        # 导入 sys 以便在文件未找到时退出

# --- 1. 从你的代码中加载预处理函数和模型 ---

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task = 'hate'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 加载模型
# 你的数据集标签是 0 (non-hate) 和 1 (hate)，所以 num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# --- 2. 加载你的 Kaggle 数据集并计算 Class Weights ---
# (使用你提供的 kagglehub 代码)

print("Downloading dataset from Kaggle Hub...")
path = kagglehub.dataset_download("vkrahul/twitter-hate-speech")
print(f"Dataset downloaded to: {path}")

data_file_path = Path(path) / "train_E6oV3lV.csv"

if not data_file_path.exists():
    # 注意：在你的原始代码中，这里有一个 'return'。
    # 因为这不在一个函数中，所以我将其改为 'sys.exit()' 来停止脚本执行。
    print(f"Error: Could not find 'train_E6oV3lV.csv' in {path}")
    sys.exit(1) # 停止脚本

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

weights = weights.sort_index() # 确保索引按 0, 1 排序
class_weights = torch.tensor(weights.values, dtype=torch.float32)
print(f"Class Weights (for labels {weights.index.values}): {class_weights}")

# 转换为 Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# --- 3. Tokenize 数据集 ---
def tokenize_function(examples):
    # 首先应用你的 preprocess 函数
    processed_texts = [preprocess(t) for t in examples['text']]
    # 然后进行 tokenization
    return tokenizer(processed_texts, padding="max_length", truncation=True, max_length=128)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 重命名 'label' 列为 'labels' 以匹配模型期望
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 划分训练集和验证集
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- 4. 自定义 Trainer 以使用 Class Weights ---
# (这是实现 "注意 class_weight 的使用" 的关键)

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # 从输入中获取 'labels'
        labels = inputs.pop("labels")
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 准备 CrossEntropyLoss
        # 将 class_weights 移动到模型所在的 device
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        
        # 计算损失
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- 5. 设置训练参数并开始训练 ---
training_args = TrainingArguments(
    output_dir="./results",               # 输出目录
    num_train_epochs=3,                 # 训练轮数
    per_device_train_batch_size=4,      # 训练 batch size
    per_device_eval_batch_size=4,       # 验证 batch size
    logging_dir='./logs',               # 日志目录
    logging_steps=50,                   # 增加日志记录的步数 (因为数据集更大了)
    evaluation_strategy="epoch",        # 每轮结束后进行验证
    save_strategy="epoch",              # 每轮结束后保存模型
    load_best_model_at_end=True,        # 训练结束后加载最佳模型
    remove_unused_columns=False,        # 必需，因为我们保留了 'labels'
)

# 实例化我们的 WeightedTrainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    class_weights=class_weights  # !! 在这里传入我们的权重 !!
)

print("Starting training...")
trainer.train()
print("Training complete.")

# --- 6. 使用微调后的模型进行推理 ---
print("\n--- Inference Example (using fine-tuned model) ---")

# 使用你原始代码中的示例文本
text = "Good night 😊"
preprocessed_text = preprocess(text)
print(f"Original: '{text}' -> Preprocessed: '{preprocessed_text}'")

encoded_input = tokenizer(preprocessed_text, return_tensors='pt')
# 将输入移动到模型所在的 device
encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}

output = model(**encoded_input)
scores = output[0][0].detach().cpu().numpy()
scores = softmax(scores)

# 你的 Kaggle 数据集标签: 0: non-hate, 1: hate
labels = ['non-hate', 'hate']

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

# 测试一个仇恨言论的例子
text_hate = "You are a terrible person."
preprocessed_text_hate = preprocess(text_hate)
print(f"\nOriginal: '{text_hate}' -> Preprocessed: '{preprocessed_text_hate}'")

encoded_input = tokenizer(preprocessed_text_hate, return_tensors='pt')
encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}

output = model(**encoded_input)
scores = output[0][0].detach().cpu().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")