import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
    set_seed,
)
from datasets import Dataset
import kagglehub
from tqdm import tqdm

from model import AmpleHateModel, create_preprocessing_function, TARGET_NER_LABELS


TRAINING_SEED = 42
set_seed(TRAINING_SEED)

def run_inference_and_analysis():
    print("--- 🚀 开始推理和错误分析 ---")
    
    # --- 1. 定义路径和模型名称 ---
    CHECKPOINT_PATH = "./amplehate_results/checkpoint-10788"
    BASE_MODEL_NAME = "bert-base-cased"
    NER_MODEL_NAME = "dslim/bert-base-NER"

    print(f"加载模型: {CHECKPOINT_PATH}")
    print(f"基础 Tokenizer: {BASE_MODEL_NAME}")

    # --- 2. 加载训练好的模型和 Tokenizer ---
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # 关键：这现在可以正常工作，因为 AmpleHateModel 已经从 model.py 导入
    model = AmpleHateModel.from_pretrained(CHECKPOINT_PATH)
    
    data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)

    training_args = TrainingArguments(
        output_dir="./amplehate_results",
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        fp16=False,
        warmup_ratio=0.1,
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


if __name__ == "__main__":
    run_inference_and_analysis()