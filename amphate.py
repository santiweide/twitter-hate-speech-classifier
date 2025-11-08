
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


# ---------------------------------------------------------------------------
# 1. The AmpleHate Model Architecture
# ---------------------------------------------------------------------------

class AmpleHateModel(BertPreTrainedModel):
    """
    Implementation of the AmpleHate model from the paper,
    using a BERT-style encoder as the backbone.
    """
    def __init__(self, config: AutoConfig, lambda_val: float = 1.0, class_weights: torch.Tensor = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Load the BERT encoder backbone 
        self.encoder = BertModel(config, add_pooling_layer=False) 

        # ... (lambda_val, relation_attention, classifier 保持不变) ...
        self.lambda_val = lambda_val
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, # 对 BERT 来说 d_model 通常叫 hidden_size
            num_heads=1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        if class_weights is not None:
            self.register_buffer('class_weights_buffer', class_weights)
        else:
            self.register_buffer('class_weights_buffer', None)        # Initialize weights
        # self.post_init()  会强制使用 BERT 的 std=0.02 初始化来重写你的 LayerNorm 和 relation_attention 的权重

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        explicit_target_mask: torch.LongTensor = None, # Provided by the preprocessor
        labels: torch.LongTensor = None,
        **kwargs,
    ):
        # 1. ... (encoder_outputs, hidden_states) ...
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state

        # 2. ... (h0, h0_q) ...
        h0 = hidden_states[:, 0, :]
        h0_q = h0.unsqueeze(1)

        # 3. ... (r_imp) ...
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            r_imp, _ = self.relation_attention(
                query=h0_q,
                key=h0_q,
                value=h0_q
            )
        
        # 4. COMPUTE EXPLICIT RELATION (r_exp)
        sample_has_explicit_target_mask = torch.any(explicit_target_mask, dim=1)
        r_exp = torch.zeros_like(r_imp)

        if torch.any(sample_has_explicit_target_mask):
            indices = sample_has_explicit_target_mask.nonzero(as_tuple=True)[0]

            h0_q_filtered = h0_q.index_select(0, indices)
            hidden_states_filtered = hidden_states.index_select(0, indices)
            
            # --- 🌟 FIX 1: Get the original padding mask for the filtered samples ---
            attention_mask_filtered = attention_mask.index_select(0, indices)
            # pad_mask_filtered is True for [PAD] tokens
            pad_mask_filtered = (attention_mask_filtered == 0)
            
            # Get the explicit target mask
            explicit_target_mask_filtered = explicit_target_mask.index_select(0, indices)
            # non_target_mask_filtered is True for non-target tokens
            non_target_mask_filtered = (explicit_target_mask_filtered == 0)

            # --- 🌟 FIX 2: Combine the masks ---
            # We want to ignore a token if it is a [PAD] OR if it's not a target
            final_key_padding_mask = pad_mask_filtered | non_target_mask_filtered
            # -----------------------------------------------------------------

            with torch.autocast(device_type='cuda', dtype=torch.float32):
                r_exp_filtered, _ = self.relation_attention(
                    query=h0_q_filtered,
                    key=hidden_states_filtered,
                    value=hidden_states_filtered,
                    key_padding_mask=final_key_padding_mask 
                )
            r_exp.index_copy_(0, indices, r_exp_filtered)

        # 5. COMBINE RELATIONS
        r = r_imp.squeeze(1) + r_exp.squeeze(1)

        # 6. DIRECT INJECTION
        z = h0 + (self.lambda_val * r)
        
        # 7. STABILIZE (Keep this from last time)
        z = self.layer_norm(z)
        
        # 8. CLASSIFICATION
        logits = self.classifier(z)

        # 9. COMPUTE LOSS (was 8)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_buffer, label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# ---------------------------------------------------------------------------
# 2. Data Pre-processing (Implements "Target Identification")
# ---------------------------------------------------------------------------

# Define the NER labels AmpleHate considers "explicit targets"
TARGET_NER_LABELS = {"ORG", "NORP", "GPE", "LOC", "EVENT"}

# --- 🌟 MODIFIED FUNCTION 🌟 ---
def create_preprocessing_function(tokenizer):
    """
    Creates a function to preprocess text data, aligning NER tags
    with the main model's tokenizer.
    ASSUMES the 'ner_results' column already exists.
    """
    def preprocess_function(examples):
        texts = examples["text"]

        # 1. Tokenize for the main model
        t5_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_offsets_mapping=True
        )

        # 2. Get pre-computed NER results from the column
        all_ner_results = examples["ner_results"] # <--- READS FROM COLUMN

        batch_explicit_masks = []

        # 3. Align NER results with T5 tokens
        for i in range(len(texts)):
            offsets = t5_inputs["offset_mapping"][i]
            # Handle cases where ner_pipeline might return a single dict instead of a list
            ner_results = all_ner_results[i] if isinstance(all_ner_results[i], list) else [all_ner_results[i]]

            explicit_mask = [0] * len(offsets)

            # Get all target entities for this sentence
            target_entities = [
                e for e in ner_results
                # --- THIS IS THE FIX ---
                # Changed 'entity' to 'entity_group' and removed .split()
                if e['entity_group'] in TARGET_NER_LABELS
                # -----------------------
            ]

            for entity in target_entities:
                ent_start, ent_end = entity["start"], entity["end"]

                # Find all tokens that overlap with this entity
                for token_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == tok_end:
                        continue

                    if max(tok_start, ent_start) < min(tok_end, ent_end):
                        explicit_mask[token_idx] = 1

            batch_explicit_masks.append(explicit_mask)

        t5_inputs["explicit_target_mask"] = batch_explicit_masks
        t5_inputs["labels"] = examples["label"]
        return t5_inputs

    return preprocess_function

# ---------------------------------------------------------------------------
# 3. Metrics Calculation
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 4. Main Training Script
# ---------------------------------------------------------------------------

def main():
    print("Setting up models and tokenizers...")

    BASE_MODEL_NAME = "bert-base-cased" # (论文使用 BERT-base )
    
    # NER Tagger from the paper 
    # Using 'dslim/bert-base-NER' as it's a widely used, high-quality NER model.
    NER_MODEL_NAME = "dslim/bert-base-NER"
    # NER_MODEL_NAME = "dslim/bert-large-NER"

    # --- Load Tokenizers and Pipelines ---
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
    
    # 2. Calculate weights (inversely proportional to frequency)
    #    weights = num_samples / (num_classes * class_counts)
    num_classes = len(class_counts)
    weights = num_samples / (num_classes * class_counts)
    
    # 3. Sort weights by label (0, then 1) and convert to a tensor
    weights = weights.sort_index()
    class_weights = torch.tensor(weights.values, dtype=torch.float32)
    print(f"Class Weights: {class_weights}") # You will see the 'hate' class has a much higher weight

    dataset = Dataset.from_pandas(df)
    
    dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
    print(f"Dataset created: {dataset}")

    # --- 🌟 NEW EFFICIENCY STEP: PRE-COMPUTE ALL NER 🌟 ---
    print("Pre-computing NER results for TRAIN set... (This will be cached by 'datasets')")
    # 1. Get all texts from the train split
    train_texts = dataset["train"]["text"]
    # 2. Run the pipeline on all texts at once for max GPU efficiency
    #    (Adjust batch_size based on your GPU VRAM)
    train_ner_results = ner_pipeline(train_texts, batch_size=64) 
    # 3. Add these results as a new column
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
        lambda_val=1.0,
        class_weights=class_weights,
    )
    data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)

    training_args = TrainingArguments(
        output_dir="./amplehate_results",
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=2e-5, # <--- 关键：匹配论文 
        max_grad_norm=1.0,  # <--- 关键：添加梯度裁剪 (Gradient Clipping)
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

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 (Macro): {eval_results['eval_f1']:.4f}")
    print(f"Recall (Macro): {eval_results['eval_recall']:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()