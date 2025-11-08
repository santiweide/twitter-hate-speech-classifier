
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    T5EncoderModel,
    T5PreTrainedModel,
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

class AmpleHateModel(T5PreTrainedModel):
    """
    Implementation of the AmpleHate model from the paper[cite: 2],
    using a T5-style encoder as the backbone.
    """
    def __init__(self, config: AutoConfig, lambda_val: float = 1.0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Load the T5 encoder backbone
        self.encoder = T5EncoderModel(config)

        # Per the paper, lambda controls the injection degree [cite: 150, 158]
        self.lambda_val = lambda_val

        # Standard attention layer for Relation Computation [cite: 93]
        # We use batch_first=True and a single head, as the paper implies
        # a simple dot-product attention, not multi-head.
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=1,
            batch_first=True
        )

        # Final classification head [cite: 147]
        self.classifier = nn.Linear(config.d_model, self.num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        explicit_target_mask: torch.LongTensor = None, # Provided by the preprocessor
        labels: torch.LongTensor = None,
        **kwargs,
    ):
        # 1. GET ENCODER HIDDEN STATES
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state

        # 2. GET IMPLICIT TARGETS (h0)
        h0 = hidden_states[:, 0, :]
        h0_q = h0.unsqueeze(1)

        # 3. COMPUTE IMPLICIT RELATION (r_imp)
        # This is computed for every sample in the batch
        r_imp, _ = self.relation_attention(
            query=h0_q,
            key=h0_q,
            value=h0_q
        )
        # Shape: [batch_size, 1, d_model]

        # 4. COMPUTE EXPLICIT RELATION (r_exp) --- PER-SAMPLE FIX ---

        # 4a. Find which samples in the batch have *any* explicit targets
        # Shape: [batch_size]
        sample_has_explicit_target_mask = torch.any(explicit_target_mask, dim=1)

        # 4b. Initialize r_exp as all zeros.
        # Samples with no targets will keep this zero value.
        r_exp = torch.zeros_like(r_imp)

        # 4c. If *at least one* sample has targets, compute attention *only for them*
        if torch.any(sample_has_explicit_target_mask):
            
            # --- Select only the samples that *have* targets ---
            # We'll use these indices to filter all our tensors
            indices = sample_has_explicit_target_mask.nonzero(as_tuple=True)[0]

            # Filter all inputs for the attention call
            h0_q_filtered = h0_q.index_select(0, indices)
            hidden_states_filtered = hidden_states.index_select(0, indices)
            
            # Create the key_padding_mask *only* for the filtered samples
            explicit_target_mask_filtered = explicit_target_mask.index_select(0, indices)
            exp_key_padding_mask_filtered = (explicit_target_mask_filtered == 0)

            # --- Compute attention on the subset ---
            # This is now safe, as every sample here is guaranteed
            # to have at least one `False` in its mask.
            r_exp_filtered, _ = self.relation_attention(
                query=h0_q_filtered,
                key=hidden_states_filtered,
                value=hidden_states_filtered,
                key_padding_mask=exp_key_padding_mask_filtered
            )
            
            # 4d. Scatter the results back into the all-zeros r_exp tensor
            # This puts the computed results into the correct rows
            r_exp.index_copy_(0, indices, r_exp_filtered)

        # --- END OF PER-SAMPLE FIX ---

        # 5. COMBINE RELATIONS
        # r_imp and r_exp both have shape [batch_size, 1, d_model]
        r = r_imp.squeeze(1) + r_exp.squeeze(1)
        # Shape: [batch_size, d_model]

        # 6. DIRECT INJECTION
        z = h0 + (self.lambda_val * r)
        
        # 7. CLASSIFICATION
        logits = self.classifier(z)

        # 8. COMPUTE LOSS
        loss = None
        if labels is not None:
            # --- FIX ---
            # We remove the torch.isnan(logits) check.
            # The robust r_exp logic prevents NaNs,
            # and removing this check fixes the ValueError.
            # -----------
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state

        h0 = hidden_states[:, 0, :]
        h0_q = h0.unsqueeze(1)

        r_imp, _ = self.relation_attention(
            query=h0_q,
            key=h0_q,
            value=h0_q
        )

        if torch.any(explicit_target_mask):
            exp_key_padding_mask = (explicit_target_mask == 0)
            r_exp, _ = self.relation_attention(
                query=h0_q,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=exp_key_padding_mask
            )
        else:
            r_exp = torch.zeros_like(r_imp)

        r = r_imp.squeeze(1) + r_exp.squeeze(1)

        z = h0 + (self.lambda_val * r)
        
        logits = self.classifier(z)

        loss = None
        if labels is not None:
            # Add a check for NaN logits *before* computing loss
            # This helps with debugging if the problem persists.
            if torch.isnan(logits).any():
                print("Warning: NaN detected in logits. Skipping loss calculation.")
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # --- END: STABILITY FIX ---

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

def create_preprocessing_function(tokenizer, ner_pipeline):
    """
    Creates a function to preprocess text data, aligning NER tags
    with the main model's tokenizer.
    """
    def preprocess_function(examples):
        texts = examples["text"]

        # 1. Tokenize for the T5 model
        t5_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_offsets_mapping=True
        )

        # 2. Run NER pipeline on the raw texts
        all_ner_results = ner_pipeline(texts)

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

    T5_MODEL_NAME = "google/mt5-small"

    # NER Tagger from the paper 
    # Using 'dslim/bert-base-NER' as it's a widely used, high-quality NER model.
    NER_MODEL_NAME = "dslim/bert-base-NER"
    # NER_MODEL_NAME = "dslim/bert-large-NER"

    # --- Load Tokenizers and Pipelines ---
    t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)

    # The NER model requires its own tokenizer and model
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)

    # We aggregate to combine B- and I- tags (e.g., B-GPE, I-GPE -> GPE)
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

    dataset = Dataset.from_pandas(df)
    
    dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
    print(f"Dataset created: {dataset}")

    print("Pre-processing data (running NER target identification)...")
    preprocess_fn = create_preprocessing_function(t5_tokenizer, ner_pipeline)
    tokenized_datasets = dataset.map(preprocess_fn, batched=True)

    tokenized_datasets.set_format("torch", columns=[
        "input_ids", "attention_mask", "explicit_target_mask", "labels"
    ])

    print(f"Loading AmpleHate model with {T5_MODEL_NAME} backbone...")
    config = AutoConfig.from_pretrained(T5_MODEL_NAME, num_labels=2)

    model = AmpleHateModel.from_pretrained(
        T5_MODEL_NAME,
        config=config,
        lambda_val=1.0
    )

    data_collator = DataCollatorWithPadding(tokenizer=t5_tokenizer)

    training_args = TrainingArguments(
        output_dir="./amplehate_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=t5_tokenizer,
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