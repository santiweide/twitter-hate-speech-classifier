import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

class AmpleHateModel(BertPreTrainedModel):
    """
    Implementation of the AmpleHate model from the paper,
    using a BERT-style encoder as the backbone.
    """
    def __init__(self, config: AutoConfig, lambda_val: float = 1.0, class_weights: torch.Tensor = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False) 

        self.lambda_val = lambda_val
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.relation_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        if class_weights is not None:
            self.register_buffer('class_weights_buffer', class_weights)
        else:
            self.register_buffer('class_weights_buffer', None)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        explicit_target_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        **kwargs,
    ):
        encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state

        # 1. GET h0 (The [CLS] token)
        h0 = hidden_states[:, 0, :]
        h0_q = h0.unsqueeze(1) # Shape: [B, 1, H]

        pad_mask = (attention_mask == 0)

        r_imp, _ = self.relation_attention(
            query=h0_q,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=pad_mask 
        )
        
        # 3. COMPUTE EXPLICIT RELATION (r_exp)
        sample_has_explicit_target_mask = torch.any(explicit_target_mask, dim=1)
        
        # r_exp starts as zeros, shape [B, 1, H]. 
        r_exp = torch.zeros_like(r_imp) 

        if torch.any(sample_has_explicit_target_mask):
            indices = sample_has_explicit_target_mask.nonzero(as_tuple=True)[0]

            h0_q_filtered = h0_q.index_select(0, indices)
            hidden_states_filtered = hidden_states.index_select(0, indices)
            
            attention_mask_filtered = attention_mask.index_select(0, indices)
            pad_mask_filtered = (attention_mask_filtered == 0)
            
            explicit_target_mask_filtered = explicit_target_mask.index_select(0, indices)
            non_target_mask_filtered = (explicit_target_mask_filtered == 0)

            final_key_padding_mask = pad_mask_filtered | non_target_mask_filtered

            r_exp_filtered, _ = self.relation_attention(
                query=h0_q_filtered,
                key=hidden_states_filtered,
                value=hidden_states_filtered,
                key_padding_mask=final_key_padding_mask 
            )
            r_exp.index_copy_(0, indices, r_exp_filtered.to(r_exp.dtype))

        r_unnormalized = r_imp.squeeze(1) + r_exp.squeeze(1)

        r_normalized = self.relation_norm(r_unnormalized)

        z = h0 + (self.lambda_val * r_normalized)
        
        z = self.layer_norm(z)
        
        logits = self.classifier(z)

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

def create_preprocessing_function(tokenizer):
    """
    Creates a function to preprocess text data, aligning NER tags
    with the main model's tokenizer.
    ASSUMES the 'ner_results' column already exists.
    """
    def preprocess_function(examples):
        texts = examples["text"]

        bert_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_offsets_mapping=True
        )

        # 2. Get pre-computed NER results from the column
        all_ner_results = examples["ner_results"] # <--- READS FROM COLUMN

        batch_explicit_masks = []

        for i in range(len(texts)):
            offsets = bert_inputs["offset_mapping"][i]
            ner_results = all_ner_results[i] if isinstance(all_ner_results[i], list) else [all_ner_results[i]]

            explicit_mask = [0] * len(offsets)

            target_entities = [
                e for e in ner_results
                if e['entity_group'] in TARGET_NER_LABELS
            ]

            for entity in target_entities:
                ent_start, ent_end = entity["start"], entity["end"]

                for token_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == tok_end:
                        continue

                    if max(tok_start, ent_start) < min(tok_end, ent_end):
                        explicit_mask[token_idx] = 1

            batch_explicit_masks.append(explicit_mask)

        bert_inputs["explicit_target_mask"] = batch_explicit_masks
        bert_inputs["labels"] = examples["label"]
        return bert_inputs

    return preprocess_function
