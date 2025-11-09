import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score

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