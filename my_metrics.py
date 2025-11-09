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



def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    
    accuracies = (preds == labels).astype(float)
    
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        
        if i == 0:
             mask = (confidences >= lo) & (confidences <= hi)
        else:
             mask = (confidences > lo) & (confidences <= hi)

        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_w = mask.mean()
            
            ece += np.abs(bin_acc - bin_conf) * bin_w
            
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_sizes.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_sizes.append(0)
            
    return ece, (bins, np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes))
