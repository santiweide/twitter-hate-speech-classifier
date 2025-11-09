# Classifiers for Twitter Hate Speech Dataset

## Problem Statement & Model Choice

Automated hate speech detection is characterized by high subjectivity, complex context-dependency, and the pervasive use of implicit or coded language. It progressed from traditional machine learning classifiers trained on lexical features  to early deep learning models like LSTMs and GRUs. The advent of Transformer-based models, particularly BERT and its variants (e.g., RoBERTa, HateBERT), established a long-standing performance baseline. However, the current SOTA is characterized by two trends: (1) the application of Large Language Models (LLMs) like Llama 3 and GPT-4 , and (2) the development of highly specialized, fine-tuned embedding models. 

Here two classifiers are implemented: NER+Bert and Roberta Embedding based model. There is a class imbalance in the training data so weighted class is applied to both fine-tuning process. NaN data is also filtered indicating there is no tweet.

Since only one dataset is provided, while evaluating the model performance, 90% of the data is used for finetuning and 10% is used for evaluating. The random seed is fixed so that the result can be replay-ed.

## Model 1: NER+Bert

Here the data is fed into NER pipeline first, then a Bert model is used for classification.

### Effectiveness and Efficiency

Evaluated on 6,392 samples:
```
Accuracy: 0.9634
F1 (Macro): 0.8664
Recall (Macro): 0.8760
```

The model demonstrates exceptionally high overall accuracy (99.66%) on the test set of 3,197 samples, making only 11 incorrect predictions.

For Efficiency, the NER preprocessing is optimized from sequencial to parallelf rolatency deduction.

### Error Analysis
Here Expected Calibration Error(ECE) is used to see %TODO, and also Slice-based (NER) Analysis is used to see if NER is making sense.

#### ECE
An ECE (Expected Calibration Error) of 0.3795 is extremely high (a perfect score is 0). This indicates that the model's confidence scores are not reliable and do not reflect the true probability of its predictions being correct.

The model is chronically overconfident. When it predicts something with high confidence (e.g., 90-100%), its actual accuracy in that bin is likely much lower, as shown by the high ECE.

#### NER-sliced

The test set is highly imbalanced in this regard. 99.4% of the data (3,178 samples) does not contain a target entity, while only 0.6% (19 samples) does. The model's performance on the tiny slice of data with target entities (has_target_entity = 1) was perfect (1.0 accuracy). 

This implies that all 11 prediction errors occurred on the 3,178 samples without target entities. The model appears to handle text with 'LOC' (Location) and 'ORG' (Organization) entities perfectly, though the sample size (n=19) is too small to draw a definitive conclusion.


#### Generalizability & Robustness 



## Model 2: Roberta

Sequence classification model `cardiffnlp/twitter-roberta-base-hate` is fine tuned on the twitter dataset here. 

### Effectiveness and Efficiency

The classifier demonstrates extremely high overall effectiveness on this dataset, achieving 98.4% accuracy. It is also highly reliable and well-calibrated, meaning its confidence scores are a trustworthy indicator of its correctness (ECE of 0.0152).

However, the error analysis reveals a critical insight: the model's "most confident errors" are almost certainly not model failures. Instead, they appear to be mislabeled examples in the ground-truth dataset. The model is confidently (and likely correctly) classifying these tweets as "non-hate," while the dataset incorrectly labels them as "hate."

Effectiveness:

Evaluated on 6,392 samples:
```
Accuracy: 0.9840
F1 (Macro): 0.9374
Recall (Macro): 0.9172
```

Expected Calibration Error (ECE): 0.0152 An ECE this low (closer to 0 is perfect) is outstanding. It means the model's predicted confidences are highly accurate. For example, when the model is 95% confident in a prediction, it is correct about 95% of the time. The diagram confirms the low ECE. The red line (average confidence) tracks the blue bars (accuracy) almost perfectly, and both are extremely close to the ideal dashed line.


#### Generalizability & Robustness 
The model's ability to generalize to datasets it wasn't trained on is very poor.

This was tested on the 'ethos' (binary) dataset, which represents out-of-distribution (OOD) data. The results show a dramatic drop in performance compared to the in-domain metrics:

Accuracy: Fell from 98.4% to 67.9%

F1 (Macro): Dropped to 66.9%

Calibration (ECE): Worsened significantly from 0.0152 to 0.3078

This high ECE score indicates the model is no longer calibrated on this new data; its confidence scores are highly unreliable. The model is over-confident in its (often incorrect) predictions.

The model appears to be overfitted to its original training data and does not generalize well to different types or phrasings of hate speech, as seen in the 'ethos' dataset.


For robustness, in contrast to its poor generalizability, the model is extremely robust to common text perturbations.

This was tested by applying two modifications to the original dataset:

Lowercase:

Accuracy: 0.9840 (No change)

F1 (Macro): 0.9374 (No change)

ECE: 0.0152 (No change)

Result: Converting all text to lowercase had zero impact on the model's performance or calibration. This is expected from a robust model like RoBERTa.

Typo Injection (5% rate):

Accuracy: 0.9797 (A negligible drop of ~0.4%)

F1 (Macro): 0.9200 (A very slight drop)

ECE: 0.0189 (Remains exceptionally well-calibrated)

Result: Introducing random spelling mistakes had a minimal, almost negligible, effect on the model's accuracy and reliability.

Conclusion: The model is highly resilient to simple noise like case changes and typos, maintaining its high performance and strong calibration.