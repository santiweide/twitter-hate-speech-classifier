# Classifiers for Twitter Hate Speech Dataset

## Problem Statement & Model Choice

Automated hate speech detection is characterized by high subjectivity, complex context-dependency, and the pervasive use of implicit or coded language. It progressed from traditional machine learning classifiers trained on lexical features  to early deep learning models like LSTMs and GRUs. The advent of Transformer-based models, particularly BERT and its variants (e.g., RoBERTa, HateBERT), established a long-standing performance baseline. However, the current SOTA is characterized by two trends: (1) the application of Large Language Models (LLMs) like Llama 3 and GPT-4 , and (2) the development of highly specialized, fine-tuned embedding models. 

Here two classifiers are implemented: NER+Bert and Roberta Embedding based model. There is a class imbalance in the training data so weighted class is applied to both fine-tuning process. NaN data is also filtered indicating there is no label or tweet.



## NER+Bert

### Implement

### Effectiveness and Efficiency
The model demonstrates exceptionally high overall accuracy (99.66%) on the test set of 3,197 samples, making only 11 incorrect predictions.

### Error Analysis
Here Expected Calibration Error(ECE) is used to see %TODO, and also Slice-based (NER) Analysis is used to see if NER is making sense.

#### ECE
An ECE (Expected Calibration Error) of 0.3795 is extremely high (a perfect score is 0). This indicates that the model's confidence scores are not reliable and do not reflect the true probability of its predictions being correct.

The model is chronically overconfident. When it predicts something with high confidence (e.g., 90-100%), its actual accuracy in that bin is likely much lower, as shown by the high ECE.

#### NER-sliced

The test set is highly imbalanced in this regard. 99.4% of the data (3,178 samples) does not contain a target entity, while only 0.6% (19 samples) does. The model's performance on the tiny slice of data with target entities (has_target_entity = 1) was perfect (1.0 accuracy). 

This implies that all 11 prediction errors occurred on the 3,178 samples without target entities. The model appears to handle text with 'LOC' (Location) and 'ORG' (Organization) entities perfectly, though the sample size (n=19) is too small to draw a definitive conclusion.


#### Generalizability & Robustness 




