# Classifiers for Twitter Hate Speech Dataset

## Problem Statement & Model Choice

Automated hate speech detection is characterized by high subjectivity, complex context-dependency, and the pervasive use of implicit or coded language. It progressed from traditional machine learning classifiers trained on lexical features  to early deep learning models like LSTMs and GRUs. The advent of Transformer-based models, particularly BERT and its variants (e.g., RoBERTa, HateBERT), established a long-standing performance baseline. However, the current SOTA is characterized by two trends: (1) the application of Large Language Models (LLMs) like Llama 3 and GPT-4 , and (2) the development of highly specialized, fine-tuned embedding models. 

Here TODO classifiers are implemented:

1) Specializing General-Purpose Embeddings with LLm

source: Specializing General-purpose LLM Embeddings for Implicit Hate Speech Detection

2) Contrastive Learning in Embedding Space

source: AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection

AmpleHate focuses on "teaching" models the fine-grained difference between hateful and non-hateful content, particularly when they are lexically similar (e.g., implicit hate vs. neutral speech).

3) Counterfactual Data Augmentation (CDA) - 代码太复杂了，算了

This section focus on generating high-quality training data. Counterfactual Data Augmentation (CDA) aims to create minimal-pair examples (e.g., a hateful tweet and its non-hateful variant) to teach models the specific boundaries of hate.

Here we implement GENES, which uniquely balances two competing goals: attribute alignment (ensuring the generated text is truly non-hateful) and semantic preservation (ensuring the original meaning is not distorted).

source: Gradient-Guided Importance Sampling for Learning Binary Energy-Based Models

TODO, implement:
Using a relatively small Flan-T5-Large model—achieved the best Macro F1-score in two of three test sets. Notably, it exceeded the performance of prompt-based CDA methods that used the much larger GPT-4o-mini. This positions GENES as a new SOTA "lightweight and open-source alternative" that excels in both effectiveness and efficiency.


4) Multilingual SOTA

5) 


## Effectiveness & Efficiency

## Rigorous Evaluation
* cross-dataset evaluation


## Generalizability & Robustness 

## Synthesis & Future Trajectories



