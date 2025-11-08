# Classifiers for Twitter Hate Speech Dataset

## Problem Statement & Model Choice

Automated hate speech detection is characterized by high subjectivity, complex context-dependency, and the pervasive use of implicit or coded language. It progressed from traditional machine learning classifiers trained on lexical features  to early deep learning models like LSTMs and GRUs. The advent of Transformer-based models, particularly BERT and its variants (e.g., RoBERTa, HateBERT), established a long-standing performance baseline. However, the current SOTA is characterized by two trends: (1) the application of Large Language Models (LLMs) like Llama 3 and GPT-4 , and (2) the development of highly specialized, fine-tuned embedding models. 

Here two classifiers are implemented: NER+Bert and 


1) Specializing General-Purpose Embeddings with LLm

source: Specializing General-purpose LLM Embeddings for Implicit Hate Speech Detection

The most significant recent breakthrough involves adapting SOTA generalist embedding models specifically for the IHS domain. A 2025 paper, "Specializing General-purpose LLM Embeddings for Implicit Hate Speech Detection," demonstrates a new SOTA by fine-tuning models like NV-Embed, Stella, and E5.   

The NV-Embed model, an ICLR 2025 paper, is a critical component. It established its own SOTA on the general-purpose Massive Text Embedding Benchmark (MTEB), setting record-high scores across 56 tasks.

2) Contrastive Learning in Embedding Space

source: AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection

AmpleHate focuses on "teaching" models the fine-grained difference between hateful and non-hateful content, particularly when they are lexically similar (e.g., implicit hate vs. neutral speech).



## Effectiveness & Efficiency

## Rigorous Evaluation
* cross-dataset evaluation


## Generalizability & Robustness 

## Synthesis & Future Trajectories



