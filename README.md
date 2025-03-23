# benchmarking_compressed_models_across_languages

# Overview
This repository contains three scripts designed for evaluating language models on different versions of the Massive Multitask Language Understanding (MMLU) dataset:
	•	arabicmmlu.py: Evaluates Arabic language models using the ArabicMMLU dataset.
	•	englishmmlu.py: Evaluates English language models using the English MMLU dataset.
	•	indicmmlu.py: Evaluates Indic language models using the Hindi MMLU dataset (Kannada ARC-C from Indic-Benchmark).
 
Each script loads the corresponding dataset, processes the test data, and evaluates model performance using causal language modeling.

# Requirements
Ensure you have the following dependencies installed before running the scripts:
pip install transformers torch pandas tqdm datasets

# Usage
Each script follows a similar workflow:
	1	Loads the dataset.
	2	Initializes a transformer-based language model.
	3	Processes and evaluates model performance.
	4	Saves the evaluation results to the results/ directory.

# Running a Script
Execute the scripts using Python or jupyter lab

# Model and Dataset Details
	•	The ArabicMMLU dataset is loaded from a CSV file (data/ArabicMMLU.csv).
	•	The English MMLU dataset is fetched via the Hugging Face datasets library (cais/mmlu).
	•	The Indic MMLU dataset (Kannada ARC-C) is retrieved from Indic-Benchmark/kannada-arc-c-2.5k.
	•	The scripts use transformer models (AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, etc.) to generate predictions.

# Output
Evaluation results are saved in the results/ directory. Each script generates a report with accuracy metrics and other relevant evaluation outputs.

# Notes
	•	The scripts are optimized for GPU usage but will fall back to CPU if a GPU is unavailable.
	•	Ensure the required datasets are accessible before execution.
	•	Modify the scripts to use different models or datasets as needed.
