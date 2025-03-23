import os
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torch.nn.utils.prune as prune
from datasets import load_dataset
from numpy import argmax

# Create output directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the English MMLU dataset
data = load_dataset("cais/mmlu", "all", split="test")

# Updated category mapping with additional keywords
categories_mapping = {
    "STEM": [
        "physics", "chemistry", "biology", "computer science", "computer",
        "math", "engineering", "algebra", "astronomy", "anatomy"
    ],
    "humanities": [
        "history", "philosophy", "law", "logic"
    ],
    "social sciences": [
        "politics", "culture", "economics", "econometrics", "geography", "psychology"
    ],
    "other (business, health, misc.)": [
        "other", "business", "health", "medicine", "clinical"
    ],
}

def get_category_group(subject):
    """
    Map a subject to a category group using substring matching.
    It replaces underscores with spaces and then checks if any keyword
    (from the updated mapping) is a substring of the subject.
    Returns "uncategorized" if no keyword matches.
    """
    subject_mod = subject.replace("_", " ").lower()
    for group, keywords in categories_mapping.items():
        for keyword in keywords:
            if keyword in subject_mod:
                return group
    return "uncategorized"

# Define a function to apply pruning
def apply_pruning(module, amount=0.4):
    print("Applying weight pruning...")
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, torch.nn.Linear):
            prune.l1_unstructured(sub_module, name="weight", amount=amount)
            prune.remove(sub_module, "weight")
    print("Pruning applied successfully.")

# Define a function to load model and tokenizer
def load_model_and_tokenizer(model_name, quantization=False, quantization_bits=8):
    print(f"Loading model: {model_name}...")
    tokenizer_class = LlamaTokenizer if 'llama' in model_name.lower() else AutoTokenizer
    model_class = LlamaForCausalLM if 'llama' in model_name.lower() else AutoModelForCausalLM
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if quantization:
        print(f"Applying {quantization_bits}-bit quantization...")
        if quantization_bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_type=torch.float16)
        else:
            raise ValueError("Invalid quantization_bits value. Only 8 or 4 are supported.")

        model = model_class.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
        print("Quantization applied successfully.")
    else:
        model = model_class.from_pretrained(model_name)

    # Special configuration for Llama models
    if 'llama' in model_name.lower():
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    if not quantization:
        model = model.to(device)
    print("Model loaded successfully.")
    return model, tokenizer

def save_final_results(model_name, accuracy, quantization_bits=None, pruning_amount=None):
    results_file = 'results/final_results.csv'  # updated filename for English MMLU

    # Define columns appropriate for English MMLU results
    columns = [
            "Model Name",
            "Accuracy on English MMLU",
            "Accuracy on English MMLU with 4-bit Quantization",
            "Accuracy on English MMLU with 8-bit Quantization",
            "Accuracy on English MMLU with 20% Pruning",
            "Accuracy on English MMLU with 40% Pruning",
            "Accuracy on English MMLU with 80% Pruning"
    ]

    # Check if file exists; create with headers if it doesn't
    if not os.path.exists(results_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(results_file, index=False)
    else:
        df = pd.read_csv(results_file)

    # Use a consistent identifier for the model (e.g., the last part of the model_name)
    model_identifier = model_name.split("/")[-1]
    
    # Find the corresponding row for the model or add a new one
    if model_identifier in df["Model Name"].values:
        row_index = df.index[df["Model Name"] == model_identifier].tolist()[0]
    else:
        row_index = len(df)
        df.loc[row_index, "Model Name"] = model_identifier

    # Update the appropriate column based on the configuration
    if quantization_bits is None and pruning_amount is None:
        df.loc[row_index, "Full_Precision"] = accuracy
    elif quantization_bits is not None and pruning_amount is None:
        if quantization_bits == 4:
            df.loc[row_index, "Quantized_4bit"] = accuracy
        elif quantization_bits == 8:
            df.loc[row_index, "Quantized_8bit"] = accuracy
    elif pruning_amount is not None and quantization_bits is None:
        if pruning_amount == 0.2:
            df.loc[row_index, "Pruning_20"] = accuracy
        elif pruning_amount == 0.4:
            df.loc[row_index, "Pruning_40"] = accuracy
        elif pruning_amount == 0.8:
            df.loc[row_index, "Pruning_80"] = accuracy

    # Save the updated dataframe
    df.to_csv(results_file, index=False)
    print(f"Final results updated in {results_file}")


# Define a function to evaluate the model and compute category-wise accuracies
def evaluate_model(model_name, quantization=False, quantization_bits=8, pruning=False, pruning_amount=0.4, batch_size=16, debug=False):
    print(f"Starting evaluation for {model_name}...")
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, quantization=quantization, quantization_bits=quantization_bits)

    # Apply pruning if specified
    if pruning:
        apply_pruning(model, amount=pruning_amount)

    correct_count = 0
    results = []
    
    # Initialize category stats with an added key for confidence_sum.
    category_stats = {group: {"correct": 0, "total": 0, "confidence_sum": 0.0} for group in categories_mapping}
    category_stats["uncategorized"] = {"correct": 0, "total": 0, "confidence_sum": 0.0}
    
    # Counter to ensure we print debug info for only the first 10 examples overall.
    debug_counter = 0

    progress_bar = tqdm(
        range(0, len(data), batch_size),
        desc="Processing Examples (Accuracy: 0.00%)",
        position=0,
        leave=True
    )

    # Process examples in batches
    for start_idx in progress_bar:
        end_idx = min(start_idx + batch_size, len(data))
        batch = data.select(range(start_idx, end_idx)).to_pandas().to_dict(orient="records")

        prompts = []
        ground_truths = []
        questions = []
        subjects = []  # store original subject from the data for mapping
        option_keys = ['A', 'B', 'C', 'D', 'E']

        # Prepare batch prompts
        for row in batch:
            choices = list(row['choices'])
            answer_index = row['answer']
            if answer_index >= len(choices):
                continue  # Skip invalid cases

            question = row['question']
            questions.append(question)
            subjects.append(row['subject'])
            options = []
            for idx, option_text in enumerate(choices):
                options.append(f"{option_keys[idx]}. {option_text}")
            prompt = (
                "Choose the correct answer for the following question:\n\n"
                f"Question: {question}\n\n"
                "Options:\n" + "\n".join(options) +
                "\nAnswer:"
            )
            prompts.append(prompt)
            ground_truths.append(option_keys[answer_index])

        if not prompts:
            continue

        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            last_token_logits = outputs.logits[:, -1, :]

            # Compute token ids for the answer options
            choice_ids = []
            for opt in option_keys:
                encoded = tokenizer.encode(" " + opt, add_special_tokens=False)
                if not encoded:
                    raise ValueError(f"Tokenization for option '{opt}' failed.")
                choice_ids.append(encoded[-1])
            choice_ids = choice_ids[:len(option_keys)]
            choice_logits = last_token_logits[:, choice_ids]

            probs = F.softmax(choice_logits, dim=-1).tolist()
            predictions = [option_keys[argmax(probs[i])] for i in range(len(probs))]

        # Print debug info for the first 10 examples overall only.
        if debug:
            for idx in range(len(prompts)):
                if debug_counter >= 10:
                    break
                print(f"\nIteration {debug_counter + 1}")
                print(f"Prompt:\n{prompts[idx]}")
                print(f"Raw Logits: {last_token_logits[idx].tolist()}")
                print(f"Choice Logits: {choice_logits[idx].tolist()}")
                print(f"Probabilities: {probs[idx]}")
                print(f"Prediction: {predictions[idx]}")
                debug_counter += 1

        # Process results for the current batch
        for i, predicted_answer in enumerate(predictions):
            is_correct = predicted_answer == ground_truths[i]
            confidence = max(probs[i])
            results.append({
                'Question': questions[i],
                'Subject': subjects[i],
                'Predicted Answer': predicted_answer,
                'Ground Truth': ground_truths[i],
                'Confidence': confidence,
                'Correct': is_correct
            })
            correct_count += 1 if is_correct else 0

            # Map subject to the corresponding group and update stats (including confidence)
            group = get_category_group(subjects[i])
            if group not in category_stats:
                category_stats[group] = {"correct": 0, "total": 0, "confidence_sum": 0.0}
            category_stats[group]["total"] += 1
            category_stats[group]["confidence_sum"] += confidence
            if is_correct:
                category_stats[group]["correct"] += 1

        current_examples = len(results)
        current_accuracy = (correct_count / current_examples * 100) if current_examples > 0 else 0.0
        progress_bar.set_description(f"Processing Examples (Accuracy: {current_accuracy:.2f}%)")

    final_accuracy = (correct_count / len(results) * 100) if results else 0.0
    print(f"\nFinal Accuracy: {final_accuracy:.2f}%")

    # Create identifiers for filenames based on model name and evaluation settings.
    model_identifier = model_name.split("/")[-1]
    if pruning:
        pruning_status = f"pruned_{pruning_amount}"
    else:
        pruning_status = "unpruned"
    quantization_status = f"quantized_{quantization_bits}bit" if quantization else "full_precision"

    results_csv_filename = f"results/evaluation_results_{model_identifier}_{pruning_status}_{quantization_status}.csv"
    category_csv_filename = f"results/category_accuracy_{model_identifier}_{pruning_status}_{quantization_status}.csv"

    # Save overall results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv_filename, index=False)
    print(f"Overall results saved to {results_csv_filename}")

    # Compute category-wise accuracy and average confidence, then save to a separate CSV file
    category_data = []
    for group, stats in category_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        avg_confidence = (stats["confidence_sum"] / total) if total > 0 else 0.0
        category_data.append({
            "Category Group": group,
            "Total": total,
            "Correct": correct,
            "Accuracy": accuracy,
            "Average Confidence": avg_confidence
        })

    df_category = pd.DataFrame(category_data)
    df_category.to_csv(category_csv_filename, index=False)
    print(f"Category-wise accuracies saved to {category_csv_filename}")

    # Update the final results CSV that aggregates evaluations across runs.
    update_final_results(model_identifier, final_accuracy, quantization, quantization_bits, pruning, pruning_amount)

