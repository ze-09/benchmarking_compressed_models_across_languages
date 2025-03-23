import os
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
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

# Load the Hindi MMLU dataset (Kannada ARC-C from Indic-Benchmark)
dataset = load_dataset("Indic-Benchmark/kannada-arc-c-2.5k")
data = pd.DataFrame(dataset['train'])

def apply_pruning(module, amount=0.4):
    print("Applying weight pruning...")
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, torch.nn.Linear):
            prune.l1_unstructured(sub_module, name="weight", amount=amount)  # Prune specified percentage of weights
            prune.remove(sub_module, "weight")  # Remove the pruning mask to free up memory
    print("Pruning applied successfully.")


def load_model_and_tokenizer(model_name, quantization=False, quantization_bits=8):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  

    model_class = AutoModelForCausalLM if 'llama' not in model_name.lower() else LlamaForCausalLM
    
    if quantization:
        print(f"Applying {quantization_bits}-bit quantization...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=(quantization_bits == 8), load_in_4bit=(quantization_bits == 4))
        model = model_class.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    else:
        model = model_class.from_pretrained(model_name).to(device)

    # Special configuration for LLaMA models & AceGPT
    if 'llama' in model_name.lower() or 'ace' in model_name.lower():
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        })
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<unk>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    print("Model loaded successfully.")
    return model, tokenizer

def evaluate_model(model_name, quantization=False, quantization_bits=8, pruning=False, pruning_amount=0.4, batch_size=16):
    print(f"Starting evaluation for {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, quantization=quantization, quantization_bits=quantization_bits)

    if pruning:
        apply_pruning(model, amount=pruning_amount)
    
    correct_count = 0
    results = []
    progress_bar = tqdm(range(0, len(data), batch_size), desc="Processing Examples", position=0, leave=True)
    
    printed_count = 0  # Counter to track printed examples

    for start_idx in progress_bar:
        end_idx = min(start_idx + batch_size, len(data))
        batch = data.iloc[start_idx:end_idx]

        prompts, ground_truths = [], []
        option_keys = ['A', 'B', 'C', 'D']
        

        for row in batch.iterrows():
            question_data = row[1]['question']
            question_text = question_data['stem']  # Extract the actual question
            choices = {ch['label']: ch['text'] for ch in question_data['choices']}  # Extract options
            answer_key = row[1]['answerKey']  # Correct answer label

            if answer_key not in choices:
                continue
            
            options_text = "\n".join([f"{k}. {v}" for k, v in choices.items() if k in option_keys])
            prompt = f"ಕೆಳಗಿನ ಪ್ರಶ್ನೆಗೆ ಸರಿಯಾದ ಉತ್ತರವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ:\n\nಪ್ರಶ್ನೆ: {question_text}\n\nಆಯ್ಕೆಗಳು:\n{options_text}\n\nಉತ್ತರ:" 
            prompts.append(prompt)
            ground_truths.append(answer_key)

        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]
            choice_logits = torch.stack(
                [last_token_logits[:, tokenizer.encode(opt, add_special_tokens=True)[-1]] for opt in option_keys], dim=1
            ).to(device)
            probs = F.softmax(choice_logits, dim=-1).tolist()
            log_probs = F.log_softmax(choice_logits, dim=-1).tolist()  # Compute log probabilities
            predictions = [option_keys[argmax(probs[i])] for i in range(len(probs))]
        
        for i, predicted_answer in enumerate(predictions):
            # Only print details for the first 10 questions overall
            if printed_count < 10:
                print("Prompt:", prompts[i])
                print("Log probabilities:", log_probs[i])
                print("Prediction:", predicted_answer.strip())
                print("Ground Truth:", ground_truths[i])
                print("-----")
                printed_count += 1

            is_correct = predicted_answer.strip().upper() == ground_truths[i].strip().upper()
            results.append({
                'Prompt': prompts[i],
                'Predicted Answer': predicted_answer.strip(),
                'Ground Truth': ground_truths[i],
                'Confidence': max(probs[i])
            })
            correct_count += int(is_correct)
        
    accuracy = (correct_count / len(data)) * 100
    print(f"Final Accuracy for {model_name}: {accuracy:.2f}%")

    # Determine configuration description for the evaluation
    if not quantization and not pruning:
        config = "Base Accuracy"
    elif quantization:
        config = f"{quantization_bits}-bit Quantization"
    elif pruning:
        config = f"{int(pruning_amount * 100)}% Pruning"
    else:
        config = "Custom Configuration"

    # Determine filename based on applied options for individual model results
    config_suffix = ""
    if quantization:
        config_suffix += f"_{quantization_bits}bitquantization"
    if pruning:
        config_suffix += f"_pruning{pruning_amount}"
    file_name = f"results/{model_name.replace('/', '_')}{config_suffix}.csv"
    
    # Save individual model results
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_name, index=False)
    
    # Append summary result to CSV file so that all evaluations are preserved
    summary_file_path = "results/model_evaluation_summary.csv"
    file_exists = os.path.exists(summary_file_path)
    with open(summary_file_path, "a", newline="") as summary_file:
        writer = csv.writer(summary_file)
        # Write header if the file does not exist or is empty
        if not file_exists or os.stat(summary_file_path).st_size == 0:
            writer.writerow(["Model", "Configuration", "Accuracy"])
        writer.writerow([model_name, config, accuracy])

