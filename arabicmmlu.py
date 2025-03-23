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

# Load the ArabicMMLU dataset
file_path = 'data/ArabicMMLU.csv'
data = pd.read_csv(file_path)

# Define mappings for Arabic translation
level_ar = {
    'Primary': 'للمرحلة الابتدائية',
    'Middle': 'للمرحلة المتوسطة',
    'High': 'للمرحلة الثانوية',
    'Univ': 'للجامعات',
    'Prof': 'للمحترفين'
}

country_ar = {
    'UAE': 'في دولة الإمارات العربية المتحدة',
    'Egypt': 'في مصر',
    'Lebanon': 'في لبنان',
    'Jordan': 'في الأردن',
    'Kuwait': 'في الكويت',
    'KSA': 'في المملكة العربية السعودية',
    'Palestine': 'في فلسطين',
    'Morocco': 'في المغرب',
}

subject_ar = {
    'Islamic Studies': 'في الدراسات الإسلامية',
    'Driving Test': 'اختبار القيادة', 
    'Natural Science': 'في العلوم الطبيعية',
    'History': 'في التاريخ',
    'General Knowledge': 'في المعلومات العامة',
    'Law': 'في القانون', 
    'Physics': 'في الفيزياء', 
    'Social Science': 'في العلوم الاجتماعية',
    'Management': 'في الإدارة', 
    'Arabic Language': 'في اللغة العربية',
    'Political Science': ' في العلوم السياسية', 
    'Philosophy': 'في الفلسفة',
    'Accounting': 'في المحاسبة',
    'Computer Science': 'في علوم الحاسوب',
    'Geography': 'في الجغرافيا',
    'Math': 'في الرياضيات',
    'Biology': 'في علم الأحياء',
    'Economics': 'في الاقتصاد',
    'Arabic Language (General)': 'في اللغة العربية (عام)',
    'Arabic Language (Grammar)': 'في اللغة العربية (قواعد النحو)',
    'Civics': 'في التربية المدنية',
}


# Define a function to apply pruning
def apply_pruning(module, amount=0.4):
    print("Applying weight pruning...")
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, torch.nn.Linear):
            prune.l1_unstructured(sub_module, name="weight", amount=amount)  # Prune specified percentage of weights
            prune.remove(sub_module, "weight")  # Remove the pruning mask to free up memory
    print("Pruning applied successfully.")


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
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            raise ValueError("Invalid quantization_bits value. Only 8 or 4 are supported.")

        model = model_class.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )
    else:
        model = model_class.from_pretrained(model_name)
    
    # Special tokens configuration for llama and AceGPT
    if 'llama' in model_name.lower() or 'ace' in model_name.lower():
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        })
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<unk>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()  # Set to evaluation mode
    print("Model loaded successfully.")
    return model, tokenizer

##################################################################################

def save_final_results(model_name, accuracy, quantization_bits=None, pruning_amount=None):
    results_file = 'results/final_model_accuracies.csv'

    # Check if file exists
    if not os.path.exists(results_file):
        # Create file with headers if it doesn't exist
        df = pd.DataFrame(columns=[
            "Model Name",
            "Accuracy on Arabic MMLU",
            "Accuracy on Arabic MMLU with 4-bit Quantization",
            "Accuracy on Arabic MMLU with 8-bit Quantization",
            "Accuracy on Arabic MMLU with 20% Pruning",
            "Accuracy on Arabic MMLU with 40% Pruning",
            "Accuracy on Arabic MMLU with 80% Pruning"
        ])
        df.to_csv(results_file, index=False)
    else:
        # Load existing file
        df = pd.read_csv(results_file)

    # Find the corresponding row for the model
    if model_name in df["Model"].values:
        row_index = df.index[df["Model"] == model_name].tolist()[0]
    else:
        # Add a new row for the model
        row_index = len(df)
        df.loc[row_index, "Model"] = model_name

    # Update the appropriate column based on the configuration
    if quantization_bits == 4:
        df.loc[row_index, "Accuracy on Arabic MMLU with 4-bit Quantization"] = accuracy
    elif quantization_bits == 8:
        df.loc[row_index, "Accuracy on Arabic MMLU with 8-bit Quantization"] = accuracy
    elif pruning_amount:
        pruning_col = f"Accuracy on Arabic MMLU with {int(pruning_amount * 100)}% Pruning"
        df.loc[row_index, pruning_col] = accuracy
    else:
        df.loc[row_index, "Accuracy on Arabic MMLU"] = accuracy

    # Save the updated dataframe back to the CSV file
    df.to_csv(results_file, index=False)

        
##################################################################################

# Define a function to evaluate the model
def evaluate_model(model_name, quantization=False, quantization_bits=8, pruning=False, pruning_amount=0.4, batch_size=16):
    print(f"Starting evaluation for {model_name}...")
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, quantization=quantization, quantization_bits=quantization_bits)

    # Apply pruning if specified
    if pruning:
        apply_pruning(model, amount=pruning_amount)

    # Map English letters to Arabic letters
    english_to_arabic = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'هـ'}

    # Initialize accuracy tracking
    correct_count = 0

    # Prepare result storage
    results = []
    category_correct = {}
    category_total = {}
    category_confidence = {}

    # Progress bar with accuracy
    progress_bar = tqdm(
        range(0, len(data), batch_size),
        desc="Processing Examples (Accuracy: 0.00%)",
        position=0,
        leave=True
    )

    # Process examples in batches
    for start_idx in progress_bar:
        end_idx = min(start_idx + batch_size, len(data))
        batch = data.iloc[start_idx:end_idx]

        prompts = []
        ground_truths = []
        questions = []
        option_keys = ['أ', 'ب', 'ج', 'د', 'هـ']

        # Prepare batch prompts
        for _, row in batch.iterrows():
            question = row['Question']
            questions.append(question)
            options = []

            for idx in range(1, 6):
                option_key = f"Option {idx}"
                if pd.notna(row.get(option_key, None)):
                    options.append(f"{option_keys[idx - 1]}. {row[option_key]}")

            if len(options) < 2 or row['Answer Key'] not in english_to_arabic:
                continue

            answer_key = english_to_arabic[row['Answer Key']]
            if answer_key not in option_keys[:len(options)]:
                continue

            subject = subject_ar.get(row['Subject'], "")
            level = level_ar.get(row['Level'], "")
            country = country_ar.get(row['Country'], "")

            prompt = f"هذا سؤال {subject} {level} {country}. اختر الإجابة الصحيحة!\n\n"
            prompt += f"سؤال: {question}\n\nاختيارات:\n"
            prompt += "\n".join(options)
            prompt += "\nالإجابة:"

            prompts.append(prompt)
            ground_truths.append(answer_key)
            
        if len(prompts)==0:
            continue
            
                    # Tokenize prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        # Ensure input_ids is also moved to the same device
        input_ids = inputs["input_ids"].to(device)

        # Tokenize batch and move inputs to GPU
        # print("Tokenizing inputs...")
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)

            # Compute last token logits
            last_token_logits = outputs.logits[:, -1, :]

            # Compute choice logits
            choice_logits = torch.stack(
                [last_token_logits[:, tokenizer.encode(opt, add_special_tokens=False)[-1]] for opt in option_keys[:len(option_keys)]], dim=1
            ).to(device)

            # Compute probabilities and predictions
            probs = F.softmax(choice_logits, dim=-1).tolist()
            predictions = [option_keys[argmax(probs[i])] for i in range(len(probs))]

            # Print logits only for the first 10 iterations
            if start_idx < 10:
                print(f"Raw Logits for Iteration {start_idx + 1}: {outputs.logits.shape}")
                print(f"Last Token Logits for Iteration {start_idx + 1}: {last_token_logits.shape}")
                print(f"Choice Logits for Iteration {start_idx + 1}: {choice_logits[0]}")

            # Optional: If needed, print prompts and predictions for debugging
            if start_idx < 10:
                print(f"Prompt {start_idx + 1}: {prompts[0]}")
                print(f"Logits: {choice_logits[0].tolist()}")
                print(f"Prediction: {predictions[0]}")

        # Process each result in batch
        for i, predicted_answer in enumerate(predictions):
            is_correct = predicted_answer == ground_truths[i]
            category = batch.iloc[i]['Group']

            if category not in category_correct:
                category_correct[category] = 0
                category_total[category] = 0
                category_confidence[category] = []

            category_correct[category] += 1 if is_correct else 0
            category_total[category] += 1
            category_confidence[category].append(max(probs[i]))

            results.append({
                'Question': questions[i],
                'Predicted Answer': predicted_answer,
                'Ground Truth': ground_truths[i],
                'Confidence': max(probs[i]),
                'Correct': is_correct
            })
            correct_count += 1 if is_correct else 0

            # Update accuracy in progress bar
            progress_bar.set_description(
                f"Processing Examples (Accuracy: {100 * correct_count / (start_idx + i + 1):.2f}%)"
            )

            
###############################################################################3

    # Save results to CSV
    quantization_suffix = f"_quant{quantization_bits}bit" if quantization else ""
    pruning_suffix = f"_pruned{int(pruning_amount * 100)}" if pruning else ""
    results_file_suffix = f"{quantization_suffix}{pruning_suffix}"

    results_df = pd.DataFrame(results)
    results_file = f'results/{model_name.replace("/", "_")}{results_file_suffix}_results.csv'
    results_df.to_csv(results_file, index=False)

    category_results = []
    for category in category_correct:
        accuracy = (category_correct[category] / category_total[category]) * 100
        avg_confidence = sum(category_confidence[category]) / len(category_confidence[category])
        category_results.append({'Category': category, 'Accuracy': accuracy, 'Avg Confidence': avg_confidence})

    # Transpose results for better readability
    transposed_results = pd.DataFrame({
        "Metric": ["Accuracy", "Avg Confidence"],
        **{category: [category_correct[category] / category_total[category] * 100, 
                      sum(category_confidence[category]) / len(category_confidence[category])] for category in category_correct}
    })
    transposed_file = f'results/{model_name.replace("/", "_")}{results_file_suffix}_transposed_results.csv'
    transposed_results.to_csv(transposed_file, index=False)

    # Print final summary
    accuracy = (correct_count / len(data)) * 100
    print(f"Final Accuracy for {model_name}: {accuracy:.2f}%")

    # Save final accuracy to consolidated results
    save_final_results(model_name, accuracy, quantization_bits=quantization_bits if quantization else None, pruning_amount=pruning_amount if pruning else None)


