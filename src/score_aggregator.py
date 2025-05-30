#!/usr/bin/env python3
import os
import re
import csv
import argparse

# --- MAPPINGS and ORDERING ---

dataset_renaming = {
    "openbookqa": "OBQA",
    "arc-easy": "ARC-E",
    "arc-challenge": "ARC-C",
    "commonsenseqa": "CSQA",
    "boolq": "BoolQ",
    "gsm8k-main": "GSM8K-M",
    "truthfulqa": "TruthfulQA",
    "winogrande": "Winogrande",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
    "bigbenchhard-date_understanding_translated": "Date",
    "bigbenchhard-disambiguation_qa_translated": "DQA",
    "bigbenchhard-geometric_shapes_translated": "GS",
    "bigbenchhard-hyperbaton_translated": "HB",
    "bigbenchhard-logical_deduction_three_objects_translated": "L-3",
    "bigbenchhard-logical_deduction_five_objects_translated": "L-5",
    "bigbenchhard-logical_deduction_seven_objects_translated": "L-7",
    "bigbenchhard-movie_recommendation_translated": "Movie",
    "bigbenchhard-penguins_in_a_table_translated": "Peng.",
    "bigbenchhard-reasoning_about_colored_objects_translated": "Reason",
    "bigbenchhard-ruin_names_translated": "RN",
    "bigbenchhard-salient_translation_error_detection_translated": "SL",
    "bigbenchhard-snarks_translated": "SN",
    "bigbenchhard-temporal_sequences_translated": "Temp.",
    "bigbenchhard-tracking_shuffled_objects_three_objects": "T-3",
    "bigbenchhard-tracking_shuffled_objects_five_objects": "T-5",
    "bigbenchhard-tracking_shuffled_objects_seven_objects": "T-7"
}

model_renaming = {
    "meta-llama-meta-llama-3.1-8b-instruct-turbo": "llama3.1:8b",
    "meta-llama-meta-llama-3.1-70b-instruct-turbo": "llama3.1:70b",
    "meta-llama-llama-3.2-3b-instruct-turbo": "llama3.2:3b",
    "meta-llama-llama-3.3-70b-instruct-turbo": "llama3.3:70b",
    "deepseek-ai-deepseek-r1-distill-qwen-14b": "deepseek-r1:14b",
    "deepseek-ai-deepseek-r1-distill-llama-70b": "deepseek-r1:70b",
    "qwen-qwen2.5-7b-instruct-turbo": "qwen2.5:7b",
    "qwen-qwen2.5-72b-instruct-turbo": "qwen2.5:72b",
    "mistralai-mistral-7b-instruct-v0.3": "mistral:7b",
    "mistralai-mistral-small-24b-instruct-2501": "mistral:24b"
}

model_order = [
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.2:3b",
    "llama3.3:70b",
    "qwen2.5:7b",
    "qwen2.5:72b",
    "mistral:7b",
    "mistral:24b",
    "deepseek-r1:14b",
    "deepseek-r1:70b"
]

# For this example we only include a subset of datasets in our output table.
dataset_order = [
    "OBQA", "CSQA", "ARC-E", "ARC-C", "BoolQ", "GSM8K-M", "Winogrande", "HellaSwag", "MMLU"
]

# --- Helper functions ---

def parse_filename(filename):
    """
    Parses a filename of the expected format:
       <dataset>_<model>_<metric>.txt
    Returns a tuple: (dataset, model, metric) or None if it doesn't match.
    """
    pattern = r'^(.*?)_(.*?)_([^_]+(?:_[^_]+)*)\.txt$'
    m = re.match(pattern, filename)
    if m:
        dataset_str = m.group(1)
        model_str = m.group(2)
        metric_str = m.group(3)
        return dataset_str, model_str, metric_str
    return None

def normalize_model(model_str):
    """
    Lowercases the model string and tries to adjust for inconsistencies.
    For example, if the model string is "meta-llama-Llama-3.1-8B-Instruct-Turbo",
    it becomes "meta-llama-llama-3.1-8b-instruct-turbo". For 3.1 models, if needed,
    we try to add the extra prefix so that it matches keys in model_renaming.
    """
    norm = model_str.lower()
    # For 3.1 models, sometimes the file might be missing the extra 'meta-llama' token.
    if norm not in model_renaming:
        if "3.1-8b" in norm:
            alt = norm.replace("meta-llama-llama", "meta-llama-meta-llama", 1)
            if alt in model_renaming:
                norm = alt
        elif "3.1-70b" in norm:
            alt = norm.replace("meta-llama-llama", "meta-llama-meta-llama", 1)
            if alt in model_renaming:
                norm = alt
    return norm

def get_metric_type(metric_str):
    """
    Returns the metric type string based on the file's metric part.
    """
    if metric_str == "accuracy":
        return "accuracy"
    elif metric_str == "rer":
        return "rer"
    elif metric_str == "llm_judge_accuracy":
        return "llm_eval"
    else:
        return None

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Main aggregation logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from inference output subdirectories and generate CSV tables."
    )
    parser.add_argument(
        "input_dir", nargs="?", default="../inference-outputs/",
        help="Path to the base directory containing subdirectories with .txt files (default: ../inference-outputs/)"
    )
    args = parser.parse_args()
    base_input_dir = args.input_dir

    # We create separate dictionaries for each language and metric.
    # For example, accuracy_bn is a dictionary where:
    #    key = model name, value = { dataset_name: value }
    results = {
        "bn": {"accuracy": {}, "rer": {}, "llm_eval": {}},
        "en": {"accuracy": {}, "rer": {}, "llm_eval": {}}
    }
    
    # Iterate over immediate subdirectories in base_input_dir.
    for subdir in os.listdir(base_input_dir):
        subdir_path = os.path.join(base_input_dir, subdir)
        if os.path.isdir(subdir_path):
            # Determine language from subdirectory name: if it contains "bn", use 'bn'; if "en", use 'en'
            language = "bn" if "bn" in subdir.lower() else "en"
            # Process each .txt file within the subdirectory.
            for f in os.listdir(subdir_path):
                if f.endswith(".txt"):
                    parsed = parse_filename(f)
                    if not parsed:
                        continue
                    dataset_key, model_raw, metric_raw = parsed
                    # Check and map dataset name.
                    dataset_lower = dataset_key.lower()
                    if dataset_lower not in dataset_renaming:
                        continue
                    dataset = dataset_renaming[dataset_lower]
                    
                    # Normalize and look up model name. We force lower-case matching.
                    norm_model_key = normalize_model(model_raw).lower()
                    if norm_model_key not in model_renaming:
                        continue
                    model = model_renaming[norm_model_key]
                    
                    # Determine metric type.
                    metric_type = get_metric_type(metric_raw)
                    if metric_type is None:
                        continue
                    
                    # Build full file path and read the value.
                    file_path = os.path.join(subdir_path, f)
                    try:
                        with open(file_path, 'r') as infile:
                            content = infile.read().strip()
                            value = float(content)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    
                    # Store the value in the appropriate dictionary.
                    # We use: results[<language>][<metric_type>][<model>][<dataset>] = value
                    if model not in results[language][metric_type]:
                        results[language][metric_type][model] = {}
                    results[language][metric_type][model][dataset] = value
    
    # Prepare output: the results should be written one directory up from this script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(os.path.dirname(script_dir), "results")
    
    # For each language and each metric type, generate a CSV file.
    for language in results:
        output_dir = os.path.join(output_base, language)
        ensure_dir(output_dir)
        for metric_type, data in results[language].items():
            filename = f"{metric_type}.csv"  # e.g. accuracy.csv, rer.csv, or llm_eval.csv
            out_path = os.path.join(output_dir, filename)
            with open(out_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header row.
                header = ["Model"] + dataset_order
                writer.writerow(header)
                
                # For each model in the specified order, create a row.
                for model in model_order:
                    row = [model]
                    # If a model is not in the dictionary, use an empty dictionary.
                    model_data = data.get(model, {})
                    for dataset in dataset_order:
                        cell = model_data.get(dataset, "")
                        row.append(cell)
                    writer.writerow(row)
            print(f"Wrote metric '{metric_type}' for language '{language}' to {out_path}")

if __name__ == "__main__":
    main()
