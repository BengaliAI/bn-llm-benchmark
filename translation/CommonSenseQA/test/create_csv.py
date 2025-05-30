import json
import random
import csv

def load_data(filename):
    """Load data from a JSONL file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]

def sample_data(english_data, bangla_data, dataset_name, sample_size=20, seed=42):
    """Sample data with a fixed seed to ensure consistency."""
    random.seed(seed)
    indices = random.sample(range(len(english_data)), sample_size)
    
    sampled_data = []
    for i in indices:
        # Remove the "qID" key from the English and Bangla dictionaries
        english_entry = {key: value for key, value in english_data[i].items() if key != "id"}
        bangla_entry = {key: value for key, value in bangla_data[i].items() if key != "id"}

        sample_id = english_data[i].get("id", i)
        
        # Append the entry to the sampled data list
        sampled_data.append({
            "Dataset Name": dataset_name,
            "Sample ID": sample_id,
            "English": json.dumps(english_entry), 
            "Bangla": json.dumps(bangla_entry)
        })
    
    return sampled_data


def save_to_csv(data, output_filename):
    """Save sampled data to a CSV file."""
    fieldnames = ["Dataset Name", "Sample ID", "English", "Bangla"]
    with open(output_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main(english_filename, bangla_filename, dataset_name, output_filename="sampled_data.csv"):
    # Load English and Bangla data
    english_data = load_data(english_filename)
    bangla_data = load_data(bangla_filename)
    
    # Check that the files have the same number of lines
    if len(english_data) != len(bangla_data):
        raise ValueError("The English and Bangla files must have the same number of lines.")
    
    # Sample data
    sampled_data = sample_data(english_data, bangla_data, dataset_name)
    
    # Save to CSV
    save_to_csv(sampled_data, output_filename)
    print(f"Sampled data saved to {output_filename}")

# Define the dataset name and replace file names with actual paths
dataset_name = "Common Sense QA"  # Replace with your desired dataset name
main('commonsenseqa_test.jsonl', 'commonsenseqa_test_4omini.jsonl', dataset_name)
