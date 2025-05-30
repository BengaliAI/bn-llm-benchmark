# download_all_datasets.py

from datasets import load_dataset
import json

# List of all dataset configs
all_configs = [
    'salient_translation_error_detection',
]

# Iterate over each dataset config, load it, and save as JSONL
for config in all_configs:
    # Load the dataset for the current config
    dataset = load_dataset('bbh.py', name=config)
    
    # Path to save the JSONL file for this config
    output_file = f'{config}.jsonl'
    
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate over the dataset and write each example as a JSON line
        for example in dataset['train']:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Dataset for '{config}' saved to {output_file}")
