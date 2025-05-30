import os
import json

def process_jsonl_file(input_file):
    # Create the 'api_responses' directory if it doesn't exist
    output_folder = 'api_responses'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the jsonl file
    with open(input_file, 'r', encoding='utf-8') as infile:
        for idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                data = json.loads(line)
                # Extract the 'body' key from the top-level JSON object
                body = data.get('response', {}).get('body')
                if body is None:
                    print(f"Warning: 'body' key not found at line {idx}")
                    continue
                # Define the output file path
                output_file = os.path.join(output_folder, f'{idx}.json')
                # Write the 'body' content to the output file
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    json.dump(body, outfile, ensure_ascii=False, indent=2)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {idx}: {e}")
            except Exception as e:
                print(f"Unexpected error at line {idx}: {e}")

# Usage
input_file = 'helpful_online_test.jsonl'  # Replace with your actual jsonl file path
process_jsonl_file(input_file)