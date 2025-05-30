import os
import json
import re

def extract_data(content, idx, log_file):
    if not content:
        log_file.write(f"Error: Content is None at index {idx}\n")
        return None

    # Extract content within <bangla_translation> tags
    match = re.search(r'<bangla_translation>(.*?)</bangla_translation>', content, re.DOTALL)
    if not match:
        log_file.write(f"Error: <bangla_translation> tags not found at index {idx}\n")
        return None
    translation = match.group(1).strip()

    # Remove <translator_notes> section if it exists
    if "<translator_notes>" in translation:
        translation = translation.split("<translator_notes>")[0].strip()

    # Fix potential issues in JSON content, such as missing or empty fields
    translation = re.sub(r'"label":\s*,', '"label": null,', translation)  # Replace missing label with null

    # Attempt to parse the translation as JSON
    try:
        json_data = json.loads(translation)
        # Extract necessary fields, providing default values where appropriate
        extracted_data = {
            'ind': json_data.get('ind', ''),
            'activity_label': json_data.get('activity_label', '').strip(),
            'ctx_a': json_data.get('ctx_a', '').strip(),
            'ctx_b': json_data.get('ctx_b', '').strip(),
            'ctx': json_data.get('ctx', '').strip(),
            'split': json_data.get('split', '').strip(),
            'split_type': json_data.get('split_type', '').strip(),
            'label': json_data.get('label', ''),  # Set label to empty string if null
            'endings': json_data.get('endings', []),
            'source_id': json_data.get('source_id', '').strip()
        }
        return extracted_data
    except json.JSONDecodeError as e:
        # If there is a JSONDecodeError, log the error
        log_file.write(f"Error decoding JSON at index {idx}: {e}\nContent: {translation}\n")
        return None
    
def extract_number(filename):
    # Extracts the first number found in the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def process_api_responses(input_folder, output_file, log_file_path):
    idx = 0
    with open(output_file, 'w', encoding='utf-8') as jsonl_file, open(log_file_path, 'w', encoding='utf-8') as log_file:
        filenames = sorted(
            [f for f in os.listdir(input_folder) if f.endswith('.json')],
            key=extract_number
        )
        
        for filename in filenames:
            if filename.endswith('.json'):
                filepath = os.path.join(input_folder, filename)
                
                with open(filepath, 'r', encoding='utf-8') as json_file:
                    try:
                        data = json.load(json_file)
                        # Extract the content from the assistant's message
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                        result = extract_data(content, idx, log_file)
                        if result:
                            jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                        else:
                            log_file.write(f"Error: Failed to extract data at index {idx} in file {filename}\n")
                    except json.JSONDecodeError as e:
                        log_file.write(f"Error decoding JSON in file {filename} at index {idx}: {e}\n")
                    except Exception as e:
                        log_file.write(f"Unexpected error in file {filename} at index {idx}: {e}\n")
                    idx += 1

# Usage
input_folder = 'api_responses'      # Folder containing your .json files
output_file = 'hellaswag_dev_translated.jsonl'  # Output .jsonl file
log_file_path = 'translation.log'   # Log file for errors
process_api_responses(input_folder, output_file, log_file_path)