import os
import json
import re

def transform_translation(translation, idx):
    if not translation:
        print(f"Error: Translation is None at index {idx}")
        return None

    # Remove the <bangla_translation> and </bangla_translation> tags
    if "<bangla_translation>" in translation:
        translation = translation.replace("<bangla_translation>", "").replace("</bangla_translation>", "")

    # Remove <translator_notes> section if it exists
    if "<translator_notes>" in translation:
        translation = translation.split("<translator_notes>")[0].strip()

    # Remove any leading or trailing whitespace
    translation = translation.strip()

    # Attempt to parse the translation as JSON
    try:
        json_data = json.loads(translation)
        return json_data
    except json.JSONDecodeError:
        # If there is a JSONDecodeError, try to fix common issues
        # Replace single quotes with double quotes
        translation_fixed = translation.replace("'", '"')

        # Remove any extraneous whitespace between JSON structural characters
        translation_fixed = re.sub(r'\s*(\{|\}|\[|\]|:|,)\s*', r'\1', translation_fixed)

        # Remove extra whitespace
        translation_fixed = re.sub(r'\s+', ' ', translation_fixed).strip()

        # Try parsing again
        try:
            json_data = json.loads(translation_fixed)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error: JSON decoding error at index {idx}: {e}")
            return None

def process_api_responses(input_folder, output_file):
    idx = 0
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for filename in os.listdir(input_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(input_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    # Extract the content from the assistant's message
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    transformed = transform_translation(content, idx)
                    if transformed:
                        jsonl_file.write(json.dumps(transformed, ensure_ascii=False) + '\n')
                    else:
                        print(f"Error: Failed to transform content at index {idx} in file {filename}")
                    idx += 1

# Usage
input_folder = 'api_responses'
output_file = 'translations.jsonl'
process_api_responses(input_folder, output_file)