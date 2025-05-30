import json

# Define the file paths
input_file_path = 'MR-GSM8K.json'  # Path to your JSON file
output_file_path = 'MR-GSM8K.jsonl'  # Path to the new JSONL file

# Load the JSON file
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Write each dictionary in the list as a separate line in the JSONL file
with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
    for entry in data:
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"Conversion complete. JSONL file saved at: {output_file_path}")
