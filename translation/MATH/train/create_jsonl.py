import os
import json
import re

def combine_json_files_to_jsonl(input_folder, output_folder, output_file="combined_output.jsonl"):
    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"The specified input folder '{input_folder}' does not exist.")
    
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize an empty list to hold all json data
    combined_data = []
    
    # Helper function to extract numeric parts for sorting
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else float('inf')  # Put non-numbered files at the end

    # Sort files numerically based on the extracted number
    for file_name in sorted(os.listdir(input_folder), key=extract_number):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_folder, file_name)
            print(file_path)  # For debugging purposes
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)  # Load JSON data from the file
                    combined_data.append(data)  # Append each JSON data to list
                except json.JSONDecodeError as e:
                    print(f"Could not decode JSON in file {file_name}: {e}")
    
    # Define the path to save the output .jsonl file
    output_path = os.path.join(output_folder, output_file)
    
    # Write each JSON object as a line in the .jsonl file in the order they were read
    with open(output_path, 'w') as output_f:
        for item in combined_data:
            json_line = json.dumps(item)
            output_f.write(json_line + '\n')
    
    print(f"Data from .json files in '{input_folder}' have been combined and saved to '{output_path}'.")

# Example usage
combine_json_files_to_jsonl(input_folder='number_theory', output_folder='number_theory_jsonl', output_file="number_theory.jsonl")