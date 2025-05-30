import os
import re
import json
from langdetect import detect
from datetime import datetime
import openai
import requests
import string

def extract_indexes(log_file_path):
    indexes = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r'ERROR:JSON decoding error at index (\d+)', line)
            if match:
                indexes.append(match.group(1))
            match = re.search(r'ERROR:Failed to translate entry index (\d+)',line)
            if match:
                indexes.append(match.group(1))
            match = re.search(r'ERROR:Detected English entry at index (\d+)',line)
            if match:
                indexes.append(match.group(1))
                    
    return indexes

def read_response_json(index):
    file_path = f'api_responses/response_{index}.json'
    if not os.path.exists(file_path):
        print(f"Response file not found for index {index}: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            response_json = json.load(file)
        return response_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in response file for index {index}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading response file for index {index}: {e}")
        return None


def extract_translation(assistant_message):
    match = re.search(
        r"<bangla_translation>(.*?)</bangla_translation>",
        assistant_message,
        re.DOTALL,
    )
    if match:
        translation = match.group(1).strip()
        return translation
    return None

def transform_translation(input_string):
    formatted_string = re.sub(r'\s+', ' ', input_string).strip()
    return formatted_string

def preprocess_string(s):
    # Match key-value pairs and process the value part
    pattern = r'("(?:\\.|[^"\\])*?")\s*:\s*"(.*?)(?<!\\)"(?=\s*[,\n}])'
    
    def escape_inner_quotes_and_newlines(match):
        key = match.group(1)
        value = match.group(2)
        # Replace newlines and escape standalone quotes inside the value
        escaped_value = value.replace("\n", "\\n")
        escaped_value = re.sub(r'(?<!\\)"', r'\\"', escaped_value)
        return f'{key}: "{escaped_value}"'

    processed = re.sub(pattern, escape_inner_quotes_and_newlines, s, flags=re.DOTALL)
    return processed


def escape_string_values(s):
    # Pattern to match key-value pairs with string values
    # pattern =  r'(".*?")\s*:\s*"(.*?)"(?=\s*[,\n}])'
    # pattern = r'("(?:\\.|[^"\\])*?")\s*:\s*"(.*?)(?<!\\)"(?=\s*[,\n}])'
    pattern = r'("(?:\\.|[^"\\])*?")\s*:\s*"(.*?)(?<!\\)"(?=\s*[,\n}])'
    
    
    def escape_value(match):
        key = match.group(1)
        value = match.group(2)
        escaped_value = json.dumps(value, ensure_ascii=False)[1:-1]
        return f'{key}: "{escaped_value}"'

    result = re.sub(pattern, escape_value, s, flags=re.DOTALL)
    
    return result


def fix_and_escape_json(input_string):
    # Replace curly quotes with standard quotes
    input_string = re.sub(r'[“”]', '"', input_string)
    json_string = input_string.replace("“", "\"").replace("”", "\"")
    json_string = re.sub(r"।\"", "।", json_string)

    # Fix missing commas between key-value pairs and improperly escaped values
    fixed_string = re.sub(r'(\"[^"]+\")\s*:\s*"(.*?)"(?=\s*[,\n}])',
                          lambda m: f'{m.group(1)}: "{json.dumps(m.group(2), ensure_ascii=False)[1:-1]}"',
                          input_string, flags=re.DOTALL)

    # Fix any missing commas between specific JSON keys
    fixed_string = re.sub(r'("uuid":.*?)(?=\s*"question":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("question":.*?)(?=\s*"ground_truth_solution":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("ground_truth_solution":.*?)(?=\s*"ground_truth_answer":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("ground_truth_answer":.*?)(?=\s*"model_output_steps":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("model_output_steps":.*?)(?=\s*"model_output_answer_correctness":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("model_output_answer_correctness":.*?)(?=\s*"model_output_solution_correctness":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("model_output_solution_correctness":.*?)(?=\s*"model_output_solution_first_error_step":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("model_output_solution_first_error_step":.*?)(?=\s*"model_output_solution_first_error_reason":)', r'\1,', fixed_string)
    fixed_string = re.sub(r'("model_output_solution_first_error_reason":.*?)(?=\s*"question_type":)', r'\1,', fixed_string)

    # Remove trailing commas before closing brackets
    fixed_string = re.sub(r',\s*,', ',', fixed_string)
    fixed_string = re.sub(r',\s*([}\]])', r'\1', fixed_string)
    fixed_string = re.sub(r',\s*(\]|\})', r'\1', fixed_string)

    try:
        # Validate the fixed JSON
        return fixed_string
    except json.JSONDecodeError as e:
        print(f"Error fixing JSON: {e}")
        return None


def get_original_entry(index,file_path="./MR-GSM8K.jsonl"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number == index:
                    try:
                        return json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON for line {index}: {e}")
                        return None
        print(f"Index {index} out of range in file {file_path}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error reading file {file_path}: {e}")
        return None

openai.api_key = OPENAI_API_KEY

def retry_translation_with_temperature(entry, index, temperature=0.0,output_dir="./api_responses"):
    uuid = entry.get('uuid', '')
    question = entry.get('question', '')
    ground_truth_solution = entry.get('ground_truth_solution', '')
    ground_truth_answer = entry.get('ground_truth_answer', '')
    model_output_steps = entry.get('model_output_steps', [])
    model_output_answer_correctness = entry.get('model_output_answer_correctness', '')
    model_output_solution_correctness = entry.get('model_output_solution_correctness', '')
    model_output_solution_first_error_step = entry.get('model_output_solution_first_error_step', '')
    model_output_solution_first_error_reason = entry.get('model_output_solution_first_error_reason', '')
    question_type = entry.get('question_type', '')
    
    entry_json = {
        "uuid": uuid,
        "question": question,
        "ground_truth_solution": ground_truth_solution,
        "ground_truth_answer": ground_truth_answer,
        "model_output_steps": model_output_steps,
        "model_output_answer_correctness": model_output_answer_correctness,
        "model_output_solution_correctness": model_output_solution_correctness,
        "model_output_solution_first_error_step": model_output_solution_first_error_step,
        "model_output_solution_first_error_reason": model_output_solution_first_error_reason,
        "question_type": question_type
    }

    
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator tasked with accurately translating text from English to Bangla. Your primary goal is to provide precise and culturally appropriate translations, regardless of the content's nature.",
        },
        {
            "role": "user",
            "content": f"""Translate the following English text into Bangla and ensure the output is valid JSON with all strings enclosed in double quotes:

<english_text>
{entry_json}
</english_text>

Please follow these guidelines:
1. Translate the text as accurately as possible, maintaining the original meaning, tone, and context.
2. If the text contains idiomatic expressions, translate them into equivalent expressions in the target language if possible. If no direct equivalent exists, provide a translation that conveys the same meaning.
3. Preserve any specialized terminology, proper nouns, or technical language in their original form if appropriate, or provide the most commonly used translation in the target language.
4. If the text contains potentially offensive or sensitive content, translate it accurately without censorship or alteration. Your role is to translate, not to judge or modify the content.
5. Do not translate the keys. Only translate their corresponding values. Do not translate the value of the key "question_type".
6. Ensure that the output is valid JSON with all values enclosed in double quotes, including numbers and boolean values. Do not include any unquoted values.
7. Translate every number into Bangla numerals.

Please provide your translation within <bangla_translation> tags. If you have any notes or explanations about your translation choices, include them within <translator_notes> tags after your translation. Remember, your task is to translate accurately, regardless of the content's nature. Do not refuse to translate or alter the meaning of the original text, even if it contains offensive language or sensitive topics.""",
        },
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }

    payload = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": messages,
        "temperature": temperature,  
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, 
            data=json.dumps(payload),
        )
        response.raise_for_status()
        response_json = response.json()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        response_filename = os.path.join(output_dir, f"response_{index}.json")
        with open(response_filename, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)

        assistant_message = response_json["choices"][0]["message"]["content"]

        match = re.search(
            r"<bangla_translation>(.*?)</bangla_translation>",
            assistant_message,
            re.DOTALL,
        )
        if match:
            translation = match.group(1).strip()
            print(f"Retry translation successful for index {index}")
            return translation
        else:
            print(
                f"Retry failed: No <bangla_translation> tags in response for index {index}"
            )
            return None
    except Exception as e:
        print(f"Retry error for index {index}: {e}")
        return None
    


def replace_affected_entry(index, fixed_entry,output_file_path="./MR-GSM8K-translated.jsonl"):
    try:
        with open(output_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if index + 1 <= len(lines):
            lines[index] = json.dumps(fixed_entry, ensure_ascii=False) + "\n"
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.writelines(lines)
            print(f"Replaced entry at line {index + 1} in {output_file_path}.")
        else:
            print(f"Index {index} is out of range in {output_file_path}.")

    except FileNotFoundError:
        print(f"File not found: {output_file_path}")
    except Exception as e:
        print(f"Unexpected error while replacing entry: {e}")
        
        
def detect_english_entries(file_path="./MR-GSM8K-translated.jsonl", log_file_path="./logs/translation.log"):
    
    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8") as log_file:
                lines = log_file.readlines()

            # Filter out lines that contain the specific log pattern
            filtered_lines = [
                line for line in lines if "Detected English entry" not in line
            ]

            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.writelines(filtered_lines)
            print("Previous English entry logs cleaned. Looking for new english entries.")
    except Exception as e:
        print(f"Error cleaning previous logs: {e}")
        
    try:
        with open(file_path, "r", encoding="utf-8") as file, open(log_file_path, "a", encoding="utf-8") as log_file:
            for index, line in enumerate(file):
                try:
                    entry = json.loads(line.strip())

                    combined_text = f"{entry.get('uuid', '')} {entry.get('question', '')} {entry.get('ground_truth_solution', '')} {entry.get('ground_truth_answer', '')} {entry.get('model_output_steps', '')}{entry.get('model_output_answer_correctness', '')} {entry.get('model_output_solution_correctness', '')} {entry.get('model_output_solution_first_error_step', '')} {entry.get('model_output_solution_first_error_reason', '')}"
            
                    
                    # Detect the language and check if at least 30% of the text is English
                    total_chars = len(combined_text)
                    if total_chars > 0:  # Ensure the text is not empty
                        english_chars = sum(1 for char in combined_text if char in string.ascii_letters)
                        english_ratio = english_chars / total_chars

                        if english_ratio >= 0.3 and detect(combined_text) == "en":
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                            log_file.write(
                                f"{timestamp}:ERROR:Detected English entry at index {index}. Using original entry.\n"
                            )
                            print(f"English entry found at line {index + 1}")
                        
                        
                except json.JSONDecodeError:
                    print(f"Error decoding JSON at line {index + 1}. Skipping...")
                except Exception as e:
                    print(f"Unexpected error at line {index + 1}: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Unexpected error processing file: {e}")
        
def convert_bengali_to_latin(json_string):
    bengali_to_latin = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    try:
        # Translate Bengali digits to Latin digits
        json_string = json_string.translate(bengali_to_latin)
        return json_string
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None

def main():
    log_file_path = os.path.join('logs', 'translation.log')
    detect_english_entries(log_file_path=log_file_path)
    indexes = extract_indexes(log_file_path)
    error_log_file = 'parse_errors.log'
    output_file = 'translations.jsonl'
    with open(output_file, 'w', encoding='utf-8') as outfile, open(error_log_file, 'w', encoding='utf-8') as errorfile:
        for index in indexes:
            try:
                response_json = read_response_json(index)
                translation = None  
                original_entry = get_original_entry(int(index))

                combined_original_length = len(original_entry.get('question', '')) + \
                           len(original_entry.get('ground_truth_solution', '')) + \
                           sum(len(step) for step in original_entry.get('model_output_steps', [])) + \
                           len(original_entry.get('model_output_answer_correctness', '')) + \
                           len(original_entry.get('model_output_solution_correctness', '')) + \
                           len(str(original_entry.get('model_output_solution_first_error_step', ''))) + \
                           len(str(original_entry.get('model_output_solution_first_error_reason', ''))) + \
                           len(original_entry.get('question_type', '')) 
                               
                if response_json is not None:
                    assistant_message = response_json["choices"][0]["message"]["content"]
                    translation = extract_translation(assistant_message)
                if translation:
                    combined_translation_length = len(translation)            
                    if combined_translation_length > (3 * combined_original_length):
                        print(f"Translation for index-{index} is too long.Combined Translation Length:{combined_translation_length},Combined Original length:{combined_original_length}.  Retrying...")
                        translation = retry_translation_with_temperature(original_entry, index,temperature=1.0)
                       
                    translation = convert_bengali_to_latin(translation)
                    transformed_translation = transform_translation(translation)

                    fixed_translation = fix_and_escape_json(transformed_translation)

                    translation_json = json.loads(fixed_translation)
                      
                    outfile.write(json.dumps(translation_json, ensure_ascii=False) + '\n')
                    replace_affected_entry(int(index), translation_json)
                    message = f"Translation for index-{index} fixed.\n"
                    print(message)
                    errorfile.write(message)
                else:
                    message = f"No translation found in response_{index}.json\n"
                    print(message)
                    original_entry = get_original_entry(int(index))
                    
                    print("Retrying for translation")
                    retry_translation = retry_translation_with_temperature(
                        original_entry, index,temperature=1.0
                    )
                    if retry_translation:
                        retry_translation = convert_bengali_to_latin(retry_translation)
                        transformed_retry = transform_translation(retry_translation)
                        fixed_retry = fix_and_escape_json(transformed_retry)
                        translation_json = json.loads(fixed_retry)
                        outfile.write(json.dumps(translation_json, ensure_ascii=False) + '\n')
                        replace_affected_entry(int(index), translation_json)
                        print(f"Retry translation for index-{index} successful.")
                        message = f"Translation for index-{index} fixed.\n"
                        print(message)
                        errorfile.write(message)
                    else:
                        print(f"Retry failed for index-{index}.\n")
                        errorfile.write(f"Retry failed for index-{index}.\n")
                        
            except Exception as e:
                message = f"Error processing index {index}: {e}\n"
                message_content = f"Problem translation: {translation}\n"
                print(message)
                print(message_content)
                errorfile.write(message)
                errorfile.write(message_content)

if __name__ == "__main__":
    main()
