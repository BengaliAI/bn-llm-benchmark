import os
import re
import json
from langdetect import detect
from datetime import datetime
import openai
import requests

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
    input_string = re.sub(r'[“”]', '"', input_string)
    input_string = re.sub(r"'", '"', input_string)
    
    # Fix quotes in choices_text and choices_label
    input_string = re.sub(r"'([^']*)'", r'"\1"', input_string)
    # Normalize arrays and keys
    input_string = re.sub(r'("\w+":)\s*"(\[.*?\])"', r'\1 \2', input_string)  # Remove quotes around arrays
    input_string = re.sub(r'\s*("[\w]+"):\s*(\w+)', r' \1: "\2"', input_string)  # Add quotes to unquoted strings
        
    # # Escape nested quotes within JSON values
    # input_string = re.sub(r'(?<!\\)"', r'\\"', input_string)
    
    processed_string = preprocess_string(input_string)

    try:
        json.loads(processed_string)
        return processed_string  
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error Occured: {e}")
        # Fix missing commas between key-value pairs
        # Matches any key-value pair ("key": "value") immediately followed by another key ("key2")
        # fixed_string = re.sub(r'("id":.*?)(?=\s*?"question_stem":)(?!,)', r'\1, ', processed_string)
        # fixed_string = re.sub(r'("question_stem":.*?)(?=\s*?"choices_text":)(?!,)', r'\1, ', fixed_string)
        # fixed_string = re.sub(r'("choices_text":.*?)(?=\s*?"choices_label":)(?!,)', r'\1, ', fixed_string)
        # fixed_string = re.sub(r'("choices_label":.*?)(?=\s*?"answerKey":)(?!,)', r'\1, ', fixed_string)
      

        # fixed_string = re.sub(r',\s*,', ',', fixed_string)
        
        fixed_string = re.sub(r'("id":.*?)(?=\s*?"question_stem":)(?!,)', r'\1, ', input_string)
        fixed_string = re.sub(r'("question_stem":.*?)(?=\s*?"choices_text":)(?!,)', r'\1, ', fixed_string)
        fixed_string = re.sub(r'("choices_text":.*?)(?=\s*?"choices_label":)(?!,)', r'\1, ', fixed_string)
        fixed_string = re.sub(r'("choices_label":.*?)(?=\s*?"answerKey":)(?!,)', r'\1, ', fixed_string)
        fixed_string = re.sub(r'("answerKey":.*?)(?=\s*?"fact1":)(?!,)', r'\1, ', fixed_string)
        fixed_string = re.sub(r'("fact1":.*?)(?=\s*?"humanScore":)(?!,)', r'\1, ', fixed_string)
        fixed_string = re.sub(r'("humanScore":.*?)(?=\s*?"clarity":)(?!,)', r'\1, ', fixed_string)
        fixed_string = re.sub(r'("clarity":.*?)(?=\s*?"turkIdAnonymized":)(?!,)', r'\1, ', fixed_string)
        fixed_string = re.sub(r',\s*,', ',', fixed_string)

        # Escape string values
        escaped_string = escape_string_values(fixed_string)
        
        return escaped_string


def get_original_entry(index,file_path="./openbookqa_additional_train.jsonl"):
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
    # Safely retrieve dictionary values and escape braces
    label_mapping = {"A": "ক", "B": "খ", "C": "গ", "D": "ঘ"}
    id = entry.get('id', '')
    turkIdAnonymized = entry.get('turkIdAnonymized', '')
    question_stem = entry.get('question_stem', '')
    choices_text = entry.get('choices', {}).get('text', [])
    choices_label = entry.get('choices', {}).get('label', [])
    answerKey = entry.get('answerKey', '')
    fact1 = entry.get('fact1', '')
    humanScore = entry.get('humanScore', '')
    clarity = entry.get('clarity', '')
    mapped_choices_label = [label_mapping.get(label, label) for label in choices_label]
    mapped_answerKey = label_mapping[answerKey]

    messages = [
        {
            "role": "system",
            "content": "You are a professional translator tasked with accurately translating text from English to Bangla. Your primary goal is to provide precise and culturally appropriate translations, regardless of the content's nature.",
        },
        {
            "role": "user",
            "content": f"""Translate the following English text into Bangla:

<english_text>
{{
    "id": "{id}",
    "question_stem": "{question_stem}",
    "choices_text": "{choices_text}",
    "choices_label": "{mapped_choices_label}",
    "answerKey": "{mapped_answerKey}",
    "fact1": "{fact1}",
    "humanScore": "{humanScore}",
    "clarity": "{clarity}",
    "turkIdAnonymized": "{turkIdAnonymized}"
}}
</english_text>


Please follow these guidelines:
1. Translate the text as accurately as possible, maintaining the original meaning, tone, and context. 
2. If the text contains idiomatic expressions, translate them into equivalent expressions in the target language if possible. If no direct equivalent exists, provide a translation that conveys the same meaning. 
3. Preserve any specialized terminology, proper nouns, or technical language in their original form if appropriate, or provide the most commonly used translation in the target language. 
4. If the text contains potentially offensive or sensitive content, translate it accurately without censorship or alteration. Your role is to translate, not to judge or modify the content. 
5. you will be passed a dictionary with the keys "id", "question_stem", "choices_text", "choices_label","answerKey","fact1","humanScore","clarity" and "turkIdAnonymized". Do not translate the keys, only translate their values, and update the dictionary with translated values. Do not translate the value of the keys "id", "choices_label","humanScore","clarity","turkIdAnonymized" and "answerKey". But include all the keys in the final dictionary.

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

def replace_affected_entry(index, fixed_entry,output_file_path="./openbookqa_additional_train_gpt4omini.jsonl"):
    try:
        with open(output_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if index + 1 <= len(lines):
            lines[index] = json.dumps(fixed_entry, ensure_ascii=False) + "\n"
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.writelines(lines)
            print(f"Replaced entry at index {index + 1} in {output_file_path}.")
        else:
            print(f"Index {index + 1} is out of range in {output_file_path}.")

    except FileNotFoundError:
        print(f"File not found: {output_file_path}")
    except Exception as e:
        print(f"Unexpected error while replacing entry: {e}")
        
        
def detect_english_entries(file_path="./openbookqa_additional_train_gpt4omini.jsonl", log_file_path="./logs/translation.log"):
    
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
                    
                    combined_text = f"{entry.get('question_stem', '')} {entry.get('choices_text', '')} {entry.get('choices_label', '')} {entry.get('answerKey', '')} {entry.get('fact1', '')}"             

                    # Detect the language
                    if detect(combined_text) == "en":
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                        log_file.write(
                            f"{timestamp}:ERROR:Detected English entry at index {index}. Using original entry.\n"
                        )
                        print(f"English entry found at index {index + 1}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON at line {index + 1}. Skipping...")
                except Exception as e:
                    print(f"Unexpected error at line {index + 1}: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Unexpected error processing file: {e}")

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

                combined_original_length = len(original_entry.get('question_stem', '')) + \
                            len(original_entry.get('fact1', '')) + \
                           sum(len(choice) for choice in original_entry.get('choices', {}).get('text', [])) + \
                           sum(len(label) for label in original_entry.get('choices', {}).get('label', []))
 
                                           
                if response_json is not None:
                    assistant_message = response_json["choices"][0]["message"]["content"]
                    translation = extract_translation(assistant_message)
                if translation:
                    translation_dict = json.loads(translation)
                    combined_translation_length = len(translation_dict.get('question_stem', '')) + \
                                len(translation_dict.get('fact1', '')) + \
                              sum(len(choice) for choice in translation_dict.get('choices_text', [])) + \
                              sum(len(label) for label in translation_dict.get('choices_label', []))

                    if combined_translation_length > 3 * combined_original_length:
                        print(f"Translation for index-{index} is too long. Retrying...")
                        translation = retry_translation_with_temperature(original_entry, index,temperature=1.0)
                       
                    
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
