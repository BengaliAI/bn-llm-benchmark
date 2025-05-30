import os
import re
import json
import openai
import requests

def extract_indexes(log_file_path):
    indexes = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r'ERROR:JSON decoding error at index (\d+)', line)
            if match:
                indexes.append(match.group(1))
    return indexes

def read_response_json(index):
    file_path = f'api_responses/response_{index}.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        response_json = json.load(file)
    return response_json

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

def escape_string_values(s):
    # Pattern to match key-value pairs with string values
    pattern = r'(".*?")\s*:\s*"(.*?)"(?=\s*[,\n}])'
    
    def escape_value(match):
        key = match.group(1)
        value = match.group(2)
        escaped_value = json.dumps(value, ensure_ascii=False)[1:-1]
        return f'{key}: "{escaped_value}"'

    result = re.sub(pattern, escape_value, s, flags=re.DOTALL)
    return result

def get_original_entry(index,file_path="./piqa_test.jsonl"):
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

def retry_translation_with_temperature(entry, index, temperature=1.0,output_dir="./api_responses"):
    goal = entry.get('goal', '')
    sol1 = entry.get('sol1', '')
    sol2 = entry.get('sol2', '')
    label = entry.get('label', '')

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
    "goal": "{goal}",
    "sol1": "{sol1}",
    "sol2": "{sol2}",
    "label": {label}
}}
</english_text>

Please follow these guidelines:
1. Translate the text as accurately as possible, maintaining the original meaning, tone, and context. 
2. If the text contains idiomatic expressions, translate them into equivalent expressions in the target language if possible. If no direct equivalent exists, provide a translation that conveys the same meaning. 
3. Preserve any specialized terminology, proper nouns, or technical language in their original form if appropriate, or provide the most commonly used translation in the target language. 
4. If the text contains potentially offensive or sensitive content, translate it accurately without censorship or alteration. Your role is to translate, not to judge or modify the content. 
5. you will be passed a dictionary with the keys "goal", "sol1", "sol2", and "label". Do not translate the keys, only translate their values, and update the dictionary with translated values. Do not translate the value of the key "label".

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
        "temperature": 1.0,  
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

def replace_affected_entry(index, fixed_entry,output_file_path="./piqa_test_gpt4omini.jsonl"):
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

def main():
    log_file_path = os.path.join('logs', 'translation.log')
    indexes = extract_indexes(log_file_path)
    error_log_file = 'parse_errors.log'
    output_file = 'translations.jsonl'
    with open(output_file, 'w', encoding='utf-8') as outfile, open(error_log_file, 'w', encoding='utf-8') as errorfile:
        for index in indexes:
            try:
                response_json = read_response_json(index)
                assistant_message = response_json["choices"][0]["message"]["content"]
                translation = extract_translation(assistant_message)
                if translation:
                    transformed_translation = transform_translation(translation)
                    fixed_translation = escape_string_values(transformed_translation)
                    translation_json = json.loads(fixed_translation)
                    outfile.write(json.dumps(translation_json, ensure_ascii=False) + '\n')
                    message = f"Translation for index-{index} fixed.\n"
                    print(message)
                    errorfile.write(message)
                else:
                    message = f"No translation found in response_{index}.json\n"
                    print(message)
                    original_entry = get_original_entry(int(index))
                    print("Retrying for translation")
                    retry_translation = retry_translation_with_temperature(
                        original_entry, index
                    )
                    
                    if retry_translation:
                        transformed_retry = transform_translation(retry_translation)
                        fixed_retry = escape_string_values(transformed_retry)
                        translation_json = json.loads(fixed_retry)
                        outfile.write(json.dumps(translation_json, ensure_ascii=False) + '\n')
                        replace_affected_entry(int(index), translation_json)
                        print(f"Retry translation for index-{index} successful.")
                        message = f"Translation for index-{index} fixed.\n"
                        errorfile.write(message)
                    else:
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
