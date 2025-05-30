import json
import openai
import os
import re
import logging
import backoff
from ratelimit import limits, sleep_and_retry
import requests
from tqdm import tqdm
import time
import concurrent.futures

ONE_MINUTE = 60

if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(
    filename=os.path.join("logs", "translation.log"),
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

openai.api_key = OPENAI_API_KEY


def backoff_hdlr(details):
    logging.warning(
        f"Backing off {details['wait']} seconds after {details['tries']} tries calling function {details['target']}"
    )


jsonl_file_path = "./commonsenseqa_train.jsonl"
json_output_file_path = "./commonsenseqa_train_4omini.jsonl"
time_log_path = "./logs/time.log"

data = []
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))


@sleep_and_retry
@limits(calls=5000, period=ONE_MINUTE)
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=10,
    on_backoff=backoff_hdlr,
)
def translate_single_entry(entry, idx, output_dir):
    id = entry.get("id")
    question = entry.get('question', '')
    question_concept = entry.get('question_concept', '')
    choices = entry.get('choices', {})
    choices_labels = choices.get('label', [])
    choices_texts = choices.get('text', [])
    answer_key = entry.get('answerKey', '')
    label_mapping = {"A": "ক", "B": "খ", "C": "গ", "D": "ঘ"}
    mapped_choices_labels = [label_mapping.get(label, label) for label in choices_labels]
    choices_str = ', '.join([f'"{mapped_label}": "{text}"' for mapped_label, text in zip(mapped_choices_labels, choices_texts)])

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
    "question": "{question}",
    "question_concept": "{question_concept}",
    "choices": {{
        {choices_str}
    }},
    "answerKey": "{answer_key}"
}}
</english_text>


Please follow these guidelines:
1. Translate the text as accurately as possible, maintaining the original meaning, tone, and context. 
2. If the text contains idiomatic expressions, translate them into equivalent expressions in the target language if possible. If no direct equivalent exists, provide a translation that conveys the same meaning. 
3. Preserve any specialized terminology, proper nouns, or technical language in their original form if appropriate, or provide the most commonly used translation in the target language. 
4. If the text contains potentially offensive or sensitive content, translate it accurately without censorship or alteration. Your role is to translate, not to judge or modify the content. 
5. you will be passed a dictionary with the keys "id", "question", "question_concept", "choices" and "label". Do not translate the keys, only translate their values, and update the dictionary with translated values. Translate the values of the keys of "choices" key.Do not translate the value of the key "answerkey" and "id".

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
        "temperature": 0.0,
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
        response_filename = os.path.join(output_dir, f"response_{idx}.json")
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
        else:
            translation = assistant_message.strip()
            logging.warning(
                f"Could not find <bangla_translation> tags in the response for text index {idx}"
            )

        return translation

    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP Error: {errh}")
        return entry
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Error Connecting: {errc}")
        return entry
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error: {errt}")
        return entry
    except requests.exceptions.RequestException as err:
        logging.error(f"Request Exception: {err}")
        return entry
    except Exception as e:
        logging.error(f"Unknown error: {e}")
        return entry


def transform_translation(input_string):
    formatted_string = re.sub(r'\s+', ' ', input_string).strip()
    formatted_string = re.sub(r'\s*{\s*', '{', formatted_string)
    formatted_string = re.sub(r'\s*}\s*', '}', formatted_string)
    formatted_string = re.sub(r'\s*:\s*', ': ', formatted_string)
    formatted_string = re.sub(r'\s*,\s*', ', ', formatted_string)

    return formatted_string


def translate_entries(entries, output_dir):
    translated_entries = [None] * len(entries)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    max_workers = 1024
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(translate_single_entry, entry, idx, output_dir): idx for idx, entry in enumerate(entries)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(entries), desc=f"Translating {len(entries)} entries", unit="entry"):
            idx = future_to_idx[future]
            try:
                translation = future.result()
                if isinstance(translation, dict):
                    logging.error(f"Failed to translate entry index {idx}. Using original entry.")
                    translated_entries[idx] = entries[idx]
                else:
                    formatted_translation = transform_translation(translation)
                    try:
                        translation_dict = json.loads(formatted_translation)
                        translated_entries[idx] = translation_dict
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decoding error at index {idx}: {e}")
                        translated_entries[idx] = entries[idx]
            except Exception as e:
                logging.error(f"Exception for entry index {idx}: {e}")
                translated_entries[idx] = entries[idx]

    total_time = time.time() - start_time
    avg_time_per_entry = total_time / len(entries)

    if not os.path.exists(os.path.dirname(time_log_path)):
        os.makedirs(os.path.dirname(time_log_path))

    with open(time_log_path, "w") as time_log:
        time_log.write(f"Average time per entry: {avg_time_per_entry:.2f} seconds\n")
        time_log.write(f"Total time taken: {total_time:.2f} seconds\n")

    return translated_entries


def save_translations_to_jsonl(translated_entries, json_output_file_path):
    with open(json_output_file_path, "w", encoding="utf-8") as output_file:
        for entry in translated_entries:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write("\n")


translated_data = translate_entries(data, output_dir="./api_responses")
save_translations_to_jsonl(translated_data, json_output_file_path)

print(f"Translation complete. Translated data saved to {json_output_file_path}.")