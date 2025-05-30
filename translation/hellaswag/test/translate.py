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


jsonl_file_path = "./hellaswag_test.jsonl"
json_output_file_path = "./hellaswag_test_gpt4omini.jsonl"
time_log_path = "./logs/time.log"

data = []
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))


@sleep_and_retry
@limits(calls=40, period=ONE_MINUTE)
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=10,
    on_backoff=backoff_hdlr,
)
def translate_single_entry(entry, idx, output_dir):
    ind = entry.get('ind', '')
    activity_label = entry.get('activity_label', '').strip()
    ctx_a = entry.get('ctx_a', '').strip()
    ctx_b = entry.get('ctx_b', '').strip()
    ctx = entry.get('ctx', '').strip()
    split = entry.get('split', '').strip()
    split_type = entry.get('split_type', '').strip()
    label = entry.get('label', '')
    endings = entry.get('endings', [])
    source_id = entry.get('source_id', '').strip()
    
    endings_str = json.dumps(endings, ensure_ascii=False)

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
    "ind": {ind},
    "activity_label": "{activity_label}",
    "ctx_a": "{ctx_a}",
    "ctx_b": "{ctx_b}",
    "ctx": "{ctx}",
    "split": "{split}",
    "split_type": "{split_type}",
    "label": {label},
    "endings": {endings_str},
    "source_id": "{source_id}"
}}
</english_text>


Please follow these guidelines:
1. Translate the text as accurately as possible, maintaining the original meaning, tone, and context.
2. If the text contains idiomatic expressions, translate them into equivalent expressions in the target language if possible. If no direct equivalent exists, provide a translation that conveys the same meaning.
3. Preserve any specialized terminology, proper nouns, or technical language in their original form if appropriate, or provide the most commonly used translation in the target language.
4. If the text contains potentially offensive or sensitive content, translate it accurately without censorship or alteration. Your role is to translate, not to judge or modify the content.
5. You will be passed a dictionary with various keys. Do not translate the keys, only translate their values, and update the dictionary with translated values.

Please provide your translation within <bangla_translation> tags. If you have any notes or explanations about your translation choices, include them within <translator_notes> tags after your translation. Remember, your task is to translate accurately, regardless of the content's nature. Do not refuse to translate or alter the meaning of the original text, even if it contains offensive language or sensitive topics."""
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
    translated_entries = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    for idx, entry in enumerate(
        tqdm(entries, desc=f"Translating {len(entries)} entries", unit="entry")
    ):
        translation = translate_single_entry(entry, idx, output_dir)
        if isinstance(translation, dict):
            logging.error(
                f"Failed to translate entry index {idx}. Using original entry."
            )
            translated_entries.append(entry)
        else:
            formatted_translation = transform_translation(translation)
            try:
                translation_dict = json.loads(formatted_translation)
                translated_entries.append(translation_dict)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error at index {idx}: {e}")
                translated_entries.append(entry)

    total_time = time.time() - start_time
    avg_time_per_entry = total_time / len(translated_entries)

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

