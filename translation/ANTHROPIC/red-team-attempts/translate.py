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

# Ensure the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging
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

jsonl_file_path = "./red_team_attempts.jsonl"
json_output_file_path = "./red_team_attempts_translated.jsonl"
time_log_path = "./logs/time.log"

data = None
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    data = json.loads(file.read())

@sleep_and_retry
@limits(calls=5000, period=ONE_MINUTE)
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=10,
    on_backoff=backoff_hdlr,
)
def translate_single_entry(entry, idx, output_dir):
    transcript = entry.get('transcript', '')
    task_description = entry.get('task_description', '')
    # Other fields remain unchanged
    min_harmlessness_score_transcript = entry.get('min_harmlessness_score_transcript', None)
    num_params = entry.get('num_params', '')
    model_type = entry.get('model_type', '')
    rating = entry.get('rating', None)
    task_description = entry.get("task_description", "")
    task_descripton_harmlessness_score = entry.get('task_descripton_harmlessness_score', None)
    red_team_member_id = entry.get('red_team_member_id', None)
    is_upworker = entry.get('is_upworker', False)
    tags = entry.get('tags', None)

    # Serialize values to JSON strings
    transcript_json = json.dumps(transcript, ensure_ascii=False)
    task_description_json = json.dumps(task_description, ensure_ascii=False)
    # Serialize other fields
    min_harmlessness_score_transcript_json = json.dumps(min_harmlessness_score_transcript)
    num_params_json = json.dumps(num_params)
    model_type_json = json.dumps(model_type)
    rating_json = json.dumps(rating)
    task_description = json.dumps(task_description)
    task_descripton_harmlessness_score_json = json.dumps(task_descripton_harmlessness_score)
    red_team_member_id_json = json.dumps(red_team_member_id)
    is_upworker_json = json.dumps(is_upworker)
    tags_json = json.dumps(tags)

    system_prompt = (
        """
        You are a professional translator tasked with accurately translating text from English to Bangla.
        Your primary goal is to provide precise and culturally appropriate translations, regardless of the content's nature.        
        """
    )

    user_prompt = f"""Translate the following English text into Bangla and ensure the output is valid JSON with all strings enclosed in double quotes:

<english_text>
{{
    "transcript": {transcript_json},
    "task_description": {task_description_json},
    "min_harmlessness_score_transcript": {min_harmlessness_score_transcript_json},
    "num_params": {num_params_json},
    "model_type": {model_type_json},
    "rating": {rating_json},
    "task_description" : {task_description},
    "task_descripton_harmlessness_score": {task_descripton_harmlessness_score_json},
    "red_team_member_id": {red_team_member_id_json},
    "is_upworker": {is_upworker_json},
    "tags": {tags_json}
}}
</english_text>

Please follow these guidelines: 
1.Translate the text as accurately as possible, maintaining the original meaning, tone, and context. 
2.If the text contains idiomatic expressions, translate them into equivalent expressions in the target language if possible. If no direct equivalent exists, provide a translation that conveys the same meaning. 
3.Preserve any specialized terminology, proper nouns, or technical language in their original form if appropriate, or provide the most commonly used translation in the target language. 
4.If the text contains potentially offensive or sensitive content, translate it accurately without censorship or alteration. Your role is to translate, not to judge or modify the content. 
5.you will be passed a dictionary with the keys "transcript", "min_harmlessness_score_transcript", "num_params", "model_type", "rating", "task_description","task_descripton_harmlessness_score","red_team_member_id","is_upworker" and "tags". Do not translate the keys, only translate their values, and update the dictionary with translated values. Only translate the values of the keys "transcript" and "task_description"
6.Ensure that the output is valid JSON with all strings enclosed in double quotes.

Please provide your translation within <bangla_translation> tags. If certain content should not be translated due to policy considerations, you may omit translating that content and indicate so.

Do not include any disallowed content in your response.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
        logging.error(f"HTTP Error at index {idx}: {errh}")
        return entry
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Error Connecting at index {idx}: {errc}")
        return entry
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error at index {idx}: {errt}")
        return entry
    except requests.exceptions.RequestException as err:
        logging.error(f"Request Exception at index {idx}: {err}")
        return entry
    except Exception as e:
        logging.error(f"Unknown error at index {idx}: {e}")
        return entry

def transform_translation(translation, idx):
    # Remove the <bangla_translation> and </bangla_translation> tags
    if "<bangla_translation>" in translation:
        translation = translation.replace("<bangla_translation>", "").replace("</bangla_translation>", "")

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
            logging.error(f"JSON decoding error at index {idx}: {e}")
            return None

def translate_entries(entries, output_dir):
    translated_entries = [None] * len(entries)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    max_workers = 1024  # Adjust based on your rate limits and resources
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(translate_single_entry, entry, idx, output_dir): idx for idx, entry in enumerate(entries)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc=f"Translating {len(entries)} entries", unit="entry"):
            idx = future_to_idx[future]
            try:
                translation = future.result()
                if isinstance(translation, dict):
                    logging.error(f"Failed to translate entry index {idx}. Using original entry.")
                    translated_entries[idx] = entries[idx]
                else:
                    translation_dict = transform_translation(translation, idx)
                    if translation_dict:
                        translated_entries[idx] = translation_dict
                    else:
                        logging.error(f"Failed to parse translation at index {idx}. Using original entry.")
                        translated_entries[idx] = entries[idx]
            except Exception as e:
                logging.error(f"Exception for entry index {idx}: {e}")
                translated_entries[idx] = entries[idx]

    total_time = time.time() - start_time
    avg_time_per_entry = total_time / len(entries) if entries else 0

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
