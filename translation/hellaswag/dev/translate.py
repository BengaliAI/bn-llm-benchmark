import json
import openai
import os
import re
import logging
import requests
from tqdm import tqdm
import time

# Constants
BATCH_SIZE = 50  # Maximum tasks per batch
POLL_INTERVAL = 5  # Seconds to wait before polling batch status again

# Ensure logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure logging
logging.basicConfig(
    filename=os.path.join("logs", "translation.log"),
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

# File paths
jsonl_file_path = "./hellaswag_dev.jsonl"
json_output_file_path = "./hellaswag_dev_gpt4omini.jsonl"
time_log_path = "./logs/time.log"

# Read data from the JSONL file
data = []
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))
data = data[:5]

def transform_translation(input_string):
    formatted_string = re.sub(r'\s+', ' ', input_string).strip()
    formatted_string = re.sub(r'\s*{\s*', '{', formatted_string)
    formatted_string = re.sub(r'\s*}\s*', '}', formatted_string)
    formatted_string = re.sub(r'\s*:\s*', ': ', formatted_string)
    formatted_string = re.sub(r'\s*,\s*', ', ', formatted_string)
    return formatted_string

def create_task(entry, global_idx):
    ind = entry.get('ind', '')
    activity_label = entry.get('activity_label', '')
    ctx_a = entry.get('ctx_a', '')
    ctx_b = entry.get('ctx_b', '')
    ctx = entry.get('ctx', '')
    split = entry.get('split', '')
    split_type = entry.get('split_type', '')
    label = entry.get('label', '')
    endings = entry.get('endings', '')
    source_id = entry.get('source_id', '')

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
    "ctx_b": {ctx_b},
    "ctx": {ctx},
    "split": {split},
    "split_type": {split_type},
    "label": {label},
    "endings": {endings},
    "source_id": {source_id}
}}
</english_text>

Please follow these guidelines:
1. Translate the text as accurately as possible, maintaining the original meaning, tone, and context.
2. If the text contains idiomatic expressions, translate them into equivalent expressions in the target language if possible. If no direct equivalent exists, provide a translation that conveys the same meaning.
3. Preserve any specialized terminology, proper nouns, or technical language in their original form if appropriate, or provide the most commonly used translation in the target language.
4. If the text contains potentially offensive or sensitive content, translate it accurately without censorship or alteration. Your role is to translate, not to judge or modify the content.
5. You will be passed a dictionary with the keys "ind", "activity_label", "ctx_a", "ctx_b", "split", "split_type", "label", "endings", and "source_id". Do not translate the keys, only translate their values, and update the dictionary with translated values. Do not translate the value of the key "label".

Please provide your translation within <bangla_translation> tags. If you have any notes or explanations about your translation choices, include them within <translator_notes> tags after your translation. Remember, your task is to translate accurately, regardless of the content's nature. Do not refuse to translate or alter the meaning of the original text, even if it contains offensive language or sensitive topics.""",
        },
    ]

    task = {
        "task_id": str(global_idx),
        "task": {
            "messages": messages,
            "temperature": 0.0,
        },
    }
    return task

def submit_and_process_batch(entries_batch, batch_idx, output_dir, batch_start):
    tasks = []
    for idx, entry in enumerate(entries_batch):
        global_idx = batch_start + idx
        task = create_task(entry, global_idx)
        tasks.append(task)

    batch_payload = {
        "tasks": tasks,
        "model": "gpt-4o-mini-2024-07-18",
        "type": "chat.completion",
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }

    # Submit the batch
    try:
        response = requests.post(
            "https://api.openai.com/v1/batches",
            headers=headers,
            data=json.dumps(batch_payload),
        )
        response.raise_for_status()
        response_json = response.json()
        batch_id = response_json["id"]

        # Save batch response
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        batch_response_filename = os.path.join(output_dir, f"batch_{batch_idx}_response.json")
        with open(batch_response_filename, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)
    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP Error when submitting batch {batch_idx}: {errh}")
        return []
    except Exception as e:
        logging.error(f"Error when submitting batch {batch_idx}: {e}")
        return []

    # Poll for batch status
    batch_status_url = f"https://api.openai.com/v1/batches/{batch_id}"
    batch_completed = False
    while not batch_completed:
        try:
            status_response = requests.get(
                batch_status_url,
                headers=headers,
            )
            status_response.raise_for_status()
            status_json = status_response.json()
            status = status_json["status"]
            if status == "completed":
                batch_completed = True
            elif status == "failed":
                logging.error(f"Batch {batch_id} failed")
                return []
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            logging.error(f"Error when polling batch {batch_idx}: {e}")
            return []

    # Retrieve the results
    results_url = f"https://api.openai.com/v1/batches/{batch_id}/results"
    try:
        results_response = requests.get(
            results_url,
            headers=headers,
        )
        results_response.raise_for_status()
        results_json = results_response.json()

        # Save results
        results_filename = os.path.join(output_dir, f"batch_{batch_idx}_results.json")
        with open(results_filename, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error when retrieving results for batch {batch_idx}: {e}")
        return []

    # Process the results
    translated_entries = []
    results_dict = {}
    for result in results_json["data"]:
        task_id = int(result["task_id"])
        assistant_message = result["message"]["content"]
        results_dict[task_id] = assistant_message

    for idx, entry in enumerate(entries_batch):
        global_idx = batch_start + idx
        assistant_message = results_dict.get(global_idx, "")
        if assistant_message == "":
            logging.error(f"No result for task_id {global_idx}")
            translated_entries.append(entry)
        else:
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
                    f"Could not find <bangla_translation> tags in the response for task_id {global_idx}"
                )
            formatted_translation = transform_translation(translation)
            try:
                translation_dict = json.loads(formatted_translation)
                translated_entries.append(translation_dict)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error for task_id {global_idx}: {e}")
                translated_entries.append(entry)

    return translated_entries

def translate_entries(entries, output_dir):
    batch_size = BATCH_SIZE
    translated_entries = []
    num_batches = len(entries) // batch_size + (1 if len(entries) % batch_size != 0 else 0)

    start_time = time.time()
    global_idx = 0

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(entries))
        entries_batch = entries[batch_start:batch_end]
        batch_translated_entries = submit_and_process_batch(entries_batch, batch_idx, output_dir, batch_start)
        translated_entries.extend(batch_translated_entries)
        global_idx += len(entries_batch)

    total_time = time.time() - start_time
    avg_time_per_entry = total_time / 1

    # Log timing information
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

# Run the translation process
translated_data = translate_entries(data, output_dir="./api_responses")
save_translations_to_jsonl(translated_data, json_output_file_path)

print(f"Translation complete. Translated data saved to {json_output_file_path}.")
