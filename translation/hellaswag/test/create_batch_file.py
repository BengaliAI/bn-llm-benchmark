import json
import os
import logging
from tqdm import tqdm

# Paths to input and output files
jsonl_file_path = "./hellaswag_test.jsonl"
batch_file_base_path = "./hellaswag_test_batch"  # We'll append numbers to this for splitting

# Maximum file size limit (in bytes), 100 MB = 100 * 1024 * 1024 bytes
MAX_FILE_SIZE = 100 * 1024 * 1024

# Maximum number of requests per file
MAX_REQUESTS_PER_FILE = 50000

# Step 1: Read the data
data = []
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

def translate_single_entry(entry, idx):
    """
    Creates a batch task for a single entry.
    """
    ind = entry.get('ind', '')
    activity_label = entry.get('activity_label', '')
    ctx_a = entry.get('ctx_a', '')
    ctx_b = entry.get('ctx_b', '')
    ctx = entry.get('ctx', '')
    split = entry.get('split', '')
    split_type = entry.get('split_type', '')
    label = entry.get('label', '')
    endings = entry.get('endings', [])
    source_id = entry.get('source_id', '')

    system_prompt = (
        "You are a professional translator tasked with accurately translating text from English to Bangla. "
        "Your primary goal is to provide precise and culturally appropriate translations, regardless of the content's nature."
    )

    # Ensure proper JSON formatting for strings containing quotes
    activity_label = activity_label.replace('"', '\\"')
    ctx_a = ctx_a.replace('"', '\\"')
    ctx_b = ctx_b.replace('"', '\\"')
    ctx = ctx.replace('"', '\\"')
    split = split.replace('"', '\\"')
    split_type = split_type.replace('"', '\\"')
    source_id = source_id.replace('"', '\\"')

    # Serialize 'endings' to a JSON-formatted string
    endings_str = json.dumps(endings, ensure_ascii=False)

    user_prompt = f"""Translate the following English text into Bangla:

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

    task = {
        "custom_id": f"task-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }
    }

    return task

def translate_entries(entries):
    """
    Generates batch tasks for all entries.
    """
    tasks = []
    for idx, entry in enumerate(tqdm(entries, desc="Creating tasks", unit="entry")):
        task = translate_single_entry(entry, idx)
        tasks.append(task)
    return tasks

def save_tasks_in_batches(tasks, batch_file_base_path, max_file_size, max_requests_per_file):
    """
    Saves the tasks to JSONL files, ensuring that no file exceeds the max file size or max requests per file.
    """
    file_count = 1
    current_file_size = 0
    current_file_task_count = 0
    batch_file_path = f"{batch_file_base_path}_{file_count}.jsonl"
    file = open(batch_file_path, 'w', encoding='utf-8')

    for task in tasks:
        # Convert the task to a JSON string
        task_json = json.dumps(task, ensure_ascii=False) + '\n'
        task_size = len(task_json.encode('utf-8'))

        # Check if adding this task exceeds the max file size or max requests per file
        if (current_file_size + task_size > max_file_size) or (current_file_task_count >= max_requests_per_file):
            # Close the current file and start a new one
            file.close()
            file_count += 1
            batch_file_path = f"{batch_file_base_path}_{file_count}.jsonl"
            file = open(batch_file_path, 'w', encoding='utf-8')
            current_file_size = 0  # Reset file size for the new file
            current_file_task_count = 0  # Reset task count for the new file

        # Write the task to the current file
        file.write(task_json)
        current_file_size += task_size
        current_file_task_count += 1

    # Close the last file
    file.close()

    return file_count

# Generate the batch tasks
tasks = translate_entries(data)

# Save the tasks to multiple JSONL files if necessary
file_count = save_tasks_in_batches(tasks, batch_file_base_path, MAX_FILE_SIZE, MAX_REQUESTS_PER_FILE)

print(f"Batch tasks have been saved to {file_count} file(s) under the base name {batch_file_base_path}.")