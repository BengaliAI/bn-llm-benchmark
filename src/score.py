import os
import re
import csv
import argparse
from metrics import accuracy, response_error_rate


def clean_response(response):
    """
    Remove <think>...</think> content from the response if present.
    """
    if "<think>" in response and "</think>" in response:
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    return response.strip()


def extract_options(prompt, lang):
    """
    Extract the options from the prompt.

    For Bengali (lang == "bn"), extract everything after "বিকল্পসমূহ:".
    For English (lang == "en"), extract everything after "Options:".
    If the delimiter is not found, return the entire prompt (stripped).
    """
    if lang == "bn":
        delimiter = "বিকল্পসমূহ:"
    else:
        delimiter = "Options:"
    idx = prompt.find(delimiter)
    if idx != -1:
        return prompt[idx + len(delimiter) :].strip()
    else:
        return prompt.strip()


def process_csv_file(csv_path, lang, dataset=None):
    """
    Reads a CSV file and extracts the cleaned responses, options list, and answer keys.

    Expected CSV columns:
      - "Model Response" : The model's raw response that will be cleaned.
      - "Prompt": The prompt from which to extract the options.
      - "Ground Truth": The answer key.
    """
    responses = []
    options_list = []
    answer_keys = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean response from "Model Response" column
            cleaned_resp = clean_response(row["Model Response"])
            responses.append(cleaned_resp)

            # Extract options from "Prompt" column based on language
            if dataset == "boolq":
                options = ["true", "false"]
                options_list.append(options)
            else:
                options = extract_options(row["Prompt"], lang)
                options_list.append(options)

            # Ground truth as the answer key from "Ground Truth" column
            answer_keys.append(row["Ground Truth"])

    return responses, options_list, answer_keys


def calculate_scores(responses, options_list, answer_keys, dataset=None,lang=None):
    """
    Calculate and return the accuracy and response error rate using the provided metrics functions.
    """
    acc = accuracy(responses, answer_keys, dataset,lang)
    rer = response_error_rate(responses, options_list, dataset,lang)
    return acc, rer


def process_folder(folder_path, avoid_dirs):
    """
    Walk through all subdirectories in folder_path (excluding those in avoid_dirs).
    For each CSV file encountered:
      - Determine the language (Bengali if the subdirectory name ends with "bn",
        English if it ends with "en", defaulting to English otherwise).
      - Process the file to extract responses, options list, and answer keys.
      - Calculate and output the accuracy and response error rate to separate .txt files.
    """
    for root, dirs, files in os.walk(folder_path):
        # Filter out directories that are in the avoid list.
        dirs[:] = [d for d in dirs if not any(avoid in d for avoid in avoid_dirs)]

        # Determine language based on the current subdirectory name.
        subdir = os.path.basename(root)
        if "bn" in subdir:
            lang = "bn"
        else:
            lang = "en"

        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                responses = None
                options_list = None
                answer_keys = None
                dataset = None
                if "boolq" in file:
                    dataset = "boolq"
                    responses, options_list, answer_keys = process_csv_file(
                        csv_path, lang, "boolq"
                    )
                elif "gsm8k" in file:
                    dataset = "gsm8k"
                    responses, options_list, answer_keys = process_csv_file(
                        csv_path, lang
                    )
                elif "commonsenseqa" in file:
                    dataset = "cqsa"
                    responses, options_list, answer_keys = process_csv_file(
                        csv_path, lang
                    )
                elif "winogrande" in file:
                    dataset = "winogrande"
                    responses, options_list, answer_keys = process_csv_file(
                        csv_path, lang
                    )
                elif "hellaswag" in file:
                    dataset = "hellaswag"
                    responses, options_list, answer_keys = process_csv_file(
                        csv_path, lang
                    )
                else:
                    responses, options_list, answer_keys = process_csv_file(
                        csv_path, lang
                    )
                acc, rer = calculate_scores(
                    responses, options_list, answer_keys, dataset,lang
                )

                # Determine output file names based on the CSV name.
                if file.endswith("_responses.csv"):
                    base_name = file[: -len("_responses.csv")]
                else:
                    base_name = file[:-4]  # remove the .csv extension

                acc_filename = f"{base_name}_accuracy.txt"
                rer_filename = f"{base_name}_rer.txt"

                acc_path = os.path.join(root, acc_filename)
                rer_path = os.path.join(root, rer_filename)

                # Write the accuracy result.
                with open(acc_path, "w", encoding="utf-8") as f_acc:
                    f_acc.write(str(acc))

                # Write the response error rate result.
                with open(rer_path, "w", encoding="utf-8") as f_rer:
                    f_rer.write(str(rer))

                print(f"Processed CSV file: {csv_path}")
                print(f"Accuracy stored in: {acc_path}")
                print(f"RER stored in: {rer_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process CSV files in subdirectories to calculate accuracy and response error rate."
    )
    parser.add_argument(
        "inference_output_directory",
        type=str,
        help="Path to the folder containing CSV files to process.",
    )
    args = parser.parse_args()

    folder_path = args.inference_output_directory
    avoid_dirs = []
    process_folder(folder_path, avoid_dirs)


if __name__ == "__main__":
    main()
