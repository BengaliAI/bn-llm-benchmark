"""
LLM Judge System

This module implements a system for evaluating LLM responses to multiple-choice questions.
It uses OpenAI's GPT models to act as an impartial judge, determining whether an LLM's
answer matches the correct answer in meaning, even if the wording differs.

The system uses a few-shot learning approach with examples to guide the judge LLM
in making consistent evaluations.

Requirements:
    - openai
    - pydantic
    - python-dotenv
    - backoff
    - requests
    - pandas

Environment Variables:
    - OPENAI_API_KEY: Your OpenAI API key
"""

import logging
import backoff
import requests
import os
import sys
from typing import Tuple, List
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import concurrent.futures
import re

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_TRIES = 10

# Load environment variables from .env file
load_dotenv(find_dotenv())

def backoff_hdlr(details):
    """Handler for logging backoff retries."""
    logger.warning(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs {kwargs}".format(**details)
    )

def handle_errors(func):
    """Decorator for handling various API and network errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as errh:
            logger.error(f"HTTP Error: {errh}")
            raise
        except requests.exceptions.ConnectionError as errc:
            logger.error(f"Error Connecting: {errc}")
            raise
        except requests.exceptions.Timeout as errt:
            logger.error(f"Timeout Error: {errt}")
            raise
        except requests.exceptions.RequestException as err:
            logger.error(f"Request Exception: {err}")
            raise
        except Exception as e:
            logger.error(f"Unknown error: {e}")
            raise
    return wrapper

# Initialize the OpenAI client
client = OpenAI()

class Evaluation(BaseModel):
    """
    Pydantic model for structuring the evaluation response.

    Attributes:
        reasoning (str): Detailed explanation of the evaluation decision.
        verdict (str): Final judgment, either "Correct" or "Incorrect".
    """
    reasoning: str
    verdict: str

def get_eval_prompt(prompt_from_response: str, ground_truth: str, model_response: str) -> str:
    """
    Generates a prompt for the judge LLM to evaluate an answer.
    """

    few_shot_example = """Example:

Prompt:
<prompt>
Question:

What is 2+2?

Options:
- 3
- 4
- 5
- 6

</prompt>

Ground Truth:
<ground_truth>
4
</ground_truth>

Model Response:
<model_response>
The answer is 4.
</model_response>

Evaluation:
{
    "reasoning": "The LLM's answer clearly indicates the correct choice even with additional phrasing.",
    "verdict": "Correct",
}
----
"""
    prompt = f"""You are an impartial judge tasked with evaluating the accuracy of an AI language model's (LLM) response to a multiple-choice question. Your goal is to determine whether the LLM's answer is correct, even if it does not exactly match the ground truth wording, as long as it conveys the same exact meaning.

Follow these instructions carefully and output only a JSON object in the exact format specified (with no additional text):

1. Accuracy: Does the LLM's answer convey the same meaning as the correct answer?
2. Relevance: Is the LLM's answer directly addressing the question?
3. Completeness: Does the LLM's answer include all necessary information to be considered correct?

Below is an example of the expected JSON format:
{{
    "reasoning": "Your detailed reasoning here...",
    "verdict": "Correct",
}}

Few-shot example:
{few_shot_example}

Now, evaluate the following instance.

Prompt:
<prompt>
{prompt_from_response}
</prompt>

Ground Truth:
<ground_truth>
{ground_truth}
</ground_truth>

Model Response:
<model_response>
{model_response}
</model_response>

Output only the JSON object.
"""
    return prompt

def evaluate_llm_answer(prompt_from_response: str, correct_answer: str, llm_answer: str) -> Evaluation:
    """
    Evaluates an LLM's answer using the judge LLM.
    """
    prompt = get_eval_prompt(prompt_from_response, correct_answer, llm_answer)
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, Exception),
        max_tries=MAX_TRIES,
        on_backoff=backoff_hdlr
    )
    @handle_errors
    def _make_api_call():
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an impartial judge tasked with evaluating the accuracy of an AI language model's (LLM) response to a question. Your goal is to determine whether the LLM's answer is correct, even if it does not exactly match the ground truth wording, as long as it conveys the same exact meaning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format=Evaluation,
            seed=42,
        )
        return completion

    try:
        completion = _make_api_call()
        evaluation = completion.choices[0].message.parsed
        logger.debug(f"Evaluation result: {evaluation}")
        return evaluation
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def process_row(idx, row, csv_file):
    """
    Process a single row: parse the prompt and evaluate the LLM's answer.
    Returns a tuple: (verdict, reasoning, error, judge_prompt)
    """
    try:
        prompt_from_response = row["Prompt"]
    except Exception as e:
        logger.error(f"Row {idx} in file {csv_file}: Failed to parse prompt. Error: {e}")
        return "", "", f"Prompt parsing error: {e}", ""
    
    llm_answer = row["Model Response"]
    correct_answer = row["Ground Truth"]
    
    try:
        # Generate the judge prompt that will be sent to the LLM judge.
        judge_prompt = get_eval_prompt(prompt_from_response, correct_answer, llm_answer)
        evaluation = evaluate_llm_answer(prompt_from_response, correct_answer, llm_answer)
        logger.info(f"Row {idx} in file {csv_file}: Successfully evaluated question.")
        return evaluation.verdict, evaluation.reasoning, "", judge_prompt
    except Exception as e:
        error_message = str(e)
        # Check if the error indicates that the token limit was reached.
        if "Could not parse response content as the length limit was reached" in error_message:
            logger.error(f"Row {idx} in file {csv_file}: Evaluation failed due to token limit. Error: {error_message}")
            return ("Incorrect",
                    "Could not parse response content as the length limit was reached",
                    error_message,
                    judge_prompt)
        else:
            logger.error(f"Row {idx} in file {csv_file}: Evaluation failed. Error: {error_message}")
            return "", "", error_message, judge_prompt

def main():
    """
    Processes a folder of CSV files with names ending in '_responses.csv'.
    For each CSV file, it evaluates each row in parallel (using threads),
    adds new columns (Judge Prompt, Judge Verdict, Judge Reasoning, Judge Error) to the data,
    writes a new CSV (tagged with _judge.csv), and creates a final text file that appends
    the overall accuracy computed from the evaluations.
    The output CSV will contain an "id" column to ensure rows are in the original order.
    """
    if len(sys.argv) < 2:
        print("Usage: python llm_eval.py folder_path")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        logger.error(f"Folder {folder_path} does not exist or is not a directory.")
        sys.exit(1)
    
    # Only process CSV files ending with _responses.csv
    csv_files = [f for f in os.listdir(folder_path) if f.endswith("_responses.csv")]
    if not csv_files:
        logger.warning("No CSV files with '_responses.csv' found in the folder.")
        sys.exit(0)
    
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read CSV file {csv_path}: {e}")
            continue
        
        n_rows = len(df)
        # Pre-allocate lists for storing results
        judge_verdicts = [None] * n_rows
        judge_reasonings = [None] * n_rows
        judge_errors = [None] * n_rows
        judge_prompts = [None] * n_rows
        
        logger.info(f"Starting evaluation on {n_rows} rows in file {csv_file}.")

        base_filename = csv_file[:-len("_responses.csv")]
        output_csv_path = os.path.join(folder_path, base_filename + "_judge.csv")
        output_txt_path = os.path.join(folder_path, base_filename + "_judge.txt")

        if os.path.exists(output_csv_path) or os.path.exists(output_txt_path):
            logger.info(f"Judge output files for {csv_file} already exist. Skipping processing.")
            continue

        max_workers = 512
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_row, idx, row, csv_file): idx
                for idx, row in df.iterrows()
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    verdict, reasoning, error, prompt_used = future.result()
                    judge_verdicts[idx] = verdict
                    judge_reasonings[idx] = reasoning
                    judge_errors[idx] = error
                    judge_prompts[idx] = prompt_used
                except Exception as e:
                    logger.error(f"Row {idx} in file {csv_file} raised an exception: {e}")
                    judge_verdicts[idx] = ""
                    judge_reasonings[idx] = ""
                    judge_errors[idx] = str(e)
                    judge_prompts[idx] = ""
        
        # Append new columns to the DataFrame
        df["Judge Verdict"] = judge_verdicts
        df["Judge Reasoning"] = judge_reasonings
        df["Judge Error"] = judge_errors
        df["Judge Prompt"] = judge_prompts

        
        # Columns order: Question ID, Prompt, Model Response, Ground Truth, Judge Prompt, Judge Verdict, Judge Reasoning, Judge Error
        cols = list(df.columns)
        if "Ground Truth" in cols and "Judge Verdict" in cols and "Judge Prompt" in cols:
            gt_index = cols.index("Ground Truth")
            cols.remove("Judge Prompt")
            cols.insert(gt_index + 1, "Judge Prompt")
            df = df[cols]
        
        total = n_rows
        correct = sum(1 for verdict in judge_verdicts if verdict == "Correct")
        accuracy = correct / total if total > 0 else 0.0

        try:
            df.to_csv(output_csv_path, index=False, encoding="utf-8")
            logger.info(f"Saved judge CSV to {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to write judge CSV file {output_csv_path}: {e}")

        # Save the resulting accuracy to a file.
        accuracy_filename = csv_file.replace("_responses.csv", "_llm_judge_accuracy.txt")
        accuracy_path = os.path.join(folder_path, accuracy_filename)
        try:
            with open(accuracy_path, "w", encoding="utf-8") as f:
                f.write(str(accuracy))
            logger.info(f"Saved llm eval accuracy to {accuracy_path}")
        except Exception as e:
            logger.error(f"Failed to write accuracy file {accuracy_path}: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()