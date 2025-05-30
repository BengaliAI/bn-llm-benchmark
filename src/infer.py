import os
import json
import csv
import logging
import time
import random
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import backoff
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
import ollama
import ast
import pandas as pd

load_dotenv(find_dotenv())
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
BASE_URL = "https://api.together.xyz/v1"
main_api_key = TOGETHER_API_KEY

MAX_RETRIES = 10
MAX_RETRY_TIME = 600  # 10 minutes
RATE_LIMIT_INITIAL_BACKOFF = 3  # seconds
GENERAL_ERROR_INITIAL_BACKOFF = 2  # seconds

@dataclass
class RequestItem:
    """Represents a single chat completion request with its metadata"""
    id: int
    messages: List[Dict[str, str]]
    model: str = DEFAULT_MODEL
    metadata: Dict[str, Any] = None
    result: Optional[ChatCompletion] = None
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attempts: int = 0
    
    @property
    def duration(self) -> float:
        """Calculate time taken for the request"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

class APIException(Exception):
    """Base exception for API errors"""
    def __init__(self, message, is_rate_limit=False, status_code=None):
        self.message = message
        self.is_rate_limit = is_rate_limit
        self.status_code = status_code
        super().__init__(self.message)

def save_result(result: RequestItem, output_file: str) -> None:
    """Save a single result to the output file"""
    result_dict = {
        'request': {
            'model': result.model,
            'messages': result.messages
        },
        'response': result.result.model_dump() if result.result else None,
        'error': result.error,
        'duration': result.duration,
        'attempts': result.attempts,
        'metadata': result.metadata
    }
    
    # Use a lock to avoid concurrent writes
    with open(output_file, 'a', encoding="utf-8") as f:
        f.write(json.dumps(result_dict) + '\n')

async def process_chat_request(
    client: AsyncOpenAI,
    request: RequestItem,
) -> RequestItem:
    """Process a chat completion request with retries for all errors"""
    if request.attempts >= MAX_RETRIES:
        logger.error(f"Request {request.id} failed after {MAX_RETRIES} attempts: {request.error}")
        return request
        
    request.attempts += 1
    backoff_time = 0
    
    try:
        # Update start time for accurate duration measurement
        if request.attempts == 1:
            request.start_time = time.time()
            
        response = await client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=0,
        )
        
        request.result = response
    
    
        request.end_time = time.time()
        return request
        
    except Exception as e:
        error_message = str(e)
        request.error = error_message
        
        # Determine error type for appropriate backoff
        is_rate_limit = "rate limit" in error_message.lower() or "429" in error_message
        status_code = None
        
        # Extract status code if present
        if "status_code=" in error_message:
            try:
                status_code = int(error_message.split("status_code=")[1].split()[0])
            except:
                pass
        
        # Calculate backoff time with exponential strategy and jitter
        if is_rate_limit:
            # Longer backoff for rate limits (exponential with base 2)
            backoff_time = RATE_LIMIT_INITIAL_BACKOFF * (2 ** (request.attempts - 1))
            logger.warning(f"Rate limit hit for request {request.id}. Attempt {request.attempts}/{MAX_RETRIES}")
        else:
            # Shorter backoff for other errors (exponential with base 1.5)
            backoff_time = GENERAL_ERROR_INITIAL_BACKOFF * (1.5 ** (request.attempts - 1))
            logger.warning(f"Request {request.id} failed with error: {error_message}. Attempt {request.attempts}/{MAX_RETRIES}")
        
        # Add jitter to prevent synchronized retries
        backoff_time = backoff_time * (0.8 + 0.4 * random.random())
        
        # Cap backoff time
        backoff_time = min(backoff_time, 60)  # Maximum 60 second backoff
        
        logger.info(f"Backing off for {backoff_time:.2f}s before retry")
        await asyncio.sleep(backoff_time)
        
        # Recursive retry with updated attempt count
        return await process_chat_request(client, request)

async def process_batch(
    batch: List[RequestItem],
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    output_file: str,
    file_lock: asyncio.Lock
) -> List[RequestItem]:
    """Process a batch of chat completion requests with concurrency control"""
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(
        api_key=main_api_key,
        base_url=BASE_URL,
    )
    
    
    # Initialize an empty list to collect all results
    all_results = []
    
    # Create a queue to collect completed results for processing
    result_queue = asyncio.Queue()
    
    # Start a task to process results as they come in
    async def process_results():
        while True:
            result = await result_queue.get()
            
            # Save result to file immediately with lock to prevent concurrent writes
            async with file_lock:
                result_dict = {
                    'request': {
                        'model': result.model,
                        'messages': result.messages
                    },
                    'response': result.result.model_dump() if result.result else None,
                    'error': result.error,
                    'duration': result.duration,
                    'attempts': result.attempts,
                    'metadata': result.metadata
                }
                
                # Append to output file
                with open(output_file, 'a', encoding="utf-8") as f:
                    f.write(json.dumps(result_dict) + '\n')
            
            # Mark task as done
            result_queue.task_done()
    
    # Start the result processor
    asyncio.create_task(process_results())
    
    async def process_with_semaphore(req: RequestItem):
        async with semaphore:
            result = await process_chat_request(client, req)
            pbar.update(1)
            duration = result.duration
            
            if result.error and not result.result:
                pbar.set_description(f"Req {req.id}: {duration:.1f}s, {req.attempts} attempts (FAILED)")
            else:
                pbar.set_description(f"Req {req.id}: {duration:.1f}s, {req.attempts} attempts")
            
            # Add result to queue for processing
            await result_queue.put(result)
            all_results.append(result)
            
            return result
    
    # Create and run tasks for all requests
    tasks = [process_with_semaphore(req) for req in batch]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    # Wait for all results to be processed
    await result_queue.join()
    
    return results

async def parallel_process_chat(
    requests: List[Dict[str, Any]],
    output_file: str = "results.jsonl",
    max_concurrency: int = 10
) -> None:
    """
    Process multiple chat completion requests in parallel and save results to a JSONL file
    
    Args:
        requests: List of request dictionaries containing messages and optional model
        output_file: Path to output JSONL file
        max_concurrency: Maximum number of concurrent requests
    """
    # Create/clear output file before starting
    with open(output_file, 'w', encoding="utf-8") as f:
        pass
    
    # Create a lock for file access
    file_lock = asyncio.Lock()
    
    # Prepare request items
    request_items = []
    for i, req in enumerate(requests, 1):
        request_items.append(RequestItem(
            id=i,
            messages=req.get("messages", []),
            model=req.get("model", DEFAULT_MODEL),
            metadata=req.get("metadata")
        ))
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # Process requests with progress bar
    total_start_time = time.time()
    with tqdm(total=len(request_items), desc="Processing chat completions") as pbar:
        results = await process_batch(request_items, semaphore, pbar, output_file, file_lock)
    
    # Calculate final statistics
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results if r.result is not None)
    failed = sum(1 for r in results if r.result is None)
    avg_duration = sum(r.duration for r in results) / len(results)
    avg_attempts = sum(r.attempts for r in results) / len(results)
    
    # Log summary
    logger.info(f"Processing complete in {total_time:.2f}s")
    logger.info(f"Successful: {successful}, Failed: {failed}")
    logger.info(f"Average request duration: {avg_duration:.2f}s")
    logger.info(f"Average attempts per request: {avg_attempts:.2f}")
    logger.info(f"Results written to {output_file}")

def run_parallel_chat_completions(
    requests: List[Dict[str, Any]],
    output_file: str = "results.jsonl",
    max_concurrency: int = 5
) -> None:
    """
    Process multiple chat completion requests in parallel
    
    Example usage:
    requests = [
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke"}
            ],
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Optional, will use DEFAULT_MODEL if not specified
        },
        # ... more requests ...
    ]
    run_parallel_chat_completions(requests, output_file="results.jsonl")
    """
    asyncio.run(parallel_process_chat(requests, output_file, max_concurrency))

def save_results_to_csv(jsonl_file, csv_file, dataset_name, model_name, system_message):
    with open(jsonl_file, "r", encoding="utf-8") as infile, open(csv_file, "a", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        for line in infile:
            data = json.loads(line.strip())

            # Extract fields
            prompt = data["request"]["messages"][1]["content"]
            model_response = data["response"]["choices"][0]["message"]["content"] if data["response"] else "EMPTY RESPONSE" 
            ground_truth = data["metadata"]["ground_truth"] if "metadata" in data and "ground_truth" in data["metadata"] else "None"
            question_id = data["metadata"]["question_id"] if "metadata" in data and "question_id" in data["metadata"] else "UNKNOWN"
        
            writer.writerow([question_id, dataset_name, model_name, system_message, prompt, model_response, ground_truth])



def query_ollama(model_name, input_text, system_message):
    try: 
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text},
            ],
            options={"temperature":0.0}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Exception: {str(e)}"


def infer(
    dataset_name,
    model_name,
    file_path,
    output_csv,
    system_message,
    input_msg,
    process_question,
    together,
    dir_save
):

    with open(file_path, "r", encoding="utf-8") as file:
        questions = [json.loads(line.strip()) for line in file]

    with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Question ID",
                "Dataset Name",
                "Model Name",
                "System Prompt",
                "Prompt",
                "Model Response",
                "Ground Truth",
            ]
        )
    qid_order = []
    qid_dummy = 1
    if together:
        requests = []
        for question in questions:
            input_text_model, ground_truth, qid = process_question(
                input_msg, question
            )
            if qid == None:
                qid = qid_dummy
                qid_dummy += 1
            
            qid_order.append(qid)
            request = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_text_model},
                ],
                "model": model_name,
                "metadata": {
                    "ground_truth": ground_truth,
                    "question_id": qid,
                }
            }
            requests.append(request)
        
        output_file_jsonl = os.path.join(dir_save, f"{dataset_name}_{model_name.replace('/','-')}_results.jsonl")
        run_parallel_chat_completions(
            requests=requests, 
            output_file=output_file_jsonl,
            max_concurrency=10
        )
            
        save_results_to_csv(output_file_jsonl, output_csv,dataset_name, model_name, system_message)
    else:
        for question in tqdm(questions, desc=f"Inferencing with {model_name}"):
            input_text_model, ground_truth, qid = process_question(
                input_msg, question
            )
            if qid == None:
                qid = qid_dummy
                qid_dummy += 1
                
            qid_order.append(qid)
            response = query_ollama(
                model_name, input_msg, system_message
            ).strip()
            with open(output_csv, "a", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        qid,
                        dataset_name,
                        model_name,
                        system_message,
                        input_text_model,
                        response,
                        ground_truth,
                    ]

                )
    df = pd.read_csv(path_csv, encoding="utf-8")
    df["Question ID"] = pd.Categorical(df["Question ID"], categories=qid_order, ordered=True)
    df_sorted = df.sort_values("Question ID")
    df_sorted.to_csv(f"{path_csv}", index=False)
       


if __name__ == "__main__":
    from prompt_types import (PromptType,
                              parse_response_rer,
                              parse_response
                              )
    import argparse
    import os
    from score import calculate_scores

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name')
    parser.add_argument('--dataset_path')
    parser.add_argument('--dir_save')
    parser.add_argument('--language', default='en')
    parser.add_argument('--together', action='store_true')
    
    parser.add_argument(
            '--model', nargs='+',
            default = [
                # "llama3:8b",
                # "llama3:70b",
                # "llama3.1:8b",
                # "llama3.1:70b",
                # "llama3.2:3b",
                # "llama3.3:70b",
                # "deepseek-r1:7b",
                # "deepseek-r1:14b",
                # "deepseek-r1:70b",
                # "qwen2.5:7b",
                # "qwen2.5:14b",
                # "qwen2.5:72b",
                # "mistral:7b",
                # "mistral-small:24b"
                # "mistral-large:123b"
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "Qwen/Qwen2.5-7B-Instruct-Turbo",
                "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "mistralai/Mistral-Small-24B-Instruct-2501"
            ]
        )

    args = parser.parse_args()
    lang = args.language
    pt = PromptType(lang)
    SYSTEM_MESSAGE = pt.get_sys_msg(args.dataset_name)
    INPUT_MESSAGE = pt.get_inp_msg(args.dataset_name)
    
    process_question = pt.get_process_func(args.dataset_name)
    dataset_folder = f"{args.dataset_name}-{args.language}"
    _dir_save = os.path.join(args.dir_save, dataset_folder)
    print('creating save dir ', _dir_save)
    os.makedirs(_dir_save, exist_ok = True)
    
    for model_name in args.model:
        model_name_file = model_name
        if args.together:
            model_name_file = model_name_file.replace("/", "-")
            
        output_csv = f"{args.dataset_name}_{model_name_file}_responses.csv"
        path_csv = os.path.join(_dir_save, output_csv)
        infer(
            args.dataset_name,
            model_name,
            args.dataset_path,
            path_csv,
            SYSTEM_MESSAGE,
            INPUT_MESSAGE,
            process_question,
            args.together,
            _dir_save
        )
        
        
        resp = parse_response(path_csv, "Model Response")
        gt = parse_response(path_csv, "Ground Truth")
        resp_rer = parse_response_rer(path_csv)

        # acc, rer = calculate_scores(resp, resp_rer, gt, args.language, args.dataset_name)
        
        # score_file = f"{args.dataset_name}_{model_name_file}_scores.txt"
        
        # path_score = os.path.join(_dir_save, score_file)
        
        # with open(path_score, "w", encoding="utf-8") as file:
        #     file.write(f"Accuracy = {acc * 100:.2f}%\n")
        #     file.write(f"Response Error Rate = {rer * 100:.2f}%\n")
            
       




