import time
import json
import os
import logging
from typing import Optional, List
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

load_dotenv()

class OpenAIBatchProcessor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def upload_file(self, file_path: str, purpose: str = "batch", retries: int = 5) -> 'File':
        """Uploads a file to OpenAI servers with retry logic."""
        for attempt in range(retries):
            try:
                with open(file_path, "rb") as file:
                    uploaded_file = self.client.files.create(
                        file=file,
                        purpose=purpose
                    )
                self.logger.info(f"File uploaded successfully: {uploaded_file.id}")
                return uploaded_file
            except OpenAIError as e:
                self.logger.error(f"Failed to upload file: {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries exceeded for file upload.")
                    raise

    def create_batch_job(self, input_file_id: str, endpoint: str, completion_window: str, retries: int = 5) -> 'BatchJob':
        """Creates a batch job with retry logic."""
        for attempt in range(retries):
            try:
                batch_job = self.client.batches.create(
                    input_file_id=input_file_id,
                    endpoint=endpoint,
                    completion_window=completion_window
                )
                self.logger.info(f"Batch job created: {batch_job.id}")
                return batch_job
            except OpenAIError as e:
                self.logger.error(f"Failed to create batch job: {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries exceeded for batch job creation.")
                    raise

    def download_results(self, output_file_id: str, result_file_name: str = "batch_job_results.jsonl", retries: int = 5) -> str:
        """Downloads the results file with retry logic."""
        for attempt in range(retries):
            try:
                result_content = self.client.files.retrieve(output_file_id).decode("utf-8")
                with open(result_file_name, "w", encoding="utf-8") as file:
                    file.write(result_content)
                self.logger.info(f"Results downloaded to {result_file_name}")
                return result_file_name
            except OpenAIError as e:
                self.logger.error(f"Failed to download results: {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries exceeded for downloading results.")
                    raise

    def load_results(self, result_file_name: str) -> Optional[List[dict]]:
        """Loads the results from the result file."""
        results = []
        try:
            with open(result_file_name, "r", encoding="utf-8") as file:
                for line in file:
                    json_object = json.loads(line.strip())
                    results.append(json_object)
            self.logger.info("Results loaded successfully.")
            return results
        except Exception as e:
            self.logger.error(f"Failed to load results from file: {e}")
            return None

    def process_batch(self, input_file_path: str, endpoint: str, completion_window: str) -> Optional[List[dict]]:
        """Processes a batch job from start to finish."""
        try:
            # Upload the input file
            uploaded_file = self.upload_file(input_file_path)

            # Create the batch job
            batch_job = self.create_batch_job(
                input_file_id=uploaded_file.id,
                endpoint=endpoint,
                completion_window=completion_window
            )

            # Monitor the batch job status
            polling_interval = 5  # Start with 5 seconds
            max_interval = 60     # Maximum interval of 60 seconds
            while batch_job.status not in ["completed", "failed", "cancelled"]:
                self.logger.info(f"Batch job status: {batch_job.status}. Waiting for {polling_interval} seconds before next check.")
                time.sleep(polling_interval)
                batch_job = self.client.batches.retrieve(batch_job.id)
                # Increase the polling interval up to the maximum
                polling_interval = min(polling_interval * 1.5, max_interval)

            # Check the final status
            if batch_job.status == "completed":
                result_file_id = batch_job.output_file_id
                result_file_name = self.download_results(result_file_id)
                results = self.load_results(result_file_name)
                return results
            else:
                self.logger.error(f"Batch job failed with status: {batch_job.status}")
                return None

        except Exception as e:
            self.logger.error(f"An error occurred during batch processing: {e}")
            return None

if __name__ == "__main__":
    # Initialize the OpenAIBatchProcessor
    api_key = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
    processor = OpenAIBatchProcessor(api_key)

    # Process the batch job
    input_file_path = "hellaswag_dev_batch.jsonl"
    output_file_path = "hellswag_dev_batch_output.jsonl"
    endpoint = "/v1/chat/completions"
    completion_window = "24h"

    # Process the batch job
    results = processor.process_batch(input_file_path, endpoint, completion_window)

    # Print the results
    if results is not None:
        with open(output_file_path, 'w') as output_file:
            json.dump(results, output_file, indent=2)
    else:
        print("Batch processing failed.")