# utils/generate.py

from pathlib import Path
from typing import Any
import time
import os
from dotenv import load_dotenv
import requests

from fireworks import Dataset, BatchInferenceJob

# Load environment variables
load_dotenv()


def get_api_key() -> str:
    """
    Get Fireworks API key from environment.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If FIREWORKS_API_KEY not found in environment
    """
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError(
            "FIREWORKS_API_KEY not found in environment. "
            "Please add it to your .env file."
        )
    return api_key


def get_account_id() -> str:
    """
    Get Fireworks account ID from environment.
    
    Returns:
        Account ID string
        
    Raises:
        ValueError: If FIREWORKS_ACCOUNT_ID not found in environment
    """
    account_id = os.getenv('FIREWORKS_ACCOUNT_ID')
    if not account_id:
        raise ValueError(
            "FIREWORKS_ACCOUNT_ID not found in environment. "
            "Please add it to your .env file."
        )
    return account_id


def submit_batch_job(
    *,
    input_file: Path,
    generation_configs: Any  # GenerationConfigs type
) -> str:
    """
    Upload file and submit a batch inference job to Fireworks using the SDK.
    
    Args:
        input_file: Path to the transformed JSONL file (OpenAI chat format)
        generation_configs: Generation config object with model, temperature, etc.
        
    Returns:
        Batch job ID string
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    api_key = get_api_key()
    
    print("Creating dataset from file...")
    dataset = Dataset.from_file(path=str(input_file))
    
    print("Uploading dataset to Fireworks...")
    dataset.sync()
    print(f"Dataset uploaded with ID: {dataset.id}")

    print("Creating batch inference job...")
    inference_parameters = {
        "max_tokens": generation_configs.max_tokens,
        "temperature": generation_configs.temperature,
        "top_p": generation_configs.top_p
    }
    
    job = BatchInferenceJob.create(
        model=generation_configs.model,
        input_dataset_id=dataset.id,
        inference_parameters=inference_parameters,
        api_key=api_key
    )
    
    print(f"Batch job created with ID: {job.id}")
    
    return job.id


def poll_and_download_results(
    *,
    batch_job_id: str,
    output_path: Path,
    poll_interval: int = 60,
    timeout: int = 86400  # 24 hours
) -> None:
    """
    Poll batch job until complete and download results.
    
    Args:
        batch_job_id: The batch job ID to poll
        output_path: Where to save the downloaded results
        poll_interval: Seconds between status checks (default: 60)
        timeout: Maximum seconds to wait (default: 86400 = 24 hours)
        
    Raises:
        TimeoutError: If job doesn't complete within timeout
        RuntimeError: If job fails
    """
    api_key = get_api_key()
    account_id = get_account_id()
    
    start_time = time.time()
    
    print(f"Polling batch job {batch_job_id}...")
    
    job = None
    while True:
        # Check if timeout reached
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Batch job did not complete within {timeout} seconds"
            )
        
        # Get job status
        job = BatchInferenceJob.get(
            job_id=batch_job_id,
            account=account_id,
            api_key=api_key
        )
        
        if job is None:
            raise RuntimeError(f"Batch job {batch_job_id} not found")
        
        print(f"Status: {job.state}")
        
        if job.state == 'JOB_STATE_COMPLETED':
            print("Batch job completed!")
            break
        elif job.state == 'JOB_STATE_FAILED':
            raise RuntimeError("Batch job failed")
        elif job.state == 'JOB_STATE_CANCELLED':
            raise RuntimeError("Batch job was cancelled")
        elif job.state in ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING']:
            # Job still running, wait and poll again
            time.sleep(poll_interval)
        else:
            print(f"Unknown status: {job.state}, continuing to poll...")
            time.sleep(poll_interval)
    
    # Download results using REST API (SDK doesn't expose dataset download)
    print("Downloading results...")
    output_dataset_id = job.output_dataset_id
    
    base_url = "https://api.fireworks.ai/inference/v1"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Download the output dataset
    download_response = requests.get(
        f"{base_url}/datasets/{output_dataset_id}:download",
        headers=headers
    )
    download_response.raise_for_status()
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(download_response.content)
    
    print(f"Results saved to {output_path}")