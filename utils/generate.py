# utils/generate.py

from pathlib import Path
import time
import os
from dotenv import load_dotenv
import subprocess
import re
from fireworks import Dataset, BatchInferenceJob    # type: ignore

from utils.config import GenerationConfigs

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
    generation_configs: GenerationConfigs,
    job_id: str
) -> None:
    """
    Upload file and submit a batch inference job to Fireworks using the SDK.
    
    Args:
        input_file: Path to the transformed JSONL file (OpenAI chat format)
        generation_configs: Generation config object with model, temperature, etc.
        job_id: Unique identifier for this batch job
        
    Returns:
        Batch job ID string
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
        "top_p": generation_configs.top_p,
        "n": generation_configs.n,
    }
    
    BatchInferenceJob.create(
        model=generation_configs.model,
        input_dataset_id=dataset.id,
        inference_parameters=inference_parameters,
        job_id=job_id,
        api_key=api_key
    )
    

def poll_and_download_results(
    *,
    batch_job_id: str,
    output_path: Path,
    poll_interval: int = 60,
    timeout: int = 86400  # 24 hours
) -> str:
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
        
        # Get job status using SDK
        job = BatchInferenceJob.get(
            job_id=batch_job_id,
            account=account_id,
            api_key=api_key
        )
        
        if job is None:
            raise RuntimeError(f"Batch job {batch_job_id} not found")
        
        # Convert state enum to string for display
        state_name = job.state
        print(f"Status: {state_name}")
        
        # Check job state (state is an enum, 3 = COMPLETED)
        if job.state == 3:  # JOB_STATE_COMPLETED
            print("Batch job completed!")
            break
        elif job.state == 4:  # JOB_STATE_FAILED
            raise RuntimeError("Batch job failed")
        elif job.state == 5:  # JOB_STATE_CANCELLED
            raise RuntimeError("Batch job was cancelled")
        elif job.state in [1, 2]:  # JOB_STATE_PENDING, JOB_STATE_RUNNING
            # Job still running, wait and poll again
            time.sleep(poll_interval)
        else:
            print(f"Unknown status: {job.state}, continuing to poll...")
            time.sleep(poll_interval)
    
    # Download results using firectl
    print("Downloading results...")
    output_dataset_id = job.output_dataset_id
    dataset_id = output_dataset_id.split('/')[-1]

    try:
        result = subprocess.run(
            ['firectl', 'download', 'dataset', dataset_id, '--output-dir', str(output_path)],
            check=True,
            capture_output=True,
            text=True
        )
        
        dataset_pattern = r'dataset/([^/]+)/BIJOutputSet\.jsonl'
        match = re.search(dataset_pattern, result.stdout)
        if match:
            dataset_id = match.group(1)
        else:
            raise Exception(f'Data downloaded to {str(output_path)} but can\'t find where!')
        
        file_path = output_path / "dataset" / dataset_id / "BIJOutputSet.jsonl"
        print(f"Results saved to {output_path}")
        return file_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download dataset: {e.stderr}")