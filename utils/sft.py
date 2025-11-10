# utils/sft.py

from pathlib import Path
from typing import Any, Tuple
import os
import json
from fireworks import LLM, Dataset  # type: ignore


def submit_sft_job(
    *,
    dataset_path: Path,
    base_model: str,
    output_model: str,
    sft_settings: Any  # SFTSettings type
) -> Tuple[str, str, str]:
    """
    Submit a supervised fine-tuning job to Fireworks.
    
    Args:
        dataset_path: Path to the transformed SFT dataset (JSONL)
        base_model: Base model to fine-tune
        output_model: Name for the output model (e.g., "experiment_1_v1")
        sft_settings: SFT training configuration
        
    Returns:
        Tuple of (sft_job_name, dataset_id, output_model_path)
        - sft_job_name: Full path like "accounts/.../supervisedFineTuningJobs/..."
        - dataset_id: Dataset ID
        - output_model_path: Full path like "accounts/.../models/ft-..."
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        RuntimeError: If job submission fails
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print("Creating dataset from file...")
    dataset = Dataset.from_file(path=str(dataset_path))
    
    print("Uploading dataset to Fireworks...")
    dataset.sync()
    dataset_id = dataset.id
    print(f"Dataset uploaded with ID: {dataset_id}")
    
    print("Creating base model LLM instance...")
    # Use output_model with "-base" suffix as unique deployment ID
    # This ties the deployment to this specific experiment/run
    base_deployment_id = f"{output_model}-base"
    
    llm = LLM(
        model=base_model,
        id=base_deployment_id,
        deployment_type=sft_settings.deployment_type,
        api_key=os.environ['FIREWORKS_API_KEYd']
    )
    
    print("Submitting fine-tuning job...")
    sft_job = llm.create_supervised_fine_tuning_job(
        output_model,  # Display name (e.g., "experiment_1_v1")
        dataset,
        epochs=sft_settings.epochs,
        learning_rate=sft_settings.learning_rate,
        lora_rank=sft_settings.lora_rank,
        max_context_length=sft_settings.max_context_length
    )
    
    # Get the full job name and output model path from the job object
    sft_job_name = sft_job.name  # Full path: accounts/.../supervisedFineTuningJobs/...
    output_model_path = sft_job.output_model  # Full path: accounts/.../models/ft-...
    
    print(f"Fine-tuning job submitted: {sft_job_name}")
    print(f"Output model will be: {output_model_path}")
    print("Job will train asynchronously on Fireworks infrastructure")
    print(f"Monitor progress at: {sft_job.url}")
    
    return sft_job_name, dataset_id, output_model_path


def transform_for_sft(
    *,
    generated_data_path: Path,
    new_system_prompt_path: str | None,
    output_path: Path
) -> None:
    """
    Transform generated data for SFT by replacing/removing system prompts.
    
    Args:
        generated_data_path: Path to generated SFT data from Stage 1
        new_system_prompt_path: Path to new system prompt file, or None to remove
        output_path: Where to save the transformed data
        
    Raises:
        FileNotFoundError: If input files don't exist
    """
    if not generated_data_path.exists():
        raise FileNotFoundError(f"Generated data not found: {generated_data_path}")
    
    # Read new system prompt if provided
    new_system_prompt = None
    if new_system_prompt_path:
        prompt_path = Path(new_system_prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found: {new_system_prompt_path}")
        with open(prompt_path, 'r') as f:
            new_system_prompt = f.read().strip()
    
    # Transform each line
    transformed_lines = []
    
    with open(generated_data_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            messages = data['messages']
            
            # Replace or remove system message
            if messages[0]['role'] == 'system':
                if new_system_prompt:
                    messages[0]['content'] = new_system_prompt
                else:
                    # Remove system message entirely
                    messages = messages[1:]
            
            # Save in tutorial format
            transformed_lines.append({"messages": messages})
    
    # Write transformed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in transformed_lines:
            f.write(json.dumps(item) + '\n')
    
    print(f"Transformed {len(transformed_lines)} examples to {output_path}")