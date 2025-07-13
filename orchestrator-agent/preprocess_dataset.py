import json
from datasets import load_dataset, Dataset

def load_jsonl_dataset(file_path: str) -> Dataset:
    """Load a JSONL file as HuggingFace Dataset."""
    return load_dataset("json", data_files=file_path)["train"]

def format_example(example: dict) -> dict:
    """Format each example into prompt-response pairs."""
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:"
    response = json.dumps(example['response'], ensure_ascii=False)
    return {"prompt": prompt, "response": response}

def format_dataset(dataset: Dataset) -> Dataset:
    """Apply formatting to dataset."""
    return dataset.map(format_example)

def save_dataset(dataset: Dataset, output_path: str) -> bool:
    """Save formatted dataset to JSONL."""
    try:
        dataset.to_json(output_path, orient="records", lines=True)
        return True
    except Exception as e:
        print(f"Error occurred: {e}")
        return False