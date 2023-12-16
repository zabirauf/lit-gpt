"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def prepare(
    destination_path: Path = Path("/notebooks/corrections-slm"),
    checkpoint_dir: Path = Path("/notebooks/lit-gpt/checkpoints/stabilityai/stablelm-zephyr-3b"),
    mask_inputs: bool = False,  # as in alpaca-lora
    ignore_index: int = -1,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_paths = [destination_path / "train.json", destination_path / "validation.json" ]
    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    print("Loading data file...")
    for data_file_path in data_file_paths:
        assert_not_missing_file(data_file_path) 

        with open(data_file_path, "r", encoding="utf-8") as file:
            data_set = json.load(file)

        print(f"Data set {data_file_path.stem} has {len(data_set):,} samples")

        print("Processing data ...")
        data_set = [
            prepare_sample(
                example=sample,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                mask_inputs=mask_inputs,
                ignore_index=ignore_index,
            )
            for sample in tqdm(data_set)
        ]
        torch.save(data_set, destination_path / f"{data_file_path.stem}.pt")

def assert_not_missing_file(file_path: Path):
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"File {file_path} not found")

def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            f"<|system|>\n{example['instruction']}<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n"
        )
    return (
        f"<|user|>\n{example['instruction']}<|endoftext|>\n<|assistant|>\n"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
