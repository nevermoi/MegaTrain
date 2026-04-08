"""Dataset implementations for training.

Supports any HuggingFace model by using the tokenizer's native chat_template.
Supports alpaca and sharegpt data formats, local JSON/JSONL files, HuggingFace Hub,
and legacy Arrow datasets via a dataset_info.json registry.
"""

import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# File extension to HuggingFace datasets type mapping
FILEEXT2TYPE = {
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".txt": "text",
}


def load_dataset_info(dataset_dir: str) -> dict:
    """Load dataset_info.json from a directory.

    Args:
        dataset_dir: Directory containing dataset_info.json

    Returns:
        Dictionary mapping dataset names to their configurations
    """
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset_info.json not found in {dataset_dir}")
    with open(info_path, "r") as f:
        return json.load(f)


def load_dataset_by_name(dataset_name: str, dataset_dir: str = "data"):
    """Load a dataset by name from dataset_info.json.

    Supports local files (JSON/JSONL) and HuggingFace Hub datasets.

    Args:
        dataset_name: Name of the dataset in dataset_info.json
        dataset_dir: Directory containing dataset_info.json and local files

    Returns:
        Tuple of (dataset, dataset_attr) where dataset_attr is the config dict
    """
    from datasets import load_dataset, load_from_disk

    dataset_info = load_dataset_info(dataset_dir)
    if dataset_name not in dataset_info:
        available = list(dataset_info.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in dataset_info.json. "
            f"Available: {available}"
        )

    attr = dataset_info[dataset_name]
    split = attr.get("split", "train")
    num_samples = attr.get("num_samples", None)

    if "hf_hub_url" in attr:
        logger.info(f"Loading from HuggingFace Hub: {attr['hf_hub_url']}")
        kwargs = {}
        if "subset" in attr:
            kwargs["name"] = attr["subset"]
        ds = load_dataset(attr["hf_hub_url"], split=split, **kwargs)

    elif "file_name" in attr:
        file_path = os.path.join(dataset_dir, attr["file_name"])
        logger.info(f"Loading from local file: {file_path}")
        ext = Path(file_path).suffix.lower()

        if ext in FILEEXT2TYPE:
            ds = load_dataset(FILEEXT2TYPE[ext], data_files=file_path, split="train")
        elif os.path.isdir(file_path):
            ds = load_from_disk(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' must have either 'hf_hub_url' or 'file_name'"
        )

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    logger.info(f"Loaded {len(ds)} samples, columns: {ds.column_names}")
    return ds, attr


def convert_alpaca(sample: dict, columns: dict) -> tuple:
    """Convert an alpaca-format sample to (query, response).

    Args:
        sample: Dataset sample
        columns: Column name mapping from dataset_info.json

    Returns:
        Tuple of (query, response, system)
    """
    prompt_col = columns.get("prompt", "instruction")
    query_col = columns.get("query", "input")
    response_col = columns.get("response", "output")
    system_col = columns.get("system", "system")

    instruction = sample.get(prompt_col, "")
    query = sample.get(query_col, "")
    response = sample.get(response_col, "")
    system = sample.get(system_col, "")

    if query:
        user_content = f"{instruction}\n{query}"
    else:
        user_content = instruction

    return user_content, response, system


def convert_sharegpt(sample: dict, columns: dict, tags: dict) -> tuple:
    """Convert a sharegpt-format sample to (query, response).

    Extracts the last user-assistant turn for single-turn training.

    Args:
        sample: Dataset sample
        columns: Column name mapping
        tags: Tag name mapping for role/content keys

    Returns:
        Tuple of (query, response, system)
    """
    messages_col = columns.get("messages", "conversations")
    system_col = columns.get("system", "system")

    role_tag = tags.get("role_tag", "from")
    content_tag = tags.get("content_tag", "value")
    user_tag = tags.get("user_tag", "human")
    assistant_tag = tags.get("assistant_tag", "gpt")
    system_tag = tags.get("system_tag", "system")

    messages = sample.get(messages_col, [])
    system = sample.get(system_col, "")

    query = ""
    response = ""

    for msg in messages:
        role = msg.get(role_tag, "")
        content = msg.get(content_tag, "")
        if role == system_tag:
            system = content
        elif role == user_tag:
            query = content
        elif role == assistant_tag:
            response = content

    return query, response, system


class ChatDataset(Dataset):
    """Universal SFT dataset supporting alpaca, sharegpt, and legacy formats.

    Works with any model (Llama, Qwen, Mistral, Phi, Gemma, etc.) by
    relying on the tokenizer's built-in chat template for formatting.

    Two modes of loading:
      1. By name: dataset_name + dataset_dir (uses dataset_info.json registry)
      2. By path: dataset_path (legacy Arrow format via load_from_disk)

    Args:
        tokenizer: HuggingFace tokenizer (with chat_template support)
        max_seq_len: Maximum sequence length
        dataset_name: Dataset name in dataset_info.json (mode 1)
        dataset_dir: Directory with dataset_info.json (mode 1)
        dataset_path: Direct path to Arrow dataset (mode 2, legacy)
        system_prompt: Optional system prompt override
        query_field: Field name for user query (legacy mode)
        response_field: Field name for assistant response (legacy mode)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        dataset_name: str = None,
        dataset_dir: str = "data",
        dataset_path: str = None,
        system_prompt: str = None,
        query_field: str = "query",
        response_field: str = "response",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt

        if dataset_name:
            self._load_by_name(dataset_name, dataset_dir)
        elif dataset_path:
            self._load_by_path(dataset_path, query_field, response_field)
        else:
            raise ValueError("Must specify either dataset_name or dataset_path")

        logger.info(f"ChatDataset: {len(self.dataset)} samples, max_seq_len={max_seq_len}")

    def _load_by_name(self, dataset_name: str, dataset_dir: str):
        """Load dataset via dataset_info.json registry."""
        self.dataset, self.attr = load_dataset_by_name(dataset_name, dataset_dir)
        self.formatting = self.attr.get("formatting", "alpaca")
        self.columns = self.attr.get("columns", {})
        self.tags = self.attr.get("tags", {})
        self.mode = "registry"

    def _load_by_path(self, dataset_path: str, query_field: str, response_field: str):
        """Load dataset from direct path (legacy Arrow format)."""
        from datasets import load_from_disk
        self.dataset = load_from_disk(dataset_path)
        self.query_field = query_field
        self.response_field = response_field
        self.mode = "legacy"

        if len(self.dataset) > 0:
            sample = self.dataset[0]
            if self.query_field not in sample:
                available = list(sample.keys())
                raise ValueError(
                    f"query_field '{self.query_field}' not found. "
                    f"Available: {available}"
                )

    def _get_query_response(self, idx: int) -> tuple:
        """Get (query, response, system) for a sample."""
        example = self.dataset[idx]

        if self.mode == "legacy":
            query = example[self.query_field]
            response = example[self.response_field]
            system = self.system_prompt or ""
            return query, response, system

        if self.formatting == "sharegpt":
            query, response, system = convert_sharegpt(
                example, self.columns, self.tags
            )
        else:
            query, response, system = convert_alpaca(example, self.columns)

        if self.system_prompt:
            system = self.system_prompt

        return query, response, system

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        query, response, system = self._get_query_response(idx)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": response})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        prompt_messages = []
        if system:
            prompt_messages.append({"role": "system", "content": system})
        prompt_messages.append({"role": "user", "content": query})

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_length = int(prompt_encoded["attention_mask"].sum().item())

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": prompt_length,
        }


# Backward-compatible alias
MetaMathDataset = ChatDataset


def collate_fn(batch):
    """Collate function for batching dataset samples."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "prompt_length": torch.tensor([x["prompt_length"] for x in batch]),
    }
