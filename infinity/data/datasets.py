"""Dataset implementations for training.

Supports any HuggingFace model by using the tokenizer's native chat_template.
"""

import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    """Universal SFT dataset using tokenizer's native chat_template.

    Works with any model (Llama, Qwen, Mistral, Phi, Gemma, etc.) by
    relying on the tokenizer's built-in chat template for formatting.

    Args:
        dataset_path: Path to the dataset (HuggingFace datasets format)
        tokenizer: HuggingFace tokenizer (with chat_template support)
        max_seq_len: Maximum sequence length
        system_prompt: Optional system prompt. If None, no system message is added.
        query_field: Field name for user query in the dataset
        response_field: Field name for assistant response in the dataset
    """

    def __init__(self, dataset_path, tokenizer, max_seq_len: int,
                 system_prompt: str = None,
                 query_field: str = "query",
                 response_field: str = "response"):
        from datasets import load_from_disk
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        self.query_field = query_field
        self.response_field = response_field

        # Verify fields exist
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            if self.query_field not in sample:
                available = list(sample.keys())
                raise ValueError(
                    f"query_field '{self.query_field}' not found in dataset. "
                    f"Available fields: {available}"
                )
            if self.response_field not in sample:
                available = list(sample.keys())
                raise ValueError(
                    f"response_field '{self.response_field}' not found in dataset. "
                    f"Available fields: {available}"
                )

        logger.info(f"ChatDataset: {len(self.dataset)} samples, max_seq_len={max_seq_len}")
        logger.info(f"  fields: query={query_field}, response={response_field}")
        if system_prompt:
            logger.info(f"  system_prompt: {system_prompt[:80]}...")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Build messages using standard chat format
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": example[self.query_field]})
        messages.append({"role": "assistant", "content": example[self.response_field]})

        # Use tokenizer's native chat template (auto-adapts to any model)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False  # chat template already has special tokens
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Compute prompt length for masking (only supervise the response)
        prompt_messages = []
        if self.system_prompt:
            prompt_messages.append({"role": "system", "content": self.system_prompt})
        prompt_messages.append({"role": "user", "content": example[self.query_field]})

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False
        )
        prompt_length = int(prompt_encoded["attention_mask"].sum().item())

        # Labels: mask prompt tokens with -100
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": prompt_length
        }


# Backward-compatible alias
MetaMathDataset = ChatDataset


def collate_fn(batch):
    """Collate function for batching dataset samples.

    Args:
        batch: List of dataset samples

    Returns:
        Dictionary with batched tensors
    """
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "prompt_length": torch.tensor([x["prompt_length"] for x in batch])
    }
