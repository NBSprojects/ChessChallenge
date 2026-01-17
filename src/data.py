"""
Data loading utilities for the Chess Challenge.

This module provides functions to load and process chess game data
from the Lichess dataset on Hugging Face.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess games.
    
    This dataset loads games from a Hugging Face dataset and prepares
    them for language modeling training.
    
    Each game is tokenized and truncated/padded to max_length.
    The labels are shifted by one position for next-token prediction.
    
    Example:
        >>> from src.tokenizer import ChessTokenizer
        >>> tokenizer = ChessTokenizer.build_vocab_from_dataset()
        >>> dataset = ChessDataset(tokenizer, max_length=256)
        >>> sample = dataset[0]
        >>> print(sample["input_ids"].shape)  # (256,)
    """
    
    def __init__(
        self,
        tokenizer,
        dataset_name: str = "dlouapre/lichess_2025-01_1M",
        split: str = "train",
        column: str = "text",
        max_length: int = 256,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the chess dataset.
        
        Args:
            tokenizer: The chess tokenizer to use.
            dataset_name: Name of the dataset on Hugging Face Hub.
            split: Dataset split to use.
            column: Column containing the game strings.
            max_length: Maximum sequence length.
            max_samples: Maximum number of samples to load.
        """
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.column = column
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = dataset
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        game = self.data[idx][self.column]
        
        # Prepend BOS token for proper language modeling
        game_with_bos = self.tokenizer.bos_token + " " + game
        
        # Tokenize
        encoding = self.tokenizer(
            game_with_bos,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Squeeze batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Labels are the same as input_ids (model will shift internally)
        labels = input_ids.clone()
        
        # Set padding tokens to -100 to ignore in loss
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ChessDataCollator:
    """
    Data collator for chess games.
    
    This collator pads sequences to the same length within a batch
    and creates the appropriate attention masks.
    """
    
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack tensors
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



def create_train_val_datasets(
    tokenizer,
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    max_length: int = 256,
    train_samples: Optional[int] = None,
    val_samples: int = 5000,
    val_ratio: float = 0.05,
    num_proc: int = 8,
):
    from datasets import load_dataset
    
    full_dataset = load_dataset(dataset_name, split="train")
    
    def tokenize_function(examples):
        texts = [tokenizer.bos_token + " " + game for game in examples["text"]]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        encodings["labels"] = [
            [-100 if m == 0 else i for i, m in zip(ids, mask)]
            for ids, mask in zip(encodings["input_ids"], encodings["attention_mask"])
        ]
        return encodings
    
    # KEY CHANGE: Disable caching entirely
    tokenized = full_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=full_dataset.column_names,
        desc="Tokenizing",
        load_from_cache_file=False,  # Don't load from cache
        keep_in_memory=True,          # Keep in RAM, don't write to disk
    )
    tokenized.set_format(type="torch")
    
    # Split
    n_train = min(train_samples, len(tokenized) - val_samples) if train_samples else int(len(tokenized) * (1 - val_ratio))
    n_val = min(val_samples, len(tokenized) - n_train)
    
    return tokenized.select(range(n_train)), tokenized.select(range(n_train, n_train + n_val))


def stream_games(
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    split: str = "train",
    column: str = "text",
) -> Iterator[str]:
    """
    Stream games from the dataset for memory-efficient processing.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        split: Dataset split to use.
        column: Column containing the game strings.
    
    Yields:
        Game strings one at a time.
    """
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    for example in dataset:
        yield example[column]


def analyze_dataset_statistics(
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    max_samples: int = 10000,
) -> Dict:
    """
    Analyze statistics of the chess dataset.
    
    Args:
        dataset_name: Name of the dataset.
        max_samples: Maximum number of samples to analyze.
    
    Returns:
        Dictionary containing dataset statistics.
    """
    from collections import Counter
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    game_lengths = []
    move_counts = Counter()
    opening_moves = Counter()
    
    for example in dataset:
        moves = example["text"].strip().split()
        game_lengths.append(len(moves))
        move_counts.update(moves)
        
        # Track common openings (first 4 moves)
        if len(moves) >= 4:
            opening = " ".join(moves[:4])
            opening_moves[opening] += 1
    
    return {
        "total_games": len(dataset),
        "avg_game_length": sum(game_lengths) / len(game_lengths),
        "min_game_length": min(game_lengths),
        "max_game_length": max(game_lengths),
        "unique_moves": len(move_counts),
        "most_common_moves": move_counts.most_common(20),
        "most_common_openings": opening_moves.most_common(10),
    }
