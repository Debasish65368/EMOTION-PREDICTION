"""Generate disk-backed MiniLM per-token embeddings in small batches.

This keeps MiniLM frozen and separates transformer feature extraction from
BiGRU training. Embeddings are stored as .npy files and opened with mmap when
training/evaluating so the whole 16k-example dataset is not held in RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from .preprocessing import preprocess_text

DATASET_NAME = "dair-ai/emotion"
TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MAX_LENGTH = 50
DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="training/cache")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_labels(dataset) -> list[str]:
    names = list(dataset["train"].features["label"].names)
    if names != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected dataset label order: {names}. Expected {EXPECTED_LABELS}. "
            "Do not continue because the trained classifier's output order is fixed."
        )
    return names


def write_split(
    split_name: str,
    split,
    tokenizer,
    transformer,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    dtype: np.dtype,
    force: bool,
) -> dict:
    n = len(split)
    embedding_path = output_dir / f"{split_name}_embeddings.npy"
    labels_path = output_dir / f"{split_name}_labels.npy"
    meta_path = output_dir / f"{split_name}_metadata.json"

    if not force and embedding_path.exists() and labels_path.exists() and meta_path.exists():
        print(f"[{split_name}] cache already exists; use --force to regenerate")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    embeddings = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=dtype,
        shape=(n, max_length, transformer.config.hidden_size),
    )
    labels = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(n,),
    )

    transformer.eval()
    total = (n + batch_size - 1) // batch_size

    try:
        for batch_index, start in enumerate(range(0, n, batch_size), start=1):
            stop = min(start + batch_size, n)
            texts = [preprocess_text(x) for x in split[start:stop]["text"]]
            batch_labels = np.asarray(split[start:stop]["label"], dtype=np.int64)

            encoded = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = transformer(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand_as(hidden)
                hidden = hidden * mask
                hidden = hidden.detach().cpu().numpy()

            embeddings[start:stop] = hidden.astype(dtype, copy=False)
            labels[start:stop] = batch_labels

            del encoded, outputs, hidden
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[{split_name}] batch {batch_index}/{total} ({stop}/{n})")

    except Exception:
        embeddings.flush()
        labels.flush()
        raise

    embeddings.flush()
    labels.flush()

    metadata = {
        "dataset": DATASET_NAME,
        "split": split_name,
        "num_examples": n,
        "num_classes": len(EXPECTED_LABELS),
        "labels": EXPECTED_LABELS,
        "transformer": TRANSFORMER_NAME,
        "max_length": max_length,
        "hidden_size": int(transformer.config.hidden_size),
        "dtype": np.dtype(dtype).name,
        "batch_size": batch_size,
        "device": str(device),
        "preprocessing": "lowercase; remove apostrophes; replace non-alphanumeric characters; collapse whitespace",
        "attention_mask_applied": True,
        "gradient_tracking": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Embedding dtype: {args.dtype}")

    dataset = load_dataset(DATASET_NAME)
    labels = validate_labels(dataset)
    print(f"Dataset splits: {', '.join(dataset.keys())}")
    print(f"Label order: {labels}")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    transformer = AutoModel.from_pretrained(TRANSFORMER_NAME).to(device)
    transformer.eval()
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)

    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            raise ValueError(f"Expected official '{split_name}' split, but dataset has {list(dataset.keys())}")
        write_split(
            split_name,
            dataset[split_name],
            tokenizer,
            transformer,
            device,
            output_dir,
            args.batch_size,
            args.max_length,
            np_dtype,
            args.force,
        )

    print("Embedding generation complete.")


if __name__ == "__main__":
    main()
