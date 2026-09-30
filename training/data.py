"""Memory-efficient Keras Sequence backed by .npy memmaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence


class MemmapSequence(Sequence):
    def __init__(self, embeddings_path: str | Path, labels_path: str | Path, batch_size: int = 32, shuffle: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.embeddings_path = Path(embeddings_path)
        self.labels_path = Path(labels_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {self.embeddings_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {self.labels_path}")

        self.x = np.load(self.embeddings_path, mmap_mode="r")
        self.y = np.load(self.labels_path, mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError("Embedding and label counts do not match.")
        self.indices = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int):
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:stop]
        return np.asarray(self.x[batch_indices], dtype=np.float32), np.asarray(self.y[batch_indices], dtype=np.int64)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    @property
    def num_examples(self) -> int:
        return len(self.indices)
