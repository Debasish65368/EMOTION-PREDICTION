"""Train the BiGRU classifier head on cached MiniLM embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from .data import MemmapSequence
from .model import build_model

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="training/cache")
    p.add_argument("--output", default="Artifacts/MiniLM_Sequence_Classifier_retrained.keras")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-weights", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    cache = Path(args.cache_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    train = MemmapSequence(cache / "train_embeddings.npy", cache / "train_labels.npy", args.batch_size, shuffle=True)
    val = MemmapSequence(cache / "validation_embeddings.npy", cache / "validation_labels.npy", args.batch_size, shuffle=False)

    model = build_model()
    model.summary()
    print(f"Train examples: {train.num_examples:,}")
    print(f"Validation examples: {val.num_examples:,}")

    class_weights = None
    if args.class_weights:
        labels = np.asarray(np.load(cache / "train_labels.npy", mmap_mode="r"))
        weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        class_weights = {int(cls): float(weight) for cls, weight in zip(np.unique(labels), weights)}
        print(f"Class weights: {class_weights}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(output, monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        train,
        validation_data=val,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(output)
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
