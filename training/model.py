"""MiniLM-embedding + BiGRU classifier definition used by the project."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 384
NUM_CLASSES = 6


def build_model(
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Build the classifier architecture recovered from the shipped .keras artifact."""
    model = keras.Sequential(
        [
            keras.Input(shape=(max_sequence_length, embedding_dim), dtype="float32"),
            layers.Masking(mask_value=0.0),
            layers.Bidirectional(
                layers.GRU(
                    64,
                    return_sequences=False,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    reset_after=True,
                )
            ),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="minilm_bigru_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"Trainable parameters: {model.count_params():,}")
