"""Evaluate an existing MiniLM + BiGRU model on the official test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model

from .data import MemmapSequence

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="training/cache")
    p.add_argument("--model", default="Artifacts/MiniLM_Sequence_Classifier.keras")
    p.add_argument("--output-dir", default="training/results")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache_dir)
    test = MemmapSequence(cache / "test_embeddings.npy", cache / "test_labels.npy", args.batch_size, shuffle=False)
    y_true = np.asarray(np.load(cache / "test_labels.npy", mmap_mode="r"), dtype=np.int64)

    model = load_model(args.model, compile=False)
    probabilities = model.predict(test, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "model": str(args.model),
        "test_examples": int(len(y_true)),
        "labels": LABELS,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(LABELS)))
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Emotion Prediction — Test Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(report)
    print(f"\nArtifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
