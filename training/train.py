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





from sklearn.metrics import f1_score



class MacroF1Checkpoint(keras.callbacks.Callback):

    def __init__(self, val_sequence, filepath):

        super().__init__()

        self.val_sequence = val_sequence

        self.filepath = filepath

        self.best_f1 = -1.0



    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        y_pred_probs = self.model.predict(self.val_sequence, verbose=0)

        y_pred = np.argmax(y_pred_probs, axis=1)

        # We need the true labels. We can read them from the memmap file directly.

        y_true = np.asarray(np.load(self.val_sequence.labels_path, mmap_mode="r"), dtype=np.int64)

        

        macro_f1 = f1_score(y_true, y_pred, average="macro")

        logs["val_macro_f1"] = macro_f1

        print(f" - val_macro_f1: {macro_f1:.4f}")

        

        if macro_f1 > self.best_f1:

            print(f"\\nEpoch {epoch+1}: val_macro_f1 improved from {self.best_f1:.4f} to {macro_f1:.4f}, saving model to {self.filepath}")

            self.best_f1 = macro_f1

            self.model.save(self.filepath)



def main() -> None:

    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    np.random.seed(args.seed)



    cache = Path(args.cache_dir)

    output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)



    train = MemmapSequence(cache / "train_embeddings.npy", cache / "train_labels.npy", args.batch_size, shuffle=True)

    val = MemmapSequence(cache / "validation_embeddings.npy", cache / "validation_labels.npy", args.batch_size, shuffle=False)

    

    # Store labels_path in sequence for callback

    val.labels_path = cache / "validation_labels.npy"



    model = build_model()

    model.summary()

    print(f"Train examples: {train.num_examples:,}")

    print(f"Validation examples: {val.num_examples:,}")



    # ALWAYS use class weights for V3

    labels = np.asarray(np.load(cache / "train_labels.npy", mmap_mode="r"))

    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)

    class_weights = {int(cls): float(weight) for cls, weight in zip(np.unique(labels), weights)}

    print(f"Class weights: {class_weights}")



    callbacks = [

        MacroF1Checkpoint(val, output),

        keras.callbacks.EarlyStopping(monitor="val_macro_f1", patience=4, mode="max", restore_best_weights=False),

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

