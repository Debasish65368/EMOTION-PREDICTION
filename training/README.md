# Reproducible MiniLM + BiGRU Training

This directory restores the missing **reproducibility layer** for the shipped Emotion Prediction model.

## Recovered architecture

The existing `Artifacts/MiniLM_Sequence_Classifier.keras` artifact was inspected directly. Its architecture is:

```text
Input: (50, 384)
  ↓
Masking(mask_value=0.0)
  ↓
Bidirectional GRU(64 per direction)
  ↓
Dropout(0.3)
  ↓
Dense(32, ReLU)
  ↓
Dropout(0.3)
  ↓
Dense(6, Softmax)
```

The model is compiled with Adam (`learning_rate=1e-3`) and sparse categorical cross-entropy.

## Why embeddings are cached

`all-MiniLM-L6-v2` produces a 384-dimensional hidden state for each token. The transformer is **frozen** and used only as a feature extractor. `generate_embeddings.py` processes the dataset in small batches and writes the resulting `50 × 384` sequences to `.npy` files.

Training then reads those files with `numpy.memmap`, so the complete embedding matrix does not have to live in RAM during BiGRU training.

## Data split

The pipeline uses the dataset's official `train`, `validation`, and `test` splits. The test set is reserved for final evaluation and is not used for early stopping.

## Run

From the repository root:

```powershell
python -m pip install -r training/requirements-training.txt
python -m training.generate_embeddings --batch-size 16
python -m training.evaluate --model Artifacts/MiniLM_Sequence_Classifier.keras
python -m training.train --batch-size 32
python -m training.evaluate --model Artifacts/MiniLM_Sequence_Classifier_retrained.keras
```

Use `--device cpu` on a CPU-only setup. Use `--force` to regenerate embedding caches.

## Outputs

Generated cache files live in `training/cache/` and evaluation outputs in `training/results/`. These generated artifacts are intentionally ignored by Git; the scripts themselves are tracked.

## Historical 86% claim

The repository README contains a historical ~86% accuracy claim. The original training and evaluation scripts were lost with a deleted Antigravity workspace. This restored pipeline therefore treats the historical number as **unverified provenance** until `training/evaluate.py` produces a new measured result.
