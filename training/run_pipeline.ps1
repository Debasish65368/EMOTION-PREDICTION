param(
    [int]$BatchSize = 16,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$generateArgs = @("training/generate_embeddings.py", "--batch-size", $BatchSize, "--device", "cpu")
if ($Force) { $generateArgs += "--force" }

python @generateArgs
python -m training.evaluate --model Artifacts/MiniLM_Sequence_Classifier.keras
python -m training.train --batch-size 32
python -m training.evaluate --model Artifacts/MiniLM_Sequence_Classifier_retrained.keras
