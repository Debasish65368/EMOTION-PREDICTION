# Emotion Prediction Web App



A full-stack, single-page web app built to classify emotional sentiment from free text into six categories: **sadness**, **joy**, **love**, **anger**, **fear**, or **surprise**.



This project solves two distinct problems:

1. **Model Accuracy & Robustness:** Building a classifier capable of correctly interpreting context and compositional language, rather than relying strictly on simple keyword correlations.

2. **Production Deployment Constraints:** Deploying a complex PyTorch/Transformers pipeline within the strict **512 MB memory limit** of Render's Free tier.



To see the model in action, the project will be deployed to Render once the repository is finalized.



---



## 🚀 Deployment Status

The API is designed for deployment on Render's Free tier, which spins down after 15 minutes of inactivity. The frontend polls a /health endpoint to display a "waking the model up" state rather than a silently broken UI. Deployment will occur after repository finalization.



---



## 🧠 Model Architecture (V4)



The ONE production model (Artifacts/MiniLM_Sequence_Classifier_v4.onnx) is an end-to-end PyTorch implementation that was optimized and exported to ONNX. 



The model uses a partially fine-tuned MiniLM encoder to improve contextual representations, while out-of-domain negation cases can still fail because the training dataset contains strong domain-specific lexical priors. The final layer of the ll-MiniLM-L6-v2 encoder is unfrozen, combined with a BiGRU sequence classifier.



**The Training Pipeline:**

`	ext

Raw Text 

  -> Shared Preprocessing

  -> AutoTokenizer 

  -> MiniLM Transformer (Last layer unfrozen)

  -> Pack Padded Sequence (Attention Masking)

  -> Bidirectional GRU (64 units)

  -> Dense Classifier (Dropout -> 32 -> ReLU -> Dropout -> 6)

  -> Softmax

`



### Official Test Set Performance

Evaluated natively on the untouched 2,000-example 	est split of dair-ai/emotion:

- **Accuracy**: 89.85%

- **Macro Precision**: 84.86%

- **Macro Recall**: 88.47%

- **Macro F1**: 86.46%

- **Weighted F1**: 90.01%



*(The ONNX production export operates exactly at parity with these evaluation metrics).*



---



## ⚡ Deployment Optimization: Beating the 512 MB Limit



Deploying the native PyTorch V4 model alongside HuggingFace 	ransformers inside a FastAPI process consumed **~630 MB** of Working Set RAM at startup, fatally breaching Render's 512 MB Free Tier limit.



To deploy the V4 model without relying on lossy quantization, the production stack was completely rebuilt:



1. **ONNX Export (onnxruntime)**: The PyTorch model was traced and exported to an FP32 ONNX graph (MiniLM_Sequence_Classifier_v4.onnx). The export utilized a custom inference MaskedBiGRU to perfectly mimic PyTorch's pack_padded_sequence without triggering dynamic-shape tracing bugs.

2. **Rust Tokenizers (	okenizers)**: The HuggingFace 	ransformers library (which implicitly loads PyTorch) was entirely stripped from the production dependencies. Tokenization is handled natively by the Rust-based 	okenizers library.

3. **Single-Threaded Execution**: onnxruntime threading was heavily restricted and manual garbage collection was enforced per-request.



**Result:** The production FastAPI server now cold-starts and stabilizes at **~317 MB of RAM**, allowing the unquantized V4 model to run comfortably within the 512 MB constraint. PyTorch and Transformers are no longer required to run the application.



---



## 🛡️ Confidence Threshold Safeguard



Because a softmax classifier is always forced to pick across its 6 classes — even for genuinely ambiguous or neutral input — the backend applies a threshold guard. 



If the model is highly uncertain (top prediction < 80% **AND** a small gap to the runner-up <= 10%), the UI computes the top two leaning emotions from the model's probability breakdown and displays them directly — e.g. **"Leaning: Joy / Love"** — so the honest hedge reads as a real finding instead of a broken result.



### Known Limitations

- **No "Neutral" Class:** The dair-ai/emotion dataset has no neutral label. Since softmax always forces a choice, plainly neutral sentences (e.g. "The train arrives at 9am") can still receive a confidently wrong emotion label. The confidence threshold catches most of these, but it is a structural limitation of forced-choice classifiers on datasets without a neutral option.

- **Dataset Domain Biases:** The model is trained on heavily idiosyncratic Twitter data. While V4 massively outperforms previous baselines, adversarial out-of-domain syntactic negations may still occasionally misclassify if they heavily leverage structural priors from the training data.



---



## 🔌 API Documentation



### POST /predict

Predicts the emotion of the given text.



**Request:**

`json

{

  "text": "I am so happy today"

}

`



**Response:**

`json

{

  "text": "I am so happy today",

  "predicted_emotion": "joy",

  "confidence": 0.9190,

  "all_probabilities": {

    "sadness": 0.0100,

    "joy": 0.9190,

    "love": 0.0300,

    "anger": 0.0100,

    "fear": 0.0200,

    "surprise": 0.0210

  }

}

`



### GET /health

Reports the status of the API and ONNX Inference Session.



---



## 💻 Local Setup



`ash

git clone https://github.com/Debasish65368/EMOTION-PREDICTION.git

cd EMOTION-PREDICTION

python -m venv .venv



# Windows

.\.venv\Scripts\activate

# Linux/Mac

source .venv/bin/activate



pip install -r requirements.txt

uvicorn main:app --reload --port 8000

`

Then open http://127.0.0.1:8000 in your browser.



*(If you wish to retrain the model locally, install the dependencies in 	raining/requirements-training.txt instead).*



---



## 🛠️ Tech Stack



| Component | Choice |

|---|---|

| **Backend** | FastAPI, Uvicorn |

| **Inference Engine** | ONNX Runtime (CPUExecutionProvider) |

| **Tokenization** | HuggingFace 	okenizers (Rust) |

| **Model Architecture** | MiniLM + MaskedBiGRU |

| **Frontend** | Vanilla HTML/CSS/JS |

| **Dataset** | dair-ai/emotion (16K labeled sentences, 6 classes) |

| **Deployment** | Render (Free Tier, 512 MB RAM constraint) |

