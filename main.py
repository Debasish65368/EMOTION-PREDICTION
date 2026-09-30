from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

import onnxruntime as ort

import numpy as np

from tokenizers import Tokenizer

from fastapi.staticfiles import StaticFiles

from fastapi.responses import FileResponse

import uvicorn

import gc



from training.preprocessing import preprocess_text



app = FastAPI(title="Emotion Prediction API")



# Setup tokenizer

try:

    tokenizer = Tokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    tokenizer.enable_padding(length=50)

    tokenizer.enable_truncation(max_length=50)

except Exception as e:

    print(f"Error loading tokenizer: {e}")



# Setup ONNX session

try:

    # Disable ONNX Runtime threading completely to minimize memory

    sess_options = ort.SessionOptions()

    sess_options.intra_op_num_threads = 1

    sess_options.inter_op_num_threads = 1

    sess = ort.InferenceSession("Artifacts/MiniLM_Sequence_Classifier_v4.onnx", sess_options, providers=['CPUExecutionProvider'])

except Exception as e:

    print(f"Error loading ONNX model: {e}")



# Emotion labels mapping

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']



class TextRequest(BaseModel):

    text: str



def softmax(x):

    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    return e_x / e_x.sum(axis=-1, keepdims=True)



@app.post("/predict")

def predict_emotion(request: TextRequest):

    if not request.text or request.text.strip() == "":

        raise HTTPException(status_code=400, detail="Text cannot be empty")

        

    try:

        # Preprocessing

        clean_text = preprocess_text(request.text)

        

        # Tokenize

        encoded = tokenizer.encode(clean_text)

        input_ids = np.array([encoded.ids], dtype=np.int64)

        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        

        # Inference

        logits = sess.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})[0]

        

        # Post-processing

        probs = softmax(logits)[0]

        

        # Low confidence fallback logic

        sorted_indices = np.argsort(probs)[::-1]

        top_idx = sorted_indices[0]

        second_idx = sorted_indices[1]

        

        confidence = float(probs[top_idx])

        second_confidence = float(probs[second_idx])

        

        if confidence < 0.80 and (confidence - second_confidence) <= 0.10:

            top_emotion = emotion_labels[top_idx]

            second_emotion = emotion_labels[second_idx]

            final_emotion_label = f"Leaning: {top_emotion.capitalize()} / {second_emotion.capitalize()}"

        else:

            final_emotion_label = emotion_labels[top_idx]

        

        # Manual garbage collection to keep Render memory stable

        gc.collect()

        

        return {

            "text": request.text,

            "predicted_emotion": final_emotion_label,

            "confidence": round(confidence, 4),

            "all_probabilities": {emotion_labels[i]: round(float(probs[i]), 4) for i in range(len(emotion_labels))}

        }

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")

def health_check():

    return {"status": "healthy", "model": "ONNX MiniLM BiGRU V4"}



@app.get("/ram")

def get_ram():

    import os, psutil

    process = psutil.Process(os.getpid())

    return {"ram_mb": process.memory_info().rss / (1024 * 1024)}



# Serve static frontend

app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/")

def read_index():

    return FileResponse("static/index.html")



if __name__ == "__main__":

    # Limit workers to 1 to stay within Render's 512 MB memory limit

    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)

