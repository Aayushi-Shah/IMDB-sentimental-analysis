# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = FastAPI(title="IMDB Sentiment Analysis API")

# --------------------------
# Load Classical Models & Metrics
# --------------------------
classical_models = {}
for name in ['logistic_regression', 'naive_bayes', 'svm']:
    try:
        classical_models[name] = joblib.load(f'models/classical/{name}.pkl')
    except Exception as e:
        print(f"Error loading classical model {name}: {e}")

with open('models/classical/classical_metrics.json', 'r') as f:
    classical_metrics = json.load(f)

# --------------------------
# Load Deep Learning Models, Tokenizer & Metrics
# --------------------------
deep_models = {}
for name in ['rnn_lstm', 'bilstm', 'cnn']:
    try:
        deep_models[name] = tf.keras.models.load_model(f'models/deep/{name}.h5')
    except Exception as e:
        print(f"Error loading deep model {name}: {e}")

with open('models/deep/deep_metrics.json', 'r') as f:
    deep_metrics = json.load(f)

with open('models/deep/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 200  # same as used in training deep models

# --------------------------
# Request Schema & API Endpoint
# --------------------------
class SentimentRequest(BaseModel):
    review: str
    model: str  # Should be one of the six model names

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    review_text = request.review
    model_name = request.model

    if model_name in classical_models:
        pipeline = classical_models[model_name]
        prediction = pipeline.predict([review_text])[0]
        metrics = classical_metrics.get(model_name, {})
        # Note: Classical pipelines might not provide prediction probabilities
        prob = None
    elif model_name in deep_models:
        model = deep_models[model_name]
        seq = tokenizer.texts_to_sequences([review_text])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        prediction_prob = model.predict(padded_seq)[0][0]
        prediction = 1 if prediction_prob >= 0.5 else 0
        prob = float(prediction_prob)
        metrics = deep_metrics.get(model_name, {})
    else:
        raise HTTPException(status_code=400, detail="Model not found")

    sentiment = "Positive" if prediction == 1 else "Negative"
    response = {
        "sentiment": sentiment,
        "model": model_name,
        "metrics": metrics,
    }
    if prob is not None:
        response["probability"] = prob
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
