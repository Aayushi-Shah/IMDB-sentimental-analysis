# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Literal
import os
import pickle

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment on IMDB reviews using various NLP models.",
    version="0.2.1"
)

# ---------------------------
# Request and Response Models using Literal types for validation
# ---------------------------
class AnalyzeRequest(BaseModel):
    review: str
    # Added "gpt" to the list of valid model names:
    model: Literal["naive_bayes", "logistic_regression", "svm", "rnn", "cnn", "bert", "gpt", "ensemble"]
    remove_stopwords: Optional[bool] = True
    apply_lemmatization: Optional[bool] = True

class AnalyzeResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    accuracy: float
    details: Optional[Dict] = None  # Additional performance metrics

# ---------------------------
# Global Model Containers
# ---------------------------
classical_models = {}
deep_models = {}

# ---------------------------
# Model Loading Functions
# ---------------------------
def load_classical_models():
    model_names = ["naive_bayes", "logistic_regression", "svm"]
    model_folder = "models/classical"
    for name in model_names:
        path = os.path.join(model_folder, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                classical_models[name] = pickle.load(f)
            print(f"Loaded classical model: {name}")
        else:
            print(f"Warning: Classical model file {path} not found.")

# def load_deep_models():
#     """
#     Loads deep learning models.
#     For the sake of demonstration, we use dummy inference functions here:
#       - rnn
#       - cnn
#       - bert
#       - gpt
#       - ensemble (combining multiple model predictions)
#     In a real scenario, replace these placeholders with actual loading code
#     (e.g., torch.load(), tf.keras.models.load_model(), or huggingface transformers).
#     """
#     def dummy_inference_rnn(review: str):
#         return 1, 0.90  # Dummy: prediction=1 (positive), accuracy=90%
    
#     def dummy_inference_cnn(review: str):
#         return 0, 0.88  # Dummy: prediction=0 (negative), accuracy=88%
    
#     def dummy_inference_bert(review: str):
#         return 1, 0.92  # Dummy: prediction=1 (positive), accuracy=92%
    
#     def dummy_inference_gpt(review: str):
#         return 1, 0.94  # Dummy: prediction=1 (positive), accuracy=94%
    
#     def dummy_inference_ensemble(review: str):
#         return 1, 0.93  # Dummy: prediction=1 (positive), accuracy=93%
    
#     deep_models["rnn"] = dummy_inference_rnn
#     deep_models["cnn"] = dummy_inference_cnn
#     deep_models["bert"] = dummy_inference_bert
#     deep_models["gpt"] = dummy_inference_gpt
#     deep_models["ensemble"] = dummy_inference_ensemble
    
#     print("Loaded deep learning models: rnn, cnn, bert, gpt, ensemble")

# Load all models at startup
load_classical_models()
# load_deep_models()

# ---------------------------
# Helper Function for Classical Inference
# ---------------------------
def get_prediction_classical(model_name: str, review: str):
    model_artifacts = classical_models.get(model_name)
    if model_artifacts is None:
        raise ValueError("Classical model not found.")
    
    # If the model was saved as a dictionary containing vectorizer and classifier:
    if isinstance(model_artifacts, dict):
        vectorizer = model_artifacts.get("vectorizer")
        classifier = model_artifacts.get("classifier")
        if vectorizer is None or classifier is None:
            raise ValueError("Incomplete classical model artifacts.")
        review_transformed = vectorizer.transform([review])
        prediction = classifier.predict(review_transformed)[0]
        model_accuracy = model_artifacts.get("accuracy")
        return prediction, model_accuracy
    else:
        # Fallback: if the loaded object has a predict method (e.g., a pipeline)
        prediction = model_artifacts.predict([review])[0]
        model_accuracy = getattr(model_artifacts, "accuracy")
        return prediction, model_accuracy
        
# ---------------------------
# /analyze Endpoint
# ---------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_review(request: AnalyzeRequest):
    if not request.review:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    model_name = request.model.lower()
    
    # Check for classical or deep model
    if model_name in classical_models:
        try:
            prediction, model_accuracy = get_prediction_classical(model_name, request.review)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif model_name in deep_models:
        inference_function = deep_models[model_name]
        prediction, model_accuracy = inference_function(request.review)
    else:
        raise HTTPException(status_code=400, detail=f"Selected model '{model_name}' not supported.")
    
    # Map numeric prediction to a sentiment label. For simplicity:
    # 0: negative, 1: positive (you can extend logic if neutral is possible).
    sentiment_label = "positive" if prediction == 1 else "negative"
    
    response = AnalyzeResponse(
        sentiment=sentiment_label,
        accuracy=model_accuracy,
        details={
            "customization": {
                "remove_stopwords": request.remove_stopwords,
                "apply_lemmatization": request.apply_lemmatization
            }
        }
    )
    return response
