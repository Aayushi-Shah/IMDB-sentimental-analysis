import os
import boto3
import json
import pickle
import joblib
import keras

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="IMDB Sentiment Analysis API - S3 Model Loader")

# Enable CORS (adjust allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# S3 Configuration
# ---------------------------
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
S3_BUCKET = os.getenv('S3_BUCKET')
CLASSICAL_FOLDER = os.getenv('CLASSICAL_FOLDER')
DEEP_FOLDER = os.getenv('DEEP_FOLDER')

# Ensure required env variables are provided
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET, CLASSICAL_FOLDER, DEEP_FOLDER]):
    raise ValueError("One or more required S3 environment variables are missing.")

# Initialize boto3 S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# ---------------------------
# Local File Handling
# ---------------------------
def get_local_path(filename: str) -> str:
    """
    Returns a path in the 'models_cache' folder in the project root.
    Creates the folder if it does not exist.
    """
    local_dir = os.path.join(os.getcwd(), "models_cache")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    return os.path.join(local_dir, filename)

def download_from_s3(s3_key: str, local_path: str) -> str:
    """
    Downloads a file from S3 if not already present locally.
    """
    if not os.path.exists(local_path):
        try:
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
            file_size = os.path.getsize(local_path)
            print(f"Downloaded {s3_key} to {local_path} with size {file_size} bytes")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading {s3_key} from S3: {e}")
    return local_path

# ---------------------------
# Model Loading Functions
# ---------------------------
# Global caches
classical_models = {}
deep_models = {}
classical_metrics = {}
deep_metrics = {}
tokenizer = None

def load_classical_model(model_name: str):
    filename = f"{model_name}.pkl"
    local_path = get_local_path(filename)
    s3_key = f"{CLASSICAL_FOLDER}/{filename}"
    download_from_s3(s3_key, local_path)
    
    try:
        model = joblib.load(local_path)
        classical_models[model_name] = model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading classical model {model_name}: {e}")
    finally:
        # Delete the local file after loading it into memory
        if os.path.exists(local_path):
            os.remove(local_path)

    return model

def load_deep_model(model_name: str):
    filename = f"{model_name}.h5"
    local_path = get_local_path(filename)
    s3_key = f"{DEEP_FOLDER}/{filename}"
    download_from_s3(s3_key, local_path)
    
    try:
        model = keras.models.load_model(local_path)
        deep_models[model_name] = model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading deep model {model_name}: {e}")
    finally:
        # Delete the local file after loading it into memory
        if os.path.exists(local_path):
            os.remove(local_path)

    return model

def load_classical_metrics():
    filename = "classical_metrics.json"
    local_path = get_local_path(filename)
    s3_key = f"{CLASSICAL_FOLDER}/{filename}"
    download_from_s3(s3_key, local_path)
    
    try:
        with open(local_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading classical metrics: {e}")
    finally:
        # Delete the local file
        if os.path.exists(local_path):
            os.remove(local_path)

def load_deep_metrics():
    filename = "deep_metrics.json"
    local_path = get_local_path(filename)
    s3_key = f"{DEEP_FOLDER}/{filename}"
    download_from_s3(s3_key, local_path)
    
    try:
        with open(local_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading deep metrics: {e}")
    finally:
        # Delete the local file
        if os.path.exists(local_path):
            os.remove(local_path)

def load_tokenizer():
    filename = "tokenizer.pkl"
    local_path = get_local_path(filename)
    s3_key = f"{DEEP_FOLDER}/{filename}"
    download_from_s3(s3_key, local_path)
    
    try:
        with open(local_path, 'rb') as f:
            tk = pickle.load(f)
        return tk
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading tokenizer: {e}")
    finally:
        # Delete the local file
        if os.path.exists(local_path):
            os.remove(local_path)

# ---------------------------
# Startup Event: Load Metadata
# ---------------------------
@app.on_event("startup")
def load_initial_files():
    global classical_metrics, deep_metrics, tokenizer
    classical_metrics = load_classical_metrics()
    deep_metrics = load_deep_metrics()
    tokenizer = load_tokenizer()

# ---------------------------
# Request Model and Prediction Endpoint
# ---------------------------
class SentimentRequest(BaseModel):
    review: str
    model: str       # e.g., "logistic_regression", "rnn_lstm", etc.
    model_type: str  # "classical" or "deep"

max_len = 200  # Must match training configuration

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    review_text = request.review
    model_name = request.model
    model_type = request.model_type.lower()

    if model_type == "classical":
        model = classical_models.get(model_name) or load_classical_model(model_name)
        try:
            prediction = model.predict([review_text])[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during classical model prediction: {e}")
        metrics = classical_metrics.get(model_name, {})
        probability = None

    elif model_type == "deep":
        model = deep_models.get(model_name) or load_deep_model(model_name)
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")

        seq = tokenizer.texts_to_sequences([review_text])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        try:
            prediction_prob = model.predict(padded_seq)[0][0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during deep model prediction: {e}")

        prediction = 1 if prediction_prob >= 0.5 else 0
        probability = float(prediction_prob)
        metrics = deep_metrics.get(model_name, {})

    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'classical' or 'deep'.")

    sentiment = "Positive" if prediction == 1 else "Negative"
    return {
        "sentiment": sentiment,
        "model": model_name,
        "model_type": model_type,
        "metrics": metrics,
        "probability": probability
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
