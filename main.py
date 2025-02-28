# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment on IMDB reviews using various NLP models.",
    version="0.1.0"
)

# Define the request body model
class AnalyzeRequest(BaseModel):
    review: str
    model: str 
    remove_stopwords: Optional[bool] = True
    apply_lemmatization: Optional[bool] = True

# Define the response model
class AnalyzeResponse(BaseModel):
    sentiment: str 
    accuracy: float
    details: Optional[dict] = None 

# Create the /analyze endpoint
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_review(request: AnalyzeRequest):
    # For now, placeholder logic.
    if not request.review:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    # Placeholder: Based on the selected model, run the appropriate NLP model
    # For now, just returning dummy data.
    dummy_result = {
        "mnb": {"sentiment": "positive", "accuracy": 85.43},
        "r": {"sentiment": "positive", "accuracy": 87.68},
    }
    
    result = dummy_result.get(request.model)
    if not result:
        raise HTTPException(status_code=400, detail="Selected model not supported.")
    
    # Apply customization options if needed (this is where your preprocessing toggles would come in)
    # For now, this is just a placeholder.
    # e.g., if not request.remove_stopwords, adjust the preprocessing pipeline accordingly.
    
    response = AnalyzeResponse(
        sentiment=result["sentiment"],
        accuracy=result["accuracy"],
        details={"customization": {
            "remove_stopwords": request.remove_stopwords,
            "apply_lemmatization": request.apply_lemmatization
        }}
    )
    return response
