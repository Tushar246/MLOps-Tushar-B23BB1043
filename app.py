from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

MODEL_ID = "Tushar04913/goodreads-distilbert"

app = FastAPI(title="Goodreads Genre Classifier API")

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=device
)

class Review(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Goodreads Genre Classifier API is running"}

@app.post("/predict")
def predict(review: Review):
    result = classifier(review.text)
    return {"prediction": result}