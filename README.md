This project builds an end-to-end MLOps pipeline for text classification using DistilBERT.

It includes:

Model training

Model evaluation

Model hosting on Hugging Face

Inference pipeline

FastAPI deployment

Docker containerization


Hugging Face Model

Model available at:

 https://huggingface.co/Tushar04913/goodreads-distilbert

Running Inference Locally
python inference.py
 Running FastAPI Server
uvicorn app:app --reload

Open:

http://127.0.0.1:8000/docs



Build API container:

docker build -f Dockerfile.api -t goodreads-api .

Run container:

docker run -p 8000:8000 goodreads-api

Open:

http://localhost:8000/docs