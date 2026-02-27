import torch
from transformers import pipeline

MODEL_ID = "Tushar04913/goodreads-distilbert"

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=device
)

def predict(text):
    result = classifier(text)
    return result

if __name__ == "__main__":
    while True:
        text = input("Enter review (or type exit): ")

        if text.lower() == "exit":
            break

        prediction = predict(text)
        print("Prediction:", prediction)