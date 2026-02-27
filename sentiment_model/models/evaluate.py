import torch
from sklearn.metrics import accuracy_score, classification_report
from src.data.data_preprocessing import load_and_clean
from src.models.rnn import SentimentRNN


from data.data_preprocessing import load_and_clean
from data.dataset import IMDBDataset, pad_collate
from models.model import SentimentRNN
from configs.config import *

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("models/rnn_sentiment.pt", map_location=device)
    vocab = checkpoint["vocab"]

    df = load_and_clean(DATA_PATH)

    X = df["review"].values
    y = df["sentiment"].values

    dataset = IMDBDataset(X, y, vocab)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=pad_collate
    )

    model = SentimentRNN(len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for texts, lbls in loader:
            texts = texts.to(device)
            outputs = torch.sigmoid(model(texts))
            preds.extend((outputs > 0.5).cpu().numpy())
            labels.extend(lbls.numpy())

    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds))

if __name__ == "__main__":
    evaluate()
