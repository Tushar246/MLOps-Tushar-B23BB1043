import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.data_preprocessing import load_and_clean
from data.dataset import IMDBDataset, build_vocab, pad_collate
from models.model import SentimentRNN
from configs.config import *

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_and_clean(DATA_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        df["review"].values,
        df["sentiment"].values,
        test_size=0.2,
        stratify=df["sentiment"],
        random_state=42
    )

    vocab = build_vocab(X_train)

    train_ds = IMDBDataset(X_train, y_train, vocab)
    val_ds = IMDBDataset(X_val, y_val, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate
    )

    model = SentimentRNN(len(vocab)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for texts, labels in tqdm(train_loader):
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

    torch.save(
        {"model_state": model.state_dict(), "vocab": vocab},
        "models/rnn_sentiment.pt"
    )

    print("Model training completed and saved")

if __name__ == "__main__":
    train()
