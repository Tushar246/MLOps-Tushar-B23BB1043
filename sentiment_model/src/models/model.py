import torch
import torch.nn as nn
from configs.config import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        self.rnn = nn.RNN(
            EMBED_DIM,
            HIDDEN_DIM,
            NUM_LAYERS,
            batch_first=True
        )

        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden[-1])
        return output.squeeze()
