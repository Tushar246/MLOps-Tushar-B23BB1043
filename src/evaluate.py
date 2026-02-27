from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer
from data import load_data
from dataset import GoodreadsDataset
from utils import compute_metrics

MODEL_DIR = "distilbert-reviews-genres"

def main():
    train_texts, train_labels, test_texts, test_labels = load_data()

    unique_labels = list(set(train_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    test_labels_encoded = [label2id[l] for l in test_labels]

    test_dataset = GoodreadsDataset(test_encodings, test_labels_encoded)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(test_dataset)
    print(results)

if __name__ == "__main__":
    main()