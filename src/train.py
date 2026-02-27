import os
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from data import load_data
from dataset import GoodreadsDataset
from utils import compute_metrics

MODEL_NAME = "distilbert-base-cased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
SAVE_DIR = "distilbert-reviews-genres"
REPO_ID = "Tushar04913/goodreads-distilbert"


def main():
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_data()

    # Create label mappings
    unique_labels = list(set(train_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Tokenize
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # Encode labels
    train_labels_encoded = [label2id[l] for l in train_labels]
    test_labels_encoded = [label2id[l] for l in test_labels]

    # Create datasets
    train_dataset = GoodreadsDataset(train_encodings, train_labels_encoded)
    test_dataset = GoodreadsDataset(test_encodings, test_labels_encoded)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="steps",
        logging_steps=100,
        report_to="none",

        # Hugging Face Hub settings
        push_to_hub=True,
        hub_model_id=REPO_ID,
        hub_token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save locally
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # Push to Hugging Face
    trainer.push_to_hub()

    # Evaluate
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)

    # Save evaluation results
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/results.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()