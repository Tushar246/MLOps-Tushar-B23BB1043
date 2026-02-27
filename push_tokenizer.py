from transformers import DistilBertTokenizerFast

REPO_ID = "Tushar04913/goodreads-distilbert"

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
tokenizer.push_to_hub(REPO_ID)

print("Tokenizer uploaded successfully!")