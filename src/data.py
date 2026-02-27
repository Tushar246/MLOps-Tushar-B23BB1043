import requests
import gzip
import json
import random

GENRE_URLS = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

def load_reviews(url, head=10000, sample_size=1000):
    reviews = []
    response = requests.get(url, stream=True)

    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= head:
                break
            data = json.loads(line)
            reviews.append(data["review_text"])

    return random.sample(reviews, min(sample_size, len(reviews)))

def load_data():
    genre_reviews = {}

    for genre, url in GENRE_URLS.items():
        print(f"Loading {genre}...")
        genre_reviews[genre] = load_reviews(url)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        for review in reviews[:800]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in reviews[800:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels