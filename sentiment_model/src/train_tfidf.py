import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data.data_preprocessing import load_and_clean
from configs.config import DATA_PATH


def train():
    df = load_and_clean(DATA_PATH)

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=5
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                n_jobs=-1
            )
        )
    ])

    print("Training TF-IDF + Logistic Regression model...")
    pipeline.fit(X_train, y_train)

    joblib.dump(
        {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test
        },
        "models/tfidf_sentiment.pkl"
    )

    print("Model trained and saved!")


if __name__ == "__main__":
    train()
