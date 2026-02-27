
import joblib
from sklearn.metrics import accuracy_score, classification_report


artifact = joblib.load("models/tfidf_sentiment.pkl")

model = artifact["model"]
X_test = artifact["X_test"]
y_test = artifact["y_test"]

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
