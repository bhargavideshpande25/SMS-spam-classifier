from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sms.csv"
MODEL_PATH = BASE_DIR / "model.joblib"


def main():
    data = pd.read_csv(DATA_PATH)
    required = {"label", "message"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain 'label' and 'message' columns.")
    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].str.lower().str.strip()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    if data["label"].nunique() < 2 or len(data) < 10:
        raise ValueError("Please provide at least 10 labeled rows containing both ham and spam.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report (small demo dataset; metrics are not reliable estimates):")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix [ham, spam]:")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
