from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run 'python train.py' first.")
        return
    model = joblib.load(MODEL_PATH)
    print("SMS Spam Classifier (type 'exit' to quit)")
    while True:
        message = input("Enter an SMS message: ").strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            print("Please enter a message.")
            continue
        label = model.predict([message])[0]
        print("Prediction:", "SPAM" if label == "spam" else "HAM (not spam)")


if __name__ == "__main__":
    main()
