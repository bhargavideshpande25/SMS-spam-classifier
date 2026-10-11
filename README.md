# SMS Spam Classifier

A beginner-friendly NLP / machine-learning project that trains a TF-IDF + Multinomial Naive Bayes model to classify SMS messages as `spam` or `ham` (not spam), then provides a small command-line prediction tool.

## Requirements
- Python 3.10+

## Setup and run
```bash
python -m venv .venv
```
Activate the environment:
- Windows Command Prompt: `.venv\\Scripts\\activate`
- PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then run:
```bash
pip install -r requirements.txt
python train.py
python predict.py
```
The training script saves `model.joblib` and prints a classification report. The included CSV is a small demonstration dataset, not a production-quality corpus. For meaningful evaluation, replace it with a larger, properly sourced and licensed dataset and use a held-out test set.

## Project structure
- `data/sms.csv` — sample labeled messages
- `train.py` — train and evaluate the model
- `predict.py` — enter messages for prediction
- `requirements.txt` — dependencies
