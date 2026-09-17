"""Predict sentiment with a saved Phase 1 model."""

from __future__ import annotations

import argparse
import json

from financial_news_ml.model import FinancialSentimentClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Financial sentence to classify.")
    parser.add_argument(
        "--model",
        default="models/financial_sentiment_model.joblib",
        help="Path to the saved model artifact.",
    )
    arguments = parser.parse_args()

    classifier = FinancialSentimentClassifier.load(arguments.model)
    prediction = classifier.predict_sentiments([arguments.text])[0]
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
