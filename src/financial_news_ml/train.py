"""Train and save the Phase 1 sentiment classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from financial_news_ml.model import FinancialSentimentClassifier


def train(output_path: str | Path) -> dict[str, float | int]:
    """Train the selected model and return held-out evaluation metrics."""
    dataset = load_dataset(
        "takala/financial_phrasebank",
        "sentences_75agree",
        trust_remote_code=True,
    )
    frame = (
        dataset["train"]
        .to_pandas()
        .drop_duplicates(subset="sentence")
        .reset_index(drop=True)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        frame["sentence"],
        frame["label"],
        test_size=0.20,
        random_state=42,
        stratify=frame["label"],
    )

    classifier = FinancialSentimentClassifier(min_df=2).fit(
        X_train.tolist(), y_train.tolist()
    )
    predictions = classifier.predict(X_test.tolist())
    classifier.save(output_path)

    return {
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="models/financial_sentiment_model.joblib",
        help="Destination for the trained model artifact.",
    )
    arguments = parser.parse_args()
    metrics = train(arguments.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
