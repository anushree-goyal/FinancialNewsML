"""Reusable financial sentiment classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression

from financial_news_ml.features import ManualTfidfVectorizer


LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


class FinancialSentimentClassifier:
    """Class-balanced Logistic Regression with custom TF-IDF features."""

    def __init__(self, min_df: int = 2) -> None:
        self.vectorizer = ManualTfidfVectorizer(min_df=min_df)
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

    def fit(self, texts: list[str], labels: list[int]) -> FinancialSentimentClassifier:
        """Fit the vectorizer and classifier."""
        features = self.vectorizer.fit_transform(texts)
        self.model.fit(features, labels)
        return self

    def predict(self, texts: list[str]) -> list[int]:
        """Return numeric sentiment predictions."""
        features = self.vectorizer.transform(texts)
        return self.model.predict(features).tolist()

    def predict_sentiments(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return readable labels and class probabilities."""
        features = self.vectorizer.transform(texts)
        labels = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        return [
            {
                "label": int(label),
                "sentiment": LABEL_NAMES[int(label)],
                "probabilities": {
                    LABEL_NAMES[int(class_label)]: float(probability)
                    for class_label, probability in zip(
                        self.model.classes_, row_probabilities, strict=True
                    )
                },
            }
            for label, row_probabilities in zip(labels, probabilities, strict=True)
        ]

    def save(self, path: str | Path) -> None:
        """Serialize the fitted classifier and vectorizer."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> FinancialSentimentClassifier:
        """Load a serialized classifier."""
        classifier = joblib.load(path)
        if not isinstance(classifier, cls):
            raise TypeError("The artifact is not a FinancialSentimentClassifier.")
        return classifier
