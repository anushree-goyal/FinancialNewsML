from financial_news_ml.model import FinancialSentimentClassifier


def test_classifier_predicts_valid_labels():
    texts = [
        "profit increased strongly",
        "revenue rose significantly",
        "company issued a routine statement",
        "board scheduled the annual meeting",
        "profit declined sharply",
        "company reported a severe loss",
    ]
    labels = [2, 2, 1, 1, 0, 0]
    classifier = FinancialSentimentClassifier(min_df=1).fit(texts, labels)

    predictions = classifier.predict(["profit rose", "severe loss"])

    assert len(predictions) == 2
    assert set(predictions).issubset({0, 1, 2})


def test_classifier_can_be_saved_and_loaded(tmp_path):
    classifier = FinancialSentimentClassifier(min_df=1).fit(
        ["profit rose", "routine statement", "profit fell"],
        [2, 1, 0],
    )
    artifact = tmp_path / "model.joblib"

    classifier.save(artifact)
    restored = FinancialSentimentClassifier.load(artifact)

    assert restored.predict(["profit rose"]) == classifier.predict(["profit rose"])
