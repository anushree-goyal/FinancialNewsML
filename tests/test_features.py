import math

import pytest

from financial_news_ml.features import ManualTfidfVectorizer


def test_tokenize_lowercases_and_removes_punctuation():
    assert ManualTfidfVectorizer.tokenize("Profit rose 10%!") == [
        "profit",
        "rose",
        "10",
    ]


def test_fit_transform_has_expected_shape_and_values():
    documents = ["profit rose", "profit fell", "revenue rose"]
    vectorizer = ManualTfidfVectorizer(min_df=1)

    matrix = vectorizer.fit_transform(documents)

    assert matrix.shape == (3, 4)
    profit_column = vectorizer.vocabulary_["profit"]
    expected_idf = math.log(4 / 3) + 1
    assert matrix[0, profit_column] == pytest.approx(0.5 * expected_idf)


def test_transform_ignores_unknown_words_and_handles_empty_text():
    vectorizer = ManualTfidfVectorizer(min_df=1).fit(["profit rose"])

    matrix = vectorizer.transform(["unknown term", ""])

    assert matrix.shape == (2, 2)
    assert matrix.nnz == 0


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        ManualTfidfVectorizer().transform(["profit rose"])
