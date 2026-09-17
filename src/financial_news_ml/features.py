"""Custom TF-IDF feature extraction used by the Phase 1 baseline."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix


TOKEN_PATTERN = re.compile(r"\b\w+\b")


class ManualTfidfVectorizer:
    """Convert text to a sparse unigram TF-IDF matrix.

    The implementation uses relative term frequency and smoothed inverse
    document frequency. Vocabulary and IDF values are learned by ``fit`` and
    reused unchanged by ``transform``.
    """

    def __init__(self, min_df: int = 2) -> None:
        if min_df < 1:
            raise ValueError("min_df must be at least 1")
        self.min_df = min_df
        self.vocabulary_: dict[str, int] | None = None
        self.idf_: dict[str, float] | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Return lowercase alphanumeric tokens from text."""
        return TOKEN_PATTERN.findall(text.lower())

    def fit(self, documents: Iterable[str]) -> ManualTfidfVectorizer:
        """Learn vocabulary and IDF values from documents."""
        documents = list(documents)
        document_frequencies: Counter[str] = Counter()

        for document in documents:
            document_frequencies.update(set(self.tokenize(document)))

        terms = sorted(
            term
            for term, frequency in document_frequencies.items()
            if frequency >= self.min_df
        )
        self.vocabulary_ = {term: index for index, term in enumerate(terms)}

        document_count = len(documents)
        self.idf_ = {
            term: math.log(
                (1 + document_count) / (1 + document_frequencies[term])
            )
            + 1
            for term in terms
        }
        return self

    def transform(self, documents: Sequence[str] | Iterable[str]) -> csr_matrix:
        """Transform documents using the fitted vocabulary and IDF values."""
        if self.vocabulary_ is None or self.idf_ is None:
            raise RuntimeError("The vectorizer must be fitted before transform().")

        documents = list(documents)
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []

        for row_index, document in enumerate(documents):
            tokens = self.tokenize(document)
            if not tokens:
                continue

            counts = Counter(tokens)
            document_length = len(tokens)

            for term, count in counts.items():
                column_index = self.vocabulary_.get(term)
                if column_index is None:
                    continue

                term_frequency = count / document_length
                values.append(term_frequency * self.idf_[term])
                row_indices.append(row_index)
                column_indices.append(column_index)

        return csr_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(documents), len(self.vocabulary_)),
            dtype=np.float64,
        )

    def fit_transform(self, documents: Sequence[str] | Iterable[str]) -> csr_matrix:
        """Fit on documents and return their TF-IDF matrix."""
        documents = list(documents)
        return self.fit(documents).transform(documents)

    def get_feature_names_out(self) -> np.ndarray:
        """Return feature names ordered by matrix column."""
        if self.vocabulary_ is None:
            raise RuntimeError("The vectorizer must be fitted first.")
        return np.asarray(list(self.vocabulary_), dtype=object)
