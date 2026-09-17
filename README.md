# FinancialNewsML

FinancialNewsML is an end-to-end machine-learning project exploring whether
information extracted from financial news can help characterize and eventually
predict market reactions.

Phase 1 delivers a reproducible three-class financial sentiment classifier. It
uses a custom TF-IDF implementation and class-balanced Logistic Regression to
classify sentences as negative, neutral, or positive.

## Results

The model uses the 75%-agreement subset of Financial PhraseBank. After removing
five duplicate sentences, the dataset contains 3,448 observations.

| Model | Mean CV accuracy | Mean CV macro F1 |
|---|---:|---:|
| Unweighted Logistic Regression | 0.810 | 0.714 |
| Class-balanced Logistic Regression | **0.815** | **0.762** |

On the held-out 20% test partition, the selected class-balanced model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 0.826 |
| Macro F1 | 0.781 |
| Weighted F1 | 0.825 |

Class weighting increased negative recall from 0.464 to 0.702 while preserving
nearly the same overall accuracy. Cross-validation rebuilt TF-IDF features
inside every fold to prevent validation-data leakage.

## Methodology

1. Load Financial PhraseBank from Hugging Face.
2. Remove exact duplicate sentences before splitting.
3. Create a reproducible stratified 80/20 train-test split.
4. Build a unigram vocabulary using training data only.
5. Calculate relative term frequency and smoothed inverse document frequency.
6. Store TF-IDF features in sparse matrices.
7. Compare unweighted and class-balanced Logistic Regression with stratified
   five-fold cross-validation.
8. Evaluate the selected model using accuracy, per-class precision and recall,
   macro F1, weighted F1, and a confusion matrix.

The custom smoothed IDF calculation is:

```text
idf(term) = log((1 + number of documents) /
                (1 + documents containing term)) + 1
```

## Repository structure

```text
FinancialNewsML/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_model.ipynb
├── src/financial_news_ml/
│   ├── features.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/
├── models/                 # generated artifacts are ignored by Git
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the model

```bash
PYTHONPATH=src python -m financial_news_ml.train
```

This downloads or reads the cached dataset, trains the selected classifier,
prints held-out metrics, and creates:

```text
models/financial_sentiment_model.joblib
```

The generated artifact is ignored by Git and can be reproduced with the
training command.

## Make a prediction

After training:

```bash
PYTHONPATH=src python -m financial_news_ml.predict \
  "Operating profit increased substantially during the quarter."
```

The command returns the numeric label, readable sentiment, and probabilities
for all three classes.

## Run tests

```bash
PYTHONPATH=src pytest -q
```

## Notebooks

- `01_data_exploration.ipynb` documents dataset structure, class imbalance,
  duplicate handling, and the stratified split.
- `02_baseline_model.ipynb` implements custom TF-IDF, trains the classifiers,
  performs leakage-safe cross-validation, and evaluates predictions.

## Limitations

- Financial PhraseBank contains isolated sentences rather than complete news
  articles.
- The dataset is relatively small and has a neutral-class majority.
- The tokenizer uses lowercase alphanumeric unigrams and does not represent
  word order, negation scope, or broader context.
- Sentiment is not equivalent to a tradable market signal.
- The model has not been evaluated on newer news or other financial domains.

## Roadmap

- **Phase 1 — complete:** classical financial sentiment baseline.
- **Phase 2:** transformer-based NLP and financial event extraction.
- **Phase 3:** timestamped news and market-data integration.
- **Phase 4:** market-reaction modeling with time-based validation.
- **Phase 5 — optional:** API or interactive application.
