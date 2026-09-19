# FinancialNewsML

The goal of this project is to build models that, given financial news containing information about companies, can use it to predict subsequent market behavior.

I will be working in phases, beginning with a classical NLP baseline using TF-IDF vectorization, against which subsequent models will be compared and evaluated.


## Aim

The aim is to build an end-to-end workflow for extracting useful signals from
financial news. This includes understanding the data, representing financial
language numerically, training and evaluating models, and being careful about
issues such as class imbalance and data leakage.

## Research question

Can machine-learning models extract useful information from financial news
about sentiment, event type, and subsequent market reactions?

## Phase 1: classical sentiment baseline

The first phase focuses only on sentiment classification. Given a financial
sentence, the model predicts whether it is negative, neutral, or positive.

Phase 1 includes:

- a documented and reproducible dataset-preparation process;
- a custom implementation of TF-IDF;
- a classical sentiment classifier;
- evaluation that accounts for the imbalanced sentiment classes;
- error analysis and interpretable model features; and
- reusable training and prediction code with automated tests.

For the dataset, I use the 75%-agreement subset of Financial PhraseBank from HuggingFace. It
retains more examples than the all-agreement (allagree) subset while still requiring a clear majority of annotators to agree on each label.

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
│   ├── 02_baseline_model.ipynb
│   └── 03_finbert_sentiment.ipynb
├── src/financial_news_ml/
│   ├── features.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/
├── models/                 # generated artifacts are ignored by Git
├── requirements.txt
├── requirements-phase2.txt
└── README.md
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Phase 2 has additional transformer dependencies:

```bash
python -m pip install -r requirements-phase2.txt
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
- `03_finbert_sentiment.ipynb` prepares the matching data split and tokenizer
  for the Phase 2 finance-pretrained transformer experiment.

## Limitations of Phase 1 and their possible extensions

- Financial PhraseBank contains isolated sentences rather than complete news
  articles.
- The dataset is relatively small and has a neutral-class majority (meaning a
  model that always predicts "neutral" still has a reasonable accuracy).
- The tokenizer uses lowercase alphanumeric unigrams and does not represent
  word order, negation scope, or broader context.

Some of TF-IDF's limitations include:

1. **Meaning from context missing:** TF-IDF treats a term in isolation. Phase 2
   starts to address this by using a transformer that handles context.
2. **Synonyms:** identify words that can have similar meanings in a particular
   sentence, rather than treating them as completely separate features.
3. **Word order and grammar:** preserve some information about the order and
   relationship between words, including which words act as the subject or
   object of a statement.

## Roadmap

- **Phase 1 — complete:** classical financial sentiment baseline.
- **Phase 2 — in progress:** transformer-based NLP and financial event extraction.
- **Phase 3:** timestamped news and market-data integration.
- **Phase 4:** market-reaction modeling with time-based validation.
- **Phase 5:** API or interactive application.
