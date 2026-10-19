# Megaline Mobile Plan Classification

Predicting mobile plan selection from monthly customer usage with supervised machine learning.

This project compares decision trees, random forests, and logistic regression to classify customers into Megaline's **Smart** or **Ultra** plans. It covers stratified data splitting, hyperparameter tuning, model evaluation, and comparison with a majority-class baseline.

## Business Problem

Megaline wants to help customers move from legacy plans to its newer Smart and Ultra plans. Using usage records from customers already subscribed to these plans, the objective is to build a classifier with a test accuracy of at least **75%**.

The target is the customer's observed plan. Predicting that label can support a recommendation workflow, but does not establish which plan is cheapest or best for an individual customer. Pricing, customer preferences, and the impact of recommendations are outside this study's scope.

## Dataset

The dataset contains **3,214 monthly customer usage records**, four input features, and one binary target. The supplied data are already preprocessed, and the initial inspection found no missing values.

| Column | Description | Role |
| --- | --- | --- |
| `calls` | Number of calls during the month | Feature |
| `minutes` | Total call duration in minutes | Feature |
| `messages` | Number of text messages | Feature |
| `mb_used` | Mobile internet usage in megabytes | Feature |
| `is_ultra` | Current plan: Smart = `0`, Ultra = `1` | Target |

Approximately 69% of records belong to Smart and 31% to Ultra. This class imbalance makes a majority-class baseline an essential comparison.

The full dataset is included in this repository at `data/users_behavior.csv`, so no separate download is needed after cloning.

Dataset source: TripleTen AI and Data Science course materials.

## Methodology

1. Inspect the dataset's structure, data types, and missing values.
2. Separate the four usage features from the target.
3. Split the data into **60% training, 20% validation, and 20% test** sets using stratification and `random_state=123`.
4. Compare three classification algorithms using the validation set:
   - **Decision tree:** search `max_depth` from 1 to 10.
   - **Random forest:** search `n_estimators` from 10 to 100 in steps of 10 and `max_depth` from 1 to 10.
   - **Logistic regression:** use `solver="liblinear"` as a linear comparison model.
5. Evaluate the validation-selected configurations on the test set.
6. Compare test accuracy with a `DummyClassifier(strategy="most_frequent")` baseline.

The tree-based models are fitted again on the training set with their selected hyperparameters before test evaluation. Training and validation sets are not combined for the reported results.

## Results

The following scores are recorded in the course-reviewed notebook. They will be checked again when the portfolio notebook is finalized; results may vary across dependency versions.

| Model | Configuration | Validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Majority-class baseline | Always predict Smart | — | 69.36% |
| Decision tree | `max_depth=9` | 79.32% | 79.47% |
| Random forest | `n_estimators=20`, `max_depth=10` | **80.40%** | **80.25%** |
| Logistic regression | `solver="liblinear"` | 75.12% | 76.83% |

All three trained classifiers exceed the 75% test accuracy target and outperform the majority-class baseline.

The validation-selected random forest achieves the highest accuracy among these configurations, improving on the baseline by **10.89 percentage points** on the test set. The decision tree is **0.78 percentage points** behind the random forest and remains a candidate when model simplicity matters.

The course project favors the decision tree as a simpler alternative. A deployment decision would require measured training time, inference latency, and model size; this study does not establish a performance or scalability advantage.

An additional random forest with 100 trees reached 81.18% test accuracy during exploratory comparisons. That configuration was not the validation winner, so its score is not used as the headline result. Since the test set was inspected for multiple configurations, a fresh holdout would be needed for an independent evaluation of further model-selection decisions.

## Repository Structure

The intended portfolio layout is:

```text
megaline-mobile-plan-classification/
├── notebook.ipynb
├── data/
│   └── users_behavior.csv    # Full dataset, tracked by Git
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally with Anaconda and VS Code

Use Anaconda to manage the project environment and install the **Python** and **Jupyter** extensions in VS Code.

Open **Anaconda Prompt**, then clone the repository and enter its folder:

```bat
git clone https://github.com/pedrolocosta/megaline-mobile-plan-classification.git
cd megaline-mobile-plan-classification
```

If you have already cloned the repository, open Anaconda Prompt and navigate to your existing project folder instead.

Create and activate a dedicated Conda environment, then install the project dependencies:

```bat
conda create --name megaline-ml python=3.11 pip -y
conda activate megaline-ml
python -m pip install -r requirements.txt
code .
```

For future sessions, activate the existing environment with `conda activate megaline-ml`; you do not need to create it again.

In VS Code, open the Command Palette with **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and select the Python interpreter from **megaline-ml**.

The dataset is included in `data/`. Open `notebook.ipynb` and choose **Select Kernel → Python Environments → megaline-ml**. The portfolio notebook should load the dataset relative to the repository root:

```python
from pathlib import Path
import pandas as pd

df = pd.read_csv(Path("data") / "users_behavior.csv")
```

Restart the kernel and run all cells from top to bottom to check that the analysis is reproducible without previously stored variables.

## Tools

- Python
- pandas
- scikit-learn
- Jupyter Notebook
- Anaconda / Conda
- VS Code

## Limitations and Next Steps

- Accuracy summarizes overall correctness but can hide differences between Smart and Ultra predictions. A confusion matrix, precision, recall, and F1-score would provide a fuller assessment.
- The logistic regression comparison uses unscaled features. A pipeline with feature scaling and regularization tuning would make it a stronger benchmark.
- Results come from a single random split. Cross-validation on the development data could assess sensitivity to the split, followed by evaluation on a fresh holdout.
- The dataset does not provide the customer identifiers or timestamps needed to verify separation by customer or evaluate performance over time.
- This is an offline classification study, not a deployed recommendation service or evidence of increased revenue or customer satisfaction.

## Author and Context

**Pedro Leonardo Costa** — Engineer transitioning into AI engineering.

Developed as part of the TripleTen AI and Data Science program and adapted for my machine learning portfolio. The project demonstrates a progression from exploratory data analysis to supervised model development and evaluation.

[LinkedIn](https://www.linkedin.com/in/pedrolocosta/) · [GitHub](https://github.com/pedrolocosta)
