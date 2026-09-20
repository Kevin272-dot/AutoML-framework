# XGBoost model package

Deployment-ready artifact produced by RL-AutoML.

## Model

| Field | Value |
| --- | --- |
| Model | XGBoost (`xgboost`) |
| Framework | xgboost |
| Task | classification |
| Objective | churn prediction |
| Primary metric | f1 (maximize) |
| Validation score | 0.457 |
| Test score | 0.4713 |
| Raw features | 9 |
| Transformed features | 18 |
| Training time | 5.5065 s |
| Peak memory | 883.59 MB |
| Trained at | 2026-09-19T08:31:49+00:00 |
| Run id | run_961c567816b3 |

This package was verified after being archived: the ZIP was re-opened, its checksums tested, and its model reloaded and compared against the in-process model.

## Installation

```
pip install -r requirements.txt
```

## Expected input

One row per prediction. Required columns:

| Column | Kind | Required | Nullable | Unique values |
| --- | --- | --- | --- | --- |
| `contract_type` | categorical | yes | no | 3 |
| `payment_method` | categorical | yes | no | 4 |
| `internet_service` | categorical | yes | no | 3 |
| `tenure_months` | numeric | yes | no | 72 |
| `monthly_charges` | numeric | yes | yes | 2354 |
| `total_charges` | numeric | yes | yes | 2659 |
| `support_calls` | numeric | yes | no | 8 |
| `age` | numeric | yes | yes | 72 |
| `signup_date` | datetime | yes | no | 1651 |

`churn` is the training target and is **not** an input column.

## Usage

```python
from model_loader import load_model

model = load_model("model_package")

# A DataFrame, a dict, a list of row dicts, or a path to a CSV/Parquet file
predictions = model.predict("new_rows.csv")
print(predictions)
```

From the command line:

```
python predict.py --input new_rows.csv
```

## Example

Input (`inference/example_input.csv`, synthesised from the schema -- it contains no real data):

```
contract_type,payment_method,internet_service,tenure_months,monthly_charges,total_charges,support_calls,age,signup_date
month-to-month,bank_transfer,fiber,0.0,0.0,0.0,0.0,0.0,2024-01-01
```

Output:

```
yes
```

## Preprocessing

The package includes the **fitted** preprocessor, so the transformations applied here are
exactly the ones used at training time. In order:

1. `clip_outliers`
2. `datetime_features`
3. `encode_categorical`
4. `impute_categorical`
5. `impute_numeric`
6. `log_transform`
7. `missing_indicator`
8. `scale_robust`

Nothing needs to be re-fitted before use. Unseen categories in categorical columns are
routed to the encoder's infrequent bucket rather than raising.

## Metrics

| Metric | Validation | Test |
| --- | --- | --- |
| accuracy | 0.695 | 0.6933 |
| average_precision | 0.4969 | 0.4715 |
| balanced_accuracy | 0.6264 | 0.6351 |
| f1 | 0.457 | 0.4713 |
| log_loss | 0.612 | 0.6307 |
| matthews | 0.246 | 0.2584 |
| pr_auc | 0.4969 | 0.4715 |
| precision | 0.4375 | 0.4385 |
| recall | 0.4783 | 0.5093 |
| roc_auc | 0.732 | 0.7153 |

## Training data

| Field | Value |
| --- | --- |
| Dataset | churn_strings.csv |
| SHA-256 |  |
| Rows | 3000 |
| Columns | 10 |
| Missing ratio | 0.0178 |
| Split | stratified - train 1800, validation 600, test 600 |



## Contents

```
inference/example_input.csv
inference/model_loader.py
inference/predict.py
metadata/dataset_metadata.json
metadata/feature_schema.json
metadata/model_metadata.json
model/model.joblib
preprocessing/preprocessor.joblib
requirements.txt
runtime/core/errors.py
runtime/core/logging.py
runtime/execution/runtime_transformers.py
runtime/execution/torch_models.py
```

## Notes

* `predict()` returns the original label names, not encoded integers.
* `predict_proba()` returns class probabilities for classifiers that support it, else `None`.
* Inference runs on CPU only.

