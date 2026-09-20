# XGBoost model package

Deployment-ready artifact produced by RL-AutoML.

## Model

| Field | Value |
| --- | --- |
| Model | XGBoost (`xgboost`) |
| Framework | xgboost |
| Task | classification |
| Objective | species prediction |
| Primary metric | f1 (maximize) |
| Validation score | 0.8982 |
| Test score | 0.9666 |
| Raw features | 4 |
| Transformed features | 8 |
| Training time | 9.0614 s |
| Peak memory | 867.98 MB |
| Trained at | 2026-09-19T08:30:39+00:00 |
| Run id | run_5494d62bee71 |

This package was verified after being archived: the ZIP was re-opened, its checksums tested, and its model reloaded and compared against the in-process model.

## Installation

```
pip install -r requirements.txt
```

## Expected input

One row per prediction. Required columns:

| Column | Kind | Required | Nullable | Unique values |
| --- | --- | --- | --- | --- |
| `sepal length (cm)` | numeric | yes | no | 35 |
| `sepal width (cm)` | numeric | yes | no | 23 |
| `petal length (cm)` | numeric | yes | no | 43 |
| `petal width (cm)` | numeric | yes | no | 22 |

`species` is the training target and is **not** an input column.

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
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
0.0,0.0,0.0,0.0
```

Output:

```
2
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
| accuracy | 0.9 | 0.9667 |
| average_precision | 0.9078 | 1 |
| balanced_accuracy | 0.9 | 0.9667 |
| f1 | 0.8982 | 0.9666 |
| log_loss | 0.4085 | 0.0682 |
| matthews | 0.8514 | 0.9516 |
| precision | 0.8993 | 0.9697 |
| recall | 0.9 | 0.9667 |

## Training data

| Field | Value |
| --- | --- |
| Dataset | iris.csv |
| SHA-256 |  |
| Rows | 150 |
| Columns | 5 |
| Missing ratio | 0 |
| Split | stratified - train 90, validation 30, test 30 |

## Recorded warnings

* feature-name count (12) differs from the transformed width (8); regenerating generic names
* 1 duplicate rows detected; consider whether they are legitimate

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

