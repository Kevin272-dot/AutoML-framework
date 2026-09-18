"""Preprocessing pipeline: dedupe, constant/ID removal, imputation, encoding."""

import numpy as np
import pandas as pd

from app.datasets.preprocess import preprocess_dataframe


def _base_df():
    return pd.DataFrame({
        "record_id": [f"id{i}" for i in range(10)],
        "amount": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
        "region": ["north", "south", "north", None, "east", "west", "south", "north", "east", "west"],
        "constant": ["same"] * 10,
    })


def test_duplicates_constant_and_ids_removed():
    df = pd.concat([_base_df(), _base_df().head(2)], ignore_index=True)
    processed, report = preprocess_dataframe(df)
    assert report["rows_dropped_duplicates"] == 2
    assert "constant" in report["constant_columns_removed"]
    assert "record_id" in report["identifier_columns_excluded"]
    assert "record_id" not in processed.columns
    assert "constant" not in processed.columns


def test_missing_imputed_and_no_missing_remain():
    df = _base_df()
    processed, report = preprocess_dataframe(df)
    assert report["missing_cells_before"] > 0
    assert report["missing_cells_after"] == 0
    assert report["imputed"]["amount"] == "median"
    assert report["imputed"]["region"] == "mode"
    assert not processed.isna().any().any()


def test_low_cardinality_one_hot_encoded():
    df = _base_df()
    processed, report = preprocess_dataframe(df)
    assert "region" in report["one_hot_encoded"]
    for col in processed.columns:
        if col.startswith("region_"):
            assert set(processed[col].unique()).issubset({0, 1})


def test_high_cardinality_ordinal_encoded():
    df = pd.DataFrame({
        "merchant_name": [f"shop-{i}" for i in range(50)],
        "value": np.arange(50, dtype=float),
    })
    processed, report = preprocess_dataframe(df)
    assert "merchant_name" in report["ordinal_encoded"]
    assert processed["merchant_name"].dtype.kind in "iu"


def test_outliers_flagged_not_altered():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0]})
    _, report = preprocess_dataframe(df)
    assert report["outliers"]["value"] == 1
    assert 1000.0 in df["value"].values or 1000 in df["value"].values
