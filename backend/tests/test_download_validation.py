import io

import pandas as pd
import pytest

from app.datasets.download import ValidationError, load_dataframe, validate_dataframe


def _csv(bytes_data: bytes) -> str:
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "wb") as fh:
        fh.write(bytes_data)
    return path


def test_valid_csv_passes():
    path = _csv(b"a,b,c\n1,2,x\n3,4,y\n")
    df, skipped = load_dataframe(path, "csv")
    report = validate_dataframe(df, "csv", skipped)
    assert report["rows"] == 2
    assert report["columns"] == 3


def test_empty_file_rejected():
    path = _csv(b"a,b\n")
    df, _ = load_dataframe(path, "csv")
    with pytest.raises(ValidationError) as exc:
        validate_dataframe(df, "csv")
    assert exc.value.code == "DATASET_EMPTY"


def test_malformed_csv_rejected():
    # Rows with wildly inconsistent field counts -> most rows collapse to NaN
    path = _csv(b"a,b,c\n1,2,x\n,,,,,,,,,\n,,,,,,,,,\n")
    df, skipped = load_dataframe(path, "csv")
    with pytest.raises(ValidationError) as exc:
        validate_dataframe(df, "csv", skipped)
    assert exc.value.code == "DATASET_MALFORMED"


def test_unsupported_format():
    with pytest.raises(ValidationError) as exc:
        load_dataframe("/dev/null", "xlsx")
    assert exc.value.code == "UNSUPPORTED_FORMAT"


def test_parquet_roundtrip(tmp_path):
    df_in = pd.DataFrame({"num": [1.5, 2.5, 3.5], "cat": ["a", "b", "a"]})
    pq = tmp_path / "d.parquet"
    df_in.to_parquet(pq)
    df, skipped = load_dataframe(str(pq), "parquet")
    report = validate_dataframe(df, "parquet", skipped)
    assert report["rows"] == 3
