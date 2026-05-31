import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import pytest

from dskit.features import (
    encode_categoricals,
    create_date_features,
    scale_features,
    handle_missing,
    add_polynomial_features,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(35, 10, 50),
        "salary": np.random.normal(50000, 15000, 50),
        "gender": np.random.choice(["M", "F"], 50),
        "city": np.random.choice(["NYC", "LA", "SF", "Chicago", "Boston"], 50),
        "date": pd.date_range("2020-01-01", periods=50, freq="D").astype(str),
    })


@pytest.fixture
def df_with_missing(sample_df):
    df = sample_df.copy()
    df.loc[:4, "age"] = np.nan
    df.loc[:2, "gender"] = np.nan
    return df


def test_encode_onehot_removes_original(sample_df):
    result = encode_categoricals(sample_df, cols=["gender"], method="onehot")
    assert "gender" not in result.columns


def test_encode_label_preserves_column(sample_df):
    result = encode_categoricals(sample_df, cols=["gender"], method="label")
    assert "gender" in result.columns
    assert pd.api.types.is_integer_dtype(result["gender"])


def test_create_date_features_drops_original(sample_df):
    result = create_date_features(sample_df, col="date")
    assert "date" not in result.columns
    assert "date_year" in result.columns
    assert "date_is_weekend" in result.columns


def test_create_date_features_keep_original(sample_df):
    result = create_date_features(sample_df, col="date", drop_original=False)
    assert "date" in result.columns


def test_scale_standard_mean_zero(sample_df):
    result = scale_features(sample_df, cols=["age", "salary"], method="standard")
    assert abs(result["age"].mean()) < 1e-9


def test_scale_minmax_range(sample_df):
    result = scale_features(sample_df, cols=["age"], method="minmax")
    assert result["age"].min() >= 0.0
    assert result["age"].max() <= 1.0


def test_handle_missing_median_no_nulls(df_with_missing):
    result = handle_missing(df_with_missing, strategy="median")
    assert result["age"].isnull().sum() == 0
    assert result["gender"].isnull().sum() == 0


def test_handle_missing_drop_reduces_rows(df_with_missing):
    result = handle_missing(df_with_missing, strategy="drop")
    assert len(result) < len(df_with_missing)
    assert result.isnull().sum().sum() == 0


def test_polynomial_features_count():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = add_polynomial_features(df, cols=["a", "b"], degree=2)
    assert "a_pow2" in result.columns
    assert "b_pow2" in result.columns
    assert "a_x_b" in result.columns


def test_polynomial_feature_values():
    df = pd.DataFrame({"x": [2.0, 3.0]})
    result = add_polynomial_features(df, cols=["x"], degree=2)
    assert list(result["x_pow2"]) == [4.0, 9.0]
