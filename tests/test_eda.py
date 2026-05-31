import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import pytest

from dskit.eda import auto_eda, missing_report, outlier_summary, target_analysis


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(35, 10, 100),
        "salary": np.random.normal(50000, 15000, 100),
        "score": np.concatenate([np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]]),
        "category": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.randint(0, 2, 100),
    })


@pytest.fixture
def df_with_missing(sample_df):
    df = sample_df.copy()
    df.loc[:9, "age"] = np.nan
    df.loc[:4, "category"] = np.nan
    return df


def test_auto_eda_returns_expected_keys(sample_df):
    report = auto_eda(sample_df)
    for key in ("shape", "dtypes", "missing", "numeric_stats", "outliers"):
        assert key in report


def test_auto_eda_shape(sample_df):
    assert auto_eda(sample_df)["shape"] == (100, 5)


def test_auto_eda_with_target_classification(sample_df):
    report = auto_eda(sample_df, target="target")
    assert report["target_analysis"]["type"] == "classification"


def test_auto_eda_with_target_regression(sample_df):
    report = auto_eda(sample_df, target="age")
    assert report["target_analysis"]["type"] == "regression"


def test_missing_report_no_missing(sample_df):
    assert len(missing_report(sample_df)) == 0


def test_missing_report_counts(df_with_missing):
    result = missing_report(df_with_missing)
    assert result.loc["age", "missing_count"] == 10
    assert result.loc["category", "missing_count"] == 5


def test_missing_report_sorted_descending(df_with_missing):
    result = missing_report(df_with_missing)
    assert result["missing_pct"].is_monotonic_decreasing


def test_outlier_summary_normal_low_outlier_rate():
    s = pd.Series(np.random.normal(0, 1, 1000))
    result = outlier_summary(s)
    assert result["iqr_outlier_pct"] < 10


def test_outlier_summary_detects_extremes():
    s = pd.Series(list(np.random.normal(0, 1, 95)) + [100, -100, 200, -200, 150])
    assert outlier_summary(s)["iqr_outlier_count"] > 0


def test_outlier_summary_non_numeric_returns_empty():
    assert outlier_summary(pd.Series(["a", "b", "c"])) == {}


def test_target_analysis_regression_has_skewness(sample_df):
    result = target_analysis(sample_df, "age")
    assert "skewness" in result and "kurtosis" in result


def test_target_analysis_classification_n_classes(sample_df):
    result = target_analysis(sample_df, "category")
    assert result["n_classes"] == 3
