"""
dskit.eda — Exploratory Data Analysis utilities.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def auto_eda(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    """Run comprehensive EDA on a DataFrame and return a structured report."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": missing_report(df),
        "numeric_stats": df.describe(include="number").to_dict(),
        "categoricals": {
            col: df[col].value_counts().head(10).to_dict()
            for col in df.select_dtypes(include=["object", "category"]).columns
        },
        "correlations": df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {},
        "outliers": {col: outlier_summary(df[col]) for col in numeric_cols},
    }
    if target and target in df.columns:
        report["target_analysis"] = target_analysis(df, target)
    return report


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted DataFrame summarising missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return result[result["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def outlier_summary(series: pd.Series) -> dict:
    """Detect outliers via IQR and Z-score for a numeric Series."""
    if not pd.api.types.is_numeric_dtype(series):
        return {}
    clean = series.dropna()
    if len(clean) < 4:
        return {}
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    iqr_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
    z_mask = np.abs(stats.zscore(clean)) > 3
    return {
        "iqr_outlier_count": int(iqr_mask.sum()),
        "iqr_outlier_pct": round(float(iqr_mask.mean() * 100), 2),
        "z_outlier_count": int(z_mask.sum()),
        "z_outlier_pct": round(float(z_mask.mean() * 100), 2),
    }


def target_analysis(df: pd.DataFrame, target: str) -> dict:
    """Summarise the target column — detects regression vs. classification."""
    col = df[target]
    if pd.api.types.is_numeric_dtype(col):
        return {
            "type": "regression",
            "stats": col.describe().to_dict(),
            "skewness": round(float(col.skew()), 4),
            "kurtosis": round(float(col.kurt()), 4),
        }
    return {
        "type": "classification",
        "class_distribution": col.value_counts(normalize=True).round(4).to_dict(),
        "n_classes": int(col.nunique()),
    }


def correlation_analysis(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return pairwise correlation matrix for numeric columns."""
    return df.select_dtypes(include="number").corr(method=method)
