"""
dskit.features — Feature engineering utilities.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from typing import List, Literal, Optional


def encode_categoricals(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: Literal["onehot", "label"] = "onehot",
    max_cardinality: int = 10,
) -> pd.DataFrame:
    """Encode categorical columns. Auto-detects object/category columns when cols is None."""
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if method == "onehot":
        high_card = [c for c in cols if df[c].nunique() > max_cardinality]
        low_card = [c for c in cols if df[c].nunique() <= max_cardinality]
        if low_card:
            df = pd.get_dummies(df, columns=low_card, drop_first=True)
        for col in high_card:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == "label":
        for col in cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def create_date_features(
    df: pd.DataFrame, col: str, drop_original: bool = True
) -> pd.DataFrame:
    """Extract temporal features from a datetime column."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    prefix = col + "_"
    df[prefix + "year"] = df[col].dt.year
    df[prefix + "month"] = df[col].dt.month
    df[prefix + "day"] = df[col].dt.day
    df[prefix + "dayofweek"] = df[col].dt.dayofweek
    df[prefix + "quarter"] = df[col].dt.quarter
    df[prefix + "is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
    df[prefix + "week_of_year"] = df[col].dt.isocalendar().week.astype(int)
    if drop_original:
        df = df.drop(columns=[col])
    return df


def scale_features(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: Literal["standard", "minmax", "robust"] = "standard",
) -> pd.DataFrame:
    """Scale numeric features using the specified method."""
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[method]
    df[cols] = scaler.fit_transform(df[cols])
    return df


def handle_missing(
    df: pd.DataFrame,
    strategy: Literal["mean", "median", "mode", "knn", "drop"] = "median",
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """Handle missing values across the DataFrame."""
    df = df.copy()
    if strategy == "drop":
        return df.dropna()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if strategy == "knn" and numeric_cols:
        df[numeric_cols] = KNNImputer(n_neighbors=knn_neighbors).fit_transform(df[numeric_cols])
    elif strategy in ("mean", "median") and numeric_cols:
        df[numeric_cols] = SimpleImputer(strategy=strategy).fit_transform(df[numeric_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    return df


def add_polynomial_features(
    df: pd.DataFrame, cols: List[str], degree: int = 2
) -> pd.DataFrame:
    """Add polynomial and interaction features for the specified columns."""
    df = df.copy()
    for col in cols:
        for d in range(2, degree + 1):
            df[f"{col}_pow{d}"] = df[col] ** d
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
    return df
