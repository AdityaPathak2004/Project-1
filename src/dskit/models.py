"""
dskit.models — Model comparison framework.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix
from typing import Literal, Optional

warnings.filterwarnings("ignore")

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
}

REGRESSOR = {
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
}


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: Literal["classification", "regression"] = "classification",
    cv: int = 5,
    scoring: Optional[str] = None,
) -> pd.DataFrame:
    """Train and compare multiple models using cross-validation. Returns a ranked DataFrame."""
    models = CLASSIFIERS if task == "classification" else REGRESSOR
    if scoring is None:
        scoring = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"
    cv_strategy = StratifiedKFold(n_splits=cv) if task == "classification" else KFold(n_splits=cv)
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)
        results.append({
            "Model": name,
            "Mean Score": round(scores.mean(), 4),
            "Std Dev": round(scores.std(), 4),
            "Min": round(scores.min(), 4),
            "Max": round(scores.max(), 4),
        })
    return pd.DataFrame(results).sort_values("Mean Score", ascending=False).reset_index(drop=True)


def get_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """Extract and rank feature importances from a tree-based model."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__} does not support feature_importances_")
    df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def train_and_evaluate(model, X_train, X_test, y_train, y_test, task="classification") -> dict:
    """Fit a model and return evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if task == "classification":
        return {
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }
