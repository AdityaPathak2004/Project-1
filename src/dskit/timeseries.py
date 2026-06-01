"""
dskit.timeseries — Time series analysis utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Optional

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


_FREQ_PERIOD: dict[str, int] = {
    "MS": 12, "M": 12, "QS": 4, "Q": 4, "W": 52, "D": 7, "B": 5, "H": 24,
}


def _infer_period(series: pd.Series) -> int:
    """Guess the seasonal period from the DatetimeIndex frequency string."""
    if hasattr(series.index, "freq") and series.index.freq is not None:
        freq_str = str(series.index.freq)
        for key, period in _FREQ_PERIOD.items():
            if key in freq_str:
                return period
    raise ValueError(
        "Cannot infer seasonal period from index frequency. "
        "Pass `period` explicitly."
    )


def decompose_series(
    series: pd.Series,
    model: Literal["additive", "multiplicative"] = "additive",
    period: Optional[int] = None,
):
    """Decompose a time series into trend, seasonal, and residual components.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series to decompose.
    model : {"additive", "multiplicative"}
        Use "multiplicative" when the seasonal amplitude grows with the level.
    period : int, optional
        Seasonal period. Inferred from DatetimeIndex frequency when omitted.

    Returns
    -------
    statsmodels DecomposeResult — access .trend, .seasonal, .resid attributes.
    """
    if period is None:
        period = _infer_period(series)
    return seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")


def adf_test(series: pd.Series, signif: float = 0.05) -> dict:
    """Augmented Dickey-Fuller unit-root test.

    Returns a structured dict with ADF statistic, p-value, critical values,
    and a boolean ``is_stationary`` flag (True = reject unit-root null).
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_statistic": round(float(result[0]), 4),
        "p_value": round(float(result[1]), 4),
        "n_lags_used": int(result[2]),
        "n_observations": int(result[3]),
        "critical_values": {k: round(float(v), 3) for k, v in result[4].items()},
        "is_stationary": bool(result[1] < signif),
    }


def make_stationary(
    series: pd.Series,
    max_diffs: int = 2,
    log_transform: bool = False,
) -> tuple[pd.Series, int, bool]:
    """Iteratively difference a series until the ADF test passes.

    Parameters
    ----------
    series : pd.Series
        Raw time series.
    max_diffs : int
        Maximum differencing operations before giving up.
    log_transform : bool
        Apply log transform before differencing (handles multiplicative structure).

    Returns
    -------
    (stationary_series, n_diffs_applied, log_was_applied)
    """
    s = np.log(series) if log_transform else series.copy()
    for d in range(max_diffs + 1):
        if adf_test(s)["is_stationary"]:
            return s.dropna(), d, log_transform
        s = s.diff().dropna()
    return s, max_diffs, log_transform


def arima_forecast(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    steps: int = 12,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit an ARIMA model and return a forecast DataFrame with confidence intervals.

    Parameters
    ----------
    series : pd.Series
        Training series.
    order : (p, d, q)
        ARIMA order. Default (1, 1, 1).
    steps : int
        Number of future periods to forecast.
    alpha : float
        Significance level for the confidence interval.

    Returns
    -------
    pd.DataFrame with columns: forecast, lower_ci, upper_ci.
    """
    fitted = ARIMA(series, order=order).fit()
    fcast = fitted.get_forecast(steps=steps)
    ci = fcast.conf_int(alpha=alpha)
    return pd.DataFrame(
        {
            "forecast": fcast.predicted_mean,
            "lower_ci": ci.iloc[:, 0],
            "upper_ci": ci.iloc[:, 1],
        }
    )


def rolling_stats(series: pd.Series, window: int = 12) -> pd.DataFrame:
    """Compute rolling mean and standard deviation alongside the original series."""
    return pd.DataFrame(
        {
            "original": series,
            "rolling_mean": series.rolling(window=window).mean(),
            "rolling_std": series.rolling(window=window).std(),
        }
    )
