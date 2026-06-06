"""Project 4 — Egypt national yield forecasting (Linear Regression, FAO).

Faithful to the notebook:
  - Features: lag_1, lag_2, lag_3, rolling_mean_3 (mean of the three lags)
  - Artifact is a Pipeline(StandardScaler, LinearRegression) -> predict() scales internally
  - Recursive: each predicted year is appended to history and feeds the next lags
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = _DIR / "egypt_linreg.pkl"
HISTORY_PATH = _DIR / "egypt_history.csv"   # columns: Year, Yield (actuals)

FEATURES = ["lag_1", "lag_2", "lag_3", "rolling_mean_3"]


def load_model():
    return joblib.load(MODEL_PATH)


def load_history() -> pd.DataFrame:
    """Actual Year/Yield series, sorted ascending."""
    h = pd.read_csv(HISTORY_PATH)
    return h[["Year", "Yield"]].dropna().sort_values("Year").reset_index(drop=True)


def recursive_forecast(model, forecast_years, history: pd.DataFrame):
    """Replicates the notebook loop exactly.

    forecast_years : iterable of years to predict (e.g. [2025, 2026, 2027])
    history        : actual Year/Yield df used to seed the lags
    returns        : list of (year, predicted_yield)
    """
    yields = history["Yield"].tolist()
    out = []
    for yr in forecast_years:
        lag_1, lag_2, lag_3 = yields[-1], yields[-2], yields[-3]
        rolling_mean = float(np.mean([lag_1, lag_2, lag_3]))
        x = pd.DataFrame([[lag_1, lag_2, lag_3, rolling_mean]], columns=FEATURES)
        pred = float(model.predict(x)[0])      # Pipeline scales then predicts
        out.append((yr, pred))
        yields.append(pred)                    # feed back for the next step
    return out
