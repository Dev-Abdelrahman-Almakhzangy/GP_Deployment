"""Project 3 — Wheat yield prediction (LightGBM).

Feature names are read straight from the booster, so the input form in the UI
builds itself — no manual feature list to keep in sync.
"""
from pathlib import Path
import numpy as np
import lightgbm as lgb

MODEL_PATH = Path(__file__).parent.parent / "models" / "lightgbm_yield.txt"

# How each feature should be entered in the UI.
#   "toggle"  -> Yes/No switch (binary 0/1 features)
#   "slider"  -> min/max/default/step (continuous features)
# Anything NOT listed here falls back to a plain number box, so unknown
# features won't break the app — just add/adjust entries as needed.
FEATURE_SPEC = {
    "Rainfall_mm":             {"kind": "slider", "min": 0.0, "max": 1500.0, "default": 500.0, "step": 1.0},
    "Temperature_Celsius":     {"kind": "slider", "min": -5.0, "max": 50.0,  "default": 22.0,  "step": 0.1},
    "Fertilizer_Used":         {"kind": "toggle", "default": True},
    "Irrigation_Used":         {"kind": "toggle", "default": True},
    "Weather_Condition_Rainy": {"kind": "toggle", "default": False},
}


def load_booster() -> lgb.Booster:
    # If you saved with joblib instead, swap for: joblib.load(...).booster_
    return lgb.Booster(model_file=str(MODEL_PATH))


def feature_names(booster: lgb.Booster):
    return booster.feature_name()


def predict(booster: lgb.Booster, feature_dict: dict) -> float:
    names = booster.feature_name()
    x = np.array([[feature_dict[n] for n in names]], dtype=np.float64)
    return float(booster.predict(x)[0])


def shap_contributions(booster: lgb.Booster, feature_dict: dict):
    """Return (names, shap_values, base_value) for a single prediction.

    Uses LightGBM's built-in pred_contrib (last column = base value) — avoids a
    hard dependency on the shap package, but shap.TreeExplainer works too.
    """
    names = booster.feature_name()
    x = np.array([[feature_dict[n] for n in names]], dtype=np.float64)
    contrib = booster.predict(x, pred_contrib=True)[0]   # len = n_features + 1
    return names, contrib[:-1], contrib[-1]


def global_importance(booster: lgb.Booster):
    """Overall feature importance (gain) across the whole model — not tied to
    one prediction. Returns (names, gains) so the UI can show 'in general,
    which features matter most'."""
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    return names, gains
