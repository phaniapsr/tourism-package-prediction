"""Model loading and input cleaning shared by the Streamlit app and the tests.

Kept free of Streamlit code so it can be tested without a browser.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
METRICS_PATH = ROOT / "models" / "metrics.json"
THRESHOLD = 0.5  # the cut-off the model was evaluated at

# Spelling variant present in the raw data; training cleans it, so uploads must too.
GENDER_FIXES = {"Fe Male": "Female"}


def load_artifacts():
    """Return (model, schema, metrics) from the files the pipeline committed to models/."""
    # joblib unpickles code, so only load a model we trust. This file is written by
    # our own GitHub Actions pipeline and committed to this repo; never point
    # MODEL_PATH at a file from an external or user-supplied source.
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text())
    return model, schema, metrics


def required_columns(schema: dict) -> list[str]:
    return list(schema["numeric"]) + list(schema["categorical"])


def unseen_pairs(df: pd.DataFrame, schema: dict) -> pd.Series:
    """True for rows whose Designation/ProductPitched pairing never occurs in training."""
    observed = {tuple(pair) for pair in schema["observed_designation_product"]}
    return pd.Series([(d, p) not in observed
                      for d, p in zip(df["Designation"], df["ProductPitched"])],
                     index=df.index)


def prepare_batch(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
    """Clean an uploaded table into model input.

    Returns the model-ready columns plus plain-language notes about anything the
    model will treat specially. Raises ValueError if required columns are missing.
    """
    numeric, categorical = list(schema["numeric"]), list(schema["categorical"])
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    X = df[numeric + categorical].copy()
    notes = []

    for col in categorical:
        X[col] = X[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    X["Gender"] = X["Gender"].replace(GENDER_FIXES)
    for col in categorical:
        unseen = X[col].notna() & ~X[col].isin(schema["categorical"][col])
        if unseen.any():
            examples = ", ".join(map(repr, sorted(X.loc[unseen, col].astype(str).unique())[:3]))
            notes.append(f"{int(unseen.sum())} row(s) have a value in {col} that the model has not seen "
                         f"({examples}); the model ignores it for those rows.")

    for col in numeric:
        values = pd.to_numeric(X[col], errors="coerce")
        bad = values.isna() & X[col].notna()
        if bad.any():
            notes.append(f"{int(bad.sum())} non-numeric value(s) in {col} were treated as missing.")
        X[col] = values

    blanks = int(X.isna().sum().sum())
    if blanks:
        notes.append(f"{blanks} missing value(s) were filled with the training median "
                     "(numbers) or most common value (categories).")

    odd = unseen_pairs(X, schema)
    if odd.any():
        notes.append(f"{int(odd.sum())} row(s) pair a Designation with a ProductPitched that never occurs "
                     "in the training data; treat those scores with caution.")

    return X.replace({pd.NA: np.nan}), notes


def label(probabilities) -> np.ndarray:
    return np.where(np.asarray(probabilities) >= THRESHOLD, "Likely buyer", "Unlikely buyer")
