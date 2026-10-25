"""Smoke test for the deployed app: loads the committed model and predicts.

Covers both tabs: a single prediction from the form, and a batch upload scored
from the repository dataset. Run from the repo root: python tests/smoke_test_app.py
"""
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
DATA = ROOT / "data" / "tourism.csv"


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_single_prediction():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, f"app failed to load: {at.exception}"
    assert len(at.tabs) == 2, f"expected 2 tabs, found {len(at.tabs)}"

    at.button[0].click().run()
    assert not at.exception, f"prediction failed: {at.exception}"
    probability = metric(at, "Purchase probability")
    assert 0.0 <= float(probability.rstrip("%")) / 100 <= 1.0, f"bad probability: {probability}"
    assert not any("training data" in w.value for w in at.warning), \
        "the default customer should be a combination seen in training"
    return probability


def test_batch_upload():
    raw = pd.read_csv(DATA)  # raw file: index column, CustomerID and a 'Fe Male' spelling variant
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.file_uploader[0].upload("customers.csv", raw.to_csv(index=False).encode(), "text/csv").run()
    assert not at.exception, f"batch scoring failed: {at.exception}"
    assert not at.error, f"unexpected error shown: {[e.value for e in at.error]}"

    scored = int(metric(at, "Customers scored").replace(",", ""))
    assert scored == len(raw), f"scored {scored} of {len(raw)} rows"
    results = at.dataframe[-1].value
    assert {"PurchaseProbability", "Prediction"} <= set(results.columns)
    assert results["PurchaseProbability"].between(0, 1).all()
    assert results["PurchaseProbability"].is_monotonic_decreasing, "results should be sorted high to low"

    broken = AppTest.from_file(APP, default_timeout=90).run()
    broken.file_uploader[0].upload(
        "bad.csv", raw.head(5).drop(columns=["Age"]).to_csv(index=False).encode(), "text/csv").run()
    assert not broken.exception and any("Age" in e.value for e in broken.error), \
        "a file missing a required column should show an error naming it"
    return scored


def main() -> int:
    probability = test_single_prediction()
    print(f"Single prediction OK - default customer scores {probability}")
    scored = test_batch_upload()
    print(f"Batch prediction OK - scored {scored:,} rows, and rejected a file with a missing column")
    print("App smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
