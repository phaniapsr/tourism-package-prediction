"""Smoke test for the deployed app: loads the committed model and predicts.

Run from the repo root: python tests/smoke_test_app.py
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def main() -> int:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, f"app failed to load: {at.exception}"

    at.button[0].click().run()
    assert not at.exception, f"prediction failed: {at.exception}"

    probability = next(m.value for m in at.metric if m.label == "Purchase probability")
    value = float(probability.rstrip("%")) / 100
    assert 0.0 <= value <= 1.0, f"probability out of range: {probability}"
    print(f"App smoke test passed - default customer scores {probability}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
