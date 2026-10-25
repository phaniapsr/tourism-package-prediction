"""Stage 1 - dataset registration.

Checks that data/tourism.csv has the columns in the data dictionary and a
well-formed target, then prints a summary. Exits non-zero on any failure so the
pipeline stops before bad data reaches training.
"""
import hashlib
import sys

import pandas as pd

from common import EXPECTED_COLUMNS, RAW_DATA, TARGET, TOLERATED_EXTRA_COLUMNS


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems; an empty list means the dataset is valid."""
    problems = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing expected columns: {missing}")

    unexpected = [c for c in df.columns
                  if c not in EXPECTED_COLUMNS and c not in TOLERATED_EXTRA_COLUMNS]
    if unexpected:
        problems.append(f"unexpected columns: {unexpected}")

    if df.empty:
        problems.append("dataset has no rows")

    if TARGET in df.columns:
        values = set(df[TARGET].dropna().unique())
        if not values <= {0, 1}:
            problems.append(f"{TARGET} must be 0/1, found {sorted(values)}")
        if df[TARGET].isna().any():
            problems.append(f"{TARGET} has {int(df[TARGET].isna().sum())} missing values")

    return problems


def print_summary(df: pd.DataFrame) -> None:
    digest = hashlib.sha256(RAW_DATA.read_bytes()).hexdigest()
    print(f"Dataset      : {RAW_DATA.name}")
    print(f"SHA-256      : {digest}")
    print(f"Rows x cols  : {df.shape[0]} x {df.shape[1]}")
    print(f"Duplicate rows (ignoring index column): "
          f"{df.drop(columns=TOLERATED_EXTRA_COLUMNS, errors='ignore').duplicated().sum()}")
    print("\nColumn dtypes and missing values:")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum()})
    print(info.to_string())
    print(f"\nTarget distribution ({TARGET}):")
    print(df[TARGET].value_counts(normalize=True).round(3).rename("share").to_string())
    print("\nNumeric summary:")
    print(df.describe().T.round(2).to_string())


def main() -> int:
    if not RAW_DATA.exists():
        print(f"ERROR: {RAW_DATA} not found")
        return 1

    df = pd.read_csv(RAW_DATA)
    problems = validate(df)
    if problems:
        print("Dataset validation FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("Dataset validation PASSED - all expected columns present.\n")
    print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
