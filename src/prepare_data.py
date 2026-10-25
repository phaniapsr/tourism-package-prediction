"""Stage 2 - data preparation.

Loads the raw CSV from the repository data folder, cleans it, and writes a
stratified train/test split to data/processed/. The next pipeline job receives
that folder as a workflow artifact.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (DROP_COLUMNS, PROCESSED_DIR, RANDOM_STATE, RAW_DATA, TARGET,
                    TEST_SIZE)

# Spelling variants found in the raw data, mapped to the canonical value.
GENDER_FIXES = {"Fe Male": "Female"}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    text_cols = df.select_dtypes(exclude="number").columns
    for col in text_cols:
        df[col] = df[col].str.strip()
    df["Gender"] = df["Gender"].replace(GENDER_FIXES)

    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_and_save(df: pd.DataFrame, out_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=PROCESSED_DIR, type=Path)
    args = parser.parse_args()

    raw = pd.read_csv(RAW_DATA)
    print(f"Loaded {RAW_DATA.name}: {raw.shape[0]} rows x {raw.shape[1]} columns")

    df = clean(raw)
    print(f"After cleaning: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Dropped columns: {[c for c in DROP_COLUMNS if c in raw.columns]}")
    print(f"Gender values: {sorted(df['Gender'].unique())}")

    train, test = split_and_save(df, args.out_dir)
    print(f"\nSaved to {args.out_dir}")
    for name, part in (("train", train), ("test", test)):
        print(f"  {name}.csv: {len(part)} rows, {part[TARGET].mean():.3f} positive rate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
