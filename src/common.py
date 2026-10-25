"""Shared paths and column definitions used by every pipeline stage and the app."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA = DATA_DIR / "tourism.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
MLFLOW_DIR = REPO_ROOT / "mlruns"

TARGET = "ProdTaken"

# Columns in the data dictionary. The raw CSV also carries a stale pandas index
# column ("Unnamed: 0") from an earlier export, which cleaning drops.
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
TOLERATED_EXTRA_COLUMNS = ["Unnamed: 0"]

# Identifiers carry no signal about whether a customer buys.
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
