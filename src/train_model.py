"""Stage 3 - model building with experiment tracking.

Loads the train/test split, tunes one model per family (Decision Tree, Bagging,
Random Forest, AdaBoost, Gradient Boosting, XGBoost) with randomized search,
and logs every tried parameter set and its metrics to MLflow. The winner is
picked on cross-validated F1 over the training data only, so the test set never
influences selection. The best pipeline is saved to models/ for the workflow to
commit and the Streamlit app to load.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import (CATEGORICAL_FEATURES, FEATURES, METRICS_PATH, MLFLOW_DIR,
                    MODEL_DIR, MODEL_PATH, NUMERIC_FEATURES, PROCESSED_DIR,
                    RANDOM_STATE, SCHEMA_PATH, TARGET)

EXPERIMENT = "tourism-package-prediction"
SELECTION_METRIC = "f1"


def build_preprocessor() -> ColumnTransformer:
    """Impute and encode inputs. Trees need no scaling."""
    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def candidate_models(pos_weight: float) -> dict:
    """Return {name: (estimator, param grid)}. Param keys target the 'model' step."""
    rs = RANDOM_STATE
    return {
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=rs),
            {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 2, 5, 10],
             "min_samples_split": [2, 5, 10, 20], "criterion": ["gini", "entropy"]},
        ),
        "Bagging": (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(class_weight="balanced", random_state=rs),
                random_state=rs),
            {"n_estimators": [50, 100, 200], "max_samples": [0.6, 0.8, 1.0],
             "max_features": [0.6, 0.8, 1.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=rs),
            {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20],
             "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5],
             "class_weight": ["balanced", "balanced_subsample"]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=rs),
            {"n_estimators": [50, 100, 200, 300],
             "learning_rate": [0.05, 0.1, 0.5, 1.0]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=rs),
            {"n_estimators": [100, 200, 300], "learning_rate": [0.03, 0.05, 0.1],
             "max_depth": [2, 3, 5], "subsample": [0.7, 0.85, 1.0]},
        ),
        "XGBoost": (
            XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1,
                          random_state=rs),
            {"n_estimators": [200, 400, 600], "max_depth": [3, 5, 7, 9],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0],
             "scale_pos_weight": [1.0, round(pos_weight, 2)]},
        ),
    }


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def strip_prefix(params: dict) -> dict:
    return {k.removeprefix("model__"): v for k, v in params.items()}


def log_trials(search: RandomizedSearchCV) -> None:
    """Log every parameter set the search tried as a child MLflow run."""
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"trial-{i:02d}", nested=True):
            mlflow.log_params(strip_prefix(params))
            mlflow.log_metric(f"cv_{SELECTION_METRIC}", res["mean_test_score"][i])
            mlflow.log_metric(f"cv_{SELECTION_METRIC}_std", res["std_test_score"][i])
            mlflow.log_metric("rank", int(res["rank_test_score"][i]))


def make_schema(train: pd.DataFrame) -> dict:
    """Input ranges and category choices for the app, taken from training data."""
    pairs = train[["Designation", "ProductPitched"]].drop_duplicates()
    return {
        "numeric": {c: {"min": float(train[c].min()), "max": float(train[c].max()),
                        "median": float(train[c].median())}
                    for c in NUMERIC_FEATURES},
        "categorical": {c: sorted(train[c].dropna().unique().tolist())
                        for c in CATEGORICAL_FEATURES},
        # Designation and ProductPitched move in lockstep in the data, so the
        # app can flag combinations the model has never seen.
        "observed_designation_product": sorted(pairs.values.tolist()),
    }


def to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def setup_mlflow() -> None:
    MLFLOW_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(MLFLOW_DIR / "artifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=PROCESSED_DIR, type=Path,
                        help="folder holding train.csv and test.csv")
    parser.add_argument("--n-iter", default=20, type=int,
                        help="parameter sets tried per model")
    parser.add_argument("--cv", default=5, type=int, help="cross-validation folds")
    parser.add_argument("--model-dir", default=MODEL_DIR, type=Path,
                        help="where to save the model and metrics (default: models/)")
    args = parser.parse_args()
    model_path = args.model_dir / MODEL_PATH.name
    metrics_path = args.model_dir / METRICS_PATH.name
    schema_path = args.model_dir / SCHEMA_PATH.name

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Loaded train {train.shape}, test {test.shape} from {args.data_dir}")

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    setup_mlflow()
    results, fitted = {}, {}

    with mlflow.start_run(run_name="training-pipeline"):
        mlflow.log_params({
            "train_rows": len(train), "test_rows": len(test), "cv_folds": args.cv,
            "n_iter_per_model": args.n_iter, "selection_metric": f"cv_{SELECTION_METRIC}",
            "random_state": RANDOM_STATE})

        for name, (estimator, grid) in candidate_models(pos_weight).items():
            pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
            space = {f"model__{k}": v for k, v in grid.items()}
            n_iter = min(args.n_iter, int(np.prod([len(v) for v in grid.values()])))
            search = RandomizedSearchCV(
                pipe, space, n_iter=n_iter, scoring=SELECTION_METRIC, cv=cv,
                n_jobs=-1, random_state=RANDOM_STATE, refit=True)
            search.fit(X_train, y_train)

            test_metrics = evaluate(search.best_estimator_, X_test, y_test)
            results[name] = {
                "best_params": strip_prefix(search.best_params_),
                f"cv_{SELECTION_METRIC}": search.best_score_,
                "test": test_metrics,
            }
            fitted[name] = search.best_estimator_

            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model_family", name)
                mlflow.log_params(strip_prefix(search.best_params_))
                mlflow.log_metric(f"cv_{SELECTION_METRIC}", search.best_score_)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                log_trials(search)

            print(f"{name:<17} cv_f1={search.best_score_:.4f}  "
                  f"test_f1={test_metrics['f1']:.4f}  test_recall={test_metrics['recall']:.4f}  "
                  f"test_auc={test_metrics['roc_auc']:.4f}")

        best_name = max(results, key=lambda n: results[n][f"cv_{SELECTION_METRIC}"])
        print(f"\nSelected model (best CV {SELECTION_METRIC}): {best_name}")
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics({f"selected_test_{k}": v
                            for k, v in results[best_name]["test"].items()})

        args.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[best_name], model_path)
        summary = {"selected_model": best_name,
                   "selection_metric": f"cv_{SELECTION_METRIC}", "models": results}
        metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                           default=to_jsonable) + "\n")
        schema_path.write_text(json.dumps(make_schema(train), indent=2,
                                          sort_keys=True) + "\n")
        for path in (model_path, metrics_path, schema_path):
            mlflow.log_artifact(str(path))

    print(f"Saved {model_path.name}, {metrics_path.name}, {schema_path.name} to {args.model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
