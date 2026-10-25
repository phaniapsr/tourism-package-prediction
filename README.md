# Wellness Tourism Package Prediction

An MLOps pipeline for "Visit with Us" that predicts whether a customer will buy the
new Wellness Tourism Package, so the sales team can prioritise who to contact.
GitHub Actions validates the data, trains and tracks the model, commits it to
`main`, and the Streamlit app serves it.

## Repository layout

```
.
├── .github/workflows/pipeline.yml   # the CI/CD pipeline
├── data/tourism.csv                 # raw dataset (4,128 rows)
├── src/
│   ├── common.py                    # paths and column definitions
│   ├── validate_data.py             # stage 1: register / validate the dataset
│   ├── prepare_data.py              # stage 2: clean + train/test split
│   └── train_model.py               # stage 3: tune, track with MLflow, save best model
├── models/                          # committed by the pipeline
│   ├── best_model.joblib            #   preprocessing + best estimator
│   ├── metrics.json                 #   CV and test metrics for every model
│   └── feature_schema.json          #   input choices the app uses
├── app.py                           # Streamlit app
├── tests/smoke_test_app.py          # loads the committed model and predicts
├── notebook/                        # executed notebook (submission)
├── requirements.txt                 # deployment dependencies (Streamlit Cloud)
└── requirements-train.txt           # pipeline dependencies (adds MLflow)
```

## Pipeline

`pipeline.yml` runs four chained jobs on every push to `main` that touches `data/`,
`src/`, the requirements files or the workflow (or manually via *Run workflow*):

| Job | What it does |
|---|---|
| 1. Register dataset | `validate_data.py` checks the expected columns and target, prints a summary |
| 2. Prepare data | `prepare_data.py` cleans the data, saves a stratified 80/20 split, uploads it as the `prepared-data` artifact |
| 3. Build and track model | downloads the artifact, tunes six model families, logs every trial to MLflow (uploaded as the `mlflow-runs` artifact), then commits `models/` to `main` |
| 4. Verify app | fresh checkout of `main`, installs only `requirements.txt`, runs the app smoke test |

The model commit is made by `github-actions[bot]` with `GITHUB_TOKEN`, so it does not
re-trigger the workflow. If `main` is protected and blocks pushes from Actions, allow
the bot or push the model to a branch instead.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements-train.txt

python src/validate_data.py
python src/prepare_data.py
python src/train_model.py            # writes models/ and mlruns/
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse experiments

streamlit run app.py
```

## Deploy the app (Streamlit Community Cloud)

1. Push this repository to GitHub (public).
2. At <https://share.streamlit.io> choose **Create app**, pick the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings** select **Python 3.12** so it matches the version the model was trained with.
4. Deploy. Every model commit from the pipeline redeploys the app automatically.

## Model

Six families are tuned with 5-fold randomized search on F1: Decision Tree, Bagging,
Random Forest, AdaBoost, Gradient Boosting and XGBoost. The winner is chosen on
cross-validated F1 over the training data only; the test set is used for reporting.
See `models/metrics.json` for the full comparison.
