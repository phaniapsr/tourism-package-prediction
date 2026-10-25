"""Streamlit app: will this customer buy the Wellness Tourism Package?

Loads the model the MLOps pipeline committed to models/, collects the customer
and pitch details into a dataframe, and shows the predicted purchase probability.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
METRICS_PATH = ROOT / "models" / "metrics.json"
THRESHOLD = 0.5  # the cut-off the model was evaluated at

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳",
                   layout="wide")


@st.cache_resource
def load_artifacts():
    # joblib unpickles code, so only load a model we trust. This file is written by
    # our own GitHub Actions pipeline and committed to this repo; never point
    # MODEL_PATH at a file from an external or user-supplied source.
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text())
    return model, schema, metrics


def yes_no(label: str, default: str = "No") -> int:
    return 1 if st.selectbox(label, ["No", "Yes"], index=["No", "Yes"].index(default)) == "Yes" else 0


try:
    model, schema, metrics = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in models/. Run the training pipeline first.")
    st.stop()

median = {c: v["median"] for c, v in schema["numeric"].items()}
choices = schema["categorical"]

st.title("🧳 Wellness Tourism Package Predictor")
st.write("Enter a customer's details to estimate how likely they are to buy the "
         "**Wellness Tourism Package**, so the sales team can prioritise who to contact.")

with st.sidebar:
    st.header("About the model")
    best = metrics["selected_model"]
    st.write(f"**Algorithm:** {best}")
    test = metrics["models"][best]["test"]
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        st.metric(name.replace("_", " ").title(), f"{test[name]:.3f}")
    st.caption("Scores on a held-out test set. Trained and committed automatically "
               "by the GitHub Actions pipeline.")

with st.form("customer"):
    left, middle, right = st.columns(3)

    with left:
        st.subheader("Customer")
        age = st.number_input("Age", 18, 100, int(median["Age"]))
        gender = st.selectbox("Gender", choices["Gender"])
        marital = st.selectbox("Marital status", choices["MaritalStatus"])
        occupation = st.selectbox("Occupation", choices["Occupation"])
        designation = st.selectbox("Designation", choices["Designation"])
        income = st.number_input("Monthly income", 1000, 500000,
                                 int(median["MonthlyIncome"]), step=500)
        city_tier = st.selectbox("City tier", [1, 2, 3], index=0)

    with middle:
        st.subheader("Trip preferences")
        persons = st.number_input("Number of people visiting", 1, 10,
                                  int(median["NumberOfPersonVisiting"]))
        children = st.number_input("Children under 5 visiting", 0, 5,
                                   int(median["NumberOfChildrenVisiting"]))
        trips = st.number_input("Trips per year", 0, 30, int(median["NumberOfTrips"]))
        star = st.selectbox("Preferred property star", [3, 4, 5], index=0)
        passport = yes_no("Holds a valid passport")
        own_car = yes_no("Owns a car", default="Yes")

    with right:
        st.subheader("Sales interaction")
        contact = st.selectbox("Type of contact", choices["TypeofContact"])
        product = st.selectbox("Product pitched", choices["ProductPitched"])
        duration = st.number_input("Duration of pitch (minutes)", 1, 200,
                                   int(median["DurationOfPitch"]))
        followups = st.number_input("Number of follow-ups", 0, 10,
                                    int(median["NumberOfFollowups"]))
        satisfaction = st.slider("Pitch satisfaction score", 1, 5,
                                 int(median["PitchSatisfactionScore"]))

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    # Column names must match the training data exactly.
    inputs = pd.DataFrame([{
        "Age": age, "CityTier": city_tier, "DurationOfPitch": duration,
        "NumberOfPersonVisiting": persons, "NumberOfFollowups": followups,
        "PreferredPropertyStar": star, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": satisfaction, "OwnCar": own_car,
        "NumberOfChildrenVisiting": children, "MonthlyIncome": income,
        "TypeofContact": contact, "Occupation": occupation, "Gender": gender,
        "ProductPitched": product, "MaritalStatus": marital, "Designation": designation,
    }])

    st.subheader("Input summary")
    st.dataframe(inputs, hide_index=True, width="stretch")

    probability = float(model.predict_proba(inputs)[0, 1])
    st.subheader("Prediction")
    observed = {tuple(pair) for pair in schema["observed_designation_product"]}
    if (designation, product) not in observed:
        st.warning(f"No customer with designation '{designation}' was ever pitched "
                   f"'{product}' in the training data, so treat this score with caution.")
    st.metric("Purchase probability", f"{probability:.1%}")
    st.progress(probability)
    if probability >= THRESHOLD:
        st.success("Likely to purchase the Wellness Tourism Package. Worth prioritising.")
    else:
        st.warning("Unlikely to purchase the Wellness Tourism Package.")
