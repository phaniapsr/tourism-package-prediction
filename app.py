"""Streamlit app: will this customer buy the Wellness Tourism Package?

Loads the model the MLOps pipeline committed to models/. Score one customer from a
form (Single Prediction) or a whole customer list from a CSV (Batch Prediction).
"""
import pandas as pd
import streamlit as st

from inference import (THRESHOLD, label, load_artifacts, prepare_batch,
                       required_columns, unseen_pairs)

MAX_BATCH_ROWS = 20_000

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳",
                   layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def yes_no(text: str, default: str = "No") -> int:
    return 1 if st.selectbox(text, ["No", "Yes"], index=["No", "Yes"].index(default)) == "Yes" else 0


def pick(text: str, column: str) -> str:
    """Dropdown over a column's known values, starting on its most common one."""
    return st.selectbox(text, choices[column], index=choices[column].index(default[column]))


try:
    model, schema, metrics = get_artifacts()
except FileNotFoundError:
    st.error("Model files not found in models/. Run the training pipeline first.")
    st.stop()

median = {c: v["median"] for c, v in schema["numeric"].items()}
choices = schema["categorical"]
# Most common training value per column; older schema files without it fall back to the first choice.
common = schema.get("categorical_mode", {})
default = {c: common.get(c, choices[c][0]) for c in choices}
best = metrics["selected_model"]
test = metrics["models"][best]["test"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧳 Visit with Us")
    st.subheader("Customer Targeting System")
    st.write(f"Predicts how likely a customer is to buy the **Wellness Tourism Package** "
             f"using a trained **{best}** model, so the sales team can prioritise who to contact.")
    st.divider()
    st.subheader("Model scores")
    for name, text in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1"), ("roc_auc", "ROC-AUC")):
        st.metric(text, f"{test[name]:.3f}")
    st.caption("Scores on a held-out test set. The model is trained and committed "
               "automatically by the GitHub Actions pipeline.")
    st.divider()
    st.write("Built with **Streamlit**")

# ---------------------------------------------------------------- main
st.title("🧳 Wellness Tourism Package Predictor")
st.write("Estimate whether a customer will buy the Wellness Tourism Package. "
         "Use **Single Prediction** for one customer or **Batch Prediction** for a CSV file.")

single_tab, batch_tab = st.tabs(["📊 Single Prediction", "📁 Batch Prediction"])

# ---- single prediction
with single_tab:
    st.header("Enter customer details")
    customer, trip, sales, summary = st.columns([1, 1, 1, 0.8], gap="large")

    with customer:
        st.subheader("Customer")
        age = st.number_input("Age", 18, 100, int(median["Age"]))
        gender = pick("Gender", "Gender")
        marital = pick("Marital status", "MaritalStatus")
        occupation = pick("Occupation", "Occupation")
        designation = pick("Designation", "Designation")
        income = st.number_input("Monthly income", 1000, 500000,
                                 int(median["MonthlyIncome"]), step=500)
        city_tier = st.selectbox("City tier", [1, 2, 3], index=0)

    with trip:
        st.subheader("Trip preferences")
        persons = st.number_input("Number of people visiting", 1, 10,
                                  int(median["NumberOfPersonVisiting"]))
        children = st.number_input("Children under 5 visiting", 0, 5,
                                   int(median["NumberOfChildrenVisiting"]))
        trips = st.number_input("Trips per year", 0, 30, int(median["NumberOfTrips"]))
        star = st.selectbox("Preferred property star", [3, 4, 5], index=0)
        passport = yes_no("Holds a valid passport")
        own_car = yes_no("Owns a car", default="Yes")

    with sales:
        st.subheader("Sales interaction")
        contact = pick("Type of contact", "TypeofContact")
        product = pick("Product pitched", "ProductPitched")
        duration = st.number_input("Duration of pitch (minutes)", 1, 200,
                                   int(median["DurationOfPitch"]))
        followups = st.number_input("Number of follow-ups", 0, 10,
                                    int(median["NumberOfFollowups"]))
        satisfaction = st.slider("Pitch satisfaction score", 1, 5,
                                 int(median["PitchSatisfactionScore"]))

    with summary:
        st.subheader("Summary")
        st.metric("Age", age)
        st.metric("Monthly income", f"{income:,}")
        st.metric("Product pitched", product)
        st.metric("Travelling party", persons)
        st.metric("Passport", "Yes" if passport else "No")

    if st.button("Predict", type="primary"):
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
        if unseen_pairs(inputs, schema).iloc[0]:
            st.warning(f"No customer with designation '{designation}' was ever pitched "
                       f"'{product}' in the training data, so treat this score with caution.")
        st.metric("Purchase probability", f"{probability:.1%}")
        st.progress(probability)
        if probability >= THRESHOLD:
            st.success("Likely to purchase the Wellness Tourism Package. Worth prioritising.")
        else:
            st.info("Unlikely to purchase the Wellness Tourism Package.")

# ---- batch prediction
with batch_tab:
    st.header("Score a customer list")
    columns = required_columns(schema)
    st.write("Upload a CSV with one row per customer. The file needs these columns; any extra "
             "columns (such as CustomerID) are kept in the output.")
    st.code(", ".join(columns), language="text", wrap_lines=True)

    template = pd.DataFrame([{**{c: int(median[c]) for c in schema["numeric"]},
                              **{c: default[c] for c in choices}}])[columns]
    st.download_button("Download a template CSV", template.to_csv(index=False).encode(),
                       file_name="customers_template.csv", mime="text/csv")

    upload = st.file_uploader("Upload customer CSV", type="csv")
    if upload is not None:
        try:
            data = pd.read_csv(upload)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as err:
            st.error(f"Could not read that file as a CSV: {err}")
            data = None

        if data is not None and data.empty:
            st.error("The file has no rows.")
        elif data is not None and len(data) > MAX_BATCH_ROWS:
            st.error(f"Please upload at most {MAX_BATCH_ROWS:,} rows at a time "
                     f"(this file has {len(data):,}).")
        elif data is not None:
            try:
                features, notes = prepare_batch(data, schema)
            except ValueError as err:
                st.error(str(err))
            else:
                for note in notes:
                    st.warning(note)
                probabilities = model.predict_proba(features)[:, 1]
                results = data.copy()
                results.insert(0, "Prediction", label(probabilities))
                results.insert(0, "PurchaseProbability", probabilities.round(4))   # leftmost: the signal to act on
                results = results.sort_values("PurchaseProbability", ascending=False)

                likely = int((probabilities >= THRESHOLD).sum())
                total, share, top = st.columns(3)
                total.metric("Customers scored", f"{len(results):,}")
                share.metric("Likely buyers", f"{likely:,}")
                top.metric("Share likely to buy", f"{likely / len(results):.1%}")

                st.subheader("Results (highest probability first)")
                st.dataframe(results, hide_index=True, width="stretch", column_config={
                    "PurchaseProbability": st.column_config.ProgressColumn(
                        "PurchaseProbability", min_value=0.0, max_value=1.0, format="%.2f")})
                st.download_button("Download predictions CSV",
                                   results.to_csv(index=False).encode(),
                                   file_name="predictions.csv", mime="text/csv")
