import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Concrete Strength AI", page_icon="🏗️", layout="wide")

st.title("🏗️ Concrete Compressive Strength Prediction and Mix Analysis")
st.write(
    "A student machine-learning demonstration that estimates concrete compressive "
    "strength from mix ingredients and curing age."
)

@st.cache_resource
def train_model():
    # Public UCI Machine Learning Repository dataset (ID 165)
    dataset = fetch_ucirepo(id=165)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    X.columns = [str(column).strip() for column in X.columns]
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, test_predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_predictions)),
        "R2": r2_score(y_test, test_predictions),
        "test_rows": len(y_test),
        "total_rows": len(X),
    }
    return model, X, metrics

try:
    with st.spinner("Loading public dataset and training the model..."):
        model, X_data, metrics = train_model()
except Exception as error:
    st.error("The dataset could not be loaded. Check your internet connection and try again.")
    st.caption(f"Technical detail: {error}")
    st.stop()

st.success(f"Model trained using {int(metrics['total_rows'])} dataset records.")

st.subheader("Model performance on a held-out test set")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{metrics['MAE']:.2f} MPa")
col2.metric("RMSE", f"{metrics['RMSE']:.2f} MPa")
col3.metric("R² score", f"{metrics['R2']:.3f}")
st.caption(
    f"Evaluation used {int(metrics['test_rows'])} test records not used for training. "
    "Results depend on the random split and dataset."
)

st.divider()
st.subheader("Enter concrete mix details")
st.write("Use ingredient quantities per cubic metre (kg/m³), and curing age in days.")

feature_names = list(X_data.columns)
user_values = {}

left, right = st.columns(2)
for index, feature in enumerate(feature_names):
    container = left if index % 2 == 0 else right
    with container:
        minimum = float(X_data[feature].min())
        maximum = float(X_data[feature].max())
        median = float(X_data[feature].median())

        if feature.strip().lower() == "age":
            user_values[feature] = st.slider(
                f"{feature} (days)",
                min_value=int(np.floor(minimum)),
                max_value=int(np.ceil(maximum)),
                value=int(round(median)),
                step=1,
                help="Age of the concrete specimen at testing."
            )
        else:
            user_values[feature] = st.slider(
                f"{feature} (kg/m³)",
                min_value=minimum,
                max_value=maximum,
                value=median,
                step=1.0,
                help="Ingredient mass per cubic metre. Use values within the dataset range."
            )

input_df = pd.DataFrame([user_values], columns=feature_names)

if st.button("Predict compressive strength", type="primary"):
    prediction = float(model.predict(input_df)[0])
    st.subheader("Predicted compressive strength")
    st.metric("Estimated strength", f"{prediction:.2f} MPa")
    st.info(
        "This is a machine-learning estimate based on historical dataset patterns, "
        "not a laboratory test result or a design approval."
    )

with st.expander("About this project and its limitations"):
    st.markdown(
        """
        - **Dataset:** UCI Concrete Compressive Strength dataset (ID 165).
        - **Model:** Random Forest regression.
        - **Evaluation:** 80/20 train-test split with a fixed random seed.
        - **MAE:** Mean absolute prediction error on the held-out test set; lower is better.
        - **RMSE:** Penalizes larger errors more strongly; lower is better.
        - **R²:** Measures performance relative to predicting the test-set mean; closer to 1 is generally better.
        - This is an educational prototype. Predictions are only as reliable as the dataset and model validation.
        - Do not use this app to approve concrete mixes or make structural safety decisions. Laboratory testing and applicable standards remain necessary.
        """
    )
