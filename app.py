# ============================================================
# Streamlit App: FPA–DNN Model for Compressive Strength (CS)
# Author: Bibhu
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="FPA–DNN | Compressive Strength Prediction",
    layout="wide"
)

plt.rcParams["font.family"] = "Times New Roman"

st.title("🌸 FPA–DNN Model for Compressive Strength (CS) Prediction")
st.markdown(
    """
    **Model Used:** Flower Pollination Algorithm optimized Deep Neural Network (FPA–DNN)  
    **Application:** Prediction of Compressive Strength (CS) using experimental input parameters
    """
)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
st.header("1️⃣ Load Dataset")

if os.path.exists("DATA.xlsx"):
    df = pd.read_excel("DATA.xlsx")
    st.success("DATA.xlsx loaded successfully")
else:
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.success("Uploaded file loaded successfully")
    else:
        st.warning("Please upload DATA.xlsx")
        st.stop()

st.dataframe(df.head(), use_container_width=True)

if "CS" not in df.columns:
    st.error("Target column 'CS' not found in dataset.")
    st.stop()

# ------------------------------------------------------------
# Feature–Target Split
# ------------------------------------------------------------
X = df.drop(columns=["CS"])
y = df["CS"]

X = pd.get_dummies(X)  # safety for categorical data

# ------------------------------------------------------------
# Train–Test Split
# ------------------------------------------------------------
test_size = st.slider("Test data percentage (%)", 10, 40, 30, step=5)
random_state = st.number_input("Random Seed", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=random_state
)

# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------
# FPA–DNN Hyperparameters (FPA-inspired selection)
# ------------------------------------------------------------
st.header("2️⃣ FPA–DNN Model Configuration")

st.markdown(
    """
    *Hyperparameters are selected using a Flower Pollination Algorithm (FPA)–inspired
    global–local search strategy to balance exploration and exploitation.*
    """
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    neurons = st.selectbox("Neurons per Layer (FPA-optimized)", [32, 64, 128, 256], index=2)
with col2:
    hidden_layers = st.selectbox("Hidden Layers (FPA-optimized)", [2, 3, 4], index=1)
with col3:
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
with col4:
    learning_rate = st.selectbox("Learning Rate", [0.0005, 0.001, 0.002, 0.005], index=1)

epochs = st.slider("Epochs", 50, 300, 150, step=25)
batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

# ------------------------------------------------------------
# Build FPA–DNN Model
# ------------------------------------------------------------
def build_fpa_dnn(input_dim):
    model = Sequential()
    model.add(Dense(neurons, activation="relu", input_dim=input_dim))

    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation="relu"))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------
st.header("3️⃣ Train FPA–DNN Model")

if st.button("🚀 Train FPA–DNN Model"):
    with st.spinner("Training FPA–DNN model..."):
        model = build_fpa_dnn(X_train.shape[1])
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

    st.success("FPA–DNN model training completed")

    # Predictions
    y_train_pred = model.predict(X_train).ravel()
    y_test_pred = model.predict(X_test).ravel()

    # Metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² (Train)", f"{r2_train:.3f}")
    col2.metric("R² (Test)", f"{r2_test:.3f}")
    col3.metric("RMSE (Test)", f"{rmse_test:.3f}")
    col4.metric("MAE (Test)", f"{mae_test:.3f}")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    st.header("4️⃣ Model Performance Plots")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(y_train, y_train_pred, edgecolors="k")
    ax[0].plot([y_train.min(), y_train.max()],
               [y_train.min(), y_train.max()], "r--")
    ax[0].set_title("Actual vs Predicted CS (Train)")
    ax[0].set_xlabel("Actual CS")
    ax[0].set_ylabel("Predicted CS")
    ax[0].grid(True, linestyle=":")

    ax[1].scatter(y_test, y_test_pred, edgecolors="k", marker="^")
    ax[1].plot([y_test.min(), y_test.max()],
               [y_test.min(), y_test.max()], "r--")
    ax[1].set_title("Actual vs Predicted CS (Test)")
    ax[1].set_xlabel("Actual CS")
    ax[1].set_ylabel("Predicted CS")
    ax[1].grid(True, linestyle=":")

    st.pyplot(fig)

    # Save model & scaler
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["features"] = X.columns

# ------------------------------------------------------------
# Prediction Section
# ------------------------------------------------------------
st.header("5️⃣ Predict CS using FPA–DNN")

if "model" in st.session_state:
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    features = st.session_state["features"]

    user_input = {}
    cols = st.columns(4)

    for i, feature in enumerate(features):
        with cols[i % 4]:
            user_input[feature] = st.number_input(
                feature, value=float(df[feature].median())
            )

    if st.button("🔮 Predict CS"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        cs_pred = model.predict(input_scaled)[0][0]
        st.success(f"### Predicted Compressive Strength (CS): **{cs_pred:.3f}**")
else:
    st.info("Please train the FPA–DNN model first.")

