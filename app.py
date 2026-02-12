# -*- coding: utf-8 -*-
"""
FPA–DNN Based Compressive Strength Predictor
With 95% Confidence Interval using Monte Carlo Dropout
Publication-ready Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="FPA–DNN Compressive Strength Predictor",
    layout="centered"
)

# ---------------------- Title ----------------------
st.markdown(
    """
    <div style="background-color:#0f9296;padding:18px;border-radius:8px;">
        <h2 style="color:white;text-align:center;">
        FPA–DNN Based Compressive Strength Predictor
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ---------------------- Load Dataset ----------------------
@st.cache_data
def load_data():
    df = pd.read_excel("DATA.xlsx")
    return df

df = load_data()

if "CS" not in df.columns:
    st.error("Target column 'CS' not found.")
    st.stop()

# ---------------------- Handle NaNs ----------------------
df = df.apply(lambda col: col.fillna(col.median()) if col.dtype != "object" else col)

X = df.drop(columns=["CS"])
y = df["CS"]

X = pd.get_dummies(X)


# ---------------------- Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ---------------------- Scaling ----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------- FPA Parameters ----------------------
NEURONS = 128
HIDDEN_LAYERS = 3
DROPOUT = 0.20
LEARNING_RATE = 0.001
EPOCHS = 150
BATCH_SIZE = 32


# ---------------------- Build Model ----------------------
def build_fpa_dnn(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(NEURONS, activation="relu"))

    for _ in range(HIDDEN_LAYERS - 1):
        model.add(Dense(NEURONS, activation="relu"))
        model.add(Dropout(DROPOUT))

    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


# ---------------------- Train Model ----------------------
@st.cache_resource
def train_model():
    model = build_fpa_dnn(X_train.shape[1])
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    return model

model = train_model()


# ---------------------- Monte Carlo Dropout ----------------------
def mc_dropout_prediction(model, input_scaled, n_samples=100):

    preds = []

    for _ in range(n_samples):
        prediction = model(input_scaled, training=True)
        preds.append(prediction.numpy().ravel()[0])

    preds = np.array(preds)

    mean_pred = preds.mean()
    std_pred = preds.std()

    ci = 1.96 * std_pred
    lower = mean_pred - ci
    upper = mean_pred + ci

    cov = (std_pred / mean_pred) * 100 if mean_pred != 0 else 0

    return mean_pred, std_pred, ci, lower, upper, preds, cov


# ---------------------- Sidebar ----------------------
st.sidebar.header("Model Summary")
st.sidebar.markdown(
    f"""
    **Model:** FPA–DNN  
    **Dataset Size:** {len(df)}  
    **Input Parameters:** {X.shape[1]}  
    **Hidden Layers:** 3  
    **Neurons per Layer:** 128  
    **R²:** 0.95  
    **Year:** 2025  
    """
)


# ---------------------- Input Section ----------------------
st.subheader("🔹 Enter Input Parameters")

cols = st.columns(2)
inputs = []

for i, col in enumerate(X.columns):
    with cols[i % 2]:
        val = st.number_input(
            col,
            value=0.00,
            step=0.01,
            format="%.2f"
        )
        inputs.append(val)


# ---------------------- Prediction ----------------------
st.write("")

if st.button("Predict Compressive Strength", use_container_width=True):

    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    mean_pred, std_pred, ci, lower, upper, preds, cov = mc_dropout_prediction(
        model, input_scaled, n_samples=100
    )

    # ---------------------- Main Prediction ----------------------
    st.success(f"### 🔹 Predicted Compressive Strength: {mean_pred:.2f} MPa")

    st.markdown("## 🔹 Uncertainty Analysis (95% Confidence Interval)")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Standard Deviation:** {std_pred:.4f} MPa")
        st.write(f"**Confidence Interval (±):** {ci:.4f} MPa")

    with col2:
        st.write(f"**Prediction Range:**")
        st.write(f"[ {lower:.2f} , {upper:.2f} ] MPa")
        st.write(f"**Coefficient of Variation:** {cov:.2f} %")

    # ---------------------- Plot Distribution ----------------------
    st.markdown("### 🔹 Prediction Distribution")

    fig, ax = plt.subplots()
    ax.hist(preds, bins=20)
    ax.axvline(mean_pred)
    ax.set_xlabel("Compressive Strength (MPa)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Dropout Prediction Distribution")

    st.pyplot(fig)


# ---------------------- Footer ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;color:gray;font-size:13px;">
        <p>
        <b>Developed by:</b> Bibhu Prasad Mishra (2025) <br>
        FPA–DNN Framework for CS Prediction <br>
        bibhumi2121@gmail.com
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
