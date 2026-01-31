# -*- coding: utf-8 -*-
"""
FPA–DNN Based Compressive Strength Predictor
NaN-safe | Warning-free | Publication-ready
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="FPA–DNN Compressive Strength Predictor",
    page_icon="🧱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- Title ----------------------
st.markdown(
    """
    <div style="background-color:#0f1896;padding:18px;border-radius:8px;">
        <h2 style="color:white;text-align:center;">
        FPA–DNN Based Compressive Strength Predictor
        </h2>
        <p style="color:#dcdde1;text-align:center;font-size:14px;">
        Flower Pollination Algorithm Optimized Deep Neural Network
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------------------- Load Dataset ----------------------
df = pd.read_excel("DATA.xlsx")

if "CS" not in df.columns:
    st.error("Target column 'CS' not found.")
    st.stop()

# ---------------------- Handle NaNs (CRITICAL FIX) ----------------------
# Median imputation (robust for concrete datasets)
df = df.apply(lambda col: col.fillna(col.median()) if col.dtype != "object" else col)

X = df.drop(columns=["CS"])
y = df["CS"]

# Encode categoricals (if any)
X = pd.get_dummies(X)

# Final NaN check (safety)
if X.isna().any().any() or y.isna().any():
    st.error("Dataset still contains NaN values after cleaning.")
    st.stop()

# ---------------------- Train–Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ---------------------- Scaling ----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------- FPA–Optimized Parameters (From Study) ----------------------
NEURONS = 128
HIDDEN_LAYERS = 3
DROPOUT = 0.20
LEARNING_RATE = 0.001
EPOCHS = 150
BATCH_SIZE = 32

# ---------------------- Build FPA–DNN Model ----------------------
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
model = build_fpa_dnn(X_train.shape[1])
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

# ---------------------- Evaluation ----------------------
y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)

# ---------------------- Sidebar ----------------------
st.sidebar.header("📘 Model Summary")
st.sidebar.markdown(
    f"""
    **Model:** FPA–DNN  
    **Samples:** {len(df)}  
    **Input Parameters:** {X.shape[1]}  
    **Hidden Layers:** {HIDDEN_LAYERS}  
    **Neurons/Layer:** {NEURONS}  
    **Optimizer:** Adam  
    **R² (Test):** {r2_test:.3f}  
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
            value=float(df[col].median()),
            format="%.3f"
        )
        inputs.append(val)

# ---------------------- Prediction ----------------------
st.write("")
if st.button("🔮 Predict Compressive Strength", use_container_width=True):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0][0]
    st.success(f"**Predicted Compressive Strength (CS): {pred:.2f} MPa**")

# ---------------------- Footer ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;font-size:13px;'>"
    "Developed by Bibhu Prasad Mishra (2025) | FPA–DNN Framework"
    "</p>",
    unsafe_allow_html=True
)
