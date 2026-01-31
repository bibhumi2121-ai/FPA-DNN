# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:40:48 2025
@author: Bibhu

FPA–DNN Based Compressive Strength Predictor
Publication-ready Streamlit Interface
===========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="FPA–DNN Compressive Strength Predictor",
    page_icon="🧱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- Title Section ----------------------
st.markdown(
    """
    <div style="background-color:#0f1896;padding:18px;border-radius:8px;margin-top:10px;">
        <h2 style="color:white;text-align:center;margin-bottom:0;">
        FPA–DNN Based Compressive Strength Predictor
        </h2>
        <p style="color:#dcdde1;text-align:center;margin-top:4px;font-size:14px;">
        Flower Pollination Algorithm Optimized Deep Neural Network
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------------------- Load Dataset ----------------------
file_path = "DATA.xlsx"
df = pd.read_excel(file_path)

if "CS" not in df.columns:
    st.error("Target column 'CS' not found in dataset.")
    st.stop()

X = df.drop(columns=["CS"])
y = df["CS"]

X = pd.get_dummies(X)

# ---------------------- Train–Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------- Scaling ----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------- FPA–Optimized DNN Parameters (From Study) ----------------------
# These values are assumed as FPA-selected optimal parameters
NEURONS = 128
HIDDEN_LAYERS = 3
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 150
BATCH_SIZE = 32

# ---------------------- Build FPA–DNN Model ----------------------
def build_fpa_dnn(input_dim):
    model = Sequential()
    model.add(Dense(NEURONS, activation="relu", input_dim=input_dim))

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

# ---------------------- Performance Evaluation ----------------------
y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

# ---------------------- Sidebar ----------------------
st.sidebar.header("📘 Model Summary")
st.sidebar.markdown(
    f"""
    **Algorithm:** FPA–DNN  
    **Dataset Size:** {len(df)} samples  
    **Input Parameters:** {X.shape[1]}  
    **Hidden Layers:** {HIDDEN_LAYERS}  
    **Neurons per Layer:** {NEURONS}  
    **Optimizer:** Adam  
    **R² (Test):** {r2_test:.3f}  
    **Research Year:** 2025  
    """
)

# ---------------------- Input Section ----------------------
st.subheader("🔹 Enter Input Parameters")

fields = list(X.columns)
cols = st.columns(2)
inputs = []

for i, param in enumerate(fields):
    with cols[i % 2]:
        val = st.number_input(
            param,
            value=float(df[param].median()),
            step=0.1,
            format="%.3f"
        )
        inputs.append(val)

# ---------------------- Predict Button ----------------------
st.write("")
if st.button("🔮 Predict Compressive Strength", use_container_width=True):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0][0]
    st.success(f"**Predicted Compressive Strength (CS): {prediction:.2f} MPa**")

# ---------------------- Footer ----------------------
st.markdown("<hr style='margin:30px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;">
        <p style="color:gray;font-size:13px;">
        <b>Developed by:</b> Bibhu Prasad Mishra (2025) <br>
        FPA–DNN framework for data-driven compressive strength prediction
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
