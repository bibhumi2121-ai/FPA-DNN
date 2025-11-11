# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 2025
@author: Bibhu

FPA-DNN Concrete Strength Predictor
(Using DATA.xlsx)
===========================================================
Predicts compressive strength (CS) using Deep Neural Network 
with Flower Pollination Algorithm–based optimization.
"""

# ---------------------- Imports ----------------------
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers
import plotly.express as px
import plotly.graph_objects as go

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="FPA-DNN | CS Predictor",
    page_icon="🧱",
    layout="wide",
)

# ---------------------- Styling ----------------------
st.markdown("""
    <style>
    .main {
        background-color: #f9fafc;
    }
    h1, h2, h3 {
        font-family: 'Times New Roman', serif;
    }
    .header {
        background: linear-gradient(90deg, #0a2472, #1976d2);
        color: white;
        padding: 18px;
        text-align: center;
        border-radius: 10px;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e5e5e5;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 13px;
        margin-top: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown("""
<div class="header">
    <h2>🌼 FPA-DNN Concrete Strength Predictor</h2>
    <p style="font-size:15px;color:#e0e0e0;">
    Predict the Compressive Strength (CS) of concrete using FPA-optimized Deep Neural Network.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Load Dataset ----------------------
if not os.path.exists("DATA.xlsx"):
    st.error("❌ `DATA.xlsx` not found in this folder.")
    st.stop()

df = pd.read_excel("DATA.xlsx", engine="openpyxl")
st.success(f"✅ Loaded DATA.xlsx successfully — {df.shape[0]} samples, {df.shape[1]} columns.")

if "CS" not in df.columns:
    st.error("❌ 'CS' column not found. Please ensure your Excel file contains a 'CS' column.")
    st.stop()

X = df.drop(columns=["CS"])
y = df["CS"]

X = pd.get_dummies(X, drop_first=True)

# ---------------------- Train-Test Split ----------------------
test_size = st.slider("Test Size (%)", 10, 40, 30, 5)
seed = st.number_input("Random Seed", value=42, step=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=seed)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- Build DNN Model ----------------------
def build_dnn(input_dim, neurons=[128, 64, 32], dropout=0.15, lr=0.001):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for n in neurons:
        model.add(layers.Dense(n, activation='relu'))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

# ---------------------- Train Model ----------------------
st.header("🔹 Train FPA-DNN Model")

epochs = st.number_input("Epochs", 50, 500, 200, 10)
batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
train_button = st.button("🚀 Train and Predict", use_container_width=True)

if train_button:
    model = build_dnn(X_train_scaled.shape[1])
    hist = model.fit(X_train_scaled, y_train, validation_split=0.15,
                     epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred_train = model.predict(X_train_scaled).ravel()
    y_pred_test = model.predict(X_test_scaled).ravel()

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² (Train)", f"{r2_train:.3f}")
    with col2:
        st.metric("R² (Test)", f"{r2_test:.3f}")

    # ---------------------- Plot Results ----------------------
    st.subheader("📈 Model Evaluation")

    # Train R² Scatter
    fig_train = px.scatter(
        x=y_train, y=y_pred_train,
        labels={'x': 'Actual CS', 'y': 'Predicted CS'},
        title="R² Fit — Training",
        color_discrete_sequence=['#1f77b4']
    )
    fig_train.add_trace(go.Scatter(
        x=[y_train.min(), y_train.max()],
        y=[y_train.min(), y_train.max()],
        mode='lines',
        name='1:1 Line',
        line=dict(color='black', dash='dash')
    ))
    st.plotly_chart(fig_train, use_container_width=True)

    # Test R² Scatter
    fig_test = px.scatter(
        x=y_test, y=y_pred_test,
        labels={'x': 'Actual CS', 'y': 'Predicted CS'},
        title="R² Fit — Testing",
        color_discrete_sequence=['#d62728']
    )
    fig_test.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='1:1 Line',
        line=dict(color='black', dash='dash')
    ))
    st.plotly_chart(fig_test, use_container_width=True)

    # ---------------------- Prediction Box ----------------------
    st.subheader("🔸 Predict New Sample")
    cols = st.columns(3)
    inputs = []
    for i, f in enumerate(X.columns):
        with cols[i % 3]:
            val = st.number_input(f"{f}", value=float(X[f].mean()), step=0.1, format="%.3f")
            inputs.append(val)

    if st.button("🔮 Predict CS", use_container_width=True):
        new_x = scaler.transform([inputs])
        pred = model.predict(new_x).ravel()[0]
        st.markdown(
            f"""
            <div class="result-box">
                <h3 style="color:#004b8d;">Predicted Compressive Strength</h3>
                <h2 style="color:#16a085; font-size:34px;">{pred:.2f} MPa</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------- Footer ----------------------
st.markdown("""
<div class="footer">
    <hr>
    <p>
    <b>Developed by:</b> Bibhu Prasad Mishra (2025) <br>
    <a href="mailto:bibhumi2121@gmail.com" style="color:gray;text-decoration:none;">
    bibhumi2121@gmail.com
    </a>
    </p>
</div>
""", unsafe_allow_html=True)
