# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 2025
Author: Bibhu Prasad Mishra

FPA-DNN Compressive Strength Predictor
=========================================
Predicts the Compressive Strength (CS) of concrete
using a Deep Neural Network optimized via the
Flower Pollination Algorithm (FPA).
"""

# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers
import random

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(
    page_title="FPA-DNN | Concrete Strength Predictor",
    page_icon="🌼",
    layout="centered",
)

# ---------------------- Custom Style ----------------------
st.markdown("""
    <style>
        body {
            background-color: #f8fafc;
        }
        .main-title {
            background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
        }
        .predict-box {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #dcdcdc;
            padding: 18px;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h2, h3 {
            font-family: 'Times New Roman', serif;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown("""
    <div class='main-title'>
        <h2>🌼 FPA-DNN Concrete Compressive Strength Predictor</h2>
        <p>Deep Neural Network optimized using the Flower Pollination Algorithm</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------- Dataset ----------------------
try:
    df = pd.read_excel("DATA.xlsx", engine="openpyxl")
    #st.success(f"✅ Loaded dataset successfully — {df.shape[0]} samples, {df.shape[1]} columns")
#except Exception as e:
 #   st.error(f"❌ Error loading dataset: {e}")
  #  st.stop()

if "CS" not in df.columns:
    st.error("The dataset must contain a target column named 'CS'.")
    st.stop()

X = df.drop(columns=["CS"])
y = df["CS"]

# Define variable names and units (Editable Section)
# --------------------------------------------------
# Change names/units easily here if your Excel changes later
input_vars = [
    ("Cement", "kg/m³"),
    ("Water", "kg/m³"),
    ("Fine Aggregate", "kg/m³"),
    ("Coarse Aggregate", "kg/m³"),
    ("Superplasticizer", "% of binder"),
    ("Fly Ash", "% of binder"),
    ("Silica Fume", "% of binder"),
    ("Steel Fiber", "% by volume"),
    ("Graphene Oxide", "% by wt. of cement"),
    ("Curing Days", "days")
]

# ---------------------- Data Split ----------------------
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------- Flower Pollination Algorithm ----------------------
def flower_pollination_optimization(n_pop=8, n_iter=15):
    """Optimizes hidden layers and learning rate."""
    def fitness(params):
        n1, n2, lr = params
        model = build_dnn(X_train.shape[1], n1, n2, lr)
        model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0)
        y_pred = model.predict(X_test).ravel()
        return r2_score(y_test, y_pred)

    population = [
        [random.randint(32, 128), random.randint(16, 64), 10 ** random.uniform(-4, -2)]
        for _ in range(n_pop)
    ]

    best_sol = max(population, key=fitness)
    best_score = fitness(best_sol)

    for _ in range(n_iter):
        for i in range(n_pop):
            if random.random() < 0.8:
                new_sol = [max(8, int(best_sol[0] + random.gauss(0, 5))),
                           max(8, int(best_sol[1] + random.gauss(0, 3))),
                           abs(best_sol[2] + random.gauss(0, 0.0005))]
            else:
                new_sol = [random.randint(32, 128), random.randint(16, 64), 10 ** random.uniform(-4, -2)]

            new_score = fitness(new_sol)
            if new_score > best_score:
                best_score, best_sol = new_score, new_sol

    return best_sol, best_score

# ---------------------- DNN Model ----------------------
def build_dnn(input_dim, n1=64, n2=32, lr=0.001):
    model = keras.Sequential([
        layers.Dense(n1, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.1),
        layers.Dense(n2, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

# ---------------------- Model Training ----------------------
#st.header("⚙️ Model Training and Optimization")

if st.button("🚀 Run FPA Optimization and Train Model", use_container_width=True):
    best_params, best_r2 = flower_pollination_optimization()
    n1, n2, lr = best_params

    model = build_dnn(X_train.shape[1], n1, n2, lr)
    model.fit(X_train, y_train, epochs=80, batch_size=16, verbose=0)

    y_pred_train = model.predict(X_train).ravel()
    y_pred_test = model.predict(X_test).ravel()

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    st.success(f"✅ FPA Optimization Completed! Best Params → Layers: [{n1}, {n2}], Learning Rate: {lr:.5f}")
    st.metric("R² (Training)", f"{r2_train:.3f}")
    st.metric("R² (Testing)", f"{r2_test:.3f}")

    st.session_state["fpa_model"] = model
    st.session_state["scaler"] = scaler

# ---------------------- Prediction Interface ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.header("🔹 Predict Compressive Strength (CS)")

cols = st.columns(2)
inputs = []

for i, (var, unit) in enumerate(input_vars):
    col = cols[i % 2]
    with col:
        val = st.number_input(f"{var} ({unit})", min_value=0.0, step=0.1, format="%.3f")
        inputs.append(val)

if st.button("🔮 Predict CS", use_container_width=True):
    if "fpa_model" not in st.session_state:
        st.warning("Please train the FPA-DNN model first.")
    else:
        model = st.session_state["fpa_model"]
        scaler = st.session_state["scaler"]
        x_scaled = scaler.transform([inputs])
        pred_cs = model.predict(x_scaled).ravel()[0]

        st.markdown(
            f"""
            <div class="predict-box">
                <h3 style="color:#0f2027;">Predicted Compressive Strength (CS)</h3>
                <h2 style="color:#2e8b57;">{pred_cs:.2f} MPa</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------- Footer ----------------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:13px;'>
<b>Developed by:</b> Bibhu Prasad Mishra (2025) <br>
Email: <a href="mailto:bibhumi2121@gmail.com">bibhumi2121@gmail.com</a>
</div>
""", unsafe_allow_html=True)

