# ===========================================================
# Streamlit App: FPA-DNN (Only R² Display)
# ===========================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go

# TensorFlow imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="FPA-DNN | CS Prediction", layout="wide")
st.title("🌼 FPA-DNN Dashboard — Predicting Compressive Strength (CS)")

# ---------------------- LOAD DATA ----------------------
if not os.path.exists("DATA.xlsx"):
    st.error("❌ DATA.xlsx not found in this folder.")
    st.stop()

df = pd.read_excel("DATA.xlsx", engine="openpyxl")
st.success(f"✅ Loaded DATA.xlsx successfully — {df.shape[0]} rows, {df.shape[1]} columns.")
st.dataframe(df.head())

if "CS" not in df.columns:
    st.error("❌ 'CS' column not found. Ensure your Excel has a column named 'CS'.")
    st.stop()

# ---------------------- PREPARE DATA ----------------------
X = df.drop(columns=["CS"])
y = df["CS"]
X = pd.get_dummies(X, drop_first=True)

test_size = st.slider("Test size (%)", 10, 40, 30, 5)
seed = st.number_input("Random Seed", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=seed
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- DNN + FPA ----------------------
def build_dnn(input_dim, units, dropout, lr):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for u in units:
        model.add(layers.Dense(u, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear"))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def levy_flight(rng, size, lam=1.5):
    sigma_u = (
        (np.math.gamma(1+lam)*np.sin(np.pi*lam/2)) /
        (np.math.gamma((1+lam)/2)*lam*2**((lam-1)/2))
    ) ** (1/lam)
    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1, size=size)
    return u / (np.abs(v)**(1/lam))

def map_to_hparams(z):
    n_layers = int(2 + np.floor(z[0]*3))
    units = [int(32 * 2**int(np.floor(z[i]*3))) for i in range(1, n_layers+1)]
    dropout = float(np.clip(z[4]*0.4, 0, 0.4))
    lr = 10**(np.log10(1e-4) + z[5]*(np.log10(5e-3)-np.log10(1e-4)))
    batch = int(16 * (2**int(np.clip(np.round(z[2]*3), 0, 3))))
    epochs = int(80 + np.round(z[1]*120))
    return dict(units=units, dropout=dropout, lr=lr, batch=batch, epochs=epochs)

def cv_rmse(X, y, hparams, folds=3, seed=42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rmses = []
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = build_dnn(X.shape[1], hparams["units"], hparams["dropout"], hparams["lr"])
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
        model.fit(Xtr, ytr, validation_data=(Xva, yva),
                  epochs=hparams["epochs"], batch_size=hparams["batch"], verbose=0, callbacks=[es])
        pred = model.predict(Xva, verbose=0).ravel()
        rmse = np.sqrt(np.mean((yva - pred)**2))
        rmses.append(rmse)
        keras.backend.clear_session()
    return float(np.mean(rmses))

def fpa_optimize(X, y, pop=10, iters=10, p=0.8, lam=1.5, folds=3, seed=42, status=None):
    rng = np.random.default_rng(seed)
    Z = np.random.rand(pop, 6)
    scores = np.array([cv_rmse(X, y, map_to_hparams(z), folds, seed+i) for i, z in enumerate(Z)])
    gbest = Z[np.argmin(scores)].copy()
    gbest_score = np.min(scores)

    for it in range(iters):
        for i in range(pop):
            if rng.random() < p:
                step = levy_flight(rng, size=Z[i].shape, lam=lam)
                Z[i] = Z[i] + step*(gbest - Z[i])
            else:
                jk = rng.choice(pop, size=2, replace=False)
                eps = rng.random()
                Z[i] = Z[i] + eps*(Z[jk[0]] - Z[jk[1]])
            Z[i] = np.clip(Z[i], 0, 1)
            sc = cv_rmse(X, y, map_to_hparams(Z[i]), folds, seed+i)
            if sc < scores[i]:
                scores[i] = sc
                if sc < gbest_score:
                    gbest = Z[i].copy()
                    gbest_score = sc
        if status:
            status.info(f"Iteration {it+1}/{iters} — Best CV RMSE: {gbest_score:.3f}")
    return map_to_hparams(gbest), gbest_score

# ---------------------- TRAIN MODEL ----------------------
st.header("2️⃣ Optimize and Train Model")
pop = st.number_input("Population Size", 5, 30, 10, 1)
iters = st.number_input("Iterations", 5, 40, 10, 1)
p = st.slider("Pollination Probability (p)", 0.1, 0.95, 0.8, 0.05)
lam = st.slider("Levy λ", 0.5, 2.0, 1.5, 0.1)

if st.button("🚀 Run FPA Optimization & Train"):
    status = st.empty()
    best_hp, best_cv = fpa_optimize(
        X_train_scaled, y_train.values,
        pop=pop, iters=iters, p=p, lam=lam, folds=3,
        seed=seed, status=status
    )
    st.success(f"Best CV RMSE = {best_cv:.3f}")
    st.write("**Optimal Hyperparameters:**", best_hp)

    # Train final model
    model = build_dnn(X_train_scaled.shape[1], best_hp["units"], best_hp["dropout"], best_hp["lr"])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, validation_split=0.15,
              epochs=best_hp["epochs"], batch_size=best_hp["batch"],
              verbose=0, callbacks=[es])

    y_pred_train = model.predict(X_train_scaled).ravel()
    y_pred_test = model.predict(X_test_scaled).ravel()

    # ---------------------- ONLY R² ----------------------
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    col1, col2 = st.columns(2)
    col1.metric("R² (Train)", f"{r2_train:.3f}")
    col2.metric("R² (Test)", f"{r2_test:.3f}")

    # ---------------------- PLOTS (Plotly) ----------------------
    st.header("3️⃣ R² Evaluation")

    fig_train = px.scatter(x=y_train, y=y_pred_train,
                           labels={'x':'Actual CS','y':'Predicted CS'},
                           title="R² Fit — Training", color_discrete_sequence=['#1f77b4'])
    fig_train.add_trace(go.Scatter(x=[y_train.min(), y_train.max()],
                                   y=[y_train.min(), y_train.max()],
                                   mode='lines', name='1:1 Line',
                                   line=dict(color='black', dash='dash')))
    st.plotly_chart(fig_train, use_container_width=True)

    fig_test = px.scatter(x=y_test, y=y_pred_test,
                          labels={'x':'Actual CS','y':'Predicted CS'},
                          title="R² Fit — Testing", color_discrete_sequence=['#d62728'])
    fig_test.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                  y=[y_test.min(), y_test.max()],
                                  mode='lines', name='1:1 Line',
                                  line=dict(color='black', dash='dash')))
    st.plotly_chart(fig_test, use_container_width=True)

    st.success("✅ Model trained and R² plots generated successfully!")

st.markdown("---")
st.caption("🌸 FPA-DNN automatically optimizes DNN hyperparameters to predict Compressive Strength (CS) using your local dataset.")
