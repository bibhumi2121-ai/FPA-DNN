# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:40:29 2025

@author: bibhu
"""

# app.py
# ===========================================================
# Streamlit: FPA-DNN to predict CS from DATA.xlsx
# - Reads DATA.xlsx automatically (or via uploader)
# - Uses all columns except 'CS' as features
# - FPA optimizes a compact Keras DNN
# - Shows metrics, R²-fit plot, residuals plot
# ===========================================================

import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Silence TF logs before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------#
# Page & Theme
# ------------------------------#
st.set_page_config(page_title="FPA-DNN: CS Predictor", layout="wide")
plt.rcParams["font.family"] = "Times New Roman"

st.markdown(
    """
    <style>
    .big-title {font-size: 28px; font-weight: 800; margin-bottom: 8px;}
    .subtle {color:#666;}
    .metric-card {padding:12px 16px;border-radius:14px;border:1px solid #e8e8e8;background:#fafafa;}
    .stButton>button {border-radius:12px;font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🌼 FPA-DNN Dashboard — Predicting CS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Load your dataset, optimize DNN with Flower Pollination Algorithm, and predict CS.</div>', unsafe_allow_html=True)
st.write("")

# ------------------------------#
# Utilities
# ------------------------------#
@st.cache_data(show_spinner=False)
def read_excel_bytes(b):
    return pd.read_excel(io.BytesIO(b), engine="openpyxl")

@st.cache_data(show_spinner=False)
def read_local_or_upload():
    # Try local DATA.xlsx
    if os.path.exists("DATA.xlsx"):
        try:
            df = pd.read_excel("DATA.xlsx", engine="openpyxl")
            return df, "Loaded local DATA.xlsx"
        except Exception as e:
            return None, f"Local read error: {e}"
    return None, "DATA.xlsx not found locally."

def ensure_numeric(df, target):
    # Convert non-numeric features via get_dummies; keep target numeric
    num_df = df.copy()
    if target not in num_df.columns:
        raise ValueError(f"'CS' column not found. Columns are: {list(num_df.columns)}")
    # Separate target
    y = num_df[target].astype(float)
    X = num_df.drop(columns=[target])
    # One-hot encode categoricals if any
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def build_dnn(input_dim, units_list, dropout, lr):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for u in units_list:
        model.add(layers.Dense(u, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear"))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

# --- FPA pieces (continuous vector -> hyperparameters) ---
def sample_hyperparams(rng):
    """Return a normalized vector z in [0,1]^6 we will map to hyperparams."""
    return rng.random(6)

def map_to_hparams(z):
    """
    z -> [n_layers, u1, u2, u3, dropout, lr, batch, epochs]
    Here we use 6 dims and derive batch/epochs from u1,u2 to keep search light.
    """
    # Number of hidden layers: 2..4
    n_layers = int(2 + np.floor(z[0] * 3))  # 0-1 -> 2..4
    # Units per layer: 32..256
    def units(s): return int(32 * 2 ** int(np.floor(s * 4)))  # 32,64,128,256
    u1 = units(z[1])
    u2 = units(z[2])
    u3 = units(z[3])
    units_list = [u1, u2] if n_layers == 2 else [u1, u2, u3 if n_layers >= 3 else u2]
    # Dropout: 0.0..0.4
    dropout = float(np.clip(z[4] * 0.4, 0.0, 0.4))
    # Learning rate: 1e-4 .. 5e-3 (log scale approx)
    lr = 10 ** (np.log10(1e-4) + z[5] * (np.log10(5e-3) - np.log10(1e-4)))
    # Batch size: 16..128 (powers of 2)
    batch = int(16 * (2 ** int(np.clip(np.round(z[2] * 3), 0, 3))))  # 16,32,64,128
    # Epochs: 60..240
    epochs = int(60 + np.round(z[1] * 180))
    return dict(n_layers=n_layers, units=units_list, dropout=dropout, lr=lr, batch=batch, epochs=epochs)

def levy_flight(rng, size, lam=1.5):
    # Mantegna algorithm for symmetric Levy
    sigma_u = ( np.math.gamma(1+lam) * np.sin(np.pi*lam/2) / ( np.math.gamma((1+lam)/2) * lam * 2**((lam-1)/2) ) ) ** (1/lam)
    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1, size=size)
    step = u / (np.abs(v) ** (1/lam))
    return step

def fpa_optimize(X, y, cfg, seed, status_area=None):
    """
    Flower Pollination Algorithm to minimize CV RMSE.
    cfg: dict with keys {pop, iters, p, lam, folds}
    """
    rng = np.random.default_rng(seed)
    pop = cfg["pop"]; iters = cfg["iters"]; p = cfg["p"]; lam = cfg["lam"]; folds = cfg["folds"]

    # Init population in [0,1]^6
    Z = np.array([sample_hyperparams(rng) for _ in range(pop)])
    scores = np.array([cv_rmse(X, y, map_to_hparams(z), folds, seed + i) for i, z in enumerate(Z)])
    gbest = Z[np.argmin(scores)].copy()
    gbest_score = np.min(scores)

    for it in range(iters):
        for i in range(pop):
            if rng.random() < p:
                # Global pollination: z_i + s * (gbest - z_i)
                step = levy_flight(rng, size=Z[i].shape, lam=lam)
                Z[i] = Z[i] + step * (gbest - Z[i])
            else:
                # Local pollination: z_i + epsilon*(z_j - z_k)
                jk = rng.choice(pop, size=2, replace=False)
                eps = rng.random()
                Z[i] = Z[i] + eps * (Z[jk[0]] - Z[jk[1]])

            # Bound to [0,1]
            Z[i] = np.clip(Z[i], 0, 1)

            # Evaluate
            sc = cv_rmse(X, y, map_to_hparams(Z[i]), folds, seed + 100 + it*pop + i)
            if sc < scores[i]:
                scores[i] = sc
                if sc < gbest_score:
                    gbest_score = sc
                    gbest = Z[i].copy()

        if status_area is not None:
            status_area.write(f"Iteration {it+1}/{iters} • Best CV RMSE: {gbest_score:.3f}")

    return map_to_hparams(gbest), gbest_score

def cv_rmse(X, y, hparams, folds, seed):
    # K-fold CV on training only (inner loop)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rmses = []
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = build_dnn(X.shape[1], hparams["units"], hparams["dropout"], hparams["lr"])
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
        model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva),
            epochs=hparams["epochs"],
            batch_size=hparams["batch"],
            verbose=0,
            callbacks=[es]
        )
        pred = model.predict(Xva, verbose=0).ravel()
        rmses.append(np.sqrt(mean_squared_error(yva, pred)))
        keras.backend.clear_session()
    return float(np.mean(rmses))

def metrics_report(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape)

# ------------------------------#
# Data Section
# ------------------------------#
with st.expander("① Load Data", expanded=True):
    df_local, msg = read_local_or_upload()
    st.caption(msg)
    uploaded = st.file_uploader("Or upload an Excel file", type=["xlsx", "xls"])

    if uploaded is not None:
        try:
            df = read_excel_bytes(uploaded.read())
            st.success(f"Uploaded file loaded. Shape: {df.shape}")
        except Exception as e:
            st.error(f"Upload read error: {e}")
            st.stop()
    else:
        if df_local is None:
            st.warning("Please upload your DATA.xlsx (must contain 'CS' as target).")
            st.stop()
        df = df_local
        st.success(f"DATA.xlsx loaded. Shape: {df.shape}")

    if "CS" not in df.columns:
        st.error("Column 'CS' not found. Please ensure your file has a 'CS' column.")
        st.stop()

    st.dataframe(df.head(), use_container_width=True)
    test_size_pct = st.slider("Test size (%)", 10, 40, 30, 5)
    random_seed = st.number_input("Random seed", value=42, step=1)

# ------------------------------#
# Train Section (FPA-DNN)
# ------------------------------#
with st.expander("② Train FPA-DNN", expanded=True):
    st.caption("FPA searches DNN hyperparameters; best model is retrained on the full train split.")
    # Prepare features/target
    X_df, y_series = ensure_numeric(df, target="CS")
    feature_names = list(X_df.columns)

    # Split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y_series, test_size=test_size_pct/100, random_state=random_seed
    )

    # Scale (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test  = scaler.transform(X_test_df.values)
    y_train = y_train.values.astype(float)
    y_test  = y_test.values.astype(float)

    # FPA controls
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: pop = st.number_input("Population", 5, 50, 18, 1)
    with col2: iters = st.number_input("Iterations", 5, 80, 25, 1)
    with col3: p = st.slider("Pollination prob (p)", 0.1, 0.95, 0.8, 0.05)
    with col4: lam = st.slider("Levy λ", 0.5, 2.0, 1.5, 0.1)
    with col5: folds = st.number_input("CV folds", 3, 10, 5, 1)

    btn = st.button("🚀 Run FPA Optimization & Train", type="primary")
    if btn:
        status = st.empty()
        with st.spinner("Optimizing hyperparameters with FPA…"):
            best_hp, best_cv = fpa_optimize(
                X_train, y_train,
                cfg=dict(pop=pop, iters=iters, p=p, lam=lam, folds=folds),
                seed=random_seed,
                status_area=status
            )
        st.success(f"Best CV RMSE: {best_cv:.3f}")
        st.write("**Selected Hyperparameters**:", best_hp)

        # Retrain best model on full train set
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)
        model = build_dnn(X_train.shape[1], best_hp["units"], best_hp["dropout"], best_hp["lr"])
        hist = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=best_hp["epochs"],
            batch_size=best_hp["batch"],
            verbose=0,
            callbacks=[es]
        )

        # Predictions & metrics
        yhat_tr = model.predict(X_train, verbose=0).ravel()
        yhat_te = model.predict(X_test,  verbose=0).ravel()
        mtr = metrics_report(y_train, yhat_tr)
        mte = metrics_report(y_test,  yhat_te)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><b>R² (Train)</b><br/>{mtr["R2"]:.3f}</div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><b>R² (Test)</b><br/>{mte["R2"]:.3f}</div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><b>RMSE (Test)</b><br/>{mte["RMSE"]:.3f}</div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><b>MAPE (Test)</b><br/>{mte["MAPE"]:.2f}%</div>', unsafe_allow_html=True)

        st.session_state["scaler"] = scaler
        st.session_state["model"] = model
        st.session_state["features"] = feature_names
        st.session_state["X_train_df"] = X_train_df
        st.session_state["X_test_df"] = X_test_df
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["yhat_tr"] = yhat_tr
        st.session_state["yhat_te"] = yhat_te

# ------------------------------#
# Plots Section
# ------------------------------#
with st.expander("③ Plots", expanded=True):
    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]
        yhat_tr = st.session_state["yhat_tr"]
        yhat_te = st.session_state["yhat_te"]

        c1, c2 = st.columns(2)
        with c1:
            # R² scatter (Train)
            fig, ax = plt.subplots(figsize=(5.8, 5.2))
            ax.scatter(y_train, yhat_tr, s=28, c="#1f77b4", edgecolors="white", linewidths=0.5)
            line = np.linspace(min(y_train.min(), yhat_tr.min()), max(y_train.max(), yhat_tr.max()), 100)
            ax.plot(line, line, "k--", lw=1.2)
            ax.set_title("R² Fit — Train", fontweight="bold")
            ax.set_xlabel("Actual CS"); ax.set_ylabel("Predicted CS")
            ax.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig)

        with c2:
            # R² scatter (Test)
            fig, ax = plt.subplots(figsize=(5.8, 5.2))
            ax.scatter(y_test, yhat_te, s=28, c="#d62728", edgecolors="black", linewidths=0.5, marker="^")
            line = np.linspace(min(y_test.min(), yhat_te.min()), max(y_test.max(), yhat_te.max()), 100)
            ax.plot(line, line, "k--", lw=1.2)
            ax.set_title("R² Fit — Test", fontweight="bold")
            ax.set_xlabel("Actual CS"); ax.set_ylabel("Predicted CS")
            ax.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig)

        # Residuals (Test)
        fig, ax = plt.subplots(figsize=(11.8, 4.2))
        res = y_test - yhat_te
        ax.scatter(range(len(res)), res, s=26, c="#9467bd")
        ax.axhline(0, color="k", linestyle="--", lw=1.0)
        ax.set_title("Residuals — Test", fontweight="bold")
        ax.set_xlabel("Sample Index"); ax.set_ylabel("Residual (Actual − Pred)")
        ax.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig)

# ------------------------------#
# Prediction Section
# ------------------------------#
with st.expander("④ Predict", expanded=True):
    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        feat_names = st.session_state["features"]

        st.write("**Single record input**")
        cols = st.columns(min(4, len(feat_names)))
        user_vals = {}
        for i, f in enumerate(feat_names):
            with cols[i % len(cols)]:
                # default to dataset median if available
                default_val = float(df[f].median()) if f in df.columns else 0.0
                user_vals[f] = st.number_input(f, value=default_val)

        if st.button("🔮 Predict CS (single)"):
            x_df = pd.DataFrame([user_vals])[feat_names]
            x_scaled = scaler.transform(x_df.values)
            y_pred = float(model.predict(x_scaled, verbose=0).ravel()[0])
            st.success(f"Predicted CS: **{y_pred:.3f}**")

        st.divider()
        st.write("**Batch prediction (upload CSV/XLSX with the same feature columns)**")
        up = st.file_uploader("Upload batch file", type=["csv", "xlsx"])
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    bx = pd.read_csv(up)
                else:
                    bx = read_excel_bytes(up.read())
                # Align columns
                missing = [c for c in feat_names if c not in bx.columns]
                if missing:
                    st.error(f"Missing columns in upload: {missing}")
                else:
                    x_scaled = scaler.transform(bx[feat_names].values)
                    preds = model.predict(x_scaled, verbose=0).ravel()
                    out = bx.copy()
                    out["CS_Pred"] = preds
                    st.dataframe(out.head(), use_container_width=True)
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predictions CSV", csv, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Batch prediction error: {e}")

# ------------------------------#
# Footer
# ------------------------------#
st.markdown(
    """
    <hr/>
    <div class="subtle">
    <b>Notes.</b> FPA hyperparameters: population, iterations, pollination probability (p), and Lévy λ control exploration/exploitation.
    Objective: K-fold CV RMSE minimization on the train split; final model retrained with best hyperparameters.
    </div>
    """,
    unsafe_allow_html=True,
)
