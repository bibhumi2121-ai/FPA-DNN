# ===========================================================
# Streamlit App: FPA-DNN (Local DATA.xlsx)
# Predicts 'CS' using remaining features
# ===========================================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="FPA-DNN | CS Prediction", layout="wide")
plt.rcParams["font.family"] = "Times New Roman"

st.title("🌼 FPA-DNN Dashboard — Predicting Compressive Strength (CS)")
st.markdown(
    "This app trains a Deep Neural Network optimized via the Flower Pollination Algorithm (FPA) "
    "to predict **Compressive Strength (CS)** using all other columns from your `DATA.xlsx`."
)

# ---------------------- LOAD DATA ----------------------
st.header("1️⃣ Load Dataset (Local)")

if not os.path.exists("DATA.xlsx"):
    st.error("❌ `DATA.xlsx` not found in the current folder.")
    st.stop()

df = pd.read_excel("DATA.xlsx", engine="openpyxl")
st.success(f"✅ Loaded DATA.xlsx successfully — {df.shape[0]} rows, {df.shape[1]} columns.")
st.dataframe(df.head())

if "CS" not in df.columns:
    st.error("❌ 'CS' column not found. Please ensure your Excel file contains a column named 'CS'.")
    st.stop()

# ---------------------- PREPARE DATA ----------------------
X = df.drop(columns=["CS"])
y = df["CS"]

# One-hot encode categorical variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Split
test_size = st.slider("Test size (%)", 10, 40, 30, 5)
seed = st.number_input("Random Seed", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=seed)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- BUILD DNN MODEL ----------------------
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

# ---------------------- FPA ALGORITHM ----------------------
def levy_flight(rng, size, lam=1.5):
    sigma_u = (
        (np.math.gamma(1+lam)*np.sin(np.pi*lam/2)) /
        (np.math.gamma((1+lam)/2)*lam*2**((lam-1)/2))
    ) ** (1/lam)
    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1, size=size)
    step = u / (np.abs(v) ** (1/lam))
    return step

def map_to_hparams(z):
    n_layers = int(2 + np.floor(z[0]*3))  # 2–4 layers
    units = [int(32 * 2**int(np.floor(z[i]*3))) for i in range(1, n_layers+1)]
    dropout = float(np.clip(z[4]*0.4, 0, 0.4))
    lr = 10**(np.log10(1e-4) + z[5]*(np.log10(5e-3)-np.log10(1e-4)))
    batch = int(16 * (2 ** int(np.clip(np.round(z[2]*3), 0, 3))))  # 16–128
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
        rmse = np.sqrt(mean_squared_error(yva, pred))
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

# ---------------------- RUN FPA OPTIMIZATION ----------------------
st.header("2️⃣ Optimize and Train FPA-DNN")

pop = st.number_input("Population Size", 5, 30, 10, 1)
iters = st.number_input("Iterations", 5, 40, 10, 1)
p = st.slider("Pollination Probability (p)", 0.1, 0.95, 0.8, 0.05)
lam = st.slider("Levy λ", 0.5, 2.0, 1.5, 0.1)

run = st.button("🚀 Run FPA Optimization & Train Model")
if run:
    status = st.empty()
    st.info("Optimizing hyperparameters with Flower Pollination Algorithm...")
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
    history = model.fit(X_train_scaled, y_train,
                        validation_split=0.15,
                        epochs=best_hp["epochs"],
                        batch_size=best_hp["batch"],
                        verbose=0,
                        callbacks=[es])

    y_pred_train = model.predict(X_train_scaled).ravel()
    y_pred_test = model.predict(X_test_scaled).ravel()

    # ---------------------- METRICS ----------------------
    def metrics(y_true, y_pred):
        return dict(
            R2=r2_score(y_true, y_pred),
            RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
            MAE=mean_absolute_error(y_true, y_pred),
            MAPE=np.mean(np.abs((y_true - y_pred)/y_true))*100
        )

    tr = metrics(y_train, y_pred_train)
    te = metrics(y_test, y_pred_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (Train)", f"{tr['R2']:.3f}")
    c2.metric("R² (Test)", f"{te['R2']:.3f}")
    c3.metric("RMSE (Test)", f"{te['RMSE']:.3f}")
    c4.metric("MAPE (Test)", f"{te['MAPE']:.2f}%")

    # ---------------------- PLOTS ----------------------
    st.header("3️⃣ Model Evaluation")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].scatter(y_train, y_pred_train, color="#1f77b4", edgecolors="white", label="Train")
    ax[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "k--")
    ax[0].set_title("R² Fit — Training")
    ax[0].set_xlabel("Actual CS"); ax[0].set_ylabel("Predicted CS")
    ax[0].legend(); ax[0].grid(True, linestyle=":")

    ax[1].scatter(y_test, y_pred_test, color="#d62728", edgecolors="black", label="Test")
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    ax[1].set_title("R² Fit — Testing")
    ax[1].set_xlabel("Actual CS"); ax[1].set_ylabel("Predicted CS")
    ax[1].legend(); ax[1].grid(True, linestyle=":")

    st.pyplot(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(10, 4))
    res = y_test - y_pred_test
    ax.scatter(range(len(res)), res, color="#9467bd")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_title("Residual Plot — Test Data")
    ax.set_xlabel("Sample Index"); ax.set_ylabel("Residual (Actual - Predicted)")
    ax.grid(True, linestyle=":")
    st.pyplot(fig)

    st.success("✅ Model trained successfully!")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown(
    "📘 *This app implements a Flower Pollination Algorithm (FPA) to optimize a Deep Neural Network (DNN) "
    "for predicting Compressive Strength (CS). Developed for journal-ready reproducibility.*"
)
