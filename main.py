# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import detrend, butter, filtfilt
from sklearn.cluster import KMeans


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="IMU Markov Segmentação", layout="wide")
st.title("📱 IMU Pipeline com Segmentação Markov")

fs_target = 100.0
fc_acc = 8.0
fc_gyro = 1.5
k_states = 7

n_baseline = 15
n_after = 5

seed = 42


# -----------------------------
# Funções auxiliares
# -----------------------------
def read_imu_file(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding_errors="ignore")
    df.columns = [c.strip().lower() for c in df.columns]

    df = df.rename(columns={
        "tempo": "t",
        "time": "t",
        "x": "x",
        "y": "y",
        "z": "z"
    })

    for c in ["t", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()
    return df[["t", "x", "y", "z"]]


def normalize_time(t):
    t = t - t[0]
    if np.max(t) > 1000:
        t = t / 1000.0
    return t


def lowpass(x, fs, fc):
    b, a = butter(4, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def resample(t, x, fs):
    t_uniform = np.arange(t[0], t[-1], 1 / fs)
    x_uniform = np.interp(t_uniform, t, x)
    return t_uniform, x_uniform


def norm_xyz(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


def ordered_kmeans(values, k=7):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(values.reshape(-1, 1))
    centers = km.cluster_centers_.flatten()

    order = np.argsort(centers)
    rank = np.zeros_like(order)
    rank[order] = np.arange(k)

    ordered_states = rank[labels]
    ordered_centers = centers[order]
    return ordered_states, ordered_centers


def mode_int(arr):
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]


def detect_transition(states, baseline_state):
    s = states.astype(int)
    n = len(s)

    for i in range(n_baseline, n - n_after):
        if np.all(s[i-n_baseline:i] == baseline_state) and np.all(s[i:i+n_after] > baseline_state):
            return i
    return None


# -----------------------------
# Upload
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    acc_file = st.file_uploader("Arquivo Aceleração", type=["txt", "csv"])
with col2:
    gyro_file = st.file_uploader("Arquivo Giroscópio", type=["txt", "csv"])

run = st.button("Processar")

# -----------------------------
# Pipeline
# -----------------------------
if run and acc_file and gyro_file:

    df_acc = read_imu_file(acc_file)
    df_gyr = read_imu_file(gyro_file)

    t_acc = normalize_time(df_acc["t"].values)
    t_gyr = normalize_time(df_gyr["t"].values)

    # detrend
    acc_x = detrend(df_acc["x"].values)
    acc_y = detrend(df_acc["y"].values)
    acc_z = detrend(df_acc["z"].values)

    gyr_x = detrend(df_gyr["x"].values)
    gyr_y = detrend(df_gyr["y"].values)
    gyr_z = detrend(df_gyr["z"].values)

    # resample
    t_acc_u, acc_x = resample(t_acc, acc_x, fs_target)
    _, acc_y = resample(t_acc, acc_y, fs_target)
    _, acc_z = resample(t_acc, acc_z, fs_target)

    t_gyr_u, gyr_x = resample(t_gyr, gyr_x, fs_target)
    _, gyr_y = resample(t_gyr, gyr_y, fs_target)
    _, gyr_z = resample(t_gyr, gyr_z, fs_target)

    # filtros
    acc_x = lowpass(acc_x, fs_target, fc_acc)
    acc_y = lowpass(acc_y, fs_target, fc_acc)
    acc_z = lowpass(acc_z, fs_target, fc_acc)

    gyr_x = lowpass(gyr_x, fs_target, fc_gyro)
    gyr_y = lowpass(gyr_y, fs_target, fc_gyro)
    gyr_z = lowpass(gyr_z, fs_target, fc_gyro)

    # normas
    acc_norm = norm_xyz(acc_x, acc_y, acc_z)
    gyr_norm = norm_xyz(gyr_x, gyr_y, gyr_z)

    # KMeans
    states, centers = ordered_kmeans(gyr_norm, k_states)

    # -----------------------------
    # BASELINE INICIAL (2s a 5s)
    # -----------------------------
    mask_start_baseline = (t_gyr_u >= 2.0) & (t_gyr_u <= 5.0)

    if mask_start_baseline.sum() < 20:
        st.warning("Poucas amostras na baseline inicial.")

    baseline_state_start = mode_int(states[mask_start_baseline])
    start_idx = detect_transition(states, baseline_state_start)
    start_time = t_gyr_u[start_idx] if start_idx else None

    # -----------------------------
    # BASELINE FINAL (entre -4s e -2s)
    # -----------------------------
    t_end = t_gyr_u[-1]
    mask_end_baseline = (t_gyr_u >= (t_end - 5.0)) & (t_gyr_u <= (t_end - 3.0))

    if mask_end_baseline.sum() < 20:
        st.warning("Poucas amostras na baseline final.")

    baseline_state_end = mode_int(states[mask_end_baseline])

    states_rev = states[::-1]
    end_idx_rev = detect_transition(states_rev, baseline_state_end)
    end_idx = len(states) - end_idx_rev - 1 if end_idx_rev else None
    end_time = t_gyr_u[end_idx] if end_idx else None

    # -----------------------------
    # Plot
    # -----------------------------
    st.subheader("Normas com Segmentação")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_gyr_u, gyr_norm, label="||Giro||")
    ax.plot(t_acc_u, acc_norm, label="||Acel||", alpha=0.8)

    if start_time:
        ax.axvline(start_time, linestyle="--", label="Início")

    if end_time:
        ax.axvline(end_time, linestyle="--", label="Fim")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Norma")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.write(f"Estado baseline início: {baseline_state_start}")
    st.write(f"Estado baseline final: {baseline_state_end}")
    st.write(f"Início detectado: {start_time}")
    st.write(f"Fim detectado: {end_time}")
