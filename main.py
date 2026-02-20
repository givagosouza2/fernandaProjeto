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
st.title("📱 IMU: detrend → 100 Hz → filtros → norma → K-means(7) → início/fim")

fs_target = 100.0
fc_acc = 4
fc_gyro = 2
k_states = 7

n_baseline = 15
n_after = 5

seed = st.sidebar.number_input("Seed do K-means", min_value=0, max_value=999999, value=42, step=1)


# -----------------------------
# Helpers
# -----------------------------
def read_imu_file(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding_errors="ignore")
    df.columns = [c.strip().lower() for c in df.columns]

    # map common names
    rename = {}
    for c in df.columns:
        if c in ["tempo", "time", "t"]:
            rename[c] = "t"
        elif c in ["x", "ax", "gx"]:
            rename[c] = "x"
        elif c in ["y", "ay", "gy"]:
            rename[c] = "y"
        elif c in ["z", "az", "gz"]:
            rename[c] = "z"
    df = df.rename(columns=rename)

    missing = [c for c in ["t", "x", "y", "z"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}. Encontrei: {list(df.columns)}")

    for c in ["t", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t", "x", "y", "z"]).reset_index(drop=True)
    return df[["t", "x", "y", "z"]]


def normalize_time(t_raw: np.ndarray) -> np.ndarray:
    t = np.asarray(t_raw, dtype=float)
    t = t - t[0]
    if (np.nanmax(t) - np.nanmin(t)) > 1000.0:  # heuristic ms
        t = t / 1000.0
    return t


def resample_to_fs(t: np.ndarray, x: np.ndarray, fs: float):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    # remove duplicate times
    t_unique, idx = np.unique(t, return_index=True)
    x_unique = x[idx]

    t_u = np.arange(t_unique[0], t_unique[-1], 1.0 / fs)
    x_u = np.interp(t_u, t_unique, x_unique)
    return t_u, x_u


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def norm_xyz(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def ordered_states_from_kmeans(values: np.ndarray, k: int, seed: int):
    v = values.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(v)
    centers = km.cluster_centers_.flatten()

    order = np.argsort(centers)          # low -> high
    rank = np.zeros_like(order)
    rank[order] = np.arange(k)           # cluster_id -> ordered_state

    states = rank[labels]
    centers_ord = centers[order]
    return states.astype(int), centers_ord


def mode_int(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def detect_transition_from(states: np.ndarray, baseline_state: int, start_idx: int, end_idx: int):
    """
    Procura transição baseline->superior em estados[start_idx:end_idx]
    Regra:
      - n_baseline amostras anteriores == baseline_state
      - n_after amostras seguintes > baseline_state
    Retorna índice global (no vetor states) ou None.
    """
    s = states.astype(int)
    n = len(s)
    start_idx = max(start_idx, n_baseline)
    end_idx = min(end_idx, n - n_after)

    for i in range(start_idx, end_idx):
        if np.all(s[i - n_baseline:i] == baseline_state) and np.all(s[i:i + n_after] > baseline_state):
            return i
    return None


def first_index_geq(t: np.ndarray, value: float) -> int:
    return int(np.searchsorted(t, value, side="left"))


def last_index_leq(t: np.ndarray, value: float) -> int:
    return int(np.searchsorted(t, value, side="right") - 1)


# -----------------------------
# UI
# -----------------------------
c1, c2 = st.columns(2)
with c1:
    acc_file = st.file_uploader("📄 Aceleração (Tempo, X, Y, Z)", type=["txt", "csv"], key="acc")
with c2:
    gyro_file = st.file_uploader("📄 Giroscópio (Tempo, X, Y, Z)", type=["txt", "csv"], key="gyro")

run = st.button("▶️ Processar e segmentar", type="primary", disabled=(acc_file is None or gyro_file is None))


# -----------------------------
# Pipeline
# -----------------------------
if run:
    try:
        df_acc = read_imu_file(acc_file)
        df_gyr = read_imu_file(gyro_file)
    except Exception as e:
        st.error(f"Erro ao ler arquivos: {e}")
        st.stop()

    # tempo
    t_acc = normalize_time(df_acc["t"].to_numpy())
    t_gyr = normalize_time(df_gyr["t"].to_numpy())

    # detrend
    acc_x = detrend(df_acc["x"].to_numpy(), type="linear")
    acc_y = detrend(df_acc["y"].to_numpy(), type="linear")
    acc_z = detrend(df_acc["z"].to_numpy(), type="linear")

    gyr_x = detrend(df_gyr["x"].to_numpy(), type="linear")
    gyr_y = detrend(df_gyr["y"].to_numpy(), type="linear")
    gyr_z = detrend(df_gyr["z"].to_numpy(), type="linear")

    # reamostragem 100 Hz
    t_acc_u, acc_x_u = resample_to_fs(t_acc, acc_x, fs_target)
    _,       acc_y_u = resample_to_fs(t_acc, acc_y, fs_target)
    _,       acc_z_u = resample_to_fs(t_acc, acc_z, fs_target)

    t_gyr_u, gyr_x_u = resample_to_fs(t_gyr, gyr_x, fs_target)
    _,       gyr_y_u = resample_to_fs(t_gyr, gyr_y, fs_target)
    _,       gyr_z_u = resample_to_fs(t_gyr, gyr_z, fs_target)

    # filtros
    acc_x_f = lowpass(acc_x_u, fs_target, fc_acc)
    acc_y_f = lowpass(acc_y_u, fs_target, fc_acc)
    acc_z_f = lowpass(acc_z_u, fs_target, fc_acc)

    gyr_x_f = lowpass(gyr_x_u, fs_target, fc_gyro)
    gyr_y_f = lowpass(gyr_y_u, fs_target, fc_gyro)
    gyr_z_f = lowpass(gyr_z_u, fs_target, fc_gyro)

    # normas
    acc_norm = norm_xyz(acc_x_f, acc_y_f, acc_z_f)
    gyr_norm = norm_xyz(gyr_x_f, gyr_y_f, gyr_z_f)

    # K-means (7 estados) na norma do giro
    states, centers_ord = ordered_states_from_kmeans(gyr_norm, k_states, int(seed))

    # -----------------------------------------
    # BASELINE INÍCIO: 2s a 5s
    # e varredura começa NO INÍCIO da baseline (2s)
    # -----------------------------------------
    t0 = 2.0
    t1 = 5.0
    i_bs0 = first_index_geq(t_gyr_u, t0)
    i_bs1 = last_index_leq(t_gyr_u, t1)

    if i_bs1 <= i_bs0 or (i_bs1 - i_bs0 + 1) < 20:
        st.warning("Baseline inicial (2s a 5s) ficou muito curta; verifique o sinal.")
    baseline_state_start = mode_int(states[i_bs0:i_bs1 + 1])

    # varrer a partir do início da baseline (em prática: i_bs0 + n_baseline)
    start_idx = detect_transition_from(
        states,
        baseline_state_start,
        start_idx=i_bs0 + n_baseline,
        end_idx=len(states)
    )
    start_t = None if start_idx is None else float(t_gyr_u[start_idx])

    # -----------------------------------------
    # BASELINE FINAL: (fim-4s) a (fim-2s)
    # leitura RETRÓGRADA começando no "início da baseline" NA DIREÇÃO RETRÓGRADA,
    # que é (fim-2s)
    # -----------------------------------------
    t_end = float(t_gyr_u[-1])
    tb0 = t_end - 4.0
    tb1 = t_end - 2.0

    i_be0 = first_index_geq(t_gyr_u, tb0)
    i_be1 = last_index_leq(t_gyr_u, tb1)

    if i_be1 <= i_be0 or (i_be1 - i_be0 + 1) < 20:
        st.warning("Baseline final (fim-4s a fim-2s) ficou muito curta; verifique o sinal.")
    baseline_state_end = mode_int(states[i_be0:i_be1 + 1])

    # construir leitura retrógrada iniciando em i_be1 (≈ fim-2s) e indo para trás
    # assim, a varredura começa exatamente no início da baseline na direção retrógrada
    states_rev = states[:i_be1 + 1][::-1]

    # no eixo retrógrado, baseline começa em índice 0
    end_idx_rev = detect_transition_from(
        states_rev,
        baseline_state_end,
        start_idx=0 + n_baseline,      # precisa de 15 amostras anteriores em baseline
        end_idx=len(states_rev)
    )
    end_idx = None if end_idx_rev is None else (i_be1 - end_idx_rev)
    end_t = None if end_idx is None else float(t_gyr_u[end_idx])

    # -----------------------------------------
    # Report + Plot
    # -----------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.subheader("📌 Estados de baseline")
        st.write(f"- Baseline início (2–5s): estado dominante **{baseline_state_start}**")
        st.write(f"- Baseline final (fim−4 a fim−2): estado dominante **{baseline_state_end}**")
    with right:
        st.subheader("📊 Centroides (ordenados) do K-means na norma do giro")
        centers_df = pd.DataFrame({"estado_ordenado": np.arange(k_states), "centroide_norma_giro": centers_ord})
        st.dataframe(centers_df, use_container_width=True)

    st.subheader("📈 Normas com marcações de início e fim")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_gyr_u, gyr_norm, label="||giro|| (LP 1.5 Hz)")
    ax.plot(t_acc_u, acc_norm, label="||acel|| (LP 8 Hz)", alpha=0.8)

    # marcar janelas de baseline
    ax.axvspan(t0, t1, alpha=0.12, label="baseline início (2–5s)")
    ax.axvspan(tb0, tb1, alpha=0.12, label="baseline final (fim−4 a fim−2)")

    if start_t is not None:
        ax.axvline(start_t, linestyle="--", linewidth=2, label=f"Início @ {start_t:.3f}s")
    else:
        st.warning("Início não encontrado com a regra 15→5 a partir da baseline inicial.")

    if end_t is not None:
        ax.axvline(end_t, linestyle="--", linewidth=2, label=f"Fim @ {end_t:.3f}s")
    else:
        st.warning("Fim não encontrado com a regra 15→5 na leitura retrógrada a partir da baseline final.")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Norma")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.caption(f"Regra: {n_baseline} amostras baseline seguidas por {n_after} amostras em estado superior (k={k_states}).")
