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
st.set_page_config(page_title="IMU: detrend, 100 Hz, filtros e Markov/K-means", layout="wide")
st.title("📱 IMU pipeline: detrend → 100 Hz → filtros → norma → K-means(7) → início/fim")

with st.expander("Formato esperado dos arquivos"):
    st.markdown(
        "- Cabeçalho: **Tempo, X, Y, Z**\n"
        "- Um arquivo de **aceleração** e um de **giroscópio**\n"
        "- Separador pode ser vírgula, ponto-e-vírgula, tab etc. (o app tenta inferir)\n"
    )


# -----------------------------
# Helpers
# -----------------------------
def read_imu_file(uploaded_file) -> pd.DataFrame:
    """Read txt/csv with columns like Tempo, X, Y, Z (robust separator inference)."""
    df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding_errors="ignore")
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Common column names
    # Expect: tempo, x, y, z
    col_map = {}
    for c in df.columns:
        if c in ["tempo", "time", "t"]:
            col_map[c] = "t"
        elif c in ["x", "ax", "gx"]:
            col_map[c] = "x"
        elif c in ["y", "ay", "gy"]:
            col_map[c] = "y"
        elif c in ["z", "az", "gz"]:
            col_map[c] = "z"

    df = df.rename(columns=col_map)

    missing = [c for c in ["t", "x", "y", "z"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}. Encontrei: {list(df.columns)}")

    # Coerce numeric (handles stray spaces)
    for c in ["t", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["t", "x", "y", "z"]).reset_index(drop=True)
    return df[["t", "x", "y", "z"]]


def normalize_time_to_seconds(t_raw: np.ndarray) -> np.ndarray:
    """Convert time to seconds when it looks like ms; always zero-start."""
    t = np.asarray(t_raw, dtype=float)
    t = t - t[0]

    span = np.nanmax(t) - np.nanmin(t)
    # Heuristic:
    # if time span is large (>1000), likely milliseconds
    if span > 1000.0:
        t = t / 1000.0
    return t


def lowpass_filter(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass."""
    if fc <= 0 or fc >= fs / 2:
        raise ValueError("fc deve estar entre 0 e fs/2.")
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def resample_to_fs(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation to uniform grid."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    # Ensure strictly increasing time
    order = np.argsort(t)
    t = t[order]
    x = x[order]

    # Remove duplicated timestamps (keep last)
    _, unique_idx = np.unique(t, return_index=True)
    t = t[unique_idx]
    x = x[unique_idx]

    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    x_uniform = np.interp(t_uniform, t, x)
    return t_uniform, x_uniform


def vector_norm(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def ordered_states_from_kmeans(values: np.ndarray, k: int, seed: int = 42):
    """KMeans on 1D, then remap labels by ascending centroid."""
    v = values.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(v)
    centers = km.cluster_centers_.flatten()

    # order cluster ids by center magnitude
    order = np.argsort(centers)
    rank = np.zeros_like(order)
    rank[order] = np.arange(k)  # cluster_id -> ordered_state

    ordered_states = rank[labels]
    ordered_centers = centers[order]  # centers in ordered-state order
    return ordered_states, ordered_centers, labels, centers


def mode_int(arr: np.ndarray) -> int:
    """Fast mode for small integer arrays."""
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def detect_transition(states: np.ndarray, baseline_state: int, n_baseline: int, n_after: int) -> int | None:
    """
    Find first index i where:
      - previous n_baseline samples are baseline_state
      - next n_after samples are any state > baseline_state
    Returns i (start index) or None.
    """
    s = states.astype(int)
    n = len(s)
    if n < (n_baseline + n_after + 1):
        return None

    for i in range(n_baseline, n - n_after):
        if np.all(s[i - n_baseline:i] == baseline_state) and np.all(s[i:i + n_after] > baseline_state):
            return i
    return None


# -----------------------------
# UI inputs
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    acc_file = st.file_uploader("📄 Arquivo de aceleração (Tempo, X, Y, Z)", type=["txt", "csv"], key="acc")
with col2:
    gyro_file = st.file_uploader("📄 Arquivo de giroscópio (Tempo, X, Y, Z)", type=["txt", "csv"], key="gyro")

fs_target = 100.0
fc_acc = 8.0
fc_gyro = 1.5
k_states = 7

n_baseline = 15
n_after = 5
baseline_seconds = 3.0

seed = st.sidebar.number_input("Seed do K-means", min_value=0, max_value=999999, value=42, step=1)

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

    # Time normalization
    t_acc = normalize_time_to_seconds(df_acc["t"].to_numpy())
    t_gyr = normalize_time_to_seconds(df_gyr["t"].to_numpy())

    # Detrend (per axis) on irregular samples
    acc_x = detrend(df_acc["x"].to_numpy(), type="linear")
    acc_y = detrend(df_acc["y"].to_numpy(), type="linear")
    acc_z = detrend(df_acc["z"].to_numpy(), type="linear")

    gyr_x = detrend(df_gyr["x"].to_numpy(), type="linear")
    gyr_y = detrend(df_gyr["y"].to_numpy(), type="linear")
    gyr_z = detrend(df_gyr["z"].to_numpy(), type="linear")

    # Resample each axis to 100 Hz
    t_acc_u, acc_x_u = resample_to_fs(t_acc, acc_x, fs_target)
    _,       acc_y_u = resample_to_fs(t_acc, acc_y, fs_target)
    _,       acc_z_u = resample_to_fs(t_acc, acc_z, fs_target)

    t_gyr_u, gyr_x_u = resample_to_fs(t_gyr, gyr_x, fs_target)
    _,       gyr_y_u = resample_to_fs(t_gyr, gyr_y, fs_target)
    _,       gyr_z_u = resample_to_fs(t_gyr, gyr_z, fs_target)

    # Low-pass filters
    acc_x_f = lowpass_filter(acc_x_u, fs_target, fc_acc)
    acc_y_f = lowpass_filter(acc_y_u, fs_target, fc_acc)
    acc_z_f = lowpass_filter(acc_z_u, fs_target, fc_acc)

    gyr_x_f = lowpass_filter(gyr_x_u, fs_target, fc_gyro)
    gyr_y_f = lowpass_filter(gyr_y_u, fs_target, fc_gyro)
    gyr_z_f = lowpass_filter(gyr_z_u, fs_target, fc_gyro)

    # Norms
    acc_norm = vector_norm(acc_x_f, acc_y_f, acc_z_f)
    gyr_norm = vector_norm(gyr_x_f, gyr_y_f, gyr_z_f)

    # KMeans on gyro norm
    states, ordered_centers, _, _ = ordered_states_from_kmeans(gyr_norm, k=k_states, seed=int(seed))

    # Baseline (first 3 seconds) dominant state
    baseline_mask = t_gyr_u <= baseline_seconds
    if baseline_mask.sum() < max(n_baseline + 1, 10):
        st.warning(
            f"Poucas amostras na baseline de {baseline_seconds}s (tenho {baseline_mask.sum()}). "
            "Vou usar as primeiras 300 amostras como baseline."
        )
        baseline_mask = np.zeros_like(t_gyr_u, dtype=bool)
        baseline_mask[: min(len(baseline_mask), 300)] = True

    baseline_state = mode_int(states[baseline_mask])

    # Start detection
    start_idx = detect_transition(states, baseline_state, n_baseline=n_baseline, n_after=n_after)

    # End detection (retrograde): baseline = last 3 seconds
    t_end0 = t_gyr_u[-1] - baseline_seconds
    end_baseline_mask = t_gyr_u >= t_end0
    if end_baseline_mask.sum() < max(n_baseline + 1, 10):
        end_baseline_mask = np.zeros_like(t_gyr_u, dtype=bool)
        end_baseline_mask[-min(len(end_baseline_mask), 300):] = True

    states_rev = states[::-1]
    # baseline in reversed = first 3 seconds from end => end_baseline_mask reversed
    end_baseline_state = mode_int(states[end_baseline_mask])
    end_idx_rev = detect_transition(states_rev, end_baseline_state, n_baseline=n_baseline, n_after=n_after)
    end_idx = None if end_idx_rev is None else (len(states) - end_idx_rev - 1)

    # Times
    start_t = None if start_idx is None else float(t_gyr_u[start_idx])
    end_t = None if end_idx is None else float(t_gyr_u[end_idx])

    # Sanity check
    if (start_t is not None) and (end_t is not None) and (end_t <= start_t):
        st.warning("Detectei fim <= início. Vou manter as marcações, mas vale revisar parâmetros/ruído.")

    # -----------------------------
    # Outputs
    # -----------------------------
    left, right = st.columns([1, 1])
    with left:
        st.subheader("📌 Segmentação (giroscópio)")
        st.write(f"- Estado dominante na baseline (primeiros {baseline_seconds:.1f}s): **{baseline_state}**")
        st.write(f"- Início detectado: **{start_t:.3f} s**" if start_t is not None else "- Início detectado: **não encontrado**")
        st.write(f"- Fim detectado: **{end_t:.3f} s**" if end_t is not None else "- Fim detectado: **não encontrado**")
        st.caption(f"Regra: {n_baseline} amostras baseline → {n_after} amostras em estado superior (k={k_states}).")

    with right:
        st.subheader("📊 Centros (ordenados) do K-means na norma do giro")
        centers_df = pd.DataFrame({"estado_ordenado": np.arange(k_states), "centroide_norma_giro": ordered_centers})
        st.dataframe(centers_df, use_container_width=True)

    # Plot
    st.subheader("📈 Normas (aceleração e giroscópio) com início/fim")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_gyr_u, gyr_norm, label="||giro|| (filtrado 1.5 Hz)")
    # Accel time may differ; plot on its own time
    ax.plot(t_acc_u, acc_norm, label="||acel|| (filtrado 8 Hz)", alpha=0.8)

    if start_t is not None:
        ax.axvline(start_t, linestyle="--", linewidth=2, label="Início")
    if end_t is not None:
        ax.axvline(end_t, linestyle="--", linewidth=2, label="Fim")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Norma")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Optional: show processed table
    with st.expander("Ver tabela processada (100 Hz)"):
        out = pd.DataFrame({
            "t_s": t_gyr_u,
            "acc_norm": np.interp(t_gyr_u, t_acc_u, acc_norm),  # align accel to gyro time for convenience
            "gyr_norm": gyr_norm,
            "state": states.astype(int),
        })
        st.dataframe(out, use_container_width=True)
