# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import detrend, butter, filtfilt, find_peaks
from sklearn.cluster import KMeans


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="IMU Markov + 1º Pico Local Acel", layout="wide")
st.title("📱 IMU: detrend → 100 Hz → filtros → norma → K-means(7) → início/fim → 1º pico local acel (início+2s)")

fs_target = 100.0
fc_acc = 8.0
fc_gyro = 1.5
k_states = 7

# Regra Markov
n_baseline = 15
n_after = 5

# Baselines
bs_start_t0 = 2.0  # baseline início: 2s
bs_start_t1 = 5.0  # até 5s
bs_end_back0 = 4.0  # baseline final: fim-4s
bs_end_back1 = 2.0  # até fim-2s

# Pico acel
peak_after_start_seconds = 2.0  # janela começa em início + 2s
peak_search_window_seconds = 6.0  # opcional: procurar pico local por até N s após início+2s (ajuste na sidebar)


# -----------------------------
# Helpers
# -----------------------------
def read_imu_file(uploaded_file) -> pd.DataFrame:
    """Lê txt/csv com colunas Tempo/Time/T, X, Y, Z (inferindo separador)."""
    df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding_errors="ignore")
    df.columns = [c.strip().lower() for c in df.columns]

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


def normalize_time_to_seconds(t_raw: np.ndarray) -> np.ndarray:
    """Zera tempo no início e converte para segundos se parecer ms."""
    t = np.asarray(t_raw, dtype=float)
    t = t - t[0]
    span = np.nanmax(t) - np.nanmin(t)
    if span > 1000.0:  # heurística de ms
        t = t / 1000.0
    return t


def resample_to_fs(t: np.ndarray, x: np.ndarray, fs: float):
    """Interpolação linear para grade uniforme."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    # remove tempos duplicados
    t_unique, idx = np.unique(t, return_index=True)
    x_unique = x[idx]

    t_u = np.arange(t_unique[0], t_unique[-1], 1.0 / fs)
    x_u = np.interp(t_u, t_unique, x_unique)
    return t_u, x_u


def lowpass_filter(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Butterworth LP (zero-phase)."""
    if fc <= 0 or fc >= fs / 2:
        raise ValueError("fc deve estar entre 0 e fs/2.")
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def vector_norm(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def ordered_states_from_kmeans(values: np.ndarray, k: int, seed: int):
    """KMeans 1D e reordena labels por centroide (baixo→alto) para estados 0..k-1."""
    v = values.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(v)
    centers = km.cluster_centers_.flatten()

    order = np.argsort(centers)
    rank = np.zeros_like(order)
    rank[order] = np.arange(k)

    ordered_states = rank[labels].astype(int)
    ordered_centers = centers[order]
    return ordered_states, ordered_centers


def mode_int(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def detect_transition_from(states: np.ndarray, baseline_state: int, start_idx: int, end_idx: int):
    """
    Transição baseline->superior:
      - n_baseline amostras anteriores == baseline_state
      - n_after amostras seguintes > baseline_state
    Retorna índice global ou None.
    """
    s = states.astype(int)
    n = len(s)

    start_idx = max(start_idx, n_baseline)
    end_idx = min(end_idx, n - n_after)
    if start_idx >= end_idx:
        return None

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
st.sidebar.header("Parâmetros")
seed = st.sidebar.number_input("Seed (K-means)", min_value=0, max_value=999999, value=42, step=1)
limit_peak_to_end = st.sidebar.checkbox("Limitar busca do pico até o fim detectado (end_idx)", value=True)
peak_search_window_seconds = st.sidebar.slider(
    "Janela de busca do 1º pico local (segundos após início+2s)",
    min_value=1.0,
    max_value=15.0,
    value=float(peak_search_window_seconds),
    step=0.5,
)

# Critérios do pico local
min_peak_distance_s = st.sidebar.slider("Distância mínima entre picos (s)", 0.05, 1.00, 0.25, 0.05)
prom_mult = st.sidebar.slider("Proeminência mínima (multiplicador do desvio-padrão local)", 0.0, 3.0, 0.5, 0.1)

c1, c2 = st.columns(2)
with c1:
    acc_file = st.file_uploader("📄 Aceleração (Tempo, X, Y, Z)", type=["txt", "csv"], key="acc")
with c2:
    gyro_file = st.file_uploader("📄 Giroscópio (Tempo, X, Y, Z)", type=["txt", "csv"], key="gyro")

run = st.button("▶️ Processar", type="primary", disabled=(acc_file is None or gyro_file is None))


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

    # Tempo (s)
    t_acc = normalize_time_to_seconds(df_acc["t"].to_numpy())
    t_gyr = normalize_time_to_seconds(df_gyr["t"].to_numpy())

    # Detrend
    acc_x = detrend(df_acc["x"].to_numpy(), type="linear")
    acc_y = detrend(df_acc["y"].to_numpy(), type="linear")
    acc_z = detrend(df_acc["z"].to_numpy(), type="linear")

    gyr_x = detrend(df_gyr["x"].to_numpy(), type="linear")
    gyr_y = detrend(df_gyr["y"].to_numpy(), type="linear")
    gyr_z = detrend(df_gyr["z"].to_numpy(), type="linear")

    # Reamostrar para 100 Hz
    t_acc_u, acc_x_u = resample_to_fs(t_acc, acc_x, fs_target)
    _,       acc_y_u = resample_to_fs(t_acc, acc_y, fs_target)
    _,       acc_z_u = resample_to_fs(t_acc, acc_z, fs_target)

    t_gyr_u, gyr_x_u = resample_to_fs(t_gyr, gyr_x, fs_target)
    _,       gyr_y_u = resample_to_fs(t_gyr, gyr_y, fs_target)
    _,       gyr_z_u = resample_to_fs(t_gyr, gyr_z, fs_target)

    # Filtrar
    acc_x_f = lowpass_filter(acc_x_u, fs_target, fc_acc)
    acc_y_f = lowpass_filter(acc_y_u, fs_target, fc_acc)
    acc_z_f = lowpass_filter(acc_z_u, fs_target, fc_acc)

    gyr_x_f = lowpass_filter(gyr_x_u, fs_target, fc_gyro)
    gyr_y_f = lowpass_filter(gyr_y_u, fs_target, fc_gyro)
    gyr_z_f = lowpass_filter(gyr_z_u, fs_target, fc_gyro)

    # Normas
    acc_norm = vector_norm(acc_x_f, acc_y_f, acc_z_f)
    gyr_norm = vector_norm(gyr_x_f, gyr_y_f, gyr_z_f)

    # Estados do giro (K-means na norma)
    states, centers_ord = ordered_states_from_kmeans(gyr_norm, k_states, int(seed))

    # -----------------------------
    # Início: baseline 2–5s e leitura começa em 2s
    # -----------------------------
    i_bs0 = first_index_geq(t_gyr_u, bs_start_t0)
    i_bs1 = last_index_leq(t_gyr_u, bs_start_t1)

    if i_bs1 <= i_bs0 or (i_bs1 - i_bs0 + 1) < 20:
        st.warning("Baseline inicial (2–5s) curta; usando primeiros 300 pontos como fallback.")
        baseline_state_start = mode_int(states[:min(len(states), 300)])
    else:
        baseline_state_start = mode_int(states[i_bs0:i_bs1 + 1])

    start_idx = detect_transition_from(
        states,
        baseline_state_start,
        start_idx=i_bs0 + n_baseline,
        end_idx=len(states),
    )
    start_t = None if start_idx is None else float(t_gyr_u[start_idx])

    # -----------------------------
    # Fim: baseline (fim−4 a fim−2) e leitura retrógrada começa em (fim−2)
    # -----------------------------
    t_end = float(t_gyr_u[-1])
    tb0 = t_end - bs_end_back0  # fim-4
    tb1 = t_end - bs_end_back1  # fim-2

    i_be0 = first_index_geq(t_gyr_u, tb0)
    i_be1 = last_index_leq(t_gyr_u, tb1)

    if i_be1 <= i_be0 or (i_be1 - i_be0 + 1) < 20:
        st.warning("Baseline final (fim−4 a fim−2) curta; usando últimos 300 pontos como fallback.")
        baseline_state_end = mode_int(states[max(0, len(states) - 300):])
        i_be1 = len(states) - 1  # garante corte válido
    else:
        baseline_state_end = mode_int(states[i_be0:i_be1 + 1])

    states_rev = states[:i_be1 + 1][::-1]
    end_idx_rev = detect_transition_from(
        states_rev,
        baseline_state_end,
        start_idx=0 + n_baseline,
        end_idx=len(states_rev),
    )
    end_idx = None if end_idx_rev is None else (i_be1 - end_idx_rev)
    end_t = None if end_idx is None else float(t_gyr_u[end_idx])

    # -----------------------------
    # 1º pico local da norma da aceleração após (início + 2s)
    # -----------------------------
    # Alinhar aceleração no tempo do gyro
    acc_norm_on_gyr = np.interp(t_gyr_u, t_acc_u, acc_norm)

    peak_idx = peak_t = peak_val = None
    search_start_t = None
    search_end_t = None

    if start_idx is None:
        st.warning("Não foi possível achar pico: início do teste não foi detectado.")
    else:
        search_start_t = float(t_gyr_u[start_idx] + peak_after_start_seconds)
        idx0 = int(np.searchsorted(t_gyr_u, search_start_t, side="left"))
        idx0 = min(idx0, len(acc_norm_on_gyr) - 1)

        # define fim da janela de busca do pico
        search_end_t = float(search_start_t + peak_search_window_seconds)
        idx1_time = int(np.searchsorted(t_gyr_u, search_end_t, side="right")) - 1
        idx1 = min(len(acc_norm_on_gyr) - 1, max(idx0, idx1_time))

        if limit_peak_to_end and (end_idx is not None):
            idx1 = min(idx1, end_idx)

        if idx0 >= idx1:
            st.warning("Janela inválida para buscar o pico local (idx0 >= idx1).")
        else:
            seg = acc_norm_on_gyr[idx0:idx1 + 1]

            # Critérios de pico local
            min_distance = max(1, int(min_peak_distance_s * fs_target))
            local_std = float(np.std(seg[:min(len(seg), int(2.0 * fs_target))])) if len(seg) > 5 else float(np.std(seg))
            min_prominence = max(prom_mult * local_std, 1e-9)

            peaks, props = find_peaks(seg, distance=min_distance, prominence=min_prominence)

            if len(peaks) == 0:
                st.warning("Não encontrei pico local na janela (início+2s → início+2s+janela) com os critérios atuais.")
            else:
                peak_idx = idx0 + int(peaks[0])  # PRIMEIRO pico local
                peak_t = float(t_gyr_u[peak_idx])
                peak_val = float(acc_norm_on_gyr[peak_idx])

    # -----------------------------
    # Painel de resultados
    # -----------------------------
    a, b, c = st.columns(3)
    with a:
        st.subheader("📌 Início")
        st.write(f"Baseline início (2–5s): **estado {baseline_state_start}**")
        st.write(f"Início detectado: **{start_t:.3f}s**" if start_t is not None else "Início detectado: **não encontrado**")
    with b:
        st.subheader("📌 Fim")
        st.write(f"Baseline final (fim−4 a fim−2): **estado {baseline_state_end}**")
        st.write(f"Fim detectado: **{end_t:.3f}s**" if end_t is not None else "Fim detectado: **não encontrado**")
    with c:
        st.subheader("🏔️ 1º pico local acel")
        if peak_t is None or start_t is None:
            st.write("Pico: **não encontrado**")
        else:
            st.write(f"Janela começa em: **{search_start_t:.3f}s** (início+2s)")
            st.write(f"Pico em: **{peak_t:.3f}s** | Δt vs início = **{(peak_t - start_t):.3f}s**")
            st.write(f"Valor (||acel||): **{peak_val:.6f}**")

    st.subheader("📊 Centroides (ordenados) do K-means na norma do giro")
    centers_df = pd.DataFrame({"estado_ordenado": np.arange(k_states), "centroide_norma_giro": centers_ord})
    st.dataframe(centers_df, use_container_width=True)

    # -----------------------------
    # Plot
    # -----------------------------
    st.subheader("📈 Normas com marcações (início, fim, início da busca do pico, 1º pico local)")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t_gyr_u, gyr_norm, label="||giro|| (LP 1.5 Hz)")
    ax.plot(t_acc_u, acc_norm, label="||acel|| (LP 8 Hz)", alpha=0.8)

    # Baselines
    ax.axvspan(bs_start_t0, bs_start_t1, alpha=0.12, label="baseline início (2–5s)")
    ax.axvspan(tb0, tb1, alpha=0.12, label="baseline final (fim−4 a fim−2)")

    # Início/Fim
    if start_t is not None:
        ax.axvline(start_t, linestyle="--", linewidth=2, label=f"Início @ {start_t:.3f}s")
    if end_t is not None:
        ax.axvline(end_t, linestyle="--", linewidth=2, label=f"Fim @ {end_t:.3f}s")

    # Início da busca do pico (início+2s)
    if search_start_t is not None:
        ax.axvline(search_start_t, linestyle="-.", linewidth=2, label=f"Início busca pico @ {search_start_t:.3f}s")

    # Pico local
    if peak_t is not None:
        ax.axvline(peak_t, linestyle=":", linewidth=2, label=f"1º pico local @ {peak_t:.3f}s")
        ax.plot(peak_t, peak_val, "o", markersize=7)

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Norma")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    with st.expander("Ver tabela processada (tempo do gyro, 100 Hz)"):
        out = pd.DataFrame(
            {
                "t_s": t_gyr_u,
                "gyr_norm": gyr_norm,
                "acc_norm": acc_norm_on_gyr,
                "state": states.astype(int),
            }
        )
        st.dataframe(out, use_container_width=True)
