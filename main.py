
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
st.set_page_config(page_title="IMU Markov + Métricas (A1/A2/G1/G2)", layout="wide")
st.title(
    "Análise Fernanda"
)

fs_target = 100.0
fc_acc = 8.0
fc_gyro = 1.5
k_states = 7

# Regra Markov
n_baseline = 15
n_after = 5

# Baselines (definição do usuário)
bs_start_t0 = 2  # baseline início: 2s
bs_start_t1 = 5   # até 5s
bs_end_back0 = 7 # baseline final: fim-4s
bs_end_back1 = 5  # até fim-2s

# Janelas dos picos de aceleração (A1 e A2)
peak_window_seconds = 1.25  # 2 segundos


# -----------------------------
# Helpers
# -----------------------------
def read_imu_file(uploaded_file) -> pd.DataFrame:
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
    t = np.asarray(t_raw, dtype=float)
    t = t - t[0]
    span = np.nanmax(t) - np.nanmin(t)
    if span > 1000.0:
        t = t / 1000.0
    return t


def resample_to_fs(t: np.ndarray, x: np.ndarray, fs: float):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    t_unique, idx = np.unique(t, return_index=True)
    x_unique = x[idx]

    t_u = np.arange(t_unique[0], t_unique[-1], 1.0 / fs)
    x_u = np.interp(t_u, t_unique, x_unique)
    return t_u, x_u


def lowpass_filter(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    if fc <= 0 or fc >= fs / 2:
        raise ValueError("fc deve estar entre 0 e fs/2.")
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def vector_norm(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def ordered_states_from_kmeans(values: np.ndarray, k: int, seed: int):
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


def two_largest_peaks_global(y: np.ndarray, t: np.ndarray, fs: float, min_dist_s: float, prom_mult: float):
    """
    Encontra os DOIS maiores picos locais de y (por amplitude).
    Retorna lista de dicts [{"idx","t","val","prom"}, ...] ordenada por val desc.
    """
    min_distance = max(1, int(min_dist_s * fs))
    local_std = float(np.std(y)) if len(y) > 3 else 0.0
    min_prominence = max(prom_mult * local_std, 1e-9)

    peaks, props = find_peaks(y, distance=min_distance, prominence=min_prominence)
    if len(peaks) == 0:
        return []

    vals = y[peaks]
    order = np.argsort(vals)[::-1]
    top = order[:2]

    out = []
    for j in top:
        p = int(peaks[j])
        out.append(
            {
                "idx": p,
                "t": float(t[p]),
                "val": float(y[p]),
                "prom": float(props["prominences"][j]) if "prominences" in props else np.nan,
            }
        )
    out.sort(key=lambda d: d["val"], reverse=True)
    return out


def window_max(y: np.ndarray, idx0: int, idx1: int):
    """Retorna (idx_max, val_max) do máximo em y[idx0:idx1+1]."""
    idx0 = int(max(0, idx0))
    idx1 = int(min(len(y) - 1, idx1))
    if idx0 > idx1:
        return None, None
    seg = y[idx0:idx1 + 1]
    rel = int(np.argmax(seg))
    idx = idx0 + rel
    return idx, float(y[idx])


# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Parâmetros")

seed = st.sidebar.number_input("Seed (K-means)", min_value=0, max_value=999999, value=42, step=1)

# Picos do giro (global, pega os 2 maiores por amplitude)
gyro_min_peak_distance_s = st.sidebar.slider("Giro: distância mínima entre picos (s)", 0.05, 2.00, 0.50, 0.05)
gyro_prom_mult = st.sidebar.slider("Giro: proeminência mínima (mult. do desvio-padrão do sinal)", 0.0, 3.0, 0.3, 0.1)

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
    t_end_record = float(t_gyr_u[-1])
    tb0 = t_end_record - bs_end_back0
    tb1 = t_end_record - bs_end_back1

    i_be0 = first_index_geq(t_gyr_u, tb0)
    i_be1 = last_index_leq(t_gyr_u, tb1)

    if i_be1 <= i_be0 or (i_be1 - i_be0 + 1) < 20:
        st.warning("Baseline final (fim−4 a fim−2) curta; usando últimos 300 pontos como fallback.")
        baseline_state_end = mode_int(states[max(0, len(states) - 300):])
        i_be1 = len(states) - 1
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

    # Fim do teste (para métricas): se detectou end_idx, usa ele; senão, usa fim do registro
    if end_idx is not None:
        test_end_t = float(t_gyr_u[end_idx])
        test_end_idx = int(end_idx)
    else:
        test_end_t = float(t_gyr_u[-1])
        test_end_idx = len(t_gyr_u) - 1

    # -----------------------------
    # A1 e A2 como MÁXIMO da norma de aceleração nas janelas
    # -----------------------------
    # alinhar aceleração no tempo do gyro
    acc_norm_on_gyr = np.interp(t_gyr_u, t_acc_u, acc_norm)

    # Janela A1: início → início+2s
    A1_idx = A1_t = A1_val = None
    A1_win0_t = A1_win1_t = None

    if start_idx is None:
        st.warning("Não foi possível achar A1: início do teste não foi detectado.")
    else:
        A1_win0_t = float(t_gyr_u[start_idx])
        A1_win1_t = float(A1_win0_t + peak_window_seconds)

        idx0 = int(np.searchsorted(t_gyr_u, A1_win0_t, side="left"))
        idx1 = int(np.searchsorted(t_gyr_u, A1_win1_t, side="right")) - 1

        A1_idx, A1_val = window_max(acc_norm_on_gyr, idx0, idx1)
        if A1_idx is not None:
            A1_t = float(t_gyr_u[A1_idx])

    # Janela A2: (fim−2s) → fim do teste
    A2_idx = A2_t = A2_val = None
    A2_win1_t = test_end_t
    A2_win0_t = max(0.0, float(test_end_t - peak_window_seconds))

    idx0 = int(np.searchsorted(t_gyr_u, A2_win0_t, side="left"))
    idx1 = int(np.searchsorted(t_gyr_u, A2_win1_t, side="right")) - 1
    idx1 = min(idx1, test_end_idx)

    A2_idx, A2_val = window_max(acc_norm_on_gyr, idx0, idx1)
    if A2_idx is not None:
        A2_t = float(t_gyr_u[A2_idx])

    # -----------------------------
    # Picos do giroscópio: dois maiores por amplitude -> rotular por ordem temporal
    # G1 = o que ocorre primeiro; G2 = o que ocorre por último
    # -----------------------------
    gyro_top2_amp = two_largest_peaks_global(
        y=gyr_norm,
        t=t_gyr_u,
        fs=fs_target,
        min_dist_s=gyro_min_peak_distance_s,
        prom_mult=gyro_prom_mult,
    )

    G1 = G2 = None
    if len(gyro_top2_amp) >= 2:
        g_sorted = sorted(gyro_top2_amp[:2], key=lambda d: d["t"])
        G1, G2 = g_sorted[0], g_sorted[1]
    elif len(gyro_top2_amp) == 1:
        G1 = gyro_top2_amp[0]

    # -----------------------------
    # Métricas solicitadas (tabela)
    # -----------------------------
    dur_mov = None
    dur_levantar = None
    dur_ida = None
    dur_volta = None
    dur_sentar = None

    if (start_t is not None) and (test_end_t is not None):
        dur_mov = test_end_t - start_t

    if (A1_t is not None) and (start_t is not None):
        dur_levantar = A1_t - start_t

    if (G1 is not None) and (A1_t is not None):
        dur_ida = G1["t"] - A1_t

    if (G2 is not None) and (G1 is not None):
        dur_volta = G2["t"] - G1["t"]

    if (test_end_t is not None) and (A2_t is not None):
        dur_sentar = test_end_t - A2_t

    metrics = {
        "inicio_teste_s": start_t,
        "fim_teste_s": test_end_t,
        "A1_t_s": A1_t,
        "A2_t_s": A2_t,
        "G1_t_s": (None if G1 is None else G1["t"]),
        "G2_t_s": (None if G2 is None else G2["t"]),
        "Duração do movimento (fim - início) [s]": dur_mov,
        "Duração para levantar (A1 - início) [s]": dur_levantar,
        "Duração da caminhada de ida (G1 - A1) [s]": dur_ida,
        "Duração da caminhada de volta (G2 - G1) [s]": dur_volta,
        "Duração para sentar (fim - A2) [s]": dur_sentar,
        "Amplitude A1 (||acel|| max na janela início)": A1_val,
        "Amplitude A2 (||acel|| max na janela final)": A2_val,
        "Amplitude G1 (||giro|| em G1)": (None if G1 is None else G1["val"]),
        "Amplitude G2 (||giro|| em G2)": (None if G2 is None else G2["val"]),
    }
    metrics_df = pd.DataFrame([metrics])

    st.subheader("📋 Tabela de métricas (A1, A2, G1, G2)")
    st.dataframe(metrics_df, use_container_width=True)

    csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar tabela (CSV)", data=csv_bytes, file_name="metricas_tug.csv", mime="text/csv")

    # -----------------------------
    # Plot
    # -----------------------------
    st.subheader("📈 Normas + marcações (início/fim/A1/A2/G1/G2)")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(12, 5))
    
        ax.plot(t_gyr_u, gyr_norm, '-k')
    
        # Baselines
        #ax.axvspan(bs_start_t0, bs_start_t1, alpha=0.12, label="baseline início (2–5s)")
        #ax.axvspan(t_end_record - bs_end_back0, t_end_record - bs_end_back1, alpha=0.12, label="baseline final (fim−4 a fim−2)")
    
        # Início/Fim
        if start_t is not None:
            ax.axvline(start_t, linestyle="--", linewidth=2, label=f"Início @ {start_t:.3f}s")
        ax.axvline(test_end_t, linestyle="--", linewidth=2, label=f"Fim (teste) @ {test_end_t:.3f}s")
    
        # Janelas de A1 e A2
        #if A1_win0_t is not None and A1_win1_t is not None:
            #ax.axvspan(A1_win0_t, A1_win1_t, alpha=0.10, label="janela A1 (0–2s)")
        #ax.axvspan(A2_win0_t, A2_win1_t, alpha=0.10, label="janela A2 (−2–0s)")
    
        # A1/A2
        #if A1_t is not None:
            #ax.axvline(A1_t, linestyle=":", linewidth=2, label=f"A1 (max) @ {A1_t:.3f}s")
            #ax.plot(A1_t, A1_val, "o", markersize=7)
    
        #if A2_t is not None:
            #ax.axvline(A2_t, linestyle=":", linewidth=2, label=f"A2 (max) @ {A2_t:.3f}s")
            #ax.plot(A2_t, A2_val, "o", markersize=7)
    
        # G1/G2
        if G1 is not None:
            #ax.axvline(G1["t"], linestyle="-.", linewidth=2, label=f"G1 @ {G1['t']:.3f}s")
            ax.plot(G1["t"], G1["val"], "s", markersize=7)
    
        if G2 is not None:
            #ax.axvline(G2["t"], linestyle="-", linewidth=2, label=f"G2 @ {G2['t']:.3f}s")
            ax.plot(G2["t"], G2["val"], "s", markersize=7)
    
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Norma")
        ax.grid(True, alpha=0.3)
        #ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t_gyr_u, acc_norm_on_gyr,'-k')
    
        # Baselines
        #ax.axvspan(bs_start_t0, bs_start_t1, alpha=0.12, label="baseline início (2–5s)")
        #ax.axvspan(t_end_record - bs_end_back0, t_end_record - bs_end_back1, alpha=0.12, label="baseline final (fim−4 a fim−2)")
    
        # Início/Fim
        if start_t is not None:
            ax.axvline(start_t, linestyle="--", linewidth=2, label=f"Início @ {start_t:.3f}s")
        ax.axvline(test_end_t, linestyle="--", linewidth=2, label=f"Fim (teste) @ {test_end_t:.3f}s")
    
        # Janelas de A1 e A2
        #if A1_win0_t is not None and A1_win1_t is not None:
            #ax.axvspan(A1_win0_t, A1_win1_t, alpha=0.10, label="janela A1 (0–2s)")
        #ax.axvspan(A2_win0_t, A2_win1_t, alpha=0.10, label="janela A2 (−2–0s)")
    
        # A1/A2
        if A1_t is not None:
            #ax.axvline(A1_t, linestyle=":", linewidth=2, label=f"A1 (max) @ {A1_t:.3f}s")
            ax.plot(A1_t, A1_val, "o", markersize=7)
    
        if A2_t is not None:
            #ax.axvline(A2_t, linestyle=":", linewidth=2, label=f"A2 (max) @ {A2_t:.3f}s")
            ax.plot(A2_t, A2_val, "o", markersize=7)
    
        # G1/G2
        #if G1 is not None:
            #ax.axvline(G1["t"], linestyle="-.", linewidth=2, label=f"G1 @ {G1['t']:.3f}s")
            #ax.plot(G1["t"], G1["val"], "s", markersize=7)
    
        #if G2 is not None:
            #ax.axvline(G2["t"], linestyle="-.", linewidth=2, label=f"G2 @ {G2['t']:.3f}s")
            #ax.plot(G2["t"], G2["val"], "s", markersize=7)
    
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Norma")
        #ax.grid(True, alpha=0.3)
        #ax.legend()
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
