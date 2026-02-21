# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import detrend, butter, filtfilt, find_peaks
from sklearn.cluster import KMeans


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("IMU TUG – Detecção + A1/A2/G1/G2 + Ajuste Manual")

fs_target = 100.0
fc_acc = 8.0
fc_gyro = 1.5
k_states = 7

peak_window = 1.5  # segundos


# -------------------------------------------------
# FUNÇÕES
# -------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def vector_norm(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


def lowpass(x, fs, fc, order=4):
    b, a = butter(order, fc/(fs/2), btype="low")
    return filtfilt(b, a, x)


def resample_to_fs(t, x, fs):
    order = np.argsort(t)
    t = t[order]
    x = x[order]

    t_unique, idx = np.unique(t, return_index=True)
    x_unique = x[idx]

    t_u = np.arange(t_unique[0], t_unique[-1], 1/fs)
    x_u = np.interp(t_u, t_unique, x_unique)
    return t_u, x_u


def read_file(file):
    df = pd.read_csv(file, sep=None, engine="python", encoding_errors="ignore")
    df.columns = [c.strip().lower() for c in df.columns]

    rename = {}
    for c in df.columns:
        if c in ["tempo","time","t"]: rename[c]="t"
        if c in ["x","ax","gx"]: rename[c]="x"
        if c in ["y","ay","gy"]: rename[c]="y"
        if c in ["z","az","gz"]: rename[c]="z"

    df=df.rename(columns=rename)
    return df[["t","x","y","z"]].dropna()


def two_largest_peaks(y, t):
    peaks, _ = find_peaks(y)
    if len(peaks)==0:
        return []
    vals=y[peaks]
    order=np.argsort(vals)[::-1][:2]
    out=[]
    for i in order:
        p=peaks[i]
        out.append({"idx":p,"t":t[p],"val":y[p]})
    return out


def window_max(y, t, t0, t1):
    idx0=int(np.searchsorted(t,t0,"left"))
    idx1=int(np.searchsorted(t,t1,"right")-1)
    idx0=clamp(idx0,0,len(y)-1)
    idx1=clamp(idx1,0,len(y)-1)
    if idx0>idx1:
        return None,None
    seg=y[idx0:idx1+1]
    rel=np.argmax(seg)
    idx=idx0+rel
    return idx,y[idx]


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
delta_start=st.sidebar.number_input("Δ início (s)",-10.0,10.0,0.0,0.1)
delta_end=st.sidebar.number_input("Δ fim (s)",-10.0,10.0,0.0,0.1)


acc_file=st.file_uploader("Aceleração",type=["txt","csv"])
gyro_file=st.file_uploader("Giroscópio",type=["txt","csv"])

run=st.button("Processar",disabled=(acc_file is None or gyro_file is None))


# -------------------------------------------------
# PIPELINE
# -------------------------------------------------
if run:

    df_acc=read_file(acc_file)
    df_gyr=read_file(gyro_file)

    # tempo
    t_acc=df_acc["t"].values
    t_gyr=df_gyr["t"].values

    t_acc=t_acc-t_acc[0]
    t_gyr=t_gyr-t_gyr[0]

    # detrend
    ax=detrend(df_acc["x"])
    ay=detrend(df_acc["y"])
    az=detrend(df_acc["z"])

    gx=detrend(df_gyr["x"])
    gy=detrend(df_gyr["y"])
    gz=detrend(df_gyr["z"])

    # resample
    t_acc_u,ax_u=resample_to_fs(t_acc,ax,fs_target)
    _,ay_u=resample_to_fs(t_acc,ay,fs_target)
    _,az_u=resample_to_fs(t_acc,az,fs_target)

    t_gyr_u,gx_u=resample_to_fs(t_gyr,gx,fs_target)
    _,gy_u=resample_to_fs(t_gyr,gy,fs_target)
    _,gz_u=resample_to_fs(t_gyr,gz,fs_target)

    # filtros
    ax_f=lowpass(ax_u,fs_target,fc_acc)
    ay_f=lowpass(ay_u,fs_target,fc_acc)
    az_f=lowpass(az_u,fs_target,fc_acc)

    gx_f=lowpass(gx_u,fs_target,fc_gyro)
    gy_f=lowpass(gy_u,fs_target,fc_gyro)
    gz_f=lowpass(gz_u,fs_target,fc_gyro)

    acc_norm=vector_norm(ax_f,ay_f,az_f)
    gyr_norm=vector_norm(gx_f,gy_f,gz_f)

    # início e fim automáticos simples
    start_auto=t_gyr_u[np.argmax(gyr_norm>np.percentile(gyr_norm,75))]
    end_auto=t_gyr_u[-1]

    # aplicar delta
    start=clamp(start_auto+delta_start,0,t_gyr_u[-1])
    end=clamp(end_auto+delta_end,0,t_gyr_u[-1])

    if start>=end:
        st.error("Início >= Fim após ajuste.")
        st.stop()

    # -------------------------------------------------
    # A1 = MAIOR pico entre início e início+2s
    # -------------------------------------------------
    A1_idx,A1_val=window_max(acc_norm,t_gyr_u,start,start+peak_window)
    A1_t=t_gyr_u[A1_idx] if A1_idx is not None else None

    # -------------------------------------------------
    # A2 = MAIOR pico entre fim-2s e fim
    # -------------------------------------------------
    A2_idx,A2_val=window_max(acc_norm,t_gyr_u,end-peak_window,end)
    A2_t=t_gyr_u[A2_idx] if A2_idx is not None else None

    # -------------------------------------------------
    # G1 e G2
    # -------------------------------------------------
    peaks=two_largest_peaks(gyr_norm,t_gyr_u)
    if len(peaks)>=2:
        peaks_sorted=sorted(peaks,key=lambda x:x["t"])
        G1,G2=peaks_sorted[0],peaks_sorted[1]
    elif len(peaks)==1:
        G1=peaks[0]
        G2=None
    else:
        G1=G2=None

    # -------------------------------------------------
    # TABELA
    # -------------------------------------------------
    dur_mov=end-start
    dur_lev=A1_t-start if A1_t else None
    dur_ida=G1["t"]-A1_t if G1 and A1_t else None
    dur_volta=G2["t"]-G1["t"] if G2 and G1 else None
    dur_sentar=end-A2_t if A2_t else None

    df_metrics=pd.DataFrame([{
        "Início (ajustado)":start,
        "Fim (ajustado)":end,
        "A1_t":A1_t,
        "A2_t":A2_t,
        "G1_t":G1["t"] if G1 else None,
        "G2_t":G2["t"] if G2 else None,
        "Duração movimento":dur_mov,
        "Duração levantar":dur_lev,
        "Duração ida":dur_ida,
        "Duração volta":dur_volta,
        "Duração sentar":dur_sentar,
        "Amp A1":A1_val,
        "Amp A2":A2_val,
        "Amp G1":G1["val"] if G1 else None,
        "Amp G2":G2["val"] if G2 else None,
    }])

    st.dataframe(df_metrics)

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        fig,ax=plt.subplots(figsize=(12,5))
        ax.plot(t_gyr_u,gyr_norm,"-k")
        #ax.plot(t_gyr_u,acc_norm,label="||acel||")
    
        ax.axvline(start,color="green",label="Início ajustado")
        ax.axvline(end,color="red",label="Fim ajustado")
    
        #if A1_t: ax.axvline(A1_t,color="blue",linestyle=":")
        #if A2_t: ax.axvline(A2_t,color="purple",linestyle=":")
    
        if G1: ax.axvline(G1["t"],color="black",linestyle="-.")
        if G2: ax.axvline(G2["t"],color="black",linestyle="-.")
    
        #ax.legend()
        #ax.grid()
        st.pyplot(fig)
    with col2:
        fig,ax=plt.subplots(figsize=(12,5))
        #ax.plot(t_gyr_u,gyr_norm,label="||giro||")
        ax.plot(t_acc_u,acc_norm,'-k')
    
        ax.axvline(start,color="green",label="Início ajustado")
        ax.axvline(end,color="red",label="Fim ajustado")
    
        if A1_t: ax.axvline(A1_t,color="blue",linestyle=":")
        if A2_t: ax.axvline(A2_t,color="purple",linestyle=":")
    
        #if G1: ax.axvline(G1["t"],color="black",linestyle="-.")
        #if G2: ax.axvline(G2["t"],color="black",linestyle="-.")
    
        st.pyplot(fig)

    
