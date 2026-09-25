# -*- coding: utf-8 -*-
"""
Stock Pulse CEP | Complex Event Processing (tanpa SQL/EPL)
------------------------------------------------------------
Lanjutan dari dashboard "Stock Pulse | Real-Time Analytics".

Bedanya dengan versi sebelumnya:
- Versi sebelumnya  : monitoring 1 variabel (harga) dengan rolling stats,
                       z-score, dan Bollinger Bands -> anomali per TITIK data.
- Versi ini (CEP)   : mengevaluasi KOMBINASI beberapa kondisi/event, yaitu
                       4 pola CEP di bawah ini, semuanya ditulis langsung
                       sebagai fungsi Python/pandas (tanpa EPL/SQL):

    1. Threshold + Durasi   -> breach UCL/LCL bertahan >= n event beruntun
    2. Trend                -> rolling mean naik/turun konsisten >= n event
    3. Sequence+Correlation -> anomali harga diikuti anomali volume
                                (2 sub-stream dikorelasikan lewat timestamp)
    4. Absence              -> tidak ada data baru masuk melebihi toleransi
                                (heartbeat/feed hilang)
"""

from collections import deque
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 1. Konfigurasi Halaman Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stock Pulse CEP | Complex Event Processing",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Stock Pulse CEP | Complex Event Processing")
st.caption(
    "Lanjutan dari Stock Pulse Real-Time Analytics — mendeteksi POLA dari "
    "kombinasi beberapa event (threshold+durasi, trend, sequence+correlation, "
    "absence), bukan cuma anomali satu titik data. Semua ditulis langsung di "
    "Python/pandas, tanpa bahasa query EPL/SQL."
)

# ---------------------------------------------------------
# 2. Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Dashboard")

TICKER = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")
WINDOW_SIZE = st.sidebar.number_input("Window Size (rolling)", min_value=5, max_value=50, value=20)
MAX_BUFFER = st.sidebar.number_input("Max Buffer Size", min_value=20, max_value=500, value=100)
POLL_SECONDS = st.sidebar.number_input("Interval Update (Detik)", min_value=5, max_value=300, value=60)
Z_THRESH = st.sidebar.slider("Ambang Z-Score (anomali titik)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Parameter Pola CEP")
MIN_CONSECUTIVE = st.sidebar.number_input(
    "Threshold: min. event beruntun", min_value=2, max_value=20, value=3,
    help="Pola Threshold+Durasi baru menyala kalau breach UCL/LCL bertahan minimal segini event berturut-turut."
)
TREND_LEN = st.sidebar.number_input(
    "Trend: min. event konsisten", min_value=2, max_value=20, value=3,
    help="Pola Trend baru menyala kalau rolling mean naik/turun konsisten minimal segini event berturut-turut."
)
SEQ_WINDOW = st.sidebar.number_input(
    "Sequence: window korelasi (event)", min_value=2, max_value=30, value=5,
    help="Anomali volume dianggap 'menyusul' anomali harga jika terjadi dalam N event setelahnya."
)
ABSENCE_TOLERANCE = st.sidebar.slider(
    "Absence: toleransi (x interval update)", min_value=1.5, max_value=10.0, value=3.0, step=0.5,
    help="Alert 'data hilang' menyala jika tidak ada data baru selama lebih dari toleransi x interval update."
)

is_running = st.sidebar.toggle("Jalankan Real-Time Update", value=True)

if st.sidebar.button("🔄 Reset / Reload Data"):
    st.session_state.clear()
    st.rerun()

# ---------------------------------------------------------
# 3. Fungsi Utilitas Data
# ---------------------------------------------------------
def fetch_initial_data(ticker, max_buffer):
    """Mengambil data historis awal (harga + volume) untuk mengisi buffer."""
    raw = yf.download(tickers=ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if raw.empty:
        return []

    close = raw["Close"]
    vol = raw["Volume"] if "Volume" in raw.columns else pd.Series(np.nan, index=raw.index)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]

    df_init = pd.DataFrame({
        "timestamp": close.index,
        "price": close.values,
        "volume": vol.values,
    }).dropna(subset=["timestamp", "price"]).drop_duplicates(subset="timestamp")

    return df_init.tail(max_buffer).to_dict("records")


def get_latest_data(ticker):
    """Mengambil 1 data point (harga + volume) menit terbaru."""
    raw = yf.download(tickers=ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if raw.empty:
        return None

    close = raw["Close"]
    vol = raw["Volume"] if "Volume" in raw.columns else pd.Series(np.nan, index=raw.index)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]

    close = close.dropna()
    if close.empty:
        return None

    last_ts = close.index[-1]
    return {
        "timestamp": last_ts,
        "price": float(close.iloc[-1]),
        "volume": float(vol.reindex(close.index).iloc[-1]) if not vol.empty else np.nan,
    }


# ---------------------------------------------------------
# 4. Logika Complex Event Processing (murni Python/pandas)
# ---------------------------------------------------------
def consecutive_true_count(mask: pd.Series) -> pd.Series:
    """Panjang 'streak' True berturut-turut yang berakhir di tiap baris.
    [F,T,T,T,F,T] -> [0,1,2,3,0,1]. Dipakai untuk pola Threshold+Durasi."""
    counts = np.zeros(len(mask), dtype=int)
    running = 0
    for i, v in enumerate(mask.to_numpy()):
        running = running + 1 if v else 0
        counts[i] = running
    return pd.Series(counts, index=mask.index)


def consecutive_same_sign(diff: pd.Series) -> pd.Series:
    """Panjang streak arah yang konsisten (semua naik / semua turun).
    0 atau NaN memutus streak. Dipakai untuk pola Trend."""
    sign = np.sign(diff.fillna(0)).to_numpy()
    counts = np.zeros(len(sign), dtype=int)
    running, prev = 0, 0
    for i, s in enumerate(sign):
        if s != 0 and s == prev:
            running += 1
        elif s != 0:
            running = 1
        else:
            running = 0
        counts[i] = running
        prev = s
    return pd.Series(counts, index=diff.index)


def analyze_buffer(buffer_data, window_size, z_thresh, min_consecutive, trend_len, seq_window):
    """Menghitung rolling statistics + 4 pola CEP di atasnya."""
    df = pd.DataFrame(list(buffer_data))
    if df.empty:
        return df

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = np.nan

    # --- Statistik dasar harga ---
    df["rolling_mean"] = df["price"].rolling(window_size).mean()
    df["rolling_sd"] = df["price"].rolling(window_size).std()
    df["upper_bound"] = df["rolling_mean"] + 2 * df["rolling_sd"]
    df["lower_bound"] = df["rolling_mean"] - 2 * df["rolling_sd"]

    prior_mean = df["price"].shift(1).rolling(window_size).mean()
    prior_sd = df["price"].shift(1).rolling(window_size).std()
    df["z_score"] = (df["price"] - prior_mean) / prior_sd
    df["anomaly"] = df["z_score"].abs() > z_thresh

    # --- Statistik volume (bahan pola sequence/correlation) ---
    vol_prior_mean = df["volume"].shift(1).rolling(window_size).mean()
    vol_prior_sd = df["volume"].shift(1).rolling(window_size).std()
    df["vol_z_score"] = (df["volume"] - vol_prior_mean) / vol_prior_sd
    df["vol_anomaly"] = df["vol_z_score"].abs() > z_thresh

    # POLA 1 — Threshold + Durasi
    df["streak_upper"] = consecutive_true_count(df["price"] > df["upper_bound"])
    df["streak_lower"] = consecutive_true_count(df["price"] < df["lower_bound"])
    df["threshold_pattern"] = (df["streak_upper"] >= min_consecutive) | (df["streak_lower"] >= min_consecutive)
    df["threshold_side"] = np.select(
        [df["streak_upper"] >= min_consecutive, df["streak_lower"] >= min_consecutive],
        ["atas (UCL)", "bawah (LCL)"], default="-"
    )

    # POLA 2 — Trend
    mean_diff = df["rolling_mean"].diff()
    df["trend_streak"] = consecutive_same_sign(mean_diff)
    df["trend_pattern"] = df["trend_streak"] >= trend_len
    df["trend_direction"] = np.select([mean_diff > 0, mean_diff < 0], ["naik", "turun"], default="-")

    # POLA 3 — Sequence + Correlation (anomali harga -> anomali volume, dalam window)
    price_anomaly_recent = (
        df["anomaly"].shift(1).rolling(seq_window, min_periods=1).max().fillna(0).astype(bool)
    )
    df["sequence_pattern"] = df["vol_anomaly"].fillna(False) & price_anomaly_recent

    return df


def compute_absence_alert(last_timestamp, poll_seconds, tolerance):
    """POLA 4 — Absence: tidak ada event baru dalam window yang diharapkan."""
    if last_timestamp is None:
        return False, 0.0
    now = pd.Timestamp.now(tz=last_timestamp.tzinfo) if last_timestamp.tzinfo else pd.Timestamp.now()
    gap = (now - last_timestamp).total_seconds()
    return gap > poll_seconds * tolerance, gap


def build_event_log(df, max_rows=20):
    """Mengumpulkan semua pola yang terdeteksi jadi satu log event."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "pola", "deskripsi", "harga"])

    events = []
    for _, row in df.iterrows():
        if row.get("threshold_pattern"):
            n = int(max(row["streak_upper"], row["streak_lower"]))
            events.append({
                "timestamp": row["timestamp"], "pola": "Threshold+Durasi",
                "deskripsi": f"Harga di {row['threshold_side']} batas selama {n} event berturut-turut",
                "harga": row["price"],
            })
        if row.get("trend_pattern"):
            events.append({
                "timestamp": row["timestamp"], "pola": "Trend",
                "deskripsi": f"Rolling mean {row['trend_direction']} konsisten {int(row['trend_streak'])} periode",
                "harga": row["price"],
            })
        if row.get("sequence_pattern"):
            events.append({
                "timestamp": row["timestamp"], "pola": "Sequence+Correlation",
                "deskripsi": "Anomali volume menyusul anomali harga sebelumnya (dalam window)",
                "harga": row["price"],
            })
    if not events:
        return pd.DataFrame(columns=["timestamp", "pola", "deskripsi", "harga"])

    log = pd.DataFrame(events).sort_values("timestamp", ascending=False)
    return log.head(max_rows).reset_index(drop=True)


# ---------------------------------------------------------
# 5. Pengelolaan Buffer Data (Session State)
# ---------------------------------------------------------
if "buffer" not in st.session_state:
    with st.spinner("Mengunduh data awal..."):
        initial_records = fetch_initial_data(TICKER, MAX_BUFFER)
        st.session_state.buffer = deque(initial_records, maxlen=MAX_BUFFER)

latest_point = get_latest_data(TICKER)
if latest_point and len(st.session_state.buffer) > 0:
    if st.session_state.buffer[-1]["timestamp"] != latest_point["timestamp"]:
        st.session_state.buffer.append(latest_point)

# ---------------------------------------------------------
# 6. Analisis
# ---------------------------------------------------------
df_analyzed = analyze_buffer(
    st.session_state.buffer, WINDOW_SIZE, Z_THRESH, MIN_CONSECUTIVE, TREND_LEN, SEQ_WINDOW
)

# ---------------------------------------------------------
# 7. Tampilan Dashboard
# ---------------------------------------------------------
if not df_analyzed.empty:
    latest = df_analyzed.iloc[-1]
    is_absent, gap_seconds = compute_absence_alert(latest["timestamp"], POLL_SECONDS, ABSENCE_TOLERANCE)

    if is_absent:
        st.error(
            f"🛑 **Pola Absence terdeteksi** — tidak ada data baru selama "
            f"{gap_seconds:,.0f} detik (toleransi: {POLL_SECONDS * ABSENCE_TOLERANCE:,.0f} detik). "
            "Kemungkinan feed data/API bermasalah."
        )

    st.markdown("#### 📊 Statistik Terkini")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Harga Terakhir", f"${latest['price']:,.2f}")
    c2.metric("Rolling Mean", f"${latest['rolling_mean']:,.2f}" if pd.notna(latest['rolling_mean']) else "-")
    c3.metric("Rolling SD", f"{latest['rolling_sd']:.4f}" if pd.notna(latest['rolling_sd']) else "-")
    c4.metric("Z-Score Harga", f"{latest['z_score']:.2f}" if pd.notna(latest['z_score']) else "-")
    c5.metric("Z-Score Volume", f"{latest['vol_z_score']:.2f}" if pd.notna(latest['vol_z_score']) else "-")

    st.markdown("#### 🧩 Status Pola CEP")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.write("**Threshold + Durasi**")
        if latest["threshold_pattern"]:
            n = int(max(latest["streak_upper"], latest["streak_lower"]))
            st.error(f"⚠️ Breach {latest['threshold_side']} — {n}x beruntun")
        else:
            st.success("✅ Normal")
    with p2:
        st.write("**Trend**")
        if latest["trend_pattern"]:
            st.warning(f"📈 {latest['trend_direction'].capitalize()} — {int(latest['trend_streak'])}x beruntun")
        else:
            st.success("✅ Tidak ada trend kuat")
    with p3:
        st.write("**Sequence + Correlation**")
        if latest["sequence_pattern"]:
            st.error("🔗 Anomali harga → volume terdeteksi")
        else:
            st.success("✅ Tidak ada pola sequence")
    with p4:
        st.write("**Absence**")
        if is_absent:
            st.error(f"🛑 Data hilang {gap_seconds:,.0f}s")
        else:
            st.success(f"✅ Update {gap_seconds:,.0f}s lalu")

    st.markdown("---")

    # -----------------------------------------------------
    # Grafik: Harga (+ pola) di atas, Volume (+ sequence) di bawah
    # -----------------------------------------------------
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.06, subplot_titles=(f"Pergerakan Harga: {TICKER}", "Volume")
    )

    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["price"],
                              mode="lines", name="Price", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["rolling_mean"],
                              mode="lines", name="Rolling Mean", line=dict(color="#ff7f0e", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["upper_bound"],
                              mode="lines", name="Upper Bound", line=dict(dash="dash", color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_analyzed["timestamp"], y=df_analyzed["lower_bound"],
                              mode="lines", name="Lower Bound", line=dict(dash="dash", color="gray")), row=1, col=1)

    point_anomalies = df_analyzed[df_analyzed["anomaly"]]
    if not point_anomalies.empty:
        fig.add_trace(go.Scatter(x=point_anomalies["timestamp"], y=point_anomalies["price"],
                                  mode="markers", name="Anomali Titik (Z-Score)",
                                  marker=dict(color="red", size=8, symbol="x")), row=1, col=1)

    threshold_events = df_analyzed[df_analyzed["threshold_pattern"]]
    if not threshold_events.empty:
        fig.add_trace(go.Scatter(x=threshold_events["timestamp"], y=threshold_events["price"],
                                  mode="markers", name="Pola Threshold+Durasi",
                                  marker=dict(color="darkred", size=13, symbol="diamond",
                                              line=dict(color="white", width=1))), row=1, col=1)

    trend_events = df_analyzed[df_analyzed["trend_pattern"]]
    if not trend_events.empty:
        fig.add_trace(go.Scatter(x=trend_events["timestamp"], y=trend_events["price"],
                                  mode="markers", name="Pola Trend",
                                  marker=dict(color="#9467bd", size=8, symbol="triangle-up")), row=1, col=1)

    seq_events = df_analyzed[df_analyzed["sequence_pattern"]]
    if not seq_events.empty:
        fig.add_trace(go.Scatter(x=seq_events["timestamp"], y=seq_events["price"],
                                  mode="markers", name="Pola Sequence+Correlation",
                                  marker=dict(color="black", size=13, symbol="star")), row=1, col=1)

    fig.add_trace(go.Bar(x=df_analyzed["timestamp"], y=df_analyzed["volume"],
                          name="Volume", marker=dict(color="#a3c4dc")), row=2, col=1)
    if not seq_events.empty:
        fig.add_trace(go.Scatter(x=seq_events["timestamp"], y=seq_events["volume"],
                                  mode="markers", name="Volume anomali (sequence)",
                                  marker=dict(color="black", size=11, symbol="star")), row=2, col=1)

    fig.update_layout(template="plotly_white", height=620, margin=dict(l=20, r=20, t=50, b=20),
                       legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0))
    fig.update_yaxes(title_text="Harga (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Waktu", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # Event Pattern Log (pengganti output query EPL, versi Python biasa)
    # -----------------------------------------------------
    st.markdown("#### 📜 Event Pattern Log")
    st.caption("Daftar semua kejadian pola CEP yang terdeteksi pada buffer saat ini (terbaru di atas).")
    event_log = build_event_log(df_analyzed, max_rows=20)
    if event_log.empty:
        st.info("Belum ada pola CEP yang terdeteksi pada buffer saat ini.")
    else:
        st.dataframe(event_log.round({"harga": 2}), use_container_width=True, hide_index=True)

    with st.expander("📄 Lihat Data Mentah Terakhir (10 Baris)", expanded=False):
        st.dataframe(df_analyzed.tail(10).round(4), use_container_width=True)

else:
    st.warning("Data belum tersedia atau gagal mengunduh data.")

# ---------------------------------------------------------
# 8. Loop Auto-Refresh
# ---------------------------------------------------------
if is_running:
    time.sleep(POLL_SECONDS)
    st.rerun()
