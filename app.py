# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from skimage.restoration import denoise_tv_chambolle

st.set_page_config(page_title="SENSE Energy Analysis", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stSidebarContent"] { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #f4f6f9;
        border: 1px solid #dde2ea;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# CORE FUNCTIONS (EXACT MATH)
# ----------------------------
def remove_spikes_with_time(time, data, threshold=0.3):
    time = np.asarray(time)
    data = np.asarray(data)

    if len(time) != len(data):
        raise ValueError("Time and data arrays must have the same length")

    valid = ~np.isnan(data)
    time_clean = time[valid]
    data_clean = data[valid]

    if len(data_clean) < 3:
        return time_clean, data_clean

    diff_prev = np.abs(data_clean[1:-1] - data_clean[:-2])
    diff_next = np.abs(data_clean[1:-1] - data_clean[2:])
    is_spike = (diff_prev > threshold) | (diff_next > threshold)

    mask = np.ones(len(data_clean), dtype=bool)
    mask[1:-1][is_spike] = False

    return time_clean[mask], data_clean[mask]


def resample_temperature(T_sample, T_ref, time_sample, time_ref):
    return np.interp(time_ref, time_sample, T_sample)


def calculate_energy(T_data, time_data, T_block_data, C, K, T0):
    dt = np.diff(time_data)

    K_int = np.cumsum((T_data[1:] - T_block_data[1:]) * dt)
    K_int = np.concatenate(([0], K_int))

    C_int = T_data - T0

    U = C * C_int + K * K_int

    return U, C_int, K_int


def mask_time_range(time, data, t_min, t_max):
    time = np.asarray(time)
    data = np.asarray(data)
    mask = (time >= t_min) & (time <= t_max)
    return time[mask], data[mask]


def _gaussian_smooth(y, weight):
    """Gaussian-smooth a signal. weight controls width (higher = smoother)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    sigma = max(1.0, weight * n / 10.0)
    radius = min(int(4 * sigma), n // 2 - 1)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    y_padded = np.pad(y, radius, mode='reflect')
    return np.convolve(y_padded, kernel, mode='valid')[:n]


def compute_power(x, y, tv_weight=0.3, gauss_weight=0.1):
    """Differentiate energy → TV denoise → Gaussian smooth."""
    y = np.asarray(y, dtype=float)
    dy = np.gradient(y, x)
    dy_tv = denoise_tv_chambolle(dy, weight=tv_weight)
    return _gaussian_smooth(dy_tv, gauss_weight)


def analyze(data_dict, C, K, zero_range, ch_labels=None, time_range=None,
            gauss_weight=0.1, tv_weight=0.3, return_data=False):
    tzeroing = zero_range

    if 'BlockRef' not in data_dict:
        return {'plots': []}

    block_time = data_dict['BlockRef']['time']
    block_temp = data_dict['BlockRef']['temp']

    if len(block_time) == 0:
        return {'plots': []}

    sumupR = 0
    countR = 0
    for i in range(len(block_time)):
        if block_time[i] > tzeroing[0] and block_time[i] < tzeroing[1]:
            sumupR += block_temp[i]
            countR += 1
    T0avg = sumupR / countR

    plots = []

    for ch_name in ['Ch0', 'Ch1', 'Ch2', 'Ch3']:
        if ch_name not in data_dict:
            continue

        ch_data = data_dict[ch_name]
        if len(ch_data['time']) == 0:
            continue

        ch_time = ch_data['time']
        ch_temp = ch_data['temp']

        ch_time_clean, ch_temp_clean = remove_spikes_with_time(
            ch_time, ch_temp, threshold=0.3
        )

        if len(ch_time_clean) < 2:
            continue

        ch_temp_resampled = resample_temperature(
            ch_temp_clean, block_temp, ch_time_clean, block_time
        )

        sumup = 0
        count = 0
        for i in range(len(block_time)):
            if block_time[i] > tzeroing[0] and block_time[i] < tzeroing[1]:
                sumup += ch_temp_resampled[i] - block_temp[i]
                count += 1
        average = sumup / count
        T_zerod = ch_temp_resampled - average

        # Apply time range mask after zeroing
        if time_range is not None:
            t_min, t_max = time_range
            t_masked, T_zerod_masked = mask_time_range(block_time, T_zerod, t_min, t_max)
            _, block_temp_masked = mask_time_range(block_time, block_temp, t_min, t_max)
        else:
            t_masked, T_zerod_masked = block_time, T_zerod
            block_temp_masked = block_temp

        energy, cap_term, cond_term = calculate_energy(
            T_zerod_masked, t_masked, block_temp_masked, C, K, T0avg
        )

        power = compute_power(t_masked, energy, tv_weight=tv_weight, gauss_weight=gauss_weight)

        label = ch_labels.get(ch_name, ch_name) if ch_labels else ch_name
        plots.append({
            'x': t_masked,
            'y': energy,
            'power': power,
            'label': label,
        })

    results = {'plots': plots}

    if return_data:
        results["data"] = {}

    return results


# ----------------------------
# Helpers
# ----------------------------
# Sensor → internal channel name, in order

def _to_float64(arr: pd.Series) -> np.ndarray:
    return pd.to_numeric(arr, errors="coerce").astype("float64").to_numpy()



def _drop_nans(t, T, label, warnings):
    valid = ~np.isnan(t) & ~np.isnan(T)
    dropped = int(len(t) - valid.sum())
    if dropped:
        warnings.append(f"{label}: dropped {dropped} rows with NaNs.")
    return t[valid], T[valid]


def infer_schema_and_build(df: pd.DataFrame):
    """
    Returns (data_dict, schema, ch_labels, warnings_list)
      data_dict: {"Ch0": {"time":..., "temp":...}, ..., "BlockRef": {...}}
      ch_labels: {"Ch0": "A1", "Ch1": "A2", ...}  — display names
    """
    warnings = []
    cols = [c.strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    # ── Schema A: single time column + Ch0–Ch3 + BlockRef ────────────────────
    time_candidates = ["time", "Time", "t", "seconds", "sec"]
    time_col = None
    for c in df.columns:
        if c in time_candidates or c.lower() in [x.lower() for x in time_candidates]:
            time_col = c
            break

    if time_col and all(ch in df.columns for ch in ["Ch0", "Ch1", "Ch2", "Ch3", "BlockRef"]):
        t = _to_float64(df[time_col])
        data_dict = {
            "Ch0": {"time": t, "temp": _to_float64(df["Ch0"])},
            "Ch1": {"time": t, "temp": _to_float64(df["Ch1"])},
            "Ch2": {"time": t, "temp": _to_float64(df["Ch2"])},
            "Ch3": {"time": t, "temp": _to_float64(df["Ch3"])},
            "BlockRef": {"time": t, "temp": _to_float64(df["BlockRef"])},
        }
        ch_labels = {"Ch0": "Ch0", "Ch1": "Ch1", "Ch2": "Ch2", "Ch3": "Ch3"}
        return data_dict, "standard", ch_labels, warnings

    # ── Per-channel detection (flexible suffixes, any naming convention) ───────
    # Accepted suffixes for time, temperature, and annotation columns
    TIME_SFXS = ["_Timestamp (s)", "_Timestamp", "_Time", "_time", "_timestamp"]
    TEMP_SFXS = ["_Temp (C)", "_Temperature (C)", "_Temp", "_Temperature", "_temp"]
    ANN_SFXS  = ["_Annotations", "_Annotation", "_annotations", "_annotation"]

    def _match_suffix(col, sfx_list):
        for sfx in sfx_list:
            if col.endswith(sfx):
                return sfx
        return None

    # Build groups keyed by prefix
    groups = {}
    for c in cols:
        for sfx_list, key in [(TIME_SFXS, "time_col"), (TEMP_SFXS, "temp_col"), (ANN_SFXS, "ann_col")]:
            sfx = _match_suffix(c, sfx_list)
            if sfx:
                prefix = c[: -len(sfx)]
                groups.setdefault(prefix, {})[key] = c
                break

    # Only keep groups that have both a time and a temp column
    valid = {p: g for p, g in groups.items() if "time_col" in g and "temp_col" in g}

    if len(valid) >= 2:
        # 1. Prefer name-based BlockRef detection ("block" anywhere in prefix)
        block_prefix = next((p for p in valid if "block" in p.lower()), None)

        # 2. Fallback: TagID annotation heuristic (channels with TagID → sensors; without → BlockRef)
        if block_prefix is None:
            tagged = {
                p for p, g in valid.items()
                if g.get("ann_col") and
                df[g["ann_col"]].astype(str).str.contains("TagID:", na=False).any()
            }
            untagged = [p for p in valid if p not in tagged]
            if tagged and untagged:
                block_prefix = untagged[0]

        if block_prefix:
            sensor_prefixes = [p for p in valid if p != block_prefix]
            if sensor_prefixes:
                bg = valid[block_prefix]
                br_t, br_T = _drop_nans(
                    _to_float64(df[bg["time_col"]]),
                    _to_float64(df[bg["temp_col"]]),
                    "BlockRef", warnings,
                )
                data_dict = {"BlockRef": {"time": br_t, "temp": br_T}}
                ch_labels = {}
                ch_names = ["Ch0", "Ch1", "Ch2", "Ch3"]

                for idx, prefix in enumerate(sensor_prefixes[:4]):
                    ch_name = ch_names[idx]
                    g = valid[prefix]
                    ch_t, ch_T = _drop_nans(
                        _to_float64(df[g["time_col"]]),
                        _to_float64(df[g["temp_col"]]),
                        prefix, warnings,
                    )
                    data_dict[ch_name] = {"time": ch_t, "temp": ch_T}
                    ch_labels[ch_name] = prefix

                for ch_name in ch_names[len(sensor_prefixes):]:
                    data_dict[ch_name] = {"time": np.array([]), "temp": np.array([])}

                return data_dict, "sense_multi", ch_labels, warnings

    raise ValueError(
        "Could not detect CSV format. Supported formats:\n"
        "  • Single shared time column with Ch0, Ch1, Ch2, Ch3, BlockRef columns\n"
        "  • Per-channel columns where each channel has a timestamp and temperature column\n"
        "    (e.g. Ch0_Time + Ch0_Temp, BlockRef_Time + BlockRef_Temp).\n"
        "    BlockRef is identified by its name containing 'block', or by having no TagID: entries in its annotation column."
    )


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]

PLOT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#fafafa",
    font=dict(color="#444444", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor="#eeeeee", linecolor="#cccccc", zerolinecolor="#cccccc"),
    yaxis=dict(gridcolor="#eeeeee", linecolor="#cccccc", zerolinecolor="#cccccc"),
    margin=dict(l=60, r=20, t=30, b=50),
    hovermode="x unified",
)

# ----------------------------
# Sidebar (always visible)
# ----------------------------
def _inline(label):
    """Render a right-aligned label for use beside a collapsed number_input."""
    st.markdown(
        f"<p style='margin:0;padding-top:8px;font-size:14px'>{label}</p>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.header("Parameters")

    st.markdown("#### Zeroing window")
    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("Start (s)")
    z0 = _r.number_input("Start (s)", value=0.0, step=10.0, label_visibility="collapsed")
    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("End (s)")
    z1 = _r.number_input("End (s)", value=100.0, step=10.0, label_visibility="collapsed")
    zero_range = (float(z0), float(z1))

    st.divider()

    st.markdown("#### Sample volume")
    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("Volume (mL)")
    V = _r.number_input("Volume (mL)", value=1.0, step=0.5, min_value=0.0, label_visibility="collapsed")

    C_calc = 3.2 + 5.5 * float(V)
    K_calc = 0.025 + 0.007 * float(V)

    _ck_placeholder = st.empty()

    use_override = st.checkbox("Override C / K", value=False)
    if use_override:
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("C (J/K)")
        C = _r.number_input("C (J/K)", value=float(C_calc), step=0.1, label_visibility="collapsed")
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("K (W/K)")
        K = _r.number_input("K (W/K)", value=float(K_calc), step=0.001, format="%.6f", label_visibility="collapsed")
    else:
        C, K = float(C_calc), float(K_calc)

    with _ck_placeholder.container():
        c1, c2 = st.columns(2)
        c1.metric("C (J/K)", f"{C:.3f}")
        c2.metric("K (W/K)", f"{K:.4f}")

    st.divider()

    st.markdown("#### Analysis window")
    use_time_mask = st.checkbox("Limit time range", value=False)
    if use_time_mask:
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("Start (s)")
        t_min = _r.number_input("t_min", value=0.0, step=10.0, label_visibility="collapsed")
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("End (s)")
        t_max = _r.number_input("t_max", value=3000.0, step=10.0, label_visibility="collapsed")
        time_range = (float(t_min), float(t_max))
    else:
        time_range = None

    st.divider()

    st.markdown("#### Power smoothing")
    gauss_weight = st.slider("Gaussian width", min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                             help="Gaussian smoothing width before differentiation. Higher = smoother.")
    tv_weight = st.slider("TV weight", min_value=0.01, max_value=10.0, value=0.3, step=0.05,
                          help="Total Variation regularization strength. Higher = smoother/more piecewise-constant.")

# ----------------------------
# UI
# ----------------------------
st.title("SENSE Energy Analysis")

st.markdown("""
<style>
  [data-testid="stFileUploaderDropzone"] {
      border: 2px dashed #94a3b8 !important;
      border-radius: 10px !important;
      min-height: 80px !important;
      transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
      background-color: #eff6ff !important;
      border-color: #2563eb !important;
      border-width: 3px !important;
      box-shadow: 0 0 0 4px rgba(37,99,235,0.2) !important;
  }
</style>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if not uploaded:
    st.info("Upload a CSV file to get started.")
    st.stop()

df = load_csv(uploaded)
df.columns = df.columns.str.strip()

try:
    data_dict, schema, ch_labels, schema_warnings = infer_schema_and_build(df)
except Exception as e:
    st.error(str(e))
    st.stop()


# Temperature plot (full width)
st.subheader("Temperature traces")
fig1 = go.Figure()

if schema == "standard":
    t = data_dict["BlockRef"]["time"]
    for i, ch_name in enumerate(["Ch0", "Ch1", "Ch2", "Ch3", "BlockRef"]):
        label = ch_labels.get(ch_name, ch_name) if ch_name != "BlockRef" else "BlockRef"
        t_p, temp_p = mask_time_range(t, data_dict[ch_name]["temp"], *time_range) if time_range else (t, data_dict[ch_name]["temp"])
        fig1.add_trace(go.Scatter(
            x=t_p, y=temp_p, name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
        ))
else:
    i = 0
    for ch_name, sensor in ch_labels.items():
        if len(data_dict[ch_name]["time"]) > 0:
            t_ch, temp_ch = data_dict[ch_name]["time"], data_dict[ch_name]["temp"]
            if time_range:
                t_ch, temp_ch = mask_time_range(t_ch, temp_ch, *time_range)
            fig1.add_trace(go.Scatter(
                x=t_ch, y=temp_ch,
                name=sensor, line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            ))
            i += 1
    br_t, br_T = data_dict["BlockRef"]["time"], data_dict["BlockRef"]["temp"]
    if time_range:
        br_t, br_T = mask_time_range(br_t, br_T, *time_range)
    fig1.add_trace(go.Scatter(
        x=br_t, y=br_T,
        name="BlockRef", line=dict(color=PALETTE[i % len(PALETTE)], width=1.8, dash="dash"),
    ))

fig1.update_layout(**PLOT_LAYOUT,
    xaxis_title="Time (s)", yaxis_title="Temperature (°C)", height=350)
st.plotly_chart(fig1, use_container_width=True)

st.divider()


# ----------------------------
# Run analysis
# ----------------------------
results = analyze(data_dict, C, K, zero_range, ch_labels=ch_labels,
                  time_range=time_range, gauss_weight=gauss_weight, tv_weight=tv_weight)

if not results["plots"]:
    st.warning("No channels with sufficient data to analyze.")
    st.stop()

# Power plot (top)
st.subheader("Power traces")
fig3 = go.Figure()

for i, p in enumerate(results["plots"]):
    fig3.add_trace(go.Scatter(
        x=p["x"], y=p["power"], name=p["label"],
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
    ))

fig3.update_layout(**PLOT_LAYOUT,
    xaxis_title="Time (s)", yaxis_title="Power (W)", height=400)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Energy plot
st.subheader("Energy traces")
fig2 = go.Figure()

for i, p in enumerate(results["plots"]):
    fig2.add_trace(go.Scatter(
        x=p["x"], y=p["y"], name=p["label"],
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
    ))

fig2.update_layout(**PLOT_LAYOUT,
    xaxis_title="Time (s)", yaxis_title="Energy (J)", height=400)
st.plotly_chart(fig2, use_container_width=True)

# Final energy metrics
st.markdown("**Final energy values**")
metric_cols = st.columns(len(results["plots"]))
for col, p in zip(metric_cols, results["plots"]):
    col.metric(p["label"], f"{p['y'][-1]:.3f} J")

st.divider()

# Export
out = pd.DataFrame({"Time (s)": results["plots"][0]["x"]})
for p in results["plots"]:
    out[p["label"] + "_Energy (J)"] = p["y"]
    out[p["label"] + "_Power (W)"]  = p["power"]

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results (CSV)",
    data=csv_bytes,
    file_name="energy_traces.csv",
    mime="text/csv",
)
