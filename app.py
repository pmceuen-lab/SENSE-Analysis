# app.py
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="SENSE Analysis", layout="wide")

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

    diff_prev = data_clean[1:-1] - data_clean[:-2]   # signed
    diff_next = data_clean[2:]   - data_clean[1:-1]  # signed
    # True spike: large jump that immediately reverses (opposite sign differences)
    is_spike = (np.abs(diff_prev) > threshold) & (np.abs(diff_next) > threshold) & (diff_prev * diff_next < 0)

    mask = np.ones(len(data_clean), dtype=bool)
    mask[1:-1][is_spike] = False

    return time_clean[mask], data_clean[mask]


def resample_temperature(T_sample, T_ref, time_sample, time_ref):
    return np.interp(time_ref, time_sample, T_sample)


def calculate_energy(T_data, time_data, T_block_data, C, K, T0, alpha_env=0.0, Te=23.0):
    dt = np.diff(time_data)
    Kb = (1.0 - alpha_env) * K
    Ke = alpha_env * K

    K_int = np.cumsum((T_data[1:] - T_block_data[1:]) * dt)
    K_int = np.concatenate(([0], K_int))

    Ke_int = np.cumsum((T_data[1:] - T0) * dt)
    Ke_int = np.concatenate(([0], Ke_int))

    C_int = T_data - T0

    U = C * C_int + Kb * K_int + Ke * Ke_int

    return U, C_int, K_int


def mask_time_range(time, data, t_min, t_max):
    time = np.asarray(time)
    data = np.asarray(data)
    mask = (time >= t_min) & (time <= t_max)
    return time[mask], data[mask]


def _drop_close_points(t, *arrays, min_dt=0.01):
    """Remove points where the gap from the previous kept point is < min_dt seconds."""
    t = np.asarray(t, dtype=float)
    keep = np.ones(len(t), dtype=bool)
    last = t[0]
    for i in range(1, len(t)):
        if t[i] - last < min_dt:
            keep[i] = False
        else:
            last = t[i]
    return (t[keep],) + tuple(np.asarray(a)[keep] for a in arrays)


def compute_power(x, y, tv_weight=0.3, gauss_sigma=None):
    """Differentiate energy → TV denoise → optional Gaussian smoothing.
    Interpolates onto a uniform grid before differentiating to avoid gradient
    spikes from non-uniform channel timestamps, then resamples back."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = float(np.median(np.diff(x)))
    if dt <= 0:
        return np.zeros_like(y)
    # Uniform grid over the same time span
    x_uni = np.arange(x[0], x[-1] + dt * 0.5, dt)
    y_uni = np.interp(x_uni, x, y)
    # Differentiate on uniform grid — no non-uniform spacing artefacts
    dy_uni = np.gradient(y_uni, dt)
    result_uni = denoise_tv_chambolle(dy_uni, weight=tv_weight)
    if gauss_sigma is not None and dt > 0:
        result_uni = gaussian_filter1d(result_uni, sigma=gauss_sigma / dt)
    # Resample back to original (native) time points
    return np.interp(x, x_uni, result_uni)


@st.cache_data(show_spinner="Analysing…")
def analyze(data_dict, C, K, zero_range, ch_labels=None, time_range=None,
            tv_weight=0.3, final_zero_range=None, selected_channels=None,
            gauss_sigma=None, temp_sigma=None, alpha_env=0.0, Te=23.0,
            ref_channel=None):
    tzeroing = zero_range

    if 'BlockRef' not in data_dict:
        return {'plots': []}

    block_time = data_dict['BlockRef']['time']
    block_temp = data_dict['BlockRef']['temp']

    if len(block_time) == 0:
        return {'plots': []}

    # Clean block temp — glitches propagate to ALL channels via interpolation.
    # Block changes very slowly so even a tiny sign-reversing jump is an artifact.
    block_time, block_temp = remove_spikes_with_time(block_time, block_temp, threshold=0.05)
    # Also remove anomalously close block timestamps — near-duplicate readings
    # create a near-vertical segment in the interpolated block that hits every channel.
    (block_time, block_temp) = _drop_close_points(block_time, block_temp, min_dt=0.1)

    sumupR = 0
    countR = 0
    for i in range(len(block_time)):
        if block_time[i] > tzeroing[0] and block_time[i] < tzeroing[1]:
            sumupR += block_temp[i]
            countR += 1
    if countR == 0:
        raise ValueError(
            f"Zeroing window ({tzeroing[0]:.1f}–{tzeroing[1]:.1f} s) contains no data points. "
            "Adjust the Zero start/end values."
        )
    T0avg = sumupR / countR

    # Pre-compute reference channel resampled to block time (if requested)
    _ref_temp_resampled = None
    if ref_channel and ref_channel in data_dict:
        _rd = data_dict[ref_channel]
        if ref_channel == 'BlockRef':
            _ref_temp_resampled = block_temp.copy()
        elif len(_rd['time']) >= 2:
            _rt_clean, _rT_clean = remove_spikes_with_time(_rd['time'], _rd['temp'], threshold=0.3)
            if len(_rt_clean) >= 2:
                _ref_temp_resampled = resample_temperature(
                    _rT_clean, block_temp, _rt_clean, block_time
                )
                if temp_sigma is not None and len(block_time) > 1:
                    _dt = float(block_time[1] - block_time[0])
                    _ref_temp_resampled = gaussian_filter1d(_ref_temp_resampled, sigma=temp_sigma / _dt)

    plots = []

    all_ch = [k for k in data_dict if k != 'BlockRef']
    ch_to_process = [c for c in all_ch if selected_channels is None or c in selected_channels]

    for ch_name in ch_to_process:
        ch_data = data_dict[ch_name]
        if len(ch_data['time']) == 0:
            continue

        ch_time = ch_data['time']
        ch_temp = ch_data['temp']

        # Strip NaNs, then remove only extreme isolated glitches (5°C threshold
        # with sign-reversal required — fast legitimate risers are preserved).
        ch_time_clean, ch_temp_clean = remove_spikes_with_time(
            ch_time, ch_temp, threshold=5.0
        )

        # Sort by time — np.interp requires monotonically increasing x
        _sort_idx = np.argsort(ch_time_clean)
        ch_time_clean = ch_time_clean[_sort_idx]
        ch_temp_clean = ch_temp_clean[_sort_idx]

        if len(ch_time_clean) < 2:
            continue

        # Interpolate block onto channel's native time grid — block is slow so
        # this loses little, while interpolating the channel onto the block grid
        # can distort fast transients.
        block_temp_on_ch = np.interp(ch_time_clean, block_time, block_temp)

        if temp_sigma is not None:
            _dt_ch = float(np.median(np.diff(ch_time_clean)))
            if _dt_ch > 0:
                ch_temp_clean = gaussian_filter1d(ch_temp_clean, sigma=temp_sigma / _dt_ch)

        # Ref subtraction: interpolate pre-computed ref (on block_time) to channel grid
        _using_ref = _ref_temp_resampled is not None
        if _using_ref:
            _ref_on_ch = np.interp(ch_time_clean, block_time, _ref_temp_resampled)
            ch_temp_working = ch_temp_clean - _ref_on_ch
        else:
            ch_temp_working = ch_temp_clean

        # Zeroing on channel's native time axis
        sumup = 0
        count = 0
        for i in range(len(ch_time_clean)):
            if ch_time_clean[i] > tzeroing[0] and ch_time_clean[i] < tzeroing[1]:
                sumup += ch_temp_working[i] if _using_ref else ch_temp_working[i] - block_temp_on_ch[i]
                count += 1
        average = sumup / count

        if final_zero_range is not None:
            sumup2 = 0
            count2 = 0
            for i in range(len(ch_time_clean)):
                if ch_time_clean[i] > final_zero_range[0] and ch_time_clean[i] < final_zero_range[1]:
                    sumup2 += ch_temp_working[i] if _using_ref else ch_temp_working[i] - block_temp_on_ch[i]
                    count2 += 1
            if count2 == 0:
                raise ValueError(
                    f"Final zeroing window ({final_zero_range[0]:.1f}–{final_zero_range[1]:.1f} s) "
                    "contains no data points. Adjust the Final zero start/end values."
                )
            average_final = sumup2 / count2
            t_mid1 = (tzeroing[0] + tzeroing[1]) / 2
            t_mid2 = (final_zero_range[0] + final_zero_range[1]) / 2
            correction = average + (average_final - average) * (ch_time_clean - t_mid1) / (t_mid2 - t_mid1)
            T_zerod = ch_temp_working - correction
        else:
            T_zerod = ch_temp_working - average

        # Apply time range mask after zeroing
        if time_range is not None:
            t_min, t_max = time_range
            t_masked, T_zerod_masked = mask_time_range(ch_time_clean, T_zerod, t_min, t_max)
            _, block_temp_masked = mask_time_range(ch_time_clean, block_temp_on_ch, t_min, t_max)
        else:
            t_masked, T_zerod_masked = ch_time_clean, T_zerod
            block_temp_masked = block_temp_on_ch

        # Drop points closer than 0.3 s to avoid gradient spikes
        t_masked, T_zerod_masked, block_temp_masked = _drop_close_points(
            t_masked, T_zerod_masked, block_temp_masked
        )

        # When using ref subtraction, pass zeros for T_block so loss term uses
        # the differential directly: K*∫ΔT dt rather than K*∫(T-T_block) dt
        _T_block_for_energy = np.zeros_like(block_temp_masked) if _using_ref else block_temp_masked
        _T0_for_energy = 0.0 if _using_ref else T0avg
        energy, cap_term, cond_term = calculate_energy(
            T_zerod_masked, t_masked, _T_block_for_energy, C, K, _T0_for_energy,
            alpha_env=alpha_env, Te=Te,
        )

        power = compute_power(t_masked, energy, tv_weight=tv_weight,
                              gauss_sigma=gauss_sigma)

        label = ch_labels.get(ch_name, ch_name) if ch_labels else ch_name
        plots.append({
            'x': t_masked,
            'y': energy,
            'power': power,
            'temp': T_zerod_masked,
            'block_temp': block_temp_masked,
            'label': label,
        })

    results = {'plots': plots}

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
    t_out, T_out = t[valid], T[valid]
    # Ensure monotonically increasing timestamps for np.interp
    sort_idx = np.argsort(t_out)
    return t_out[sort_idx], T_out[sort_idx]


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

    if time_col and "BlockRef" in df.columns:
        # Shared time column: any remaining numeric column is a sensor (up to 24)
        sensor_cols = [
            c for c in df.columns
            if c not in [time_col, "BlockRef"]
            and pd.api.types.is_numeric_dtype(df[c])
        ][:24]
        t = _to_float64(df[time_col])
        data_dict = {"BlockRef": {"time": t, "temp": _to_float64(df["BlockRef"])}}
        ch_labels = {}
        for idx, col in enumerate(sensor_cols):
            ch_name = f"Ch{idx}"
            data_dict[ch_name] = {"time": t, "temp": _to_float64(df[col])}
            ch_labels[ch_name] = col
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
        block_prefixes = [p for p in valid if "block" in p.lower()]
        block_prefix = block_prefixes[0] if block_prefixes else None

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

        # 3. Fallback: all channels tagged, none qualifies as BlockRef —
        #    synthesise a flat-zero reference using the first channel's time axis.
        #    The zeroing step absorbs the absolute temperature offset, so
        #    T_zerod ≈ ΔT_reaction and the K-integral (T_zerod − 0) remains correct.
        if block_prefix is None:
            sensor_prefixes = sorted(valid.keys())
            first_g = valid[sensor_prefixes[0]]
            br_t = _to_float64(df[first_g["time_col"]])
            br_t = br_t[~np.isnan(br_t)]
            data_dict = {"BlockRef": {"time": br_t, "temp": np.zeros(len(br_t))}}
            ch_labels = {}
            for idx, prefix in enumerate(sensor_prefixes[:24]):
                ch_name = f"Ch{idx}"
                g = valid[prefix]
                ch_t, ch_T = _drop_nans(
                    _to_float64(df[g["time_col"]]),
                    _to_float64(df[g["temp_col"]]),
                    prefix, warnings,
                )
                data_dict[ch_name] = {"time": ch_t, "temp": ch_T}
                ch_labels[ch_name] = prefix
            warnings.append(
                "No BlockRef channel detected — all channels loaded as sensors. "
                "A synthetic zero reference is used; analysis results are still valid."
            )
            return data_dict, "sense_multi", ch_labels, warnings

        if block_prefix:
            sensor_prefixes = [p for p in valid if p not in block_prefixes]
            if sensor_prefixes:
                # Build BlockRef: average all block channels onto the first one's time axis
                bg = valid[block_prefix]
                br_t, br_T = _drop_nans(
                    _to_float64(df[bg["time_col"]]),
                    _to_float64(df[bg["temp_col"]]),
                    "BlockRef", warnings,
                )
                if len(block_prefixes) > 1:
                    extra_Ts = []
                    for bp in block_prefixes[1:]:
                        _bt, _bT = _drop_nans(
                            _to_float64(df[valid[bp]["time_col"]]),
                            _to_float64(df[valid[bp]["temp_col"]]),
                            bp, warnings,
                        )
                        if len(_bt) >= 2:
                            extra_Ts.append(np.interp(br_t, _bt, _bT))
                    if extra_Ts:
                        br_T = np.mean([br_T] + extra_Ts, axis=0)
                    warnings.append(
                        f"{len(block_prefixes)} BlockRef channels detected "
                        f"({', '.join(block_prefixes)}) — temperatures averaged."
                    )
                data_dict = {"BlockRef": {"time": br_t, "temp": br_T}}
                ch_labels = {}

                for idx, prefix in enumerate(sensor_prefixes[:24]):
                    ch_name = f"Ch{idx}"
                    g = valid[prefix]
                    ch_t, ch_T = _drop_nans(
                        _to_float64(df[g["time_col"]]),
                        _to_float64(df[g["temp_col"]]),
                        prefix, warnings,
                    )
                    data_dict[ch_name] = {"time": ch_t, "temp": ch_T}
                    ch_labels[ch_name] = prefix

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


PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#be185d", "#15803d", "#b45309", "#6d28d9",
    "#f97316", "#84cc16", "#06b6d4", "#a855f7", "#ec4899",
    "#22c55e", "#eab308", "#3b82f6", "#ef4444", "#8b5cf6",
    "#14b8a6", "#f43f5e", "#64748b", "#0ea5e9",
]

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
# Heat-table helper
# ----------------------------
_HEAT_TABLE_CSS = """
<style>
.ht-wrap table {
    border-collapse: collapse;
    width: 100%;
    font-family: sans-serif;
}
.ht-wrap th, .ht-wrap td {
    border: 1.5px solid #aaa;
    padding: 6px 16px;
    text-align: center;
    font-size: 14px;
}
.ht-wrap th {
    background: #f0f2f6;
    font-size: 15px;
    font-weight: 700;
}
.ht-wrap td {
    font-size: 14px;
}
</style>
"""

def _render_heat_table(plots):
    """Display final heat values as a row × column well-plate table."""
    _rre = re.compile(r'^([A-Za-z]+)(\d+)')
    _grid, _rows, _cols, _other = {}, set(), set(), []
    for _p in plots:
        _m = _rre.match(str(_p["label"]))
        if _m:
            _r, _c = _m.group(1).upper(), int(_m.group(2))
            _rows.add(_r); _cols.add(_c)
            _grid[(_r, _c)] = _p["y"][-1]
        else:
            _other.append(_p)

    st.markdown("**Final heat values (J)**")
    if _rows:
        _df = pd.DataFrame(index=sorted(_rows), columns=sorted(_cols), dtype=float)
        _df.index.name = None
        for (_r, _c), _v in _grid.items():
            _df.loc[_r, _c] = _v
        _html = _df.style.format("{:.1f}", na_rep="—").to_html()
        st.markdown(_HEAT_TABLE_CSS + f"<div class='ht-wrap'>{_html}</div>",
                    unsafe_allow_html=True)
        if _other:
            _oc = st.columns(len(_other))
            for _col, _p in zip(_oc, _other):
                _col.metric(_p["label"], f"{_p['y'][-1]:.1f} J")
    else:
        _mc = st.columns(len(plots))
        for _col, _p in zip(_mc, plots):
            _col.metric(_p["label"], f"{_p['y'][-1]:.1f} J")


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
    _plot_view = st.radio(
        "Plot view",
        options=["All-in-one", "Array", "Row", "Column"],
        index=1,
        horizontal=False,
        help="All-in-one: all channels overlaid on one graph.\n"
             "Array: each channel on its own plot in a well-plate grid.\n"
             "Row: one graph per row letter (A, B, C, D).\n"
             "Column: one graph per column number (1–6).",
    )
    row_plot_mode    = (_plot_view == "Row")
    column_plot_mode = (_plot_view == "Column")
    array_plot_mode  = (_plot_view == "Array")
    _use_downsample = st.checkbox("Downsample traces", value=True,
        help="LTTB downsampling reduces browser load on large datasets. "
             "Disable if you see artifacts in fast transients.")
    st.divider()
    st.markdown("#### Reference channel")
    _use_ref = st.toggle("Subtract reference", value=False,
                         help="Subtract a chosen channel's temperature from all other channels before analysis.")
    if _use_ref and st.session_state.get("_ch_keys"):
        _ck = st.session_state["_ch_keys"]
        def _ref_sort_key(k):
            _m = re.match(r'^([A-Za-z]+)(\d+)$', k)
            if _m:
                return (0, _m.group(1).upper(), int(_m.group(2)))
            _m2 = re.match(r'^(\D+?)(\d+)$', k)
            if _m2:
                return (1, _m2.group(1).upper(), int(_m2.group(2)))
            return (2, k, 0)
        _ref_options = sorted(
            [k for k in _ck if k != "BlockRef"],
            key=_ref_sort_key
        ) + ["BlockRef"]
        _ref_disp = st.session_state.get("_ch_labels", {})
        ref_channel = st.selectbox(
            "Reference channel",
            options=_ref_options,
            format_func=lambda k: _ref_disp.get(k, k),
            label_visibility="collapsed",
        )
    else:
        ref_channel = None
    st.divider()
    st.markdown("#### Temperature smoothing")
    _temp_smooth = st.toggle("Smooth T(t)", value=False)
    if _temp_smooth:
        temp_sigma = st.slider("σ (s)", min_value=0.5, max_value=30.0, value=2.0, step=0.5,
                               help="Gaussian smoothing applied to raw temperature before analysis.")
    else:
        temp_sigma = None
    st.divider()
    st.header("Parameters")

    st.markdown("#### Sample parameters")
    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("Volume (mL)")
    V = _r.number_input("Volume (mL)", value=1.0, step=0.5, min_value=0.0, label_visibility="collapsed")
    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("C′ (J/mL·K)")
    C_prime = _r.number_input("C_prime", value=4.18, step=0.1, min_value=0.0, label_visibility="collapsed",
                               help="Heat capacity of the solvent (e.g. water ≈ 4.18 J/mL·K)")

    _l, _r = st.columns([1.4, 1.6])
    with _l:
        st.markdown(
            "<p style='margin:0;padding-top:8px;font-size:14px'>α(env) "
            "<span title='Environment loss fraction: α = Ke / (Ke + Kb)."
            " K is split into Kb = (1−α)·K (loss to block) and Ke = α·K (loss to environment)."
            " Adds Ke·∫(T − T₀) dt to the energy (zeroed at baseline)."
            " Default α = 0 disables this correction.' "
            "style='cursor:help;color:#fff;background:#aab;border-radius:50%;"
            "width:15px;height:15px;display:inline-flex;align-items:center;"
            "justify-content:center;font-size:10px;vertical-align:middle;"
            "margin-left:2px;flex-shrink:0'>?</span></p>",
            unsafe_allow_html=True,
        )
    alpha_env = _r.number_input(
        "alpha_env", value=0.0, step=0.01, min_value=0.0, max_value=1.0,
        label_visibility="collapsed",
    )

    C_calc = 3.2 + (float(C_prime) + 1.3) * float(V)
    K_calc = 0.025 + 0.007 * float(V)
    tau_calc = C_calc / K_calc

    _ck_placeholder = st.empty()

    use_override = st.checkbox("Override C / τ", value=False)
    if use_override:
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("C (J/K)")
        C = _r.number_input("C (J/K)", value=float(C_calc), step=0.1, label_visibility="collapsed")
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("τ (s)")
        tau = _r.number_input("tau (s)", value=float(tau_calc), step=1.0, min_value=0.1, label_visibility="collapsed")
        K = C / tau
    else:
        C, K = float(C_calc), float(K_calc)
        tau = tau_calc

    with _ck_placeholder.container():
        c1, c2 = st.columns(2)
        c1.metric("C (J/K)", f"{C:.3f}",
                  help="C = 3.2 + (C′ + 1.3) × V")
        c2.metric("τ (s)", f"{tau:.1f}",
                  help="τ = C / K,  where K = 0.025 + 0.007 × V")

    _t_data_min = st.session_state.get("t_min_data", 0.0)
    _t_data_max = st.session_state.get("t_max_data", 3000.0)

    st.divider()
    st.markdown("#### Analysis window")
    use_time_mask = st.checkbox("Limit time range", value=False)
    if use_time_mask:
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("Start (s)")
        t_min = _r.number_input("t_min", value=_t_data_min, step=10.0, label_visibility="collapsed")
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("End (s)")
        t_max = _r.number_input("t_max", value=_t_data_max, step=10.0, label_visibility="collapsed")
        time_range = (float(t_min), float(t_max))
        a_start, a_end = float(t_min), float(t_max)
    else:
        time_range = None
        a_start, a_end = _t_data_min, _t_data_max

    st.divider()
    st.markdown("#### Zeroing window")

    # Initialise zeroing inputs in session_state on first render so that
    # toggling the analysis window does not reset these values.
    if "z0" not in st.session_state:
        st.session_state["z0"] = 0.0
    if "z1" not in st.session_state:
        st.session_state["z1"] = 100.0

    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("Zero start (s)")
    z0 = _r.number_input("Start (s)", key="z0", step=10.0, label_visibility="collapsed")
    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("Duration (s)")
    z_dur = _r.number_input("Duration (s)", key="z1", min_value=1.0, step=10.0, label_visibility="collapsed")
    zero_range = (float(z0), float(z0) + float(z_dur))

    use_final_zero = st.checkbox("Final zeroing window", value=False,
                                 help="Define a second zeroing window after the reaction. "
                                      "A linear drift correction is applied between the two windows.")
    if use_final_zero:
        # Defaults: if analysis window is limited use its end; otherwise
        # anchor to the current zeroing end so values are within the data.
        if "fz0" not in st.session_state:
            st.session_state["fz0"] = (a_end - 100.0) if use_time_mask else (float(z0) + float(z_dur) + 100.0)
        if "fz1" not in st.session_state:
            st.session_state["fz1"] = 100.0
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("Final start (s)")
        fz0 = _r.number_input("Final start (s)", key="fz0", step=10.0, label_visibility="collapsed")
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("Duration (s)")
        fz_dur = _r.number_input("Final duration (s)", key="fz1", min_value=1.0, step=10.0, label_visibility="collapsed")
        final_zero_range = (float(fz0), float(fz0) + float(fz_dur))
    else:
        # Clear final-zero state so defaults recalculate if re-enabled
        st.session_state.pop("fz0", None)
        st.session_state.pop("fz1", None)
        final_zero_range = None

    st.divider()

    st.markdown("#### Power smoothing")
    tv_weight = st.slider("TV weight", min_value=0.01, max_value=2.0, value=0.3, step=0.05,
                          help="Total Variation regularization strength. Higher = smoother/more piecewise-constant.")
    use_gauss = st.checkbox("Gaussian smoothing", value=False,
                            help="Apply a Gaussian filter to the power trace after TV denoising.")
    if use_gauss:
        gauss_sigma = st.slider("Sigma (s)", min_value=1.0, max_value=100.0, value=5.0, step=1.0,
                                help="Standard deviation of the Gaussian kernel in seconds. Higher = smoother.")
    else:
        gauss_sigma = None

    st.divider()
    st.markdown("#### Display")
    show_points = st.checkbox("Show data points", value=False)

# If downsampling is disabled, replace _ds with a pass-through
if not _use_downsample:
    def _ds(x, y, n=1800):
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

# ----------------------------
# UI
# ----------------------------
st.title("SENSE Analysis")

st.markdown("""
<style>
  [data-testid="stFileUploaderDropzone"] {
      border: 2px dashed #94a3b8 !important;
      border-radius: 10px !important;
      min-height: 80px !important;
      transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
  }
  [data-testid="stFileUploaderDropzone"].drag-active {
      background-color: #eff6ff !important;
      border-color: #2563eb !important;
      border-width: 3px !important;
      box-shadow: 0 0 0 4px rgba(37,99,235,0.25) !important;
  }
</style>
""", unsafe_allow_html=True)

# Inject JS drag-over highlighting via same-origin iframe trick
components.html("""<script>
(function() {
  function attach() {
    try {
      var doc = window.parent.document;
      doc.querySelectorAll('[data-testid="stFileUploaderDropzone"]').forEach(function(z) {
        if (z._dl) return;
        z._dl = true;
        z.addEventListener('dragenter', function(e) { e.preventDefault(); z.classList.add('drag-active'); });
        z.addEventListener('dragover',  function(e) { e.preventDefault(); z.classList.add('drag-active'); });
        z.addEventListener('dragleave', function(e) {
          if (!z.contains(e.relatedTarget)) z.classList.remove('drag-active');
        });
        z.addEventListener('drop', function() { z.classList.remove('drag-active'); });
      });
    } catch(e) {}
  }
  var n = 0;
  (function loop() { attach(); if (++n < 20) setTimeout(loop, 400); })();
  try {
    new MutationObserver(attach).observe(window.parent.document.body, {childList:true, subtree:true});
  } catch(e) {}
})();
</script>""", height=0)

_up_col, _dl_col, _ = st.columns([2, 1, 2])
with _up_col:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
_dl_placeholder = _dl_col.empty()

if not uploaded:
    st.info("Upload a CSV file to get started.")
    st.stop()

df = load_csv(uploaded)
if st.button("🔄 Clear analysis cache", help="Force re-analysis with current code (use after app updates)"):
    st.cache_data.clear()
    st.rerun()
df.columns = df.columns.str.strip()

try:
    data_dict, schema, ch_labels, schema_warnings = infer_schema_and_build(df)
except Exception as e:
    st.error(str(e))
    st.stop()

# Store data time range for sidebar defaults, and reset widget state on new file
_all_times = np.concatenate([v["time"] for v in data_dict.values() if len(v["time"]) > 0])
st.session_state["t_min_data"] = float(np.nanmin(_all_times))
st.session_state["t_max_data"] = float(np.nanmax(_all_times))
_all_channels = [k for k in data_dict if k != "BlockRef"]
st.session_state["_ch_keys"] = list(data_dict.keys())
st.session_state["_ch_labels"] = ch_labels

if st.session_state.get("_loaded_file") != uploaded.name:
    st.session_state["_loaded_file"] = uploaded.name
    for _k in ("t_min", "t_max", "z0", "z1", "fz0", "fz1"):
        st.session_state.pop(_k, None)
    # Reset channel selections and open the panel on new file
    st.session_state["sel"] = set(_all_channels)
    for _ch in _all_channels:
        st.session_state[f"chk_{_ch}"] = True
    st.session_state["show_ch_select"] = True

# Ensure sel exists (e.g. on very first run before any file was loaded)
if "sel" not in st.session_state:
    st.session_state["sel"] = set(_all_channels)

# ----------------------------
# Channel selection toggle
# ----------------------------
_show_sel = st.session_state.get("show_ch_select", False)
_tog_col, _ = st.columns([2, 5])
with _tog_col:
    _tog_label = "← Back to plots" if _show_sel else "⚙ Select channels"
    if st.button(_tog_label, use_container_width=True):
        st.session_state["show_ch_select"] = not _show_sel
        st.rerun()

if _show_sel:
    st.subheader("Available channels")
    _sa_col, _sn_col, _ = st.columns([1, 1, 4])
    # Update chk_* keys BEFORE checkboxes render so they take effect immediately
    if _sa_col.button("Select all", use_container_width=True):
        st.session_state["sel"] = set(_all_channels)
        for _ch in _all_channels:
            st.session_state[f"chk_{_ch}"] = True
    if _sn_col.button("Select none", use_container_width=True):
        st.session_state["sel"] = set()
        for _ch in _all_channels:
            st.session_state[f"chk_{_ch}"] = False

    # Split channels into well-plate grid and others
    _sel_re = re.compile(r'^([A-Za-z]+)(\d+)')
    _sel_grid = {}   # (row_letter, col_num) -> (orig_i, ch_key)
    _sel_other = []  # (orig_i, ch_key) for non-matching channels
    for _i, _ch in enumerate(_all_channels):
        _m = _sel_re.match(ch_labels.get(_ch, _ch))
        if _m:
            _sel_grid[(_m.group(1).upper(), int(_m.group(2)))] = (_i, _ch)
        else:
            _sel_other.append((_i, _ch))

    if _sel_grid:
        _sel_rows = sorted({r for r, _ in _sel_grid})
        _sel_cols = sorted({c for _, c in _sel_grid})
        # Column header row
        _hcols = st.columns([0.4] + [1] * len(_sel_cols))
        _hcols[0].markdown("")
        for _ci, _cn in enumerate(_sel_cols):
            _hcols[_ci + 1].markdown(
                f"<div style='text-align:center;font-weight:600;font-size:13px'>{_cn}</div>",
                unsafe_allow_html=True)
        # One row per well-plate row
        for _sr in _sel_rows:
            _rcols = st.columns([0.4] + [1] * len(_sel_cols))
            _rcols[0].markdown(
                f"<div style='font-weight:600;font-size:14px;padding-top:6px'>{_sr}</div>",
                unsafe_allow_html=True)
            for _ci, _sc in enumerate(_sel_cols):
                with _rcols[_ci + 1]:
                    if (_sr, _sc) in _sel_grid:
                        _i, _ch = _sel_grid[(_sr, _sc)]
                        _display = ch_labels.get(_ch, _ch)
                        _color = PALETTE[_i % len(PALETTE)]
                        if f"chk_{_ch}" not in st.session_state:
                            st.session_state[f"chk_{_ch}"] = (_ch in st.session_state["sel"])
                        _sw_col, _cb_col = st.columns([0.18, 1])
                        with _sw_col:
                            st.markdown(
                                f"<div style='background:{_color};width:12px;height:12px;"
                                f"border-radius:2px;margin-top:10px'></div>",
                                unsafe_allow_html=True)
                        with _cb_col:
                            _val = st.checkbox(_display, key=f"chk_{_ch}")
                        if _val:
                            st.session_state["sel"].add(_ch)
                        else:
                            st.session_state["sel"].discard(_ch)
                    else:
                        st.empty()

    if _sel_other:
        if _sel_grid:
            st.markdown("**Other channels**")
        for _i, _ch in _sel_other:
            _display = ch_labels.get(_ch, _ch)
            _color = PALETTE[_i % len(PALETTE)]
            if f"chk_{_ch}" not in st.session_state:
                st.session_state[f"chk_{_ch}"] = (_ch in st.session_state["sel"])
            _sw_col, _cb_col = st.columns([0.06, 1])
            with _sw_col:
                st.markdown(
                    f"<div style='background:{_color};width:14px;height:14px;"
                    f"border-radius:3px;margin-top:10px'></div>",
                    unsafe_allow_html=True)
            with _cb_col:
                _val = st.checkbox(_display, key=f"chk_{_ch}")
            if _val:
                st.session_state["sel"].add(_ch)
            else:
                st.session_state["sel"].discard(_ch)
    st.stop()

selected_channels = st.session_state.get("sel", set(_all_channels))
if not selected_channels:
    st.warning("No channels selected. Click **⚙ Select channels** to choose which channels to analyze.")
    st.stop()

# ----------------------------
# Run analysis
# ----------------------------
if zero_range[1] <= zero_range[0]:
    st.warning(f"Zeroing window: end ≤ start — values swapped automatically.")
    zero_range = (zero_range[1], zero_range[0])

if time_range is not None and time_range[1] <= time_range[0]:
    st.warning(f"Analysis window: end ≤ start — values swapped automatically.")
    time_range = (time_range[1], time_range[0])

if final_zero_range is not None and final_zero_range[1] <= final_zero_range[0]:
    st.warning("Final zeroing window: end ≤ start — values swapped automatically.")
    final_zero_range = (final_zero_range[1], final_zero_range[0])

try:
    results = analyze(data_dict, C, K, zero_range, ch_labels=ch_labels,
                      time_range=time_range, tv_weight=tv_weight,
                      final_zero_range=final_zero_range,
                      selected_channels=frozenset(selected_channels),
                      gauss_sigma=gauss_sigma,
                      temp_sigma=temp_sigma,
                      alpha_env=float(alpha_env),
                      ref_channel=ref_channel)
except ValueError as e:
    st.error(str(e))
    st.stop()

if not results["plots"]:
    st.warning("No channels with sufficient data to analyze.")
    st.stop()


_trace_mode = "lines+markers" if show_points else "lines"
_marker = dict(size=3) if show_points else {}

def _ds(x, y, n=1800):
    """Downsample x/y to at most n points using LTTB (Largest Triangle Three Buckets).
    Preserves visual shape — peaks and inflections are kept, flat regions thinned."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    length = len(x)
    if length <= n:
        return x, y
    # Always keep first and last point
    selected = [0]
    bucket_size = (length - 2) / (n - 2)
    a = 0  # previously selected index
    for i in range(n - 2):
        # Next bucket boundaries
        b_start = int((i + 1) * bucket_size) + 1
        b_end   = int((i + 2) * bucket_size) + 1
        b_end   = min(b_end, length - 1)
        # Average of the bucket after this one (point C)
        c_start = int((i + 2) * bucket_size) + 1
        c_end   = min(int((i + 3) * bucket_size) + 1, length)
        c_x = x[c_start:c_end].mean() if c_start < length else x[-1]
        c_y = y[c_start:c_end].mean() if c_start < length else y[-1]
        # Pick point in current bucket with largest triangle area vs A and C
        if b_start >= b_end:
            selected.append(b_start if b_start < length else length - 1)
            continue
        ax, ay = x[a], y[a]
        area = 0.5 * np.abs(
            (ax - c_x) * (y[b_start:b_end] - ay)
            - (x[b_start:b_end] - ax) * (c_y - ay)
        )
        a = b_start + int(area.argmax())
        selected.append(a)
    selected.append(length - 1)
    idx = np.array(selected)
    return x[idx], y[idx]

# Time unit (set by widget in Array section, persists via session state)
_time_mins = st.session_state.get("_time_mins", False)
_t_div   = 60.0 if _time_mins else 1.0
_t_label = "Time (min)" if _time_mins else "Time (s)"

if column_plot_mode:
    # --- Column Plot: group channels by trailing digit(s) ---
    _col_re = re.compile(r'^[A-Za-z]+(\d+)')
    col_groups: dict = {}
    for _orig_i, _p in enumerate(results["plots"]):
        _m = _col_re.search(str(_p["label"]))
        _key = _m.group(1) if _m else _p["label"]
        col_groups.setdefault(_key, []).append((_orig_i, _p))

    _sorted_col_keys = sorted(col_groups.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    _block_x = results["plots"][0]["x"]
    _block_y = results["plots"][0]["block_temp"]

    def _col_grid(data_key, ylabel, uiprefix, height=320):
        for _ri in range(0, len(_sorted_col_keys), 2):
            _rcols = st.columns(2)
            for _ci, _key in enumerate(_sorted_col_keys[_ri:_ri + 2]):
                with _rcols[_ci]:
                    _fig = go.Figure()
                    for _oi, _p in col_groups[_key]:
                        _dx, _dy = _ds(_p["x"], _p[data_key])
                        _fig.add_trace(go.Scatter(
                            x=_dx / _t_div, y=_dy, name=_p["label"],
                            mode=_trace_mode, marker=_marker,
                            line=dict(color=PALETTE[_oi % len(PALETTE)], width=2, simplify=False),
                        ))
                    if data_key == "temp" and ref_channel is None:
                        _bx, _by = _ds(_block_x, _block_y)
                        _fig.add_trace(go.Scatter(
                            x=_bx / _t_div, y=_by, name="BlockRef",
                            mode=_trace_mode, marker=_marker,
                            line=dict(color="#888888", width=1.5, dash="dash", simplify=False),
                        ))
                    _fig.update_layout(**PLOT_LAYOUT,
                        title=dict(text=f"Column {_key}", font=dict(size=14)),
                        xaxis_title=_t_label, yaxis_title=ylabel, height=height,
                        uirevision=f"{uiprefix}_{_key}")
                    st.plotly_chart(_fig, use_container_width=True)

    st.subheader("Column plot — Temperature")
    _col_grid("temp", "Temperature (°C)", "col_temp")
    st.divider()
    st.subheader("Column plot — Heat")
    _col_grid("y", "Heat (J)", "col_heat")
    _render_heat_table(results["plots"])
    st.divider()
    st.subheader("Column plot — Power")
    _col_grid("power", "Power (W)", "col_power")

elif row_plot_mode:
    # --- Row Plot: group channels by leading letter(s) ---
    _row_re = re.compile(r'^([A-Za-z]+)')
    row_groups: dict = {}
    for _orig_i, _p in enumerate(results["plots"]):
        _m = _row_re.match(str(_p["label"]))
        _key = _m.group(1).upper() if _m else _p["label"]
        row_groups.setdefault(_key, []).append((_orig_i, _p))

    _sorted_keys = sorted(row_groups.keys())
    _block_x = results["plots"][0]["x"]
    _block_y = results["plots"][0]["block_temp"]

    # Temperature
    st.subheader("Row plot — Temperature")
    for _ri in range(0, len(_sorted_keys), 2):
        _rcols = st.columns(2)
        for _ci, _key in enumerate(_sorted_keys[_ri:_ri + 2]):
            with _rcols[_ci]:
                _fig = go.Figure()
                for _oi, _p in row_groups[_key]:
                    _dx, _dy = _ds(_p["x"], _p["temp"])
                    _fig.add_trace(go.Scatter(
                        x=_dx / _t_div, y=_dy, name=_p["label"],
                        mode=_trace_mode, marker=_marker,
                        line=dict(color=PALETTE[_oi % len(PALETTE)], width=1.8),
                    ))
                if ref_channel is None:
                    _bx, _by = _ds(_block_x, _block_y)
                    _fig.add_trace(go.Scatter(
                        x=_bx / _t_div, y=_by, name="BlockRef",
                        mode=_trace_mode, marker=_marker,
                        line=dict(color="#888888", width=1.5, dash="dash", simplify=False),
                    ))
                _fig.update_layout(**PLOT_LAYOUT,
                    title=dict(text=f"Row {_key}", font=dict(size=14)),
                    xaxis_title=_t_label, yaxis_title="Temperature (°C)", height=320,
                    uirevision=f"row_temp_{_key}")
                st.plotly_chart(_fig, use_container_width=True)

    st.divider()

    # Heat
    st.subheader("Row plot — Heat")
    for _ri in range(0, len(_sorted_keys), 2):
        _rcols = st.columns(2)
        for _ci, _key in enumerate(_sorted_keys[_ri:_ri + 2]):
            with _rcols[_ci]:
                _fig = go.Figure()
                for _oi, _p in row_groups[_key]:
                    _dx, _dy = _ds(_p["x"], _p["y"])
                    _fig.add_trace(go.Scatter(
                        x=_dx / _t_div, y=_dy, name=_p["label"],
                        mode=_trace_mode, marker=_marker,
                        line=dict(color=PALETTE[_oi % len(PALETTE)], width=2, simplify=False),
                    ))
                _fig.update_layout(**PLOT_LAYOUT,
                    title=dict(text=f"Row {_key}", font=dict(size=14)),
                    xaxis_title=_t_label, yaxis_title="Heat (J)", height=320,
                    uirevision=f"row_heat_{_key}")
                st.plotly_chart(_fig, use_container_width=True)

    _render_heat_table(results["plots"])

    st.divider()

    # Power
    st.subheader("Row plot — Power")
    for _ri in range(0, len(_sorted_keys), 2):
        _rcols = st.columns(2)
        for _ci, _key in enumerate(_sorted_keys[_ri:_ri + 2]):
            with _rcols[_ci]:
                _fig = go.Figure()
                for _oi, _p in row_groups[_key]:
                    _dx, _dy = _ds(_p["x"], _p["power"])
                    _fig.add_trace(go.Scatter(
                        x=_dx / _t_div, y=_dy, name=_p["label"],
                        mode=_trace_mode, marker=_marker,
                        line=dict(color=PALETTE[_oi % len(PALETTE)], width=2, simplify=False),
                    ))
                _fig.update_layout(**PLOT_LAYOUT,
                    title=dict(text=f"Row {_key}", font=dict(size=14)),
                    xaxis_title=_t_label, yaxis_title="Power (W)", height=320,
                    uirevision=f"row_power_{_key}")
                st.plotly_chart(_fig, use_container_width=True)

elif array_plot_mode:
    # --- Array plot: each channel in its own subplot, well-plate grid ---
    _arr_re = re.compile(r'^([A-Za-z]+)(\d+)')
    _arr_rows, _arr_cols, _arr_map, _arr_other = set(), set(), {}, []
    for _orig_i, _p in enumerate(results["plots"]):
        _m = _arr_re.match(str(_p["label"]))
        if _m:
            _r, _c = _m.group(1).upper(), int(_m.group(2))
            _arr_rows.add(_r); _arr_cols.add(_c)
            _arr_map[(_r, _c)] = (_orig_i, _p)
        else:
            _arr_other.append((_orig_i, _p))

    _sorted_arr_rows = sorted(_arr_rows)   # A, B, C, D  → screen rows (down)
    _sorted_arr_cols = sorted(_arr_cols)   # 1, 2, …, 6  → screen columns (across)
    _n_scr_cols = max(len(_sorted_arr_cols), 1)
    _block_x = results["plots"][0]["x"]
    _block_y = results["plots"][0]["block_temp"]
    _arr_layout = {**PLOT_LAYOUT,
                   "margin": dict(l=45, r=8, t=28, b=28)}

    def _arr_grid(data_key, ylabel, uiprefix, height_per_row=200):
        n_r = len(_sorted_arr_rows)
        n_c = len(_sorted_arr_cols)
        if n_r == 0 or n_c == 0:
            return
        if _arr_axis_mode == "Scale individually":
            shared_x, shared_y = False, False
        elif _arr_axis_mode == "Common range":
            shared_x, shared_y = "columns", True
        else:  # Shared (row Y, col X)
            shared_x, shared_y = "columns", "rows"
        _h_sp, _v_sp = 0.02, 0.06
        fig = make_subplots(
            rows=n_r, cols=n_c,
            shared_xaxes=shared_x,
            shared_yaxes=shared_y,
            horizontal_spacing=_h_sp,
            vertical_spacing=_v_sp,
        )
        # Edge labels: column numbers across top, row letters down left
        _pw = (1 - (n_c - 1) * _h_sp) / n_c
        _ph = (1 - (n_r - 1) * _v_sp) / n_r
        for ci, _sc in enumerate(_sorted_arr_cols):
            fig.add_annotation(
                x=ci * (_pw + _h_sp) + _pw / 2, y=1.06,
                xref="paper", yref="paper",
                text=f"<b>{_sc}</b>", showarrow=False,
                font=dict(size=22, color="#222222", family="sans-serif"),
                xanchor="center", yanchor="bottom",
            )
        for ri, _sr in enumerate(_sorted_arr_rows):
            fig.add_annotation(
                x=1.04, y=1 - ri * (_ph + _v_sp) - _ph / 2,
                xref="paper", yref="paper",
                text=f"<b>{_sr}</b>", showarrow=False,
                font=dict(size=22, color="#222222", family="sans-serif"),
                xanchor="left", yanchor="middle",
            )
        for ri, _sr in enumerate(_sorted_arr_rows):
            for ci, _sc in enumerate(_sorted_arr_cols):
                if (_sr, _sc) in _arr_map:
                    _oi, _p = _arr_map[(_sr, _sc)]
                    _dx, _dy = _ds(_p["x"], _p[data_key])
                    fig.add_trace(go.Scattergl(
                        x=_dx / _t_div, y=_dy,
                        mode=_trace_mode, marker=_marker,
                        line=dict(color=PALETTE[_oi % len(PALETTE)], width=2),
                        showlegend=False,
                    ), row=ri + 1, col=ci + 1)
                    if data_key == "temp" and ref_channel is None:
                        _bx, _by = _ds(_block_x, _block_y)
                        fig.add_trace(go.Scattergl(
                            x=_bx / _t_div, y=_by,
                            mode=_trace_mode, marker=_marker,
                            line=dict(color="#888888", width=1.2, dash="dash"),
                            showlegend=False,
                        ), row=ri + 1, col=ci + 1)
        fig.update_layout(
            paper_bgcolor=PLOT_LAYOUT["paper_bgcolor"],
            plot_bgcolor=PLOT_LAYOUT["plot_bgcolor"],
            font=PLOT_LAYOUT["font"],
            showlegend=False,
            height=max(n_r * height_per_row, 200),
            margin=dict(l=40, r=80, t=75, b=30),
            hovermode="x unified",
            uirevision=uiprefix,
        )
        fig.update_xaxes(
            gridcolor="#e0e0e0", linecolor="#aaaaaa", zerolinecolor="#cccccc",
            showline=True, mirror=True, linewidth=1.2,
            tickfont=dict(size=13),
        )
        fig.update_yaxes(
            gridcolor="#e0e0e0", linecolor="#aaaaaa", zerolinecolor="#cccccc",
            showline=True, mirror=True, linewidth=1.2,
            tickfont=dict(size=13),
        )
        if _arr_axis_mode == "Common range":
            _all_y = [v for (_, _p) in _arr_map.values() for v in _p[data_key] if np.isfinite(v)]
            if _all_y:
                _pad = (max(_all_y) - min(_all_y)) * 0.05 or 0.5
                fig.update_yaxes(range=[min(_all_y) - _pad, max(_all_y) + _pad])
        st.plotly_chart(fig, use_container_width=True)
        if _arr_other:
            st.markdown("**Other channels**")
            _oc = st.columns(min(len(_arr_other), 4))
            for _ci2, (_oi2, _p2) in enumerate(_arr_other):
                with _oc[_ci2 % 4]:
                    _fig = go.Figure()
                    _fig.add_trace(go.Scatter(
                        x=_p2["x"], y=_p2[data_key],
                        mode=_trace_mode, marker=_marker,
                        line=dict(color=PALETTE[_oi2 % len(PALETTE)], width=2, simplify=False),
                        showlegend=False,
                    ))
                    _fig.update_layout(**_arr_layout,
                        title=dict(text=_p2["label"], font=dict(size=13)),
                        xaxis_title="", yaxis_title=ylabel, height=200,
                        uirevision=f"{uiprefix}_other{_ci2}")
                    st.plotly_chart(_fig, use_container_width=True)

    _ctrl_l, _ctrl_r = st.columns([3, 1])
    with _ctrl_l:
        _arr_axis_mode = st.radio(
            "Axis scaling",
            options=["Common range", "Shared (row Y, col X)", "Scale individually"],
            index=0, horizontal=True,
            help="Common range: all panels share one Y-axis range.\n"
                 "Shared: Y shared within each row, X shared within each column.\n"
                 "Scale individually: each panel auto-scales independently.",
        )
    with _ctrl_r:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        st.toggle("Minutes", key="_time_mins", help="Display time axis in minutes instead of seconds.")

    st.subheader("Array plot — Temperature")
    _arr_grid("temp", "T (°C)", "arr_temp")
    st.divider()
    st.subheader("Array plot — Heat")
    _arr_grid("y", "Heat (J)", "arr_heat")
    _render_heat_table(results["plots"])
    st.divider()
    st.subheader("Array plot — Power")
    _arr_grid("power", "P (W)", "arr_power")

else:
    # --- All-in-one plots ---
    st.subheader("Zeroed temperature traces")
    fig1 = go.Figure()
    for i, p in enumerate(results["plots"]):
        _dx, _dy = _ds(p["x"], p["temp"])
        fig1.add_trace(go.Scatter(
            x=_dx / _t_div, y=_dy, name=p["label"],
            mode=_trace_mode, marker=_marker,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8, simplify=False),
        ))
    if ref_channel is None:
        _bx, _by = _ds(results["plots"][0]["x"], results["plots"][0]["block_temp"])
        fig1.add_trace(go.Scatter(
            x=_bx / _t_div, y=_by,
            name="BlockRef",
            mode=_trace_mode, marker=_marker,
            line=dict(color="#888888", width=1.5, dash="dash", simplify=False),
        ))
    fig1.update_layout(**PLOT_LAYOUT,
        xaxis_title=_t_label, yaxis_title="Temperature (°C)", height=350,
        uirevision="temp")
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    st.subheader("Heat traces")
    fig2 = go.Figure()
    for i, p in enumerate(results["plots"]):
        _dx, _dy = _ds(p["x"], p["y"])
        fig2.add_trace(go.Scatter(
            x=_dx / _t_div, y=_dy, name=p["label"],
            mode=_trace_mode, marker=_marker,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2, simplify=False),
        ))
    fig2.update_layout(**PLOT_LAYOUT,
        xaxis_title=_t_label, yaxis_title="Heat (J)", height=400,
        uirevision="heat")
    st.plotly_chart(fig2, use_container_width=True)

    _render_heat_table(results["plots"])

    st.divider()

    st.subheader("Power traces")
    fig3 = go.Figure()
    for i, p in enumerate(results["plots"]):
        _dx, _dy = _ds(p["x"], p["power"])
        fig3.add_trace(go.Scatter(
            x=_dx / _t_div, y=_dy, name=p["label"],
            mode=_trace_mode, marker=_marker,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2, simplify=False),
        ))
    fig3.update_layout(**PLOT_LAYOUT,
        xaxis_title=_t_label, yaxis_title="Power (W)", height=400,
        uirevision="power")
    st.plotly_chart(fig3, use_container_width=True)

# Export — each channel has its own time axis so use per-channel time columns
_max_len = max(len(p["x"]) for p in results["plots"])
def _pad(arr):
    n = _max_len - len(arr)
    return arr if n == 0 else np.concatenate([arr, np.full(n, np.nan)])

_out_dict = {}
for p in results["plots"]:
    _lbl = p["label"]
    _out_dict[_lbl + "_Time (s)"]  = _pad(p["x"])
    _out_dict[_lbl + "_Temp (°C)"] = _pad(p["temp"])
    _out_dict[_lbl + "_Heat (J)"]  = _pad(p["y"])
    _out_dict[_lbl + "_Power (W)"] = _pad(p["power"])
out = pd.DataFrame(_out_dict)

analysis_window = (
    f"{time_range[0]:.1f} - {time_range[1]:.1f} s" if time_range else "full range"
)
param_header = "\n".join([
    f"# Source file,{uploaded.name}",
    f"# C (J/K),{C:.4f}",
    f"# tau (s),{tau:.2f}",
    f"# K (W/K),{K:.6f}",
    f"# C' (J/mL·K),{float(C_prime):.4f}",
    f"# Volume (mL),{float(V):.4f}",
    f"# alpha(env),{float(alpha_env):.4f}",
    f"# Te (C),23.0",
    f"# Zero start (s),{zero_range[0]:.1f}",
    f"# Zero end (s),{zero_range[1]:.1f}",
    f"# Analysis window,{analysis_window}",
    f"# TV weight,{tv_weight:.3f}",
]) + "\n"
csv_bytes = (param_header + out.to_csv(index=False)).encode("utf-8")
export_name = uploaded.name.rsplit(".", 1)[0] + "_analysis.csv"
with _dl_placeholder.container():
    st.markdown("<div style='padding-top:1.6rem'></div>", unsafe_allow_html=True)
    st.download_button(
        "Download results (CSV)",
        data=csv_bytes,
        file_name=export_name,
        mime="text/csv",
        use_container_width=True,
    )
