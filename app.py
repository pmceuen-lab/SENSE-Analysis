# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
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

    diff_prev = np.abs(data_clean[1:-1] - data_clean[:-2])
    diff_next = np.abs(data_clean[1:-1] - data_clean[2:])
    is_spike = (diff_prev > threshold) | (diff_next > threshold)

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

    Ke_int = np.cumsum((T_data[1:] - Te) * dt)
    Ke_int = np.concatenate(([0], Ke_int))

    C_int = T_data - T0

    U = C * C_int + Kb * K_int + Ke * Ke_int

    return U, C_int, K_int


def mask_time_range(time, data, t_min, t_max):
    time = np.asarray(time)
    data = np.asarray(data)
    mask = (time >= t_min) & (time <= t_max)
    return time[mask], data[mask]


def _drop_close_points(t, *arrays, min_dt=0.3):
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
    gauss_sigma is in seconds; converted to samples using median dt."""
    y = np.asarray(y, dtype=float)
    dy = np.gradient(y, x)
    result = denoise_tv_chambolle(dy, weight=tv_weight)
    if gauss_sigma is not None:
        dt = float(np.median(np.diff(x)))
        if dt > 0:
            result = gaussian_filter1d(result, sigma=gauss_sigma / dt)
    return result


def analyze(data_dict, C, K, zero_range, ch_labels=None, time_range=None,
            tv_weight=0.3, final_zero_range=None, selected_channels=None,
            gauss_sigma=None, alpha_env=0.0, Te=23.0):
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
    if countR == 0:
        raise ValueError(
            f"Zeroing window ({tzeroing[0]:.1f}–{tzeroing[1]:.1f} s) contains no data points. "
            "Adjust the Zero start/end values."
        )
    T0avg = sumupR / countR

    plots = []

    all_ch = [k for k in data_dict if k != 'BlockRef']
    ch_to_process = [c for c in all_ch if selected_channels is None or c in selected_channels]

    for ch_name in ch_to_process:
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

        if final_zero_range is not None:
            # Compute offset in final zeroing window
            sumup2 = 0
            count2 = 0
            for i in range(len(block_time)):
                if block_time[i] > final_zero_range[0] and block_time[i] < final_zero_range[1]:
                    sumup2 += ch_temp_resampled[i] - block_temp[i]
                    count2 += 1
            if count2 == 0:
                raise ValueError(
                    f"Final zeroing window ({final_zero_range[0]:.1f}–{final_zero_range[1]:.1f} s) "
                    "contains no data points. Adjust the Final zero start/end values."
                )
            average_final = sumup2 / count2
            # Linear correction: interpolate offset between midpoints of both windows
            t_mid1 = (tzeroing[0] + tzeroing[1]) / 2
            t_mid2 = (final_zero_range[0] + final_zero_range[1]) / 2
            correction = average + (average_final - average) * (block_time - t_mid1) / (t_mid2 - t_mid1)
            T_zerod = ch_temp_resampled - correction
        else:
            T_zerod = ch_temp_resampled - average

        # Apply time range mask after zeroing
        if time_range is not None:
            t_min, t_max = time_range
            t_masked, T_zerod_masked = mask_time_range(block_time, T_zerod, t_min, t_max)
            _, block_temp_masked = mask_time_range(block_time, block_temp, t_min, t_max)
        else:
            t_masked, T_zerod_masked = block_time, T_zerod
            block_temp_masked = block_temp

        # Drop points closer than 0.3 s to avoid gradient spikes
        t_masked, T_zerod_masked, block_temp_masked = _drop_close_points(
            t_masked, T_zerod_masked, block_temp_masked
        )

        energy, cap_term, cond_term = calculate_energy(
            T_zerod_masked, t_masked, block_temp_masked, C, K, T0avg,
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
            " Adds Ke·∫(T − Te) dt to the energy, where Te = 23 °C."
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
        st.session_state["z0"] = a_start
    if "z1" not in st.session_state:
        st.session_state["z1"] = a_start + 50.0

    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("Zero start (s)")
    z0 = _r.number_input("Start (s)", key="z0", step=10.0, label_visibility="collapsed")
    _l, _r = st.columns([1.4, 1.6])
    with _l: _inline("Zero end (s)")
    z1 = _r.number_input("End (s)", key="z1", step=10.0, label_visibility="collapsed")
    zero_range = (float(z0), float(z1))

    use_final_zero = st.checkbox("Final zeroing window", value=False,
                                 help="Define a second zeroing window after the reaction. "
                                      "A linear drift correction is applied between the two windows.")
    if use_final_zero:
        # Defaults: if analysis window is limited use its end; otherwise
        # anchor to the current zeroing end so values are within the data.
        if "fz0" not in st.session_state:
            st.session_state["fz0"] = (a_end - 50.0) if use_time_mask else (float(z1) + 100.0)
        if "fz1" not in st.session_state:
            st.session_state["fz1"] = a_end if use_time_mask else (float(z1) + 200.0)
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("Final start (s)")
        fz0 = _r.number_input("Final start (s)", key="fz0", step=10.0, label_visibility="collapsed")
        _l, _r = st.columns([1.4, 1.6])
        with _l: _inline("Final end (s)")
        fz1 = _r.number_input("Final end (s)", key="fz1", step=10.0, label_visibility="collapsed")
        final_zero_range = (float(fz0), float(fz1))
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

    for _i, _ch in enumerate(_all_channels):
        _display = ch_labels.get(_ch, _ch)
        _color = PALETTE[_i % len(PALETTE)]
        # Re-initialise key from sel set if widget state was lost (e.g. view switch)
        if f"chk_{_ch}" not in st.session_state:
            st.session_state[f"chk_{_ch}"] = (_ch in st.session_state["sel"])
        _sw_col, _cb_col = st.columns([0.06, 1])
        with _sw_col:
            st.markdown(
                f"<div style='background:{_color};width:14px;height:14px;"
                f"border-radius:3px;margin-top:10px'></div>",
                unsafe_allow_html=True,
            )
        with _cb_col:
            _val = st.checkbox(_display, key=f"chk_{_ch}")
        # Keep sel set in sync
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
                      selected_channels=selected_channels,
                      gauss_sigma=gauss_sigma,
                      alpha_env=float(alpha_env))
except ValueError as e:
    st.error(str(e))
    st.stop()

if not results["plots"]:
    st.warning("No channels with sufficient data to analyze.")
    st.stop()

# Temperature plot
_trace_mode = "lines+markers" if show_points else "lines"
_marker = dict(size=3) if show_points else {}

st.subheader("Zeroed temperature traces")
fig1 = go.Figure()
for i, p in enumerate(results["plots"]):
    fig1.add_trace(go.Scatter(
        x=p["x"], y=p["temp"], name=p["label"],
        mode=_trace_mode, marker=_marker,
        line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
    ))
fig1.add_trace(go.Scatter(
    x=results["plots"][0]["x"],
    y=results["plots"][0]["block_temp"],
    name="BlockRef",
    mode=_trace_mode, marker=_marker,
    line=dict(color="#888888", width=1.5, dash="dash"),
))
fig1.update_layout(**PLOT_LAYOUT,
    xaxis_title="Time (s)", yaxis_title="Temperature (°C)", height=350,
    uirevision="temp")
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# Heat plot
st.subheader("Heat traces")
fig2 = go.Figure()

for i, p in enumerate(results["plots"]):
    fig2.add_trace(go.Scatter(
        x=p["x"], y=p["y"], name=p["label"],
        mode=_trace_mode, marker=_marker,
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
    ))

fig2.update_layout(**PLOT_LAYOUT,
    xaxis_title="Time (s)", yaxis_title="Heat (J)", height=400,
    uirevision="heat")
st.plotly_chart(fig2, use_container_width=True)

# Final heat metrics
st.markdown("**Final heat values**")
metric_cols = st.columns(len(results["plots"]))
for col, p in zip(metric_cols, results["plots"]):
    col.metric(p["label"], f"{p['y'][-1]:.3f} J")

st.divider()

# Power plot
st.subheader("Power traces")
fig3 = go.Figure()

for i, p in enumerate(results["plots"]):
    fig3.add_trace(go.Scatter(
        x=p["x"], y=p["power"], name=p["label"],
        mode=_trace_mode, marker=_marker,
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
    ))

fig3.update_layout(**PLOT_LAYOUT,
    xaxis_title="Time (s)", yaxis_title="Power (W)", height=400,
    uirevision="power")
st.plotly_chart(fig3, use_container_width=True)

# Export
out = pd.DataFrame({
    "Time (s)":        results["plots"][0]["x"],
    "BlockRef_Temp (°C)": results["plots"][0]["block_temp"],
})
for p in results["plots"]:
    out[p["label"] + "_Temp (°C)"]  = p["temp"]
    out[p["label"] + "_Heat (J)"]   = p["y"]
    out[p["label"] + "_Power (W)"]  = p["power"]

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
