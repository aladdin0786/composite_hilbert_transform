#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Hilbert-transform-based composite attribute for boundary enhancement
in a 2D seismic section from the Baku Archipelago.

REVISED VERSION (addresses reviewer comments).

Input:
    my_data_1.sgy   # expected in the same folder as this script

Main outputs (in outputs_hilbert_composite_baku/):
    fig01_full_section_auto_window.png
    fig02_selected_seismic_section.png
    fig03_hilbert_attributes.png
    fig04_composite_attribute.png
    fig05_composite_overlay_on_seismic.png
    fig06a_hilbert_component_curves.png        (supplementary)
    fig06b_frequency_composite_curves.png      (supplementary)
    fig07_attribute_redundancy.png             (NEW: inter-component crossplots + correlation matrix)
    fig08_weight_sensitivity.png               (NEW: robustness of the composite to the weights)
    fig09_synthetic_validation.png             (NEW: controlled test on a known model)
    fig10_synthetic_detection_scores.png       (NEW: quantitative detection comparison)
    hilbert_attribute_metrics.csv
    automatic_interval_scores.csv
    attribute_correlation_matrix.csv           (NEW)
    weight_sensitivity_overlap.csv             (NEW)
    synthetic_detection_scores.csv             (NEW)
    deep_vs_selected_stats.csv                 (NEW)
    selected_interval_summary.txt
    attribute_arrays_selected_interval.npz

Install required packages if needed:
    pip install numpy scipy matplotlib pandas segyio

-------------------------------------------------------------------------------
WHAT CHANGED RELATIVE TO THE FIRST VERSION (reviewer-driven):
-------------------------------------------------------------------------------
1. PHASE DISCONTINUITY redefined. The old version mixed the *temporal* phase
   gradient (which is proportional to instantaneous frequency and is large in
   every wavelet cycle) with the lateral one. The attribute now uses ONLY the
   lateral phase gradient |d(phase)/dx| (wrap-safe via sin/cos) and is
   ENERGY-WEIGHTED by the normalized envelope, so it responds to lateral phase
   breaks where reflection energy exists, i.e. a genuine discontinuity sense,
   and is no longer collinear with the frequency-anomaly component.

2. INSTANTANEOUS FREQUENCY now uses the analytic-signal-derivative form
   f = (x*H' - H*x') / (2*pi*(x^2 + H^2)), which is mathematically identical to
   d(phase)/dt but avoids phase-unwrapping spikes. (The old unwrap+gradient
   method is still available via FREQ_METHOD = "unwrap".)

3. ENVELOPE normalization is now amplitude-scale-invariant: the envelope is
   divided by its own median before the log compression, so the log "knee" no
   longer depends on the arbitrary SEG-Y amplitude scaling.

4. ENVELOPE GRADIENT is now sampling-invariant: vertical and lateral gradients
   are each normalized before being combined, so the edge strength does not
   depend on dt vs trace spacing.

5. AUTOMATIC INTERVAL SELECTION now searches MULTIPLE window lengths, so the
   interval length is a genuine result rather than a fixed parameter. The
   best-vs-runner-up score margin is reported, and statistics of the excluded
   deep section are saved so the >MAX_INTERPRET_TIME exclusion is demonstrated,
   not just asserted.

6. WEIGHT JUSTIFICATION. A weight-sensitivity analysis (including an equal-
   weights baseline) quantifies how stable the top-percentile boundary map is
   to the choice of weights (Jaccard/IoU overlap). See fig08 + CSV.

7. REDUNDANCY. A correlation matrix and inter-component crossplots document how
   independent the four components actually are (replaces the old crossplots,
   which were correlated with the composite by construction). See fig07 + CSV.

8. INDEPENDENT VALIDATION. A synthetic 2D impedance model with a fault, a
   pinch-out (reflector termination) and a lateral facies change is built and
   run through the SAME attribute code over a NOISE SWEEP (SNR = 8,4,2,1).
   The target is reflecting-boundary detection (ground truth = the strongest
   reflectors, which include the fault/termination/facies contrasts); each
   attribute is scored by ROC AUC and precision/recall at a matched flag
   budget at every SNR. The headline is robustness to noise (how each
   attribute and the composite degrade as SNR falls). Isolating lateral
   discontinuities from structural dip is explicitly NOT claimed (no dip
   steering); it is left as a stated limitation. See fig09, fig10 + CSV.

9. Cosmetic/reporting: headless-safe plotting (no blocking windows), reduced
   post-composite smoothing, and additional saved metrics for the manuscript.
-------------------------------------------------------------------------------
"""

import os
import struct
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are written to disk, no GUI windows
import matplotlib.pyplot as plt

try:
    import segyio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "segyio is required. Install it with: pip install segyio"
    ) from exc

from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter, median_filter, binary_dilation
from scipy.stats import rankdata, spearmanr


# =============================================================================
# USER PARAMETERS
# =============================================================================

SEGY_FILE = "my_data_1.sgy"
OUTPUT_DIR = "outputs_hilbert_composite_baku"

# The section is around 13200 ms. The deeper part (below MAX_INTERPRET_TIME_MS)
# is excluded from the automatic search; the script also reports statistics for
# that excluded part so the exclusion can be justified quantitatively.
MAX_INTERPRET_TIME_MS = 8000.0
MIN_INTERPRET_TIME_MS = 300.0

# Optional manual override of the automatic interval selection.
MANUAL_TMIN_MS = None
MANUAL_TMAX_MS = None

# Automatic window selection: several candidate LENGTHS are tried, so the
# selected interval length is an output of the search, not a fixed constant.
AUTO_WINDOW_MS_LIST = [2500.0, 3000.0, 3500.0, 4000.0]
AUTO_STEP_MS = 250.0

# Instantaneous-frequency estimator: "analytic" (robust, default) or "unwrap".
FREQ_METHOD = "analytic"

# Gentle preprocessing. Keep broad enough not to destroy geological amplitude.
APPLY_BANDPASS = True
BANDPASS_LOW_HZ = 3.0
BANDPASS_HIGH_HZ = 80.0
BANDPASS_ORDER = 4

# Attribute smoothing. Small values preserve boundaries; larger values clean maps.
GAUSSIAN_SIGMA_SAMPLES = 1.0
GAUSSIAN_SIGMA_TRACES = 0.6
MEDIAN_FILTER_SIZE = (5, 3)  # (time samples, traces)

# Composite attribute weights (chosen). Sum need not be 1; normalized later.
W_ENVELOPE = 0.15
W_ENVELOPE_GRADIENT = 0.30
W_PHASE_DISCONTINUITY = 0.35
W_FREQUENCY_ANOMALY = 0.20

# Boundary overlay threshold. 90 means top 10% of composite attribute values.
BOUNDARY_PERCENTILE = 90.0

# Extra analyses.
RUN_WEIGHT_SENSITIVITY = True
RUN_SYNTHETIC_VALIDATION = True

# Weight sets tested in the sensitivity analysis (label, [E, G, D, F]).
SENSITIVITY_WEIGHTS = [
    ("chosen (0.15/0.30/0.35/0.20)", [0.15, 0.30, 0.35, 0.20]),
    ("equal (0.25 each)",            [0.25, 0.25, 0.25, 0.25]),
    ("phase-led (0.10/0.25/0.45/0.20)", [0.10, 0.25, 0.45, 0.20]),
    ("gradient-led (0.10/0.45/0.30/0.15)", [0.10, 0.45, 0.30, 0.15]),
    ("envelope-led (0.40/0.25/0.20/0.15)", [0.40, 0.25, 0.20, 0.15]),
    ("frequency-up (0.15/0.25/0.30/0.30)", [0.15, 0.25, 0.30, 0.30]),
]

# Figure settings.
DPI = 300
CMAP_SEISMIC = "gray"
CMAP_ATTRIBUTE = "viridis"
CMAP_PHASE = "twilight"
CMAP_FREQ = "plasma"
FIG_WIDTH = 11

# Visualization-only smoothing for vertical mean curves. Does not affect
# attribute calculations, saved arrays, or numerical metrics.
VISUAL_SMOOTH_SAMPLES = 7


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@dataclass
class SeismicData:
    data: np.ndarray       # shape: samples x traces
    time_ms: np.ndarray    # shape: samples
    dt_ms: float
    n_samples: int
    n_traces: int


@dataclass
class SegyBasicHeader:
    dt_us: int
    n_samples: int
    data_format: int


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def robust_clip(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip array using robust percentiles for stable plotting."""
    finite = np.isfinite(x)
    if not np.any(finite):
        return x
    lo, hi = np.nanpercentile(x[finite], [low, high])
    if np.isclose(lo, hi):
        return x
    return np.clip(x, lo, hi)


def robust_normalize(x: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Robust min-max normalization to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if not np.any(finite):
        return out
    lo, hi = np.nanpercentile(x[finite], [low, high])
    if np.isclose(lo, hi):
        return out
    out = (x - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out


# =============================================================================
# SEG-Y READING (unchanged: robust dual reader)
# =============================================================================

def get_time_axis_from_segy(segy_file: str, n_samples: int) -> tuple[np.ndarray, float]:
    """Return time axis in ms and sample interval in ms."""
    with segyio.open(segy_file, "r", ignore_geometry=True) as f:
        samples = np.asarray(f.samples, dtype=np.float64)

        # In most time-domain SEG-Y files, f.samples is already in ms.
        if len(samples) == n_samples and np.all(np.isfinite(samples)) and np.nanmax(samples) > 10:
            time_ms = samples.copy()
            diffs = np.diff(time_ms)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            dt_ms = float(np.median(diffs)) if diffs.size else 2.0
            return time_ms, dt_ms

        # Fallback: binary header interval is usually in microseconds.
        dt_us = None
        try:
            dt_us = segyio.tools.dt(f)
        except Exception:
            pass

        if dt_us is None or dt_us <= 0:
            try:
                dt_us = f.bin[segyio.BinField.Interval]
            except Exception:
                dt_us = 2000

        dt_ms = float(dt_us) / 1000.0
        if dt_ms <= 0 or not np.isfinite(dt_ms):
            dt_ms = 2.0
        time_ms = np.arange(n_samples, dtype=np.float64) * dt_ms
        return time_ms, dt_ms


def _read_segy_basic_header(segy_file: str) -> SegyBasicHeader:
    """Read the small subset of binary header fields needed by the fallback reader."""
    with open(segy_file, "rb") as f:
        f.seek(3200)
        binary_header = f.read(400)

    if len(binary_header) != 400:
        raise ValueError("SEG-Y file is too small to contain a complete binary header.")

    return SegyBasicHeader(
        dt_us=struct.unpack(">H", binary_header[16:18])[0],
        n_samples=struct.unpack(">H", binary_header[20:22])[0],
        data_format=struct.unpack(">H", binary_header[24:26])[0],
    )


def _ibm_float32_to_ieee(words: np.ndarray) -> np.ndarray:
    """Convert big-endian IBM 32-bit floating point words to IEEE float32."""
    words = words.astype(np.uint32, copy=False)
    sign = np.where((words & 0x80000000) != 0, -1.0, 1.0)
    exponent = ((words >> 24) & 0x7F).astype(np.int32) - 64
    fraction = (words & 0x00FFFFFF).astype(np.float64) / float(0x01000000)

    out = sign * fraction * np.power(16.0, exponent, dtype=np.float64)
    out[words == 0] = 0.0
    return out.astype(np.float32)


def _decode_trace_samples(raw: bytes, data_format: int) -> np.ndarray:
    """Decode one SEG-Y trace payload for the formats commonly used here."""
    if data_format == 1:
        words = np.frombuffer(raw, dtype=">u4")
        return _ibm_float32_to_ieee(words)
    if data_format == 5:
        return np.frombuffer(raw, dtype=">f4").astype(np.float32)
    if data_format == 2:
        return np.frombuffer(raw, dtype=">i4").astype(np.float32)
    if data_format == 3:
        return np.frombuffer(raw, dtype=">i2").astype(np.float32)
    raise ValueError(
        f"Unsupported SEG-Y sample format code {data_format}. "
        "This fallback supports IBM float, IEEE float, 32-bit int, and 16-bit int."
    )


def _read_segy_as_matrix_fallback(segy_file: str) -> SeismicData:
    """Read simple SEG-Y files without relying on segyio's global trace-count check."""
    header = _read_segy_basic_header(segy_file)
    bytes_per_sample = {1: 4, 2: 4, 3: 2, 5: 4}.get(header.data_format)
    if bytes_per_sample is None:
        raise ValueError(f"Unsupported SEG-Y sample format code {header.data_format}.")

    file_size = os.path.getsize(segy_file)
    trace_arrays: list[np.ndarray] = []
    trace_sample_counts: list[int] = []
    first_delay_ms = 0.0

    with open(segy_file, "rb") as f:
        pos = 3600
        trace_index = 0
        while pos + 240 <= file_size:
            f.seek(pos)
            trace_header = f.read(240)
            if len(trace_header) < 240:
                break

            n_samples = struct.unpack(">H", trace_header[114:116])[0] or header.n_samples
            dt_us = struct.unpack(">H", trace_header[116:118])[0] or header.dt_us
            if trace_index == 0:
                first_delay_ms = float(struct.unpack(">h", trace_header[108:110])[0])
                if first_delay_ms < 0 or not np.isfinite(first_delay_ms):
                    first_delay_ms = 0.0
                if header.dt_us <= 0:
                    header.dt_us = dt_us

            payload_bytes = n_samples * bytes_per_sample
            next_pos = pos + 240 + payload_bytes
            if next_pos > file_size:
                warnings.warn(
                    f"Stopping before incomplete trace {trace_index}: expected "
                    f"{payload_bytes} data bytes, but file ended early."
                )
                break

            raw = f.read(payload_bytes)
            trace_arrays.append(_decode_trace_samples(raw, header.data_format))
            trace_sample_counts.append(n_samples)
            pos = next_pos
            trace_index += 1

    if not trace_arrays:
        raise ValueError("No readable traces found in SEG-Y fallback reader.")

    n_samples = min(trace_sample_counts)
    if len(set(trace_sample_counts)) > 1:
        warnings.warn(
            "Variable trace lengths detected. Truncating all traces to the shortest "
            f"length ({n_samples} samples)."
        )

    data = np.column_stack([trace[:n_samples] for trace in trace_arrays]).astype(np.float32)
    dt_ms = float(header.dt_us) / 1000.0 if header.dt_us > 0 else 2.0
    time_ms = first_delay_ms + np.arange(n_samples, dtype=np.float64) * dt_ms

    leftover = file_size - pos
    if leftover > 0:
        warnings.warn(
            f"Ignored {leftover} trailing byte(s) after the last complete SEG-Y trace."
        )

    return SeismicData(
        data=data,
        time_ms=time_ms,
        dt_ms=dt_ms,
        n_samples=n_samples,
        n_traces=data.shape[1],
    )


def read_segy_as_matrix(segy_file: str) -> SeismicData:
    """Read SEG-Y file as a samples x traces matrix."""
    if not os.path.exists(segy_file):
        raise FileNotFoundError(
            f"Could not find {segy_file}. Put this script in the same folder as the SEG-Y file "
            f"or edit SEGY_FILE at the top of the script."
        )

    try:
        with segyio.open(segy_file, "r", ignore_geometry=True) as f:
            n_traces = f.tracecount
            first_trace = np.asarray(f.trace[0], dtype=np.float32)
            n_samples = first_trace.size
            data = np.empty((n_samples, n_traces), dtype=np.float32)
            for i in range(n_traces):
                data[:, i] = np.asarray(f.trace[i], dtype=np.float32)

        time_ms, dt_ms = get_time_axis_from_segy(segy_file, n_samples)
    except RuntimeError as exc:
        warnings.warn(
            "segyio could not open this file "
            f"({exc}). Falling back to the built-in sequential SEG-Y reader."
        )
        return _read_segy_as_matrix_fallback(segy_file)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return SeismicData(
        data=data,
        time_ms=time_ms,
        dt_ms=dt_ms,
        n_samples=n_samples,
        n_traces=n_traces,
    )


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_seismic(data: np.ndarray, dt_ms: float) -> np.ndarray:
    """Detrend traces and optionally apply a broad zero-phase bandpass filter."""
    x = data.astype(np.float64, copy=True)

    x -= np.nanmean(x, axis=0, keepdims=True)

    finite = np.isfinite(x)
    if np.any(finite):
        clip_val = np.nanpercentile(np.abs(x[finite]), 99.8)
        if clip_val > 0 and np.isfinite(clip_val):
            x = np.clip(x, -clip_val, clip_val)

    if APPLY_BANDPASS:
        fs = 1000.0 / dt_ms
        nyq = 0.5 * fs
        low = max(0.1, BANDPASS_LOW_HZ)
        high = min(BANDPASS_HIGH_HZ, 0.90 * nyq)
        if low < high < nyq:
            sos = butter(
                BANDPASS_ORDER,
                [low / nyq, high / nyq],
                btype="bandpass",
                output="sos",
            )
            try:
                x = sosfiltfilt(sos, x, axis=0)
            except ValueError:
                warnings.warn("Bandpass skipped because trace length is too short for filtfilt.")
        else:
            warnings.warn(
                f"Bandpass skipped. Requested band {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz "
                f"is not valid for dt={dt_ms:.3f} ms."
            )

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# HILBERT / INSTANTANEOUS ATTRIBUTES (revised)
# =============================================================================

def instantaneous_frequency(analytic: np.ndarray, dt_s: float, method: str = "analytic") -> np.ndarray:
    """
    Instantaneous frequency (Hz).

    method="analytic": f = (x*H' - H*x') / (2*pi*(x^2 + H^2)).
        This equals d(phase)/dt but is computed from the analytic signal and its
        time derivative, so it does not require phase unwrapping and avoids the
        large spurious spikes that unwrapping can introduce.

    method="unwrap": legacy f = (1/2pi) d/dt[ unwrap(angle) ].
    """
    real = np.real(analytic)
    imag = np.imag(analytic)

    if method == "unwrap":
        phase_unwrapped = np.unwrap(np.angle(analytic), axis=0)
        return np.gradient(phase_unwrapped, dt_s, axis=0) / (2.0 * np.pi)

    dxr = np.gradient(real, dt_s, axis=0)
    dxi = np.gradient(imag, dt_s, axis=0)
    denom = real ** 2 + imag ** 2
    pos = denom[denom > 0]
    eps = 1e-6 * float(np.nanmedian(pos)) if pos.size else 1e-12
    return (real * dxi - imag * dxr) / (2.0 * np.pi * (denom + eps))


def calculate_hilbert_attributes(
    data: np.ndarray,
    dt_ms: float,
    freq_method: str = FREQ_METHOD,
) -> dict[str, np.ndarray]:
    """Calculate analytic-signal/Hilbert-derived attributes."""
    analytic = hilbert(data, axis=0)

    envelope = np.abs(analytic)
    phase_wrapped = np.angle(analytic)
    cosine_phase = np.cos(phase_wrapped)

    dt_s = dt_ms / 1000.0

    # ---- Instantaneous frequency (robust analytic-signal-derivative form) ----
    inst_freq = instantaneous_frequency(analytic, dt_s, method=freq_method)
    inst_freq = median_filter(inst_freq, size=MEDIAN_FILTER_SIZE)
    inst_freq = gaussian_filter(inst_freq, sigma=(GAUSSIAN_SIGMA_SAMPLES, GAUSSIAN_SIGMA_TRACES))
    finite = np.isfinite(inst_freq)
    if np.any(finite):
        p1, p99 = np.nanpercentile(inst_freq[finite], [1, 99])
        inst_freq = np.clip(inst_freq, p1, p99)

    # ---- Envelope: amplitude-scale-invariant log compression ----
    envelope_s = gaussian_filter(envelope, sigma=(GAUSSIAN_SIGMA_SAMPLES, GAUSSIAN_SIGMA_TRACES))
    pos_env = envelope_s[np.isfinite(envelope_s) & (envelope_s > 0)]
    env_ref = float(np.nanmedian(pos_env)) if pos_env.size else 1.0
    if not np.isfinite(env_ref) or env_ref <= 0:
        env_ref = 1.0
    envelope_norm = robust_normalize(np.log1p(envelope_s / env_ref))
    # Soft PRESENCE gate (not a magnitude weight): ~0 in low-energy zones, ~1 once
    # energy is present, saturating thereafter. This removes phase noise from dead
    # zones without making phase discontinuity track envelope amplitude.
    env01 = robust_normalize(envelope_s)
    energy_gate = np.clip((env01 - 0.12) / 0.18, 0.0, 1.0)

    # ---- Envelope gradient: sampling-invariant edge strength ----
    # Normalize vertical and lateral gradients separately so the magnitude does
    # not depend on dt vs trace spacing.
    g_t = robust_normalize(np.abs(np.gradient(envelope_norm, axis=0)))
    g_x = robust_normalize(np.abs(np.gradient(envelope_norm, axis=1)))
    envelope_gradient = np.sqrt(g_t ** 2 + g_x ** 2) / np.sqrt(2.0)
    envelope_gradient = robust_normalize(envelope_gradient)

    # ---- Phase discontinuity: LATERAL phase change, energy-weighted ----
    # Lateral phase gradient |d(phase)/dx| via sin/cos (wrap-safe). The temporal
    # phase gradient is intentionally excluded because it equals 2*pi*f and lights
    # up every wavelet cycle (it is not a discontinuity). Energy weighting keeps
    # the attribute from responding to phase noise in low-amplitude zones.
    sin_phase = np.sin(phase_wrapped)
    cos_phase = np.cos(phase_wrapped)
    d_sin_x = np.gradient(sin_phase, axis=1)
    d_cos_x = np.gradient(cos_phase, axis=1)
    lateral_phase_grad = np.sqrt(d_sin_x ** 2 + d_cos_x ** 2)
    phase_discontinuity = lateral_phase_grad * energy_gate
    phase_discontinuity = gaussian_filter(
        phase_discontinuity, sigma=(GAUSSIAN_SIGMA_SAMPLES, GAUSSIAN_SIGMA_TRACES)
    )
    phase_discontinuity = robust_normalize(phase_discontinuity)

    # ---- Frequency anomaly: local deviation from smoothed background frequency ----
    freq_s = gaussian_filter(inst_freq, sigma=(2.0, 1.0))
    freq_background = gaussian_filter(freq_s, sigma=(20.0, 5.0))
    frequency_anomaly = robust_normalize(np.abs(freq_s - freq_background))

    # ---- Composite Hilbert-transform-based attribute ----
    composite = composite_from_components(
        envelope_norm,
        envelope_gradient,
        phase_discontinuity,
        frequency_anomaly,
        [W_ENVELOPE, W_ENVELOPE_GRADIENT, W_PHASE_DISCONTINUITY, W_FREQUENCY_ANOMALY],
    )

    return {
        "analytic_real": np.real(analytic),
        "analytic_imag": np.imag(analytic),
        "envelope": envelope,
        "envelope_norm": envelope_norm,
        "phase_wrapped": phase_wrapped,
        "cosine_phase": cosine_phase,
        "lateral_phase_grad": lateral_phase_grad,
        "phase_discontinuity": phase_discontinuity,
        "instantaneous_frequency": inst_freq,
        "frequency_anomaly": frequency_anomaly,
        "envelope_gradient": envelope_gradient,
        "composite": composite,
    }


def composite_from_components(
    envelope_norm: np.ndarray,
    envelope_gradient: np.ndarray,
    phase_discontinuity: np.ndarray,
    frequency_anomaly: np.ndarray,
    weights: list[float],
) -> np.ndarray:
    """Weighted, renormalized, lightly smoothed composite from the four components."""
    w = np.asarray(weights, dtype=float)
    raw = (
        w[0] * envelope_norm
        + w[1] * envelope_gradient
        + w[2] * phase_discontinuity
        + w[3] * frequency_anomaly
    )
    composite = robust_normalize(raw)
    composite = gaussian_filter(composite, sigma=(0.6, 0.4))  # reduced vs first version
    return robust_normalize(composite)


# =============================================================================
# AUTOMATIC INTERVAL SELECTION (revised: multi-length search + reporting)
# =============================================================================

def _score_window(w_data, w_env, w_phase):
    amp_rms = float(np.sqrt(np.nanmean(w_data ** 2)))
    env_contrast = float(np.nanpercentile(w_env, 95) - np.nanpercentile(w_env, 50))
    lateral_variability = float(np.nanmean(np.abs(np.diff(w_data, axis=1))))
    vertical_variability = float(np.nanmean(np.abs(np.diff(w_data, axis=0))))
    phase_edge_strength = float(np.nanmean(w_phase))
    return amp_rms, env_contrast, lateral_variability, vertical_variability, phase_edge_strength


def select_interesting_interval(
    data: np.ndarray,
    time_ms: np.ndarray,
    dt_ms: float,
    max_time_ms: float,
    min_time_ms: float,
    window_ms_list: list[float],
    step_ms: float,
) -> tuple[float, float, pd.DataFrame, dict]:
    """Automatically select a geologically informative window (length + position)."""
    info: dict = {}
    mask = (time_ms >= min_time_ms) & (time_ms <= max_time_ms)
    if not np.any(mask):
        return float(time_ms[0]), float(time_ms[-1]), pd.DataFrame(), info

    work_time = time_ms[mask]
    work_data = data[mask, :]

    # Attributes used only for scoring window quality.
    attrs = calculate_hilbert_attributes(work_data, dt_ms)
    env = attrs["envelope"]
    phase_disc = attrs["phase_discontinuity"]

    step_samples = max(5, int(round(step_ms / dt_ms)))
    rows = []
    for win_ms in window_ms_list:
        window_samples = max(20, int(round(win_ms / dt_ms)))
        if window_samples >= work_data.shape[0]:
            continue
        for start in range(0, work_data.shape[0] - window_samples + 1, step_samples):
            end = start + window_samples
            a, ec, lv, vv, pe = _score_window(
                work_data[start:end, :], env[start:end, :], phase_disc[start:end, :]
            )
            rows.append(
                {
                    "tmin_ms": float(work_time[start]),
                    "tmax_ms": float(work_time[end - 1]),
                    "window_ms": float(win_ms),
                    "amp_rms": a,
                    "env_contrast": ec,
                    "lateral_variability": lv,
                    "vertical_variability": vv,
                    "phase_edge_strength": pe,
                }
            )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return float(work_time[0]), float(work_time[-1]), score_df, info

    # Normalize each scoring component across ALL candidate windows.
    for col in ["amp_rms", "env_contrast", "lateral_variability",
                "vertical_variability", "phase_edge_strength"]:
        vals = score_df[col].to_numpy()
        vmin, vmax = np.nanpercentile(vals, [5, 95])
        if np.isclose(vmin, vmax):
            score_df[col + "_n"] = 0.0
        else:
            score_df[col + "_n"] = np.clip((vals - vmin) / (vmax - vmin), 0, 1)

    score_df["score"] = (
        0.25 * score_df["amp_rms_n"]
        + 0.25 * score_df["env_contrast_n"]
        + 0.15 * score_df["lateral_variability_n"]
        + 0.15 * score_df["vertical_variability_n"]
        + 0.20 * score_df["phase_edge_strength_n"]
    )
    score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

    best = score_df.iloc[0]
    best_tmin, best_tmax = float(best["tmin_ms"]), float(best["tmax_ms"])

    # Best window that does NOT substantially overlap the chosen one (>50% overlap).
    def _overlap_frac(r):
        lo = max(best_tmin, r["tmin_ms"])
        hi = min(best_tmax, r["tmax_ms"])
        inter = max(0.0, hi - lo)
        denom = min(best_tmax - best_tmin, r["tmax_ms"] - r["tmin_ms"])
        return inter / denom if denom > 0 else 1.0

    runner_score = np.nan
    runner_row = None
    for _, r in score_df.iloc[1:].iterrows():
        if _overlap_frac(r) < 0.5:
            runner_score = float(r["score"])
            runner_row = r
            break

    info["best_score"] = float(best["score"])
    info["best_window_ms"] = float(best["window_ms"])
    info["runner_up_score"] = runner_score
    if runner_row is not None:
        info["runner_up_interval"] = (float(runner_row["tmin_ms"]), float(runner_row["tmax_ms"]))
        info["score_margin"] = float(best["score"]) - runner_score
        info["relative_margin"] = (
            (float(best["score"]) - runner_score) / float(best["score"])
            if best["score"] > 0 else np.nan
        )
    return best_tmin, best_tmax, score_df, info


def crop_by_time(data: np.ndarray, time_ms: np.ndarray, tmin_ms: float, tmax_ms: float):
    mask = (time_ms >= tmin_ms) & (time_ms <= tmax_ms)
    return data[mask, :], time_ms[mask], mask


def compare_deep_vs_selected(
    data: np.ndarray,
    time_ms: np.ndarray,
    dt_ms: float,
    sel_tmin: float,
    sel_tmax: float,
    deep_threshold_ms: float,
) -> pd.DataFrame:
    """Quantify how the excluded deep section compares with the selected interval."""
    regions = {
        "selected_interval": (time_ms >= sel_tmin) & (time_ms <= sel_tmax),
        "excluded_deep": time_ms > deep_threshold_ms,
    }
    rows = []
    for name, m in regions.items():
        if not np.any(m):
            continue
        d = data[m, :]
        env = np.abs(hilbert(d, axis=0))
        rows.append(
            {
                "region": name,
                "tmin_ms": float(time_ms[m][0]),
                "tmax_ms": float(time_ms[m][-1]),
                "n_samples": int(d.shape[0]),
                "amp_rms": float(np.sqrt(np.nanmean(d ** 2))),
                "mean_envelope": float(np.nanmean(env)),
                "env_contrast_p95_minus_p50": float(
                    np.nanpercentile(env, 95) - np.nanpercentile(env, 50)
                ),
                "lateral_variability": float(np.nanmean(np.abs(np.diff(d, axis=1)))),
                "vertical_variability": float(np.nanmean(np.abs(np.diff(d, axis=0)))),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# REDUNDANCY / CORRELATION (NEW)
# =============================================================================

def attribute_correlation(attrs: dict[str, np.ndarray], max_samples: int = 50000):
    """Pearson + Spearman correlation among the four components and the composite."""
    rng = np.random.default_rng(42)
    names = ["envelope_norm", "envelope_gradient", "phase_discontinuity",
             "frequency_anomaly", "composite"]
    cols = [attrs[n].ravel() for n in names]
    finite = np.all(np.isfinite(np.vstack(cols)), axis=0)
    idx = np.where(finite)[0]
    if idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    mat = np.vstack([c[idx] for c in cols])
    pearson = np.corrcoef(mat)
    spearman = spearmanr(mat, axis=1).correlation
    pear_df = pd.DataFrame(pearson, index=names, columns=names)
    spear_df = pd.DataFrame(spearman, index=names, columns=names)
    return pear_df, spear_df, names, idx


def plot_correlation_redundancy(attrs: dict[str, np.ndarray], output_file: str) -> pd.DataFrame:
    pear_df, spear_df, names, idx = attribute_correlation(attrs)

    comp_names = ["envelope_norm", "envelope_gradient", "phase_discontinuity", "frequency_anomaly"]
    e = attrs["envelope_gradient"].ravel()[idx]
    d = attrs["phase_discontinuity"].ravel()[idx]
    f = attrs["frequency_anomaly"].ravel()[idx]
    en = attrs["envelope_norm"].ravel()[idx]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(pear_df.loc[comp_names, comp_names].to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax0.set_xticks(range(len(comp_names)))
    ax0.set_yticks(range(len(comp_names)))
    short = ["env", "env grad", "phase disc", "freq anom"]
    ax0.set_xticklabels(short, rotation=30, ha="right")
    ax0.set_yticklabels(short)
    ax0.set_title("Component correlation (Pearson)")
    for i in range(len(comp_names)):
        for j in range(len(comp_names)):
            ax0.text(j, i, f"{pear_df.loc[comp_names[i], comp_names[j]]:.2f}",
                     ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(d, f, s=2, alpha=0.2)
    ax1.set_xlabel("Phase discontinuity")
    ax1.set_ylabel("Frequency anomaly")
    ax1.set_title(f"Phase disc. vs frequency anomaly\nPearson r = {pear_df.loc['phase_discontinuity','frequency_anomaly']:.2f}")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(e, d, s=2, alpha=0.2)
    ax2.set_xlabel("Envelope gradient")
    ax2.set_ylabel("Phase discontinuity")
    ax2.set_title(f"Envelope grad. vs phase disc.\nPearson r = {pear_df.loc['envelope_gradient','phase_discontinuity']:.2f}")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(en, e, s=2, alpha=0.2)
    ax3.set_xlabel("Envelope")
    ax3.set_ylabel("Envelope gradient")
    ax3.set_title(f"Envelope vs envelope grad.\nPearson r = {pear_df.loc['envelope_norm','envelope_gradient']:.2f}")
    ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(en, f, s=2, alpha=0.2)
    ax4.set_xlabel("Envelope")
    ax4.set_ylabel("Frequency anomaly")
    ax4.set_title(f"Envelope vs frequency anomaly\nPearson r = {pear_df.loc['envelope_norm','frequency_anomaly']:.2f}")
    ax4.grid(True, alpha=0.25)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(en, d, s=2, alpha=0.2)
    ax5.set_xlabel("Envelope")
    ax5.set_ylabel("Phase discontinuity")
    ax5.set_title(f"Envelope vs phase disc.\nPearson r = {pear_df.loc['envelope_norm','phase_discontinuity']:.2f}")
    ax5.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()
    return pear_df


# =============================================================================
# WEIGHT SENSITIVITY (NEW)
# =============================================================================

def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else np.nan


def run_weight_sensitivity(
    attrs: dict[str, np.ndarray],
    weight_sets: list[tuple[str, list[float]]],
    percentile: float,
    output_file: str,
) -> pd.DataFrame:
    """Top-percentile boundary masks for several weight choices and their overlap."""
    labels = [w[0] for w in weight_sets]
    masks = []
    for _, w in weight_sets:
        comp = composite_from_components(
            attrs["envelope_norm"], attrs["envelope_gradient"],
            attrs["phase_discontinuity"], attrs["frequency_anomaly"], w,
        )
        thr = np.nanpercentile(comp, percentile)
        masks.append(comp >= thr)

    n = len(masks)
    iou = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            iou[i, j] = jaccard(masks[i], masks[j])
    iou_df = pd.DataFrame(iou, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(iou, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([lbl.split(" (")[0] for lbl in labels], rotation=30, ha="right")
    ax.set_yticklabels([lbl.split(" (")[0] for lbl in labels])
    ax.set_title(
        f"Overlap (Jaccard/IoU) of the top {100 - percentile:.0f}% boundary map\n"
        f"under different composite weights"
    )
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{iou[i, j]:.2f}", ha="center", va="center",
                    color="white" if iou[i, j] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="IoU")
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()
    return iou_df


# =============================================================================
# SYNTHETIC VALIDATION (NEW)
# =============================================================================

def ricker_wavelet(f0_hz: float, dt_s: float, length_s: float = 0.2) -> np.ndarray:
    t = np.arange(-length_s / 2.0, length_s / 2.0 + dt_s, dt_s)
    a = (np.pi * f0_hz * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def build_synthetic_validation_model(seed: int = 7):
    """
    Build a controlled, NOISE-FREE 2D model with a known reflectivity field on
    top of a gently folded, dipping, layered background that also contains:
      - a normal fault (lateral offset of all reflectors),
      - a pinch-out / reflector termination (a layer thinning to zero), and
      - a lateral facies change (an impedance contrast that changes laterally).

    The validation target is REFLECTING-BOUNDARY DETECTION: the ground truth is
    the set of strong reflectors (the largest |reflectivity| samples), which
    naturally includes the fault-juxtaposed reflectors, the terminating layer
    and the facies-contrast change. Noise is added separately at a controlled
    SNR by add_noise(), so the same model can be scored across a noise sweep.

    NOTE (honest scope): the target here is detection of reflecting boundaries,
    NOT isolation of lateral discontinuities from structural dip. Because the
    method does not perform dip steering, the lateral phase-gradient term cannot
    by itself separate a fault/termination from ordinary reflector dip; that
    harder task is discussed as a limitation rather than claimed as a result.
    """
    rng = np.random.default_rng(seed)
    dt_ms = 4.0
    n_samples = 600
    n_traces = 300
    time_ms = np.arange(n_samples) * dt_ms
    traces = np.arange(n_traces)

    base_boundaries = np.array([60, 130, 175, 240, 300, 360, 430, 500], dtype=float)
    layer_imp = np.array([2.0, 3.2, 2.6, 4.0, 3.0, 4.6, 3.4, 5.0, 3.8])  # arbitrary units

    dip = (traces - n_traces / 2.0) * 0.03
    fold = -18.0 * np.exp(-((traces - n_traces * 0.5) / (n_traces * 0.18)) ** 2)
    structure = dip + fold

    fault_trace = 150
    fault_throw = 14.0
    fault_shift = np.where(traces >= fault_trace, fault_throw, 0.0)

    termination_trace = 80   # the layer between boundaries 1 and 2 pinches out to the left
    facies_trace = 220       # impedance of layer index 3 changes laterally here

    Z = np.zeros((n_samples, n_traces))
    for it in range(n_traces):
        b = base_boundaries + structure[it] + fault_shift[it]
        if it < termination_trace:
            frac = (termination_trace - it) / termination_trace
            b[2] = b[1] + (b[2] - b[1]) * (1.0 - frac)  # thickness -> 0 to the left
        b_sorted = np.sort(b)
        layer_idx = np.searchsorted(b_sorted, np.arange(n_samples), side="right")
        imp = layer_imp.copy()
        if it >= facies_trace:
            imp[3] *= 1.35
        layer_idx = np.clip(layer_idx, 0, len(imp) - 1)
        Z[:, it] = imp[layer_idx]

    # Reflectivity and convolution (noise-free seismic).
    r = np.zeros_like(Z)
    r[1:, :] = (Z[1:, :] - Z[:-1, :]) / (Z[1:, :] + Z[:-1, :])
    w = ricker_wavelet(30.0, dt_ms / 1000.0)
    seismic_clean = np.empty_like(r)
    for it in range(n_traces):
        seismic_clean[:, it] = np.convolve(r[:, it], w, mode="same")

    # Ground truth = reflecting-boundary ZONES. Threshold the largest
    # |reflectivity| samples, then dilate to the wavelet's main-lobe extent so a
    # boundary is a band a few samples thick (the reflection event occupies the
    # wavelet main lobe), not a single peak sample. This scores all attributes
    # against the physical reflection zone rather than one peak row.
    r_abs = np.abs(r)
    nz = r_abs[r_abs > 1e-6]
    thr = np.percentile(nz, 90.0)            # top ~10% of non-zero reflectors
    boundary = r_abs >= thr
    boundary = binary_dilation(boundary, structure=np.ones((9, 3), dtype=bool))

    # Evaluation region: exclude a margin to avoid convolution/edge artifacts.
    eval_mask = np.zeros((n_samples, n_traces), dtype=bool)
    eval_mask[12:n_samples - 12, 3:n_traces - 3] = True

    return {
        "dt_ms": dt_ms,
        "time_ms": time_ms,
        "impedance": Z,
        "reflectivity": r,
        "seismic_clean": seismic_clean,
        "boundary_mask": boundary,
        "eval_mask": eval_mask,
        "fault_trace": fault_trace,
        "termination_trace": termination_trace,
        "facies_trace": facies_trace,
    }


def add_noise(seismic_clean: np.ndarray, snr: float, rng) -> np.ndarray:
    """Add band-limited (5-60 Hz) Gaussian noise at a target RMS amplitude SNR."""
    dt_ms = 4.0
    noise = rng.standard_normal(seismic_clean.shape)
    fs = 1000.0 / dt_ms
    nyq = 0.5 * fs
    sos = butter(4, [5.0 / nyq, 60.0 / nyq], btype="bandpass", output="sos")
    noise = sosfiltfilt(sos, noise, axis=0)
    s_rms = np.sqrt(np.nanmean(seismic_clean ** 2))
    n_rms = np.sqrt(np.nanmean(noise ** 2)) + 1e-12
    noise *= s_rms / (snr * n_rms)
    return seismic_clean + noise


def roc_auc_simple(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_at_topk(scores: np.ndarray, labels: np.ndarray, frac: float):
    labels = labels.astype(bool)
    k = max(1, int(round(frac * scores.size)))
    top_idx = np.argsort(scores)[::-1][:k]
    pred = np.zeros(scores.size, dtype=bool)
    pred[top_idx] = True
    tp = int(np.logical_and(pred, labels).sum())
    precision = tp / max(1, int(pred.sum()))
    recall = tp / max(1, int(labels.sum()))
    return float(precision), float(recall)


# Attributes scored on the synthetic model and the SNR levels of the sweep.
SYNTH_ATTRS = ["envelope_norm", "envelope_gradient", "phase_discontinuity",
               "frequency_anomaly", "composite"]
SYNTH_SNR_LIST = [8.0, 4.0, 2.0, 1.0]
SYNTH_REP_SNR = 2.0   # SNR used for the example maps in fig09


def run_synthetic_validation(output_dir: str) -> pd.DataFrame:
    """
    Run the SAME attribute workflow on a known model across a noise sweep and
    score boundary detection (ROC AUC + precision/recall at a matched budget)
    at each SNR. The headline result is robustness to noise: how each attribute,
    and the composite, degrade as SNR decreases.
    """
    model = build_synthetic_validation_model()
    boundary = model["boundary_mask"]
    eval_idx = np.where(model["eval_mask"].ravel())[0]
    labels = boundary.ravel()[eval_idx]
    pos_rate = float(np.clip(labels.mean(), 0.02, 0.20))  # matched flag budget

    rep_attrs, rep_noisy = None, None
    rows = []
    for snr in SYNTH_SNR_LIST:
        rng = np.random.default_rng(1000 + int(round(snr * 10)))  # reproducible
        noisy = add_noise(model["seismic_clean"], snr, rng)
        seismic = preprocess_seismic(noisy, model["dt_ms"])
        attrs = calculate_hilbert_attributes(seismic, model["dt_ms"])
        for name in SYNTH_ATTRS:
            sc = attrs[name].ravel()[eval_idx]
            auc = roc_auc_simple(sc, labels)
            prec, rec = precision_recall_at_topk(sc, labels, frac=pos_rate)
            rows.append({"snr": snr, "attribute": name, "roc_auc": auc,
                         "precision_matched": prec, "recall_matched": rec})
        if abs(snr - SYNTH_REP_SNR) < 1e-6:
            rep_attrs, rep_noisy = attrs, noisy

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(os.path.join(output_dir, "synthetic_detection_scores.csv"),
                     index=False)

    if rep_attrs is None:  # safety: fall back to the last computed attrs
        rep_attrs, rep_noisy = attrs, noisy
    _plot_synthetic_maps(
        model, rep_attrs, rep_noisy, SYNTH_REP_SNR,
        os.path.join(output_dir, "fig09_synthetic_validation.png")
    )
    _plot_synthetic_scores(
        scores_df, os.path.join(output_dir, "fig10_synthetic_detection_scores.png")
    )
    return scores_df


def _plot_synthetic_maps(model, attrs, noisy, rep_snr, output_file):
    time_ms = model["time_ms"]
    n_traces = model["seismic_clean"].shape[1]
    traces = np.arange(1, n_traces + 1)
    extent = [traces[0], traces[-1], time_ms[-1], time_ms[0]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [
        (model["impedance"], "viridis", "Synthetic impedance model", "Impedance (a.u.)", True),
        (robust_clip(noisy), "gray", f"Synthetic seismic (SNR = {rep_snr:g})", "Amplitude", False),
        (model["boundary_mask"].astype(float), "Reds",
         "Strong reflecting boundaries (ground truth)", "Mask", False),
        (robust_clip(attrs["envelope"]), "viridis", "Envelope", "Envelope", False),
        (attrs["phase_discontinuity"], "viridis",
         "Phase discontinuity (lateral, energy-gated)", "Value", False),
        (attrs["composite"], "viridis", "Composite attribute", "Value", False),
    ]
    for ax, (arr, cmap, title, cbar, robust) in zip(axes.ravel(), panels):
        plot_arr = robust_clip(arr) if robust else arr
        im = ax.imshow(plot_arr, cmap=cmap, aspect="auto", extent=extent,
                       interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Trace number")
        ax.set_ylabel("Two-way time (ms)")
        fig.colorbar(im, ax=ax, pad=0.01, label=cbar)
    fig.suptitle("Synthetic validation: known model and attribute response "
                 f"at SNR = {rep_snr:g}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def _plot_synthetic_scores(scores_df, output_file):
    snrs = sorted(scores_df["snr"].unique(), reverse=True)  # 8 -> 1 (noisier right)
    x = np.arange(len(snrs))
    attrs = [a for a in SYNTH_ATTRS if a in scores_df["attribute"].unique()]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    for name in attrs:
        sub = scores_df[scores_df["attribute"] == name].set_index("snr")
        y_auc = [sub.loc[s, "roc_auc"] for s in snrs]
        y_pr = [sub.loc[s, "precision_matched"] for s in snrs]
        is_comp = (name == "composite")
        style = dict(lw=3.0, marker="o", ms=7, color="black", zorder=5) if is_comp \
            else dict(lw=1.8, marker="o", ms=5)
        axA.plot(x, y_auc, label=name, **style)
        axB.plot(x, y_pr, label=name, **style)

    for ax, ttl, ylab in [(axA, "Boundary detection vs noise (ROC AUC)", "ROC AUC"),
                          (axB, "Detection at a matched flag budget", "Precision (= recall)")]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:g}" for s in snrs])
        ax.set_xlabel("Signal-to-noise ratio (decreasing ->)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    axA.axhline(0.5, color="gray", ls="--", lw=1, label="AUC = 0.5 (chance)")
    axA.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()


# =============================================================================
# PLOTTING (real data)
# =============================================================================

def plot_image(arr, time_ms, title, filename, cmap="viridis", robust=True,
               cbar_label=None, aspect="auto") -> None:
    traces = np.arange(1, arr.shape[1] + 1)
    plot_arr = robust_clip(arr) if robust else arr
    plt.figure(figsize=(FIG_WIDTH, 7))
    im = plt.imshow(plot_arr, cmap=cmap, aspect=aspect,
                    extent=[traces[0], traces[-1], time_ms[-1], time_ms[0]],
                    interpolation="nearest")
    plt.xlabel("Trace number")
    plt.ylabel("Two-way time (ms)")
    plt.title(title)
    cbar = plt.colorbar(im, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_full_section_with_window(data, time_ms, tmin_ms, tmax_ms, score_df, output_file) -> None:
    traces = np.arange(1, data.shape[1] + 1)
    plot_arr = robust_clip(data)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [2.2, 1.0]})

    im = axes[0].imshow(plot_arr, cmap=CMAP_SEISMIC, aspect="auto",
                        extent=[traces[0], traces[-1], time_ms[-1], time_ms[0]],
                        interpolation="nearest")
    axes[0].axhspan(tmin_ms, tmax_ms, color="yellow", alpha=0.22, label="Selected interval")
    axes[0].set_xlabel("Trace number")
    axes[0].set_ylabel("Two-way time (ms)")
    axes[0].set_title("Full seismic section with automatically selected interval")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0], pad=0.01, label="Amplitude")

    if score_df is not None and not score_df.empty:
        # Plot the best-scoring window per time-center for a clean monotone curve.
        df = score_df.copy()
        df["center_t"] = 0.5 * (df["tmin_ms"] + df["tmax_ms"])
        df = df.sort_values("center_t")
        axes[1].plot(df["score"], df["center_t"], linewidth=1.2, alpha=0.8)
        axes[1].axhspan(tmin_ms, tmax_ms, color="yellow", alpha=0.22)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Automatic interval score")
        axes[1].set_ylabel("Two-way time (ms)")
        axes[1].set_title("Window score (all lengths)")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Manual or full interval used", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_hilbert_panel(seismic, time_ms, attrs, output_file) -> None:
    traces = np.arange(1, seismic.shape[1] + 1)
    extent = [traces[0], traces[-1], time_ms[-1], time_ms[0]]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [
        (robust_clip(seismic), CMAP_SEISMIC, "Selected seismic section", "Amplitude"),
        (robust_clip(attrs["envelope"]), CMAP_ATTRIBUTE, "Instantaneous amplitude / envelope", "Envelope"),
        (attrs["cosine_phase"], CMAP_PHASE, "Cosine of instantaneous phase", "cos(phase)"),
        (robust_clip(attrs["instantaneous_frequency"], 2, 98), CMAP_FREQ, "Instantaneous frequency", "Hz"),
    ]
    for ax, (arr, cmap, title, cbar_label) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, aspect="auto", extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Trace number")
        ax.set_ylabel("Two-way time (ms)")
        fig.colorbar(im, ax=ax, pad=0.01, label=cbar_label)
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_overlay(seismic, composite, time_ms, output_file) -> None:
    traces = np.arange(1, seismic.shape[1] + 1)
    extent = [traces[0], traces[-1], time_ms[-1], time_ms[0]]
    threshold = np.nanpercentile(composite, BOUNDARY_PERCENTILE)
    boundary_mask = np.where(composite >= threshold, composite, np.nan)

    plt.figure(figsize=(FIG_WIDTH, 7))
    plt.imshow(robust_clip(seismic), cmap=CMAP_SEISMIC, aspect="auto",
               extent=extent, interpolation="nearest")
    im = plt.imshow(boundary_mask, cmap="autumn", alpha=0.55, aspect="auto",
                    extent=extent, interpolation="nearest")
    plt.xlabel("Trace number")
    plt.ylabel("Two-way time (ms)")
    plt.title(
        f"Composite Hilbert-transform-based boundary attribute over seismic section\n"
        f"Highlighted values: top {100 - BOUNDARY_PERCENTILE:.0f}%"
    )
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Composite boundary attribute")
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def smooth_curve_for_visualization(curve, window_samples) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    if window_samples <= 1 or curve.size < 3:
        return curve
    window_samples = int(window_samples)
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = min(window_samples, curve.size)
    if window_samples < 3:
        return curve
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    pad = window_samples // 2
    padded = np.pad(curve, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_split_vertical_attribute_curves(time_ms, mean_envelope, mean_envelope_gradient,
                                          mean_phase_discontinuity, mean_frequency_anomaly,
                                          mean_composite, output_dir) -> None:
    component_curves = {
        "Mean envelope": mean_envelope,
        "Mean envelope gradient": mean_envelope_gradient,
        "Mean phase discontinuity": mean_phase_discontinuity,
    }
    frequency_composite_curves = {
        "Mean frequency anomaly": mean_frequency_anomaly,
        "Mean composite attribute": mean_composite,
    }
    figure_specs = [
        (component_curves, "Vertical variation of Hilbert-derived component attributes",
         os.path.join(output_dir, "fig06a_hilbert_component_curves.png")),
        (frequency_composite_curves, "Vertical variation of frequency anomaly and composite boundary attribute",
         os.path.join(output_dir, "fig06b_frequency_composite_curves.png")),
    ]
    for curves, title, output_file in figure_specs:
        plt.figure(figsize=(8, 8))
        for name, curve in curves.items():
            plot_curve = smooth_curve_for_visualization(curve, VISUAL_SMOOTH_SAMPLES)
            plt.plot(plot_curve, time_ms, linewidth=2.0, label=name)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalized mean attribute value")
        ax.set_ylabel("Two-way time (ms)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, borderaxespad=0.0)
        plt.tight_layout()
        plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
        plt.close()


# =============================================================================
# SAVE METRICS
# =============================================================================

def save_metrics(seismic, time_ms, attrs, tmin_ms, tmax_ms, score_df, sel_info,
                 deep_df, output_dir) -> None:
    metrics = []
    for name in ["envelope_norm", "envelope_gradient", "phase_discontinuity",
                 "instantaneous_frequency", "frequency_anomaly", "composite"]:
        arr = attrs[name]
        finite = np.isfinite(arr)
        metrics.append({
            "attribute": name,
            "mean": float(np.nanmean(arr[finite])) if np.any(finite) else np.nan,
            "std": float(np.nanstd(arr[finite])) if np.any(finite) else np.nan,
            "p05": float(np.nanpercentile(arr[finite], 5)) if np.any(finite) else np.nan,
            "p50": float(np.nanpercentile(arr[finite], 50)) if np.any(finite) else np.nan,
            "p95": float(np.nanpercentile(arr[finite], 95)) if np.any(finite) else np.nan,
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "hilbert_attribute_metrics.csv"), index=False)

    if score_df is not None and not score_df.empty:
        score_df.to_csv(os.path.join(output_dir, "automatic_interval_scores.csv"), index=False)
    if deep_df is not None and not deep_df.empty:
        deep_df.to_csv(os.path.join(output_dir, "deep_vs_selected_stats.csv"), index=False)

    np.savez_compressed(
        os.path.join(output_dir, "attribute_arrays_selected_interval.npz"),
        seismic=seismic.astype(np.float32),
        time_ms=time_ms.astype(np.float32),
        envelope=attrs["envelope"].astype(np.float32),
        envelope_norm=attrs["envelope_norm"].astype(np.float32),
        phase_wrapped=attrs["phase_wrapped"].astype(np.float32),
        cosine_phase=attrs["cosine_phase"].astype(np.float32),
        instantaneous_frequency=attrs["instantaneous_frequency"].astype(np.float32),
        envelope_gradient=attrs["envelope_gradient"].astype(np.float32),
        phase_discontinuity=attrs["phase_discontinuity"].astype(np.float32),
        frequency_anomaly=attrs["frequency_anomaly"].astype(np.float32),
        composite=attrs["composite"].astype(np.float32),
    )

    summary_path = os.path.join(output_dir, "selected_interval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Hilbert-transform-based composite attribute workflow (revised)\n")
        f.write("==============================================================\n\n")
        f.write(f"SEG-Y file: {SEGY_FILE}\n")
        f.write(f"Instantaneous-frequency estimator: {FREQ_METHOD}\n")
        f.write(f"Selected interval: {tmin_ms:.1f} - {tmax_ms:.1f} ms\n")
        f.write(f"Selected interval length: {tmax_ms - tmin_ms:.1f} ms\n")
        f.write(f"Number of traces: {seismic.shape[1]}\n")
        f.write(f"Number of samples in selected interval: {seismic.shape[0]}\n")
        if sel_info:
            f.write("\nInterval-selection diagnostics:\n")
            f.write(f"  Candidate window lengths (ms): {AUTO_WINDOW_MS_LIST}\n")
            for k in ["best_score", "best_window_ms", "runner_up_score",
                      "runner_up_interval", "score_margin", "relative_margin"]:
                if k in sel_info:
                    f.write(f"  {k}: {sel_info[k]}\n")
        f.write("\nComposite formula used in the code:\n")
        f.write(
            "Composite = normalize("
            f"{W_ENVELOPE}*Envelope_norm + "
            f"{W_ENVELOPE_GRADIENT}*Envelope_gradient + "
            f"{W_PHASE_DISCONTINUITY}*Phase_discontinuity + "
            f"{W_FREQUENCY_ANOMALY}*Frequency_anomaly)\n"
        )
        f.write("\nAttribute metrics (selected interval):\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")
        if deep_df is not None and not deep_df.empty:
            f.write("\nSelected interval vs excluded deep section:\n")
            f.write(deep_df.to_string(index=False))
            f.write("\n")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Reading SEG-Y file...")
    sgy = read_segy_as_matrix(SEGY_FILE)
    print(f"Loaded: {sgy.n_traces} traces x {sgy.n_samples} samples")
    print(f"Time range: {sgy.time_ms[0]:.1f} - {sgy.time_ms[-1]:.1f} ms")
    print(f"Sample interval: {sgy.dt_ms:.3f} ms")

    print("Preprocessing seismic data...")
    data_proc = preprocess_seismic(sgy.data, sgy.dt_ms)

    sel_info: dict = {}
    if MANUAL_TMIN_MS is not None and MANUAL_TMAX_MS is not None:
        tmin_ms = float(MANUAL_TMIN_MS)
        tmax_ms = float(MANUAL_TMAX_MS)
        score_df = pd.DataFrame()
        print(f"Manual interval used: {tmin_ms:.1f} - {tmax_ms:.1f} ms")
    else:
        print("Selecting geologically informative interval automatically (multi-length search)...")
        tmin_ms, tmax_ms, score_df, sel_info = select_interesting_interval(
            data_proc, sgy.time_ms, sgy.dt_ms,
            max_time_ms=MAX_INTERPRET_TIME_MS, min_time_ms=MIN_INTERPRET_TIME_MS,
            window_ms_list=AUTO_WINDOW_MS_LIST, step_ms=AUTO_STEP_MS,
        )
        print(f"Automatically selected interval: {tmin_ms:.1f} - {tmax_ms:.1f} ms "
              f"(length {tmax_ms - tmin_ms:.1f} ms)")
        if sel_info.get("score_margin") is not None:
            print(f"  Best score {sel_info.get('best_score'):.3f}, "
                  f"runner-up {sel_info.get('runner_up_score')}, "
                  f"margin {sel_info.get('score_margin')}")

    seismic_sel, time_sel, _ = crop_by_time(data_proc, sgy.time_ms, tmin_ms, tmax_ms)

    print("Calculating Hilbert-derived attributes for selected interval...")
    attrs = calculate_hilbert_attributes(seismic_sel, sgy.dt_ms)

    print("Comparing selected interval with excluded deep section...")
    deep_df = compare_deep_vs_selected(
        data_proc, sgy.time_ms, sgy.dt_ms, tmin_ms, tmax_ms, MAX_INTERPRET_TIME_MS
    )

    print("Saving figures...")
    plot_full_section_with_window(
        data_proc, sgy.time_ms, tmin_ms, tmax_ms, score_df,
        os.path.join(OUTPUT_DIR, "fig01_full_section_auto_window.png"),
    )
    plot_image(seismic_sel, time_sel, "Selected 2D post-stack seismic time section",
               os.path.join(OUTPUT_DIR, "fig02_selected_seismic_section.png"),
               cmap=CMAP_SEISMIC, cbar_label="Amplitude")
    plot_hilbert_panel(seismic_sel, time_sel, attrs,
                       os.path.join(OUTPUT_DIR, "fig03_hilbert_attributes.png"))
    plot_image(attrs["composite"], time_sel,
               "Hilbert-transform-based composite attribute for boundary enhancement",
               os.path.join(OUTPUT_DIR, "fig04_composite_attribute.png"),
               cmap=CMAP_ATTRIBUTE, robust=False, cbar_label="Composite attribute value")
    plot_overlay(seismic_sel, attrs["composite"], time_sel,
                 os.path.join(OUTPUT_DIR, "fig05_composite_overlay_on_seismic.png"))
    plot_split_vertical_attribute_curves(
        time_sel,
        robust_normalize(np.nanmean(attrs["envelope"], axis=1)),
        robust_normalize(np.nanmean(attrs["envelope_gradient"], axis=1)),
        robust_normalize(np.nanmean(attrs["phase_discontinuity"], axis=1)),
        robust_normalize(np.nanmean(attrs["frequency_anomaly"], axis=1)),
        robust_normalize(np.nanmean(attrs["composite"], axis=1)),
        OUTPUT_DIR,
    )

    print("Computing attribute redundancy (correlation + crossplots)...")
    pear_df = plot_correlation_redundancy(
        attrs, os.path.join(OUTPUT_DIR, "fig07_attribute_redundancy.png"))
    pear_df.to_csv(os.path.join(OUTPUT_DIR, "attribute_correlation_matrix.csv"))
    print("Component correlation (Pearson):")
    print(pear_df.round(2).to_string())

    if RUN_WEIGHT_SENSITIVITY:
        print("Running weight-sensitivity analysis...")
        iou_df = run_weight_sensitivity(
            attrs, SENSITIVITY_WEIGHTS, BOUNDARY_PERCENTILE,
            os.path.join(OUTPUT_DIR, "fig08_weight_sensitivity.png"))
        iou_df.to_csv(os.path.join(OUTPUT_DIR, "weight_sensitivity_overlap.csv"))
        chosen = iou_df.index[0]
        others = iou_df.loc[chosen].drop(chosen)
        print(f"Mean top-{100 - BOUNDARY_PERCENTILE:.0f}% overlap of 'chosen' weights "
              f"with the other weight sets: {others.mean():.2f} "
              f"(range {others.min():.2f}-{others.max():.2f})")

    if RUN_SYNTHETIC_VALIDATION:
        print("Running synthetic validation...")
        syn_scores = run_synthetic_validation(OUTPUT_DIR)
        print("Synthetic boundary-detection ROC AUC by SNR (noise sweep):")
        auc_pivot = syn_scores.pivot(index="attribute", columns="snr",
                                     values="roc_auc").reindex(SYNTH_ATTRS)
        auc_pivot = auc_pivot[sorted(auc_pivot.columns, reverse=True)]
        print(auc_pivot.round(3).to_string())

    print("Saving metrics and arrays...")
    save_metrics(seismic_sel, time_sel, attrs, tmin_ms, tmax_ms,
                 score_df, sel_info, deep_df, OUTPUT_DIR)

    print("\nDONE.")
    print(f"Outputs saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("Please send back, in particular:")
    for fname in ["fig01_full_section_auto_window.png", "fig03_hilbert_attributes.png",
                  "fig04_composite_attribute.png", "fig05_composite_overlay_on_seismic.png",
                  "fig07_attribute_redundancy.png", "fig08_weight_sensitivity.png",
                  "fig09_synthetic_validation.png", "fig10_synthetic_detection_scores.png"]:
        print(f"  {fname}")
    print("  and the CSVs: hilbert_attribute_metrics.csv, attribute_correlation_matrix.csv,")
    print("  weight_sensitivity_overlap.csv, synthetic_detection_scores.csv, deep_vs_selected_stats.csv")


if __name__ == "__main__":
    main()