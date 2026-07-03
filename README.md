# Hilbert-Transform-Based Composite Attribute for Boundary Enhancement

This repository contains a Python workflow for calculating a Hilbert-transform-based composite boundary attribute from a single 2D post-stack seismic SEG-Y section from the Baku Archipelago.

The workflow reads a SEG-Y file, selects an informative interpretation interval, computes Hilbert-derived instantaneous attributes, builds a composite boundary-enhancement attribute, and saves publication-ready figures, numerical metrics, sensitivity checks, and synthetic-validation results.

> **Important:** The SEG-Y input file is not included. Place your SEG-Y file in the same folder as the script or edit the `SEGY_FILE` parameter inside the script.

---

## 1. Main script

```text
hilbert_composite_baku_archipelago.py
```

Default expected input:

```text
my_data_1.sgy
```

Default output folder:

```text
outputs_hilbert_composite_baku/
```

---

## 2. Purpose of the workflow

The script is designed for boundary enhancement in a 2D post-stack seismic time section. It is especially useful when only a single post-stack seismic line is available and direct well-log calibration is absent.

The method combines four normalized Hilbert-derived components:

1. **Normalized envelope**  
   Highlights reflection strength.

2. **Envelope gradient**  
   Highlights sharp amplitude/interface changes.

3. **Lateral phase discontinuity**  
   Highlights lateral phase breaks using a wrap-safe phase-gradient calculation.

4. **Frequency anomaly**  
   Highlights local deviations from the smoothed instantaneous-frequency background.

These components are combined into a weighted composite attribute:

```text
Composite = normalize(
    0.15 * Envelope_norm
  + 0.30 * Envelope_gradient
  + 0.35 * Phase_discontinuity
  + 0.20 * Frequency_anomaly
)
```

The top percentile of this composite attribute is then used to highlight possible boundary-like zones on the seismic section.

---

## 3. Key features

- Reads 2D SEG-Y data using `segyio`, with a fallback sequential SEG-Y reader.
- Applies optional broad band-pass filtering.
- Automatically selects an informative time interval using multiple candidate window lengths.
- Computes Hilbert/analytic-signal attributes:
  - envelope
  - wrapped phase
  - cosine phase
  - instantaneous frequency
  - envelope gradient
  - lateral phase discontinuity
  - frequency anomaly
- Builds a composite boundary-enhancement attribute.
- Generates seismic, attribute, overlay, curve, redundancy, sensitivity, and synthetic-validation figures.
- Saves numerical metrics and compressed attribute arrays.
- Includes reviewer-driven improvements:
  - lateral-only phase discontinuity
  - robust analytic instantaneous-frequency estimator
  - amplitude-scale-invariant envelope normalization
  - sampling-invariant envelope gradient
  - multi-length automatic interval selection
  - weight-sensitivity analysis
  - attribute-redundancy diagnostics
  - synthetic validation with a controlled impedance model and noise sweep

---

## 4. Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install numpy scipy matplotlib pandas segyio
```

Recommended Python version:

```text
Python 3.10+
```

The script uses a non-interactive Matplotlib backend, so it can run on servers or headless environments.

---

## 5. Input data

By default, the script expects the SEG-Y file to be located in the same folder as the script:

```text
my_data_1.sgy
```

To use a different file, edit this parameter near the top of the script:

```python
SEGY_FILE = "my_data_1.sgy"
```

The script expects a 2D post-stack seismic time section. It reads:

- number of traces
- number of samples
- sample interval
- time axis where available

If `segyio` cannot open the file because of trace-count or geometry issues, the script attempts to use its built-in sequential SEG-Y fallback reader.

---

## 6. How to run

From the folder containing the script and the SEG-Y file:

```bash
python hilbert_composite_baku_archipelago.py
```

After completion, results will be saved in:

```text
outputs_hilbert_composite_baku/
```

---

## 7. Main user parameters

The main editable parameters are located near the top of the script.

### Input and output

```python
SEGY_FILE = "my_data_1.sgy"
OUTPUT_DIR = "outputs_hilbert_composite_baku"
```

### Interpretation interval

```python
MAX_INTERPRET_TIME_MS = 8000.0
MIN_INTERPRET_TIME_MS = 300.0

MANUAL_TMIN_MS = None
MANUAL_TMAX_MS = None
```

If `MANUAL_TMIN_MS` and `MANUAL_TMAX_MS` are left as `None`, the script automatically selects the most informative interval. To force a manual interval, set values such as:

```python
MANUAL_TMIN_MS = 3000.0
MANUAL_TMAX_MS = 6500.0
```

### Automatic interval search

```python
AUTO_WINDOW_MS_LIST = [2500.0, 3000.0, 3500.0, 4000.0]
AUTO_STEP_MS = 250.0
```

The script tests several candidate interval lengths and chooses the window with the strongest combined score.

### Instantaneous-frequency method

```python
FREQ_METHOD = "analytic"
```

Available options:

```text
analytic
unwrap
```

The default `analytic` method is preferred because it avoids phase-unwrapping spikes.

### Band-pass filter

```python
APPLY_BANDPASS = True
BANDPASS_LOW_HZ = 3.0
BANDPASS_HIGH_HZ = 80.0
BANDPASS_ORDER = 4
```

### Composite weights

```python
W_ENVELOPE = 0.15
W_ENVELOPE_GRADIENT = 0.30
W_PHASE_DISCONTINUITY = 0.35
W_FREQUENCY_ANOMALY = 0.20
```

### Boundary overlay percentile

```python
BOUNDARY_PERCENTILE = 90.0
```

A value of `90.0` means that the top 10% of composite values are highlighted on the seismic overlay.

### Extra analyses

```python
RUN_WEIGHT_SENSITIVITY = True
RUN_SYNTHETIC_VALIDATION = True
```

Set either of these to `False` to reduce runtime.

---

## 8. Output files

The script saves figures, CSV files, a summary text file, and compressed NumPy arrays.

### Main figures

```text
fig01_full_section_auto_window.png
fig02_selected_seismic_section.png
fig03_hilbert_attributes.png
fig04_composite_attribute.png
fig05_composite_overlay_on_seismic.png
```

### Supplementary attribute-curve figures

```text
fig06a_hilbert_component_curves.png
fig06b_frequency_composite_curves.png
```

### Reviewer-response / robustness figures

```text
fig07_attribute_redundancy.png
fig08_weight_sensitivity.png
fig09_synthetic_validation.png
fig10_synthetic_detection_scores.png
```

### Numerical outputs

```text
hilbert_attribute_metrics.csv
automatic_interval_scores.csv
attribute_correlation_matrix.csv
weight_sensitivity_overlap.csv
synthetic_detection_scores.csv
deep_vs_selected_stats.csv
selected_interval_summary.txt
attribute_arrays_selected_interval.npz
```

---

## 9. What each main figure shows

### `fig01_full_section_auto_window.png`

Shows the full seismic section and the automatically selected interpretation interval.

### `fig02_selected_seismic_section.png`

Shows the selected seismic interval used for attribute calculation.

### `fig03_hilbert_attributes.png`

Shows the selected seismic section together with key Hilbert-derived attributes:

- envelope
- cosine phase
- instantaneous frequency

### `fig04_composite_attribute.png`

Shows the final composite boundary-enhancement attribute.

### `fig05_composite_overlay_on_seismic.png`

Overlays the strongest composite values on the seismic section. By default, the top 10% of composite values are highlighted.

### `fig06a_hilbert_component_curves.png`

Shows mean vertical curves for:

- envelope
- envelope gradient
- phase discontinuity

### `fig06b_frequency_composite_curves.png`

Shows mean vertical curves for:

- frequency anomaly
- composite attribute

### `fig07_attribute_redundancy.png`

Shows correlation and crossplot diagnostics among the attribute components.

### `fig08_weight_sensitivity.png`

Shows how stable the top-percentile boundary map is under different composite-weight choices.

### `fig09_synthetic_validation.png`

Shows the synthetic impedance model, noisy seismic response, ground-truth boundary mask, and selected attribute maps.

### `fig10_synthetic_detection_scores.png`

Shows quantitative synthetic detection scores across a noise sweep.

---

## 10. Synthetic validation

The script includes an optional synthetic validation workflow.

It builds a controlled 2D impedance model containing:

- a normal fault
- a pinch-out / reflector termination
- a lateral facies change
- layered impedance contrasts

The synthetic seismic data are generated by convolving reflectivity with a Ricker wavelet and adding band-limited noise at different signal-to-noise ratios.

The attributes are scored against a known reflecting-boundary mask using:

- ROC AUC
- matched-budget precision
- matched-budget recall

The synthetic validation is not intended to prove lithological classification. It tests whether the attribute workflow can detect known reflecting-boundary zones under controlled noise conditions.

---

## 11. Interpretation notes

The composite attribute should be interpreted as a **boundary-enhancement product**, not as a direct lithology classifier.

High composite values may indicate zones where several seismic responses coincide:

- amplitude strength
- sharp envelope change
- lateral phase disruption
- local frequency anomaly

These zones may correspond to stratigraphic boundaries, reflector terminations, geometry changes, or lithology-sensitive contrasts. However, without well logs, core data, inversion results, or horizon calibration, the output should not be interpreted as confirmed lithology.

---

## 12. Limitations

- The workflow is designed for a single 2D post-stack seismic line.
- It does not provide 3D continuity information.
- It does not perform dip steering.
- Instantaneous frequency can remain sensitive to noise and amplitude nulls.
- The composite weights are empirical and should be tested on additional data.
- The output is qualitative unless calibrated with independent geological or well information.
- The SEG-Y data are not included because they may be confidential.

---

## 13. Troubleshooting

### `FileNotFoundError: Could not find my_data_1.sgy`

Make sure the SEG-Y file is in the same folder as the script, or edit:

```python
SEGY_FILE = "my_data_1.sgy"
```

### `ImportError: segyio is required`

Install `segyio`:

```bash
pip install segyio
```

### Band-pass warning

If the sample interval is too large or the requested frequency band is invalid, the script may skip the band-pass filter. Adjust:

```python
BANDPASS_LOW_HZ
BANDPASS_HIGH_HZ
```

### Script is slow

Disable optional analyses:

```python
RUN_WEIGHT_SENSITIVITY = False
RUN_SYNTHETIC_VALIDATION = False
```

### Figures are not displayed

This is expected. The script uses a headless Matplotlib backend and saves figures directly to disk.

---

## 14. Suggested repository structure

```text
project/
├── hilbert_composite_baku_archipelago.py
├── README.md
├── requirements.txt
├── my_data_1.sgy                  # not included if confidential
└── outputs_hilbert_composite_baku/
```

Suggested `requirements.txt`:

```text
numpy
scipy
matplotlib
pandas
segyio
```

---

## 15. Data availability note

The Python workflow is provided for reproducibility. The SEG-Y input file is not included because seismic data may be confidential or subject to access restrictions. To reproduce the results, place an authorized SEG-Y file with the expected name in the script folder or update the `SEGY_FILE` parameter.

---

## 16. Citation / manuscript use

If this workflow is used in a manuscript, describe it as a reproducible Hilbert-transform-based composite attribute workflow for boundary enhancement in single-line post-stack seismic interpretation. The output should be presented as a seismic boundary-screening result rather than a direct lithological classification.

---

## 17. License

Add your preferred license here before publishing the repository.

Example:

```text
MIT License
```

or:

```text
All rights reserved.
```
