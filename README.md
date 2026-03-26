# 🧠 NeuroDyads — EEG Hyperscanning Pipeline

> **GSoC Submission - EEG Dyadic Manifold Learning**
> Dual-brain EEG preprocessing and CEBRA-based neural manifold decoding for dyadic affect perception.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `pipeline_diagram.png` | End-to-end EEG preprocessing pipeline (Phases A–E) |
| `Untitled_design.png` | NEURODYADS · CEBRA pipeline — dyadic manifold learning overview |

---

## 🗂️ Overview

This repository documents two complementary pipelines developed for the **NeuroDyads hyperscanning** project — a dual-brain EEG study of speaker–listener affect synchrony. Together they cover the full arc from raw EDF acquisition through to topologically validated neural manifold embeddings.

```
Raw EDF → [EEG Preprocessing Pipeline] → Clean .fif files
                                               ↓
                              [CEBRA Manifold Pipeline] → Decoded affect embeddings
```

---

## 📊 Pipeline 1 — EEG Preprocessing (`pipeline_diagram.png`)

**Stack:** `MNE-Python 1.11` · `Extended Picard ICA` · `n_components=0.95`
**Hardware:** Dual 64-ch EGI net · 250 Hz · 65-channel raw input per participant

### Phase A — Temporal Alignment

- Raw EDF files loaded for **Speaker** (65 ch · 303.0 s) and **Listener** (65 ch · 303.0 s)
- DIN1 segmentation applied per stream: `+147.8 s / −154.2 s` (Speaker), `+147.8 s / −153.9 s` (Listener)
- **Mutual temporal alignment** via `mutually_align_cuts`: sync hole-punching, negative trimmed `154.2 s → 149.6 s`, ±1 ms precision
- Affect blocks concatenated → **297.4 s continuous signal** per participant

### Phase B — Spatial Setup & Filtering

- Drop `EEG VREF`: 65 → 64 channels; rename `E1–E64`; apply `GSN-HydroCel-64_1.0` montage
- **1 Hz high-pass filter** (Hamming FIR · zero-phase · 53 dB stopband) — applied to ICA training copy only

### Phase C — Bad Channel Repair

| Participant | RANSAC Flagged | Interpolation Source |
|-------------|----------------|----------------------|
| Speaker | 21 / 64 channels | Reconstructed from 43 good ch |
| Listener | 4 / 64 channels | Reconstructed from 60 good ch |

- Spherical spline interpolation
- Average re-referencing (common average per time sample)

### Phase D — ICA Artifact Decomposition

| Participant | Components | Variance Explained | Components Removed |
|-------------|------------|--------------------|--------------------|
| Speaker | 10 IC | 95.71% | 5 |
| Listener | 16 IC | 95.28% | 6 |

- `PCA → Picard ICA` per participant
- **ICLabel AI** + conservative manual review (visual · variance · topography)
- ICA applied to main **unfiltered** signal; artifact subspace zeroed and projected back to 64 channels

### Phase E — Clean Outputs

- **8 clean `.fif` files** produced: `sp/ls × positive/negative/full`
- Segment lengths: `147.8 s + 149.6 s` per participant

---

## 🔬 Pipeline 2 — CEBRA Manifold Learning (`Untitled_design.png`)

**Stack:** `PyTorch 2.10+CUDA` · `MNE 1.11` · `CEBRA 0.6.0` · `NumPy 2.3.2`
**Data:** Seed = 42 · 250 Hz · 128 channels · **74,343 samples**

### 01 · Data Ingestion — 4 × FIF Files

| Stream | Condition | File | Channels | Duration | Samples |
|--------|-----------|------|----------|----------|---------|
| Speaker | Positive | `speaker_positive_clean_raw.fif` | 64 | 147.8 s | 36,944 |
| Speaker | Negative | `speaker_negative_clean_raw.fif` | 64 | 149.6 s | 37,399 |
| Listener | Positive | `listener_positive_clean_raw.fif` | 64 | 147.8 s | 36,944 |
| Listener | Negative | `listener_negative_clean_raw.fif` | 64 | 149.6 s | 37,399 |

Sampling rate: 250 Hz (verified identical across all 4 files). Loaded via `mne.io.read_raw_fif(..., preload=True)`. Arrays transposed `(channels × time) → (time × channels)`, trimmed to min-length per condition.

### 02 · Dyadic Joint Matrix Construction

| Operation | Shape |
|-----------|-------|
| `SP_pos ⊕ LS_pos` | (36,944 × 64) + (36,944 × 64) → (36,944 × 128) |
| `SP_neg ⊕ LS_neg` | (37,399 × 64) + (37,399 × 64) → (37,399 × 128) |
| `dyad_pos + dyad_neg → vstack` | **X_raw (74,343 × 128) · 76.1 MB** |

- Cols 0–63 = Speaker EEG · Cols 64–127 = Listener EEG
- Value range: `[−3.08e-3, 3.74e-3]` µV (raw)

### 03 · Z-Normalisation (StandardScaler)

| | Input Shape | Post-Norm Mean | Post-Norm Std |
|--|-------------|----------------|---------------|
| | X (74,343 × 128) | ≈ 0 · range `[−1.04e-16, 1.33e-16]` | **= 1.0000** exact across all 128 channels |

Rationale: high-impedance variance & ICA reconstruction differences → prevents high-variance channels from dominating CEBRA's InfoNCE loss.

### 04 · Behavioral Label Vector

| Label | Condition | Samples | Proportion | Balance Ratio |
|-------|-----------|---------|------------|---------------|
| 0 | Positive Affect | 36,944 | 49.7% | **0.988 — Balanced** |
| 1 | Negative Affect | 37,399 | 50.3% | No class weighting required |

### 05 · CEBRA Model Architecture & Hyperparameters

**CEBRA-Time** (Neural State)

| Parameter | Value |
|-----------|-------|
| `model_architecture` | `offset10-model` |
| `conditional` | `time` |
| `output_dimension` | `[3, 6, 10, 15]` |
| `distance` | `cosine` |
| `batch_size` | `512` |
| `learning_rate` | `3e-4` |
| `max_iterations` | `10,000` |
| `temperature_mode` | `auto` |
| `temperature` | `1.12 (init)` |
| `time_offsets` | `[10, 20]` |
| Grid | `4 dims × 2 offsets = 8 models` · Best: `output_dim=15, offset=20` · InfoNCE = **5.5362** |

**CEBRA-Delta** (Neural Velocity)

| Parameter | Value |
|-----------|-------|
| `model_architecture` | `offset10-model` |
| `conditional` | `time_delta` |
| `output_dimension` | `[3, 6, 10, 15]` |
| `distance` | `cosine` |
| `batch_size` | `512` |
| `learning_rate` | `3e-4` |
| `max_iterations` | `10,000` |
| `temperature_mode` | `auto` |
| `temperature` | `1.12 (init)` |
| `time_offsets` | `[10]` |
| Grid | `4 dims × 1 offset = 4 models` · Best: `output_dim=15` · InfoNCE = **5.5365** |

> `offset10-model` = causal 1D CNN with **10-sample (40 ms @ 250 Hz)** receptive field — captures local temporal context of neural oscillations. `cosine` distance = amplitude-invariant, encodes direction of activity patterns. `time` = pairs time-points by label; `time_delta` = pairs by label AND temporal gap (captures neural velocity / rate-of-change).

### 06 · Latent Embeddings (Best Models)

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `emb_time` | (74,343 × 15) | Neural State — Positive & Negative Affect separated on manifold |
| `emb_delta` | (74,343 × 15) | Neural Velocity — dynamic affect separation, better KNN than Time |

Visualised as 3D subspace (dims 0–2) in 4-panel manifold comparison · 16-angle inspection · 12-way 2D projection plots.

### 07 · Shuffled-Label Control Experiment

Hypothesis: if labels are permuted randomly, CEBRA produces a uniform hyperspherical cloud — not a structured manifold. Shuffle introduced **37,274 label transitions** vs. 1 in the original.

| Control | Best Config | Final Loss | Theoretical Min | Interpretation |
|---------|-------------|------------|-----------------|----------------|
| Neural State (Time) | `output_dim=3`, `offset=20` | 6.2381 | log(512) = 6.238 ✓ | Random baseline hit |
| Neural Velocity (Delta) | `output_dim=6` | 6.2383 | log(512) = 6.238 ✓ | Random baseline hit |

> InfoNCE with `batch=512` and uniform negative sampling has theoretical minimum = `log(512) ≈ 6.238`. Control models converged to exactly this value — confirming CEBRA does not extract structure from random noise.

### 08 · Comprehensive Evaluation

**KNN Decoding Accuracy (k=5, cosine metric)**

| Model | Type | KNN Accuracy | F1 (Macro) | InfoNCE Loss | Output Dim |
|-------|------|-------------|------------|--------------|------------|
| Main: Neural State | Time | 84.93% | 84.92% | 5.5362 | 15 |
| Main: Neural Velocity | **Delta** | **87.26% ★** | **87.25%** | 5.5365 | **15** |
| Control: State (Shuffled) | Time | 49.39% | 49.38% | 6.2381 | 3 |
| Control: Velocity (Shuffled) | Delta | 50.33% | 50.32% | 6.2383 | 6 |

**CEBRA Native Consistency (Between Runs R1 vs R2)**

| Model | Consistency | Interpretation |
|-------|-------------|----------------|
| Main — Time | **0.98** | Highly consistent |
| Main — Delta | **0.99** | Highly consistent |
| Control — Time | 0.16 | Random topology |
| Control — Delta | 0.14 | Random topology |

### 09 · Topological Validation — CE-KS Report

Cross-entropy distributions computed via GPU-batched cosine affinity matrices (L2-normalised embeddings; batch=2,048). KS statistic compares CE distributions between embedding pairs.

| Test Type | Comparison | KS Effect Size | p-value | Hypothesis | Status |
|-----------|-----------|---------------|---------|------------|--------|
| Intra-Run Stability | Main_Time R1 vs R2 | 0.0098 | 4.96e-01 | Identical Topology | ✅ Pass |
| Intra-Run Stability | Main_Delta R1 vs R2 | 0.0050 | 5.44e-01 | Identical Topology | ✅ Pass |
| Intra-Run Stability | Ctrl_Time R1 vs R2 | 0.6168 | 0.00e+00 | Identical Topology | ⚠️ Fail *(random topology expected)* |
| Label Dependency | Main_Time vs Ctrl_Time | **1.0000** | 0.00e+00 | Different Topology | ✅ Pass |
| Label Dependency | Main_Delta vs Ctrl_Delta | **1.0000** | 0.00e+00 | Different Topology | ✅ Pass |
| State vs Velocity | Main_Time vs Main_Delta | 0.0063 | 4.65e-01 | Different Topology | ⚠️ Fail *(similar topology)* |

> **Key finding:** KS=1.0 for label dependency tests proves the structured manifold geometry is entirely driven by behavioral labels, not data artefacts. Control's intra-run failure is expected (random topology is not reproducible). State vs Velocity sharing topology suggests both conditionals encode the same underlying neural geometry.

---

## 📋 Results Summary

```
BEST MODEL
CEBRA-Delta (Neural Velocity · output_dim=15)
  KNN Acc = 87.26%
  F1 Macro = 87.25%

CONTROL VALIDATION  ✓ Confirmed
  Shuffled labels → loss = log(512) ≈ 6.238
  KNN ≈ 50% (chance)
  Consistency ≈ 0.15

REPRODUCIBILITY
  ~0.99 CEBRA consistency (R1 vs R2)
  KS < 0.01 across main runs
  Stable manifold geometry
```

---

## 🛠️ Requirements

```txt
mne>=1.11
cebra>=0.6.0
torch>=2.10         # CUDA recommended
numpy>=2.3.2
scikit-learn        # StandardScaler, KNN
scipy               # KS statistic
matplotlib          # Embedding visualisation
```

---

## 📄 License

This project was developed as part of a **Google Summer of Code** submission. See `LICENSE` for details.

---

## 🙏 Acknowledgements

Built on [MNE-Python](https://mne.tools/), [CEBRA](https://cebra.ai/), and the [NeuroDyads](https://github.com/NeuroDyads) hyperscanning framework.
