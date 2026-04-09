# EEG-Based Emotion Recognition 
### SEED Dataset · Signal Preprocessing + Deep Learning

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/khushidua77/bci-project)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20P100%2016GB-green)](https://www.kaggle.com/docs/efficient-gpu-usage)

---

## Overview

This project implements an end-to-end **EEG-based emotion recognition pipeline** on the SJTU SEED dataset. The system converts raw EEG signals into topographic brain maps and classifies emotional states (stress vs. non-stress) using a novel multi-modal deep learning architecture called **StressNet** — a fusion of CNN and Bidirectional LSTM branches — benchmarked against EEGNet, CNN-only, LSTM-only, and classical ML baselines.

**Task:** Binary classification — Stress (negative emotion) vs. Non-Stress (neutral + positive)  
**Dataset:** SEED — 15 subjects, 50,910 trials, 62 EEG channels, 5 frequency bands  
**Best Result:** EEGNet achieves **98.27% accuracy** | StressNet achieves **99.98% ROC-AUC**

---

## Pipeline Architecture

```
Raw EEG (N, 5, 62)
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
  [Signal Preprocessing]                              [Raw EEG Sequence]
  Butterworth Bandpass Filter                         Downsampled EEG
  (delta/theta/alpha/beta/gamma)                      (N, time_steps, channels)
       │                                                      │
  Azimuthal Projection                            Bidirectional LSTM (128)
  (62ch → 2D scalp coords)                        Bidirectional LSTM (64)
       │                                                      │
  Cubic Interpolation + Gaussian Smoothing           LSTM Feature Vector (128)
       │
  64×64×5 Topographic Image
       │
  CNN Branch
  Conv2D(32) → Conv2D(32) → MaxPool → Dropout
  Conv2D(64) → Conv2D(64) → MaxPool → Dropout
  Conv2D(128) → MaxPool → GlobalAvgPool
  Dense(256) → CNN Feature Vector
       │
       └──────────────── Concatenate ──────────────────────┘
                               │
                        BatchNorm → Dense(512) → Dense(256) → Sigmoid
                               │
                        Binary Prediction
```

---

## Dataset

**SEED (SJTU Emotion EEG Dataset)**
- 15 subjects watching emotion-eliciting film clips
- 3 emotion labels: Negative (0), Neutral (1), Positive (2)
- Pre-processed into 1-second epochs with DE (Differential Entropy) features
- 5 frequency bands × 62 channels per trial
- Shape: `(50910, 5, 62)`

**Binary Remapping:**
| Original Label | Remapped | Count |
|---|---|---|
| Negative (0) | Stress (0) | 16,800 |
| Neutral (1) + Positive (2) | Non-Stress (1) | 34,110 |

**Train / Val / Test Split:** 64% / 16% / 20% (stratified, no leakage)

---

## Signal Preprocessing

### 1. Butterworth Bandpass Filtering
Applied per frequency band using a 2nd-order IIP Butterworth filter via `scipy.signal.butter` + `filtfilt` (zero-phase):

| Band | Frequency Range |
|---|---|
| Delta | 0.5 – 4 Hz |
| Theta | 4 – 7 Hz |
| Alpha | 8 – 12 Hz |
| Beta | 13 – 30 Hz |
| Gamma | 30 – 45 Hz |

### 2. Topographic Image Generation
Each trial is converted to a **64×64×5 multi-channel topographic image**:
1. 62-channel electrode positions projected onto 2D plane via azimuthal equidistant projection (10-20 international system)
2. Mean absolute amplitude per channel computed after bandpass filtering
3. Cubic interpolation onto a 64×64 grid using `scipy.interpolate.griddata`
4. Gaussian smoothing (`sigma=1.0`) for spatial continuity
5. Min-max normalization to [0, 1] per image

Images generated in parallel using `joblib.Parallel` — **50,910 images in ~5 minutes** on Kaggle P100.

---

## Models

### 1. CNN-only
Processes 64×64×5 topographic images through 3 Conv2D blocks with BatchNorm and Dropout, followed by GlobalAveragePooling and a Dense classifier.

### 2. LSTM-only
Processes raw downsampled EEG sequences through two stacked Bidirectional LSTM layers (128 and 64 units) with recurrent dropout.

### 3. StressNet (CNN + LSTM Fusion) ← Proposed
Concatenates the spatial feature vector from the CNN branch with the temporal feature vector from the LSTM branch. A fusion head (Dense 512 → 256 → Sigmoid) combines both modalities.

**Parameters:** ~880K

### 4. EEGNet (BCI-specific baseline)
Compact depthwise separable CNN designed specifically for EEG-BCIs. Uses Conv1D + DepthwiseConv1D + SeparableConv1D blocks.

**Parameters:** ~2,185 (lightweight!)

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam |
| CNN / StressNet LR | 1e-4 |
| EEGNet LR | 1e-3 |
| Batch Size | 32 |
| Max Epochs | 40 (StressNet), 15 (EEGNet) |
| Early Stopping | Patience 10 (StressNet), 5 (EEGNet) |
| Class Weighting | `sklearn.utils.class_weight.compute_class_weight` (balanced) |
| Hardware | Tesla P100 16GB (Kaggle GPU) |

---

## Results

### Test Set Performance

| Model | Accuracy | ROC-AUC | PR-AUC | F1 | Cohen κ |
|---|---|---|---|---|---|
| FFT + LDA *(baseline)* | 61.23% | — | — | — | — |
| PCA + LDA *(baseline)* | 77.28% | — | — | — | — |
| Naive Bayes + SGWT *(baseline)* | 78.20% | — | — | — | — |
| CNN-only | 95.05% | 0.9952 | 0.9977 | 0.9620 | 0.8911 |
| LSTM-only | 95.73% | 0.9963 | 0.9982 | 0.9672 | 0.9061 |
| **StressNet (CNN+LSTM Fusion)** | **97.52%** | **0.9998** | **0.9999** | **0.9811** | **0.9448** |
| EEGNet | **98.27%** | 0.9986 | 0.9993 | 0.9872 | 0.9607 |

> EEGNet achieves the highest test accuracy (98.27%) with only ~2K parameters — a remarkable efficiency result. StressNet achieves the best ROC-AUC and PR-AUC (both ≈ 0.9999), indicating near-perfect discrimination across thresholds.

---

## Project Structure

```
bci-project/
│
├── bci-project.ipynb          # Main notebook (full pipeline)
│
├── Sections:
│   ├── Section 1  — Imports & Dataset Loading
│   ├── Section 2  — EDA: Label distribution, band power, channel activity
│   ├── Section 3  — Channel positions & 10-20 system projection
│   ├── Section 4  — Bandpass filtering & topographic image generation
│   ├── Section 5  — Train/Val/Test split + normalization
│   ├── Section 6  — Load & verify splits
│   ├── Section 7  — Model architecture (CNN, LSTM, StressNet, EEGNet)
│   ├── Section 8  — Train CNN-only and LSTM-only
│   ├── Section 9  — Train StressNet (fusion)
│   ├── Section 10 — Training curves
│   ├── Section 11 — Train EEGNet
│   ├── Section 12 — Evaluation on test set (Accuracy, ROC-AUC, PR-AUC, F1, κ)
│   ├── Section 13 — Confusion matrices (all 4 models)
│   └── Section 14 — Final comparison chart
│
└── README.md
```

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
tensorflow >= 2.10
joblib
tqdm
```

Install:
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn tensorflow joblib tqdm
```

---

## How to Reproduce

1. **Kaggle (recommended):** Open the notebook directly at [kaggle.com/code/khushidua77/bci-project](https://www.kaggle.com/code/khushidua77/bci-project) and run with GPU P100 accelerator enabled.

2. **Dataset:** Add the SEED dataset from Kaggle Datasets: `daviderusso7/seed-dataset`. Ensure the path `/kaggle/input/datasets/daviderusso7/seed-dataset` contains:
   - `DatasetCaricatoNoImage.npz`
   - `LabelsNoImage.npz`
   - `SubjectsNoImage.npz`

3. **Run all cells in order.** Topographic image generation takes ~5 minutes on GPU; StressNet training runs for ~40 epochs (~50 min); EEGNet trains in ~1 minute.

---

## Key Design Decisions

- **Binary task over 3-class:** SEED's 3-class problem is collapsed to binary (stress vs. non-stress) for clinical relevance — stress detection is the actionable BCI use case.
- **No data leakage:** StandardScaler is fit only on training data; val/test are transformed using training statistics.
- **Class imbalance handling:** Balanced class weights applied during training (1:2 stress/non-stress ratio).
- **Topographic images vs. raw DE features:** Converting DE features to spatial images allows the CNN to learn scalp topology patterns that flat feature vectors cannot capture.
- **Parallel image generation:** `joblib.Parallel` reduces image generation time from ~40 min (serial) to ~5 min on 8 CPU cores.

---

## Limitations & Future Work

- **Subject-dependent evaluation:** The train/test split is random across subjects, not leave-one-subject-out (LOSO). Cross-subject generalization is an open question.
- **Binary simplification:** Collapsing neutral+positive into a single class discards nuance; a 3-class variant with oversampling would be worth exploring.
- **EEGNet adaptation:** The EEGNet implementation here uses 1D convolutions rather than the original 2D temporal-spatial formulation; the original architecture might perform differently.
- **No deployment/inference pipeline:** Models are saved as `.keras` checkpoints but no inference API or real-time BCI pipeline is implemented.
- **Potential data leakage source:** Multiple trials from the same subject appear in both train and test sets. A subject-stratified split is recommended for a rigorous evaluation.

---

## References

- Zheng, W.-L., & Lu, B.-L. (2015). *Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks.* IEEE TAFFC.
- Lawhern, V. J., et al. (2018). *EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces.* Journal of Neural Engineering.
- SEED Dataset: [bcmi.sjtu.edu.cn/~seed/seed.html](https://bcmi.sjtu.edu.cn/~seed/seed.html)

---

## Author

**Khushi Dua**  
B.Tech CSE (AI & ML), Sharda University  
[![GitHub](https://img.shields.io/badge/GitHub-khushi--dua-black?logo=github)](https://github.com/Khushi-Dua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-khushi--dua7-blue?logo=linkedin)](https://linkedin.com/in/khushi-dua7)
[![Kaggle](https://img.shields.io/badge/Kaggle-khushidua77-20BEFF?logo=kaggle)](https://www.kaggle.com/khushidua77)
