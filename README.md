# 🪐 Planet Hunters: Find New Worlds in Starlight
### **Hack4Hack 2026 · Round 1 (24-Hour Ranking Round)**
**Macro-F1 Benchmark: 0.4259 (Starter Baseline) ➔ 0.6978 (Our Triple Ensemble) (+63.8% Boost)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Models](https://img.shields.io/badge/Ensemble-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-success.svg)](https://github.com)
[![Evaluation](https://img.shields.io/badge/Metric-Macro--F1-orange.svg)](https://github.com)
[![Status](https://img.shields.io/badge/Round%201-Submitted-brightgreen.svg)](https://github.com)

---

## 📖 Overview

Space telescopes like NASA's TESS stare at hundreds of thousands of stars for 27.4 days at a time, measuring brightness every 30 minutes (1,315 cadences per star). While an exoplanet crossing its host star creates a tiny dip in brightness (<0.1%), **most dips in the sky are astrophysical impostors**:
- Grazing and eclipsing binaries
- Blended background binaries mimicking shallow transits
- Starspots rotating into view
- Fast multi-periodic stellar pulsations ($\delta$ Scuti, RR Lyrae)
- Magnetic flares with rapid reconnection
- Telescope thruster momentum dumps and pointing glitches

This repository contains our complete, physics-grounded machine learning solution for **Hack4Hack 2026 (Round 1)**.

> 📚 **Looking for the in-depth breakdown?**  
> Check out [**`Full_Explanation.md`**](Full_Explanation.md) for a complete, plain-English explanation of how every feature, formula, and model works!

---

## 🏆 Performance Progression

| Model / Pipeline | Validation Macro-F1 | Key Improvements |
| :--- | :---: | :--- |
| **Organizer Starter Baseline** | `0.4259` | 15 naive statistical moments without detrending or centroid tracking |
| **Initial Feature Extraction (V1)** | `0.6716` | Box Least Squares, Fourier decomposition, 79 features |
| **Cotrended Astronomical Pipeline (V2)** | `0.6929` | 16-channel CCD common drift cancellation + 88 deep features |
| **Triple Ensemble (LGBM + XGB + CatBoost)** | `0.6942` | 15-model diversity (5 folds × 3 gradient boosting architectures) |
| **Threshold-Calibrated Ensemble (Final)** | **`0.6978`** | Powell optimization on out-of-fold probability distributions |

### Class-by-Class Breakdown (18,000 Validation Stars)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`instrument_glitch`** | 0.8261 | 0.7978 | **0.8117** | 1,721 |
| **`pulsating_star`** | 0.9048 | 0.7324 | **0.8095** | 1,973 |
| **`eclipsing_binary`** | 0.8898 | 0.6898 | **0.7771** | 2,247 |
| **`flare_star`** | 0.9036 | 0.6236 | **0.7379** | 1,804 |
| **`quiet_star`** | 0.5370 | 0.8598 | **0.6611** | 3,624 |
| **`background_binary`** | 0.6932 | 0.6241 | **0.6568** | 1,604 |
| **`planet`** | 0.6745 | 0.5343 | **0.5963** | 2,145 |
| **`starspots`** | 0.5392 | 0.5253 | **0.5322** | 2,882 |
| **Macro Average** | **0.7460** | **0.6734** | **0.6978** | **18,000** |

---

## 🛠️ Key Architectural Breakthroughs

1. **Independent Sector & CCD Cotrending**:  
   Precomputed channel-level instrumental baselines across all 16 `(camera, ccd)` detectors. Erased ~22.7 millipixels of telescope pointing drift, preventing models from memorizing Sector A's specific pointing quirks.
2. **Two-Scale Time Series Decomposition**:  
   Split light curves into a **Trend** component ($\sim 1$-day rolling median filter) for stellar rotation / spots, and a **Flattened** component for transits, eclipses, and flares.
3. **Centroid Astrometry (Background Binary Smoker)**:  
   Measured in-transit 2D centroid displacement $\Delta \vec{c}$ and signal-to-noise ratio. Real exoplanets produce $\Delta \vec{c} \approx 0$; blended background binaries produce statistically significant excursions ($t > 5$).
4. **Vectorized Box Least Squares (BLS)**:  
   Periodic transit search across $P \in [0.5, 13.0]$ days, extracting physical SNR ($\delta \sqrt{N_{in}} / \text{MAD}$), $\Delta \chi^2$, U-shape vs V-shape bottom flatness, and secondary eclipse depth at phase $\phi = 0.5$.
5. **Harmonic Sparsity & Spectral Gini**:  
   Gini coefficient and top-3 power concentration ratios from Fast Fourier Transforms to distinguish multi-periodic pulsations from flat white noise.
6. **Empirical Noise Floor Mapping**:  
   Fitted 2nd-order polynomial for quiet star photon noise across $T_{\text{mag}}$, avoiding misclassification of quiet stars as micro-variables.

---

## 📁 Repository Structure

```text
├── Full_Explanation.md        # Comprehensive, plain-English breakdown of all features and models
├── PROBLEM_STATEMENT.md       # Original Hack4Hack 2026 problem statement & rules
├── README.md                  # This file
├── features.py                # Core feature engineering library (88 astrophysical features)
├── run_feature_extraction.py  # Parallel multi-core extractor for train & test sets
├── train_ensemble.py          # 5-fold Triple Gradient Boosting trainer + threshold optimizer
├── submission.csv             # Final verified test set submission (7,000 stars)
├── sample_submission.csv      # Hackathon submission template
├── train_stars.csv            # Training stellar metadata and ground-truth labels (18,000 stars)
├── test_stars.csv             # Test stellar metadata (7,000 stars)
├── train_cadences.csv         # Train sector cadence timings and quality flags (1,315 cadences)
├── test_cadences.csv          # Test sector cadence timings and quality flags (1,315 cadences)
└── starter_notebook.ipynb     # Original starter baseline notebook
```

*(Note: Raw binary `.npy` flux arrays and intermediate `.parquet` caches are omitted from git history via `.gitignore` to preserve repository lightweightness).*

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/SaquibSadiq/Hack4Hack-planet-hunters-round1.git
cd Hack4Hack-planet-hunters-round1
pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost astropy pyarrow joblib
```

### 2. Feature Extraction
Run the parallelized feature extractor (extracts all 88 features across 25,000 stars in ~35 seconds on a modern multi-core processor):
```bash
python run_feature_extraction.py
```

### 3. Ensemble Training & Inference
Train the 15-model 5-fold ensemble, calibrate class thresholds, and generate the final submission:
```bash
python train_ensemble.py
```
This generates `submission.csv` with full compliance verification against `sample_submission.csv`.

---

## 👥 Authors
- **Saquib Sadiq** ([@SaquibSadiq](https://github.com/SaquibSadiq))
- Team *ZYNTRA* · Hack4Hack 2026
