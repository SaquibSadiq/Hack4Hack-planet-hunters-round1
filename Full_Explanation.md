# 🪐 Planet Hunters (Hack4Hack 2026) — The Complete Guide
> **"Explain it like I don't know shit" edition.**  
> Everything you ever wanted to know about how this project finds exoplanets, sorts out astrophysical impostors, and beats the baseline by +64% — without needing a degree in astrophysics or machine learning.

---

## 🧭 Table of Contents
1. [The Big Picture: What Are We Doing Here?](#1-the-big-picture-what-are-we-doing-here)
2. [The 8 Celestial Suspects (Meet the Characters)](#2-the-8-celestial-suspects-meet-the-characters)
3. [Why the Starter Baseline Sucked (0.426 Macro-F1)](#3-why-the-starter-baseline-sucked-0426-macro-f1)
4. [The Secret Sauce: Physics-Grounded Feature Engineering](#4-the-secret-sauce-physics-grounded-feature-engineering)
   - [A. The CCD "Cotrending" Trick](#a-the-ccd-cotrending-trick-canceling-the-telescopes-wobble)
   - [B. Box Least Squares (BLS) & Transit Geometry](#b-box-least-squares-bls--transit-geometry)
   - [C. The Centroid Shift (Busting the Sneakiest Impostor)](#c-the-centroid-shift-busting-the-sneakiest-impostor)
   - [D. Fourier Transforms & Spectral Gini (Finding Stellar Heartbeats)](#d-fourier-transforms--spectral-gini-finding-stellar-heartbeats)
   - [E. Autocorrelation & Runs Analysis (Listening to Starspots)](#e-autocorrelation--runs-analysis-listening-to-starspots)
   - [F. Flare Shape Profiling (Magnetic Fireworks)](#f-flare-shape-profiling-magnetic-fireworks)
   - [G. The Empirical Noise Floor Curve](#g-the-empirical-noise-floor-curve)
5. [The ML Engine: Triple Gradient Boosting Ensemble](#5-the-ml-engine-triple-gradient-boosting-ensemble)
6. [The Hackathon Secret Weapon: Class Multiplier Optimization](#6-the-hackathon-secret-weapon-class-multiplier-optimization)
7. [The Scorecard: Before vs After](#7-the-scorecard-before-vs-after)
8. [How to Reproduce Everything from Scratch](#8-how-to-reproduce-everything-from-scratch)

---

## 1. The Big Picture: What Are We Doing Here?

Imagine you are standing on a hill at night, looking at a streetlight 10 miles away. 

Every once in a while, a tiny moth flies right in front of the bulb. The light dims by **0.05%** for a few minutes, then goes back to normal. If that moth circles the bulb on a regular schedule, you can figure out how fast it's flying and how big it is.

That is how NASA's **TESS** (Transiting Exoplanet Survey Satellite) hunts for alien planets orbiting distant stars:
- The telescope stares at thousands of stars for 27.4 days straight.
- Every 30 minutes, it measures how bright each star is (called **flux**). Over 27.4 days, that gives us **1,315 readings per star**.
- It also tracks the physical position of the star's light on the camera sensor down to thousandths of a pixel (called **centroids**).

### The Catch
Space is chaotic. Out of thousands of flickering signals, **most dips are NOT planets**. 
Stars pulse, spin, spit massive magnetic flares, eclipse each other, or the telescope itself fires its thrusters and shakes.

Our job in this hackathon was to build an automated AI system that looks at **1,315 brightness points** and **centroids** for any star and accurately classifies it into one of **8 classes**.

---

## 2. The 8 Celestial Suspects (Meet the Characters)

Here are the 8 things you can see in the data, in order of importance:

| Label | What It Actually Is | How to Spot It in Plain English |
| :--- | :--- | :--- |
| **`planet`** 🪐 | A genuine exoplanet crossing in front of its host star. | **Small, flat-bottomed U-shaped dips** repeating on a strict clock. Depth is identical every time. The star's center on the camera **does not budge**. |
| **`eclipsing_binary`** ☀️☀️ | Two stars locked in orbit, taking turns passing in front of each other. | **Deep, steep V-shaped dips**. You often see two dips per cycle (a big one and a smaller one halfway between at phase 0.5) because one star is hotter than the other. |
| **`background_binary`** 🎭 | A faint eclipsing binary hiding **behind or next to** the target star, blended into the same pixel. | **The hardest trap in astronomy!** Because the light is blended, the eclipse looks shallow—just like a planet! But there's a dead giveaway: **when the dip happens, the center of light (centroid) moves!** |
| **`pulsating_star`** 💓 | A star vibrating like a giant bell (e.g. $\delta$ Scuti or RR Lyrae). | **Fast, rhythmic, high-frequency waves** repeating every few hours. In frequency analysis, it looks like sharp, razor-thin radio towers. |
| **`flare_star`** 💥 | A small red dwarf with magnetic storms blowing off fireworks. | **Sudden positive spikes!** The star gets bright in one reading (fast rise) and slowly cools down over several hours (slow decay). Negative dips are rare. |
| **`starspots`** 🌑 | Giant magnetic dark spots (like sunspots) on a spinning star. | **Smooth, rolling hills** over several days as the dark spot rotates into view and out of view. |
| **`instrument_glitch`** 🛠️ | Not a star at all — the telescope or camera acted up. | Sudden jumps or steps, ramps, or brief dips that happen at the exact same minute the telescope fired its thrusters (momentum dumps). |
| **`quiet_star`** 🌌 | A calm, boring star with nothing happening. | Flat line with pure white noise static. |

---

## 3. Why the Starter Baseline Sucked (0.426 Macro-F1)

The hackathon organizers provided a basic starter notebook. It took the 1,315 flux points and calculated 15 dumb summary statistics:
- Mean, standard deviation, minimum, maximum, skewness, percentiles.

### Why did it fail?
1. **It had zero sense of time or rhythm:** A random dip from a glitch looks identical in minimum flux to a repeating planetary transit.
2. **It completely ignored the centroid:** It couldn't tell a real planet from a blended background binary, because both have shallow dips.
3. **It didn't remove telescope drift:** When the camera heats up, the baseline drifts. The simple model thought that slow drift was starspots!
4. **It fell into the "Quiet Star Trap":** Because quiet stars make up 20% of the dataset, whenever the baseline was confused, it guessed `quiet_star`. Its recall on planets was an abysmal **25%**!

To win, we had to teach the computer how an astronomer thinks.

---

## 4. The Secret Sauce: Physics-Grounded Feature Engineering

Instead of feeding raw numbers into a model, we built **88 handcrafted physical features** that isolate the signature of each suspect.

```
                    ┌──────────────────────────────────────────────┐
                    │          Raw Flux & Centroids (N=1315)       │
                    └──────────────────────┬───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
          [Flux Cotrending & Splitting]              [Centroid Cotrending]
           ├── Trend: Starspot ACF                    ├── In-Transit Shift Δc
           ├── Flat: BLS, FFT, Flares                 ├── Dip Centroid SNR
           └── Outliers: Quality Dumps                └── Dip vs Baseline t-test
```

### A. The CCD "Cotrending" Trick: Canceling the Telescope's Wobble
- Stars on the same detector channel (`camera`, `ccd`) share the exact same telescope pointing wobble, scattered light, and thermal expansion.
- **The danger:** Training stars and test stars come from **different observation sectors**. If your model memorizes Sector 1's wobble, it crashes on Sector 2.
- **Our fix:** We grouped all stars by their camera/CCD and computed the median drift across all stars on that detector. We then **subtracted that common trend** from every single star.
- **Result:** We erased ~22.7 millipixels of artificial camera drift, leaving only genuine star behavior!

---

### B. Box Least Squares (BLS) & Transit Geometry
How do you find a tiny planet transit buried under noise?
- We ran a **Box Least Squares (BLS)** grid search over orbital periods from 0.5 days to 13.0 days.
- Once the best period $P$ was found, we "folded" the 27.4 days of data over that period so all repeating dips stacked directly on top of each other.
- Then we measured:
  1. **Physical Transit SNR:** $\text{Depth} \times \sqrt{N_{\text{in-transit}}} / \text{MAD}$. Is the dip statistically real or just random noise?
  2. **Secondary Eclipse Depth:** We checked phase 0.5 (halfway across the orbit). Planets have **zero** secondary eclipse; eclipsing binary stars have a noticeable dip.
  3. **Bottom Flatness (U-shape vs V-shape):** Planets have a flat bottom (the planet is fully in front of the star). Grazing binary stars form a sharp V.
  4. **Odd vs. Even Consistency:** We compared odd-numbered transits against even-numbered transits. If they have different depths, it's two alternating stars, not a planet!

---

### C. The Centroid Shift (Busting the Sneakiest Impostor)
Background binaries are the #1 false positive in exoplanet hunting.
- When an exoplanet passes in front of its star, the center of light stays dead center.
- But when an eclipse happens on a faint background star, the background light turns off, so the combined center of light **hops toward the foreground star!**
- We calculated the 2D vector shift:
  $$\Delta c = \sqrt{(\bar{c}_{0,\text{in}} - \bar{c}_{0,\text{out}})^2 + (\bar{c}_{1,\text{in}} - \bar{c}_{1,\text{out}})^2}$$
- **The result:** For planets, $\Delta c \approx 0$. For background binaries, $\Delta c$ spikes to 2–5+ millipixels with massive statistical significance ($t > 5$).

---

### D. Fourier Transforms & Spectral Gini (Finding Stellar Heartbeats)
Pulsating stars oscillate like musical notes.
- We ran a Fast Fourier Transform (FFT) across frequencies from 0.03 to 24 cycles/day.
- **Spectral Gini Coefficient:** In quiet stars, noise is spread evenly across all frequencies (low Gini $\approx 0.5$). In pulsating stars, 80%+ of all energy is concentrated into 1–3 razor-thin harmonic spikes (high Gini $\approx 0.7 - 0.95$).
- **Band Power Ratios:** We separated power into Low ($f < 1$/day $\rightarrow$ rotation/spots), Mid ($1 \le f < 6$/day $\rightarrow$ RR Lyrae), and High ($f \ge 6$/day $\rightarrow$ fast $\delta$ Scuti).

---

### E. Autocorrelation & Runs Analysis (Listening to Starspots)
Starspots don't beep; they roll smoothly like waves on a beach.
- We computed multi-lag autocorrelation ($r_1, r_2, r_5, r_{10}$).
- For quiet white noise, correlation drops to 0 immediately. For starspots, $r_1$ and $r_2$ stay near 0.85–0.95.
- We also measured **Zero-Crossings**: white noise crosses the zero line ~600 times a sector. Smooth starspots only cross ~20–50 times!

---

### F. Flare Shape Profiling (Magnetic Fireworks)
Cosmic rays strike a camera for exactly 1 cadence (one 30-minute exposure).
- A true stellar flare has physical kinetics: **rapid magnetic reconnection (1 cadence rise) followed by plasma cooling (exponential decay over 2 to 6 cadences)**.
- We created a flare candidate tracker that counts multi-cadence positive excursions $> 4\sigma$ and measures the ratio of decay time to rise time.

---

### G. The Empirical Noise Floor Curve
A 1.0 ppt flicker on a bright star ($T_{\text{mag}} = 9$) is massive variability. But on a dim star ($T_{\text{mag}} = 15$), 1.0 ppt is just normal photon shot noise!
- We fitted a 2nd-order polynomial curve of quiet-star noise as a function of stellar magnitude:
  $$\log_{10}(\sigma_{\text{expected}}) = 0.005987 \cdot T_{\text{mag}}^2 + 0.08996 \cdot T_{\text{mag}} - 5.1358$$
- Any star whose fluctuations exceed this curve is flagged as intrinsically active!

---

## 5. The ML Engine: Triple Gradient Boosting Ensemble

Instead of trusting one model, we built a committee of **15 models** using 5-Fold Stratified Cross-Validation across three distinct architectures:

```
                          ┌────────────────────────┐
                          │   88 Physics Features  │
                          └───────────┬────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
    [5x LightGBM]              [5x XGBoost]              [5x CatBoost]
  Fast, leaf-wise splits     Depth-wise, histogram     Symmetric oblivious trees
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
                        [Weighted Probability Blend]
                        (40% LGB + 35% XGB + 25% CB)
```

1. **LightGBM (40% weight):** Super fast, leaf-wise tree growth. Excels at finding sharp cutoff boundaries (e.g. centroid SNR $> 3.5$).
2. **XGBoost (35% weight):** Depth-wise tree growth with exact histogram regularization. Extremely robust against overfitting noisy features.
3. **CatBoost (25% weight):** Uses symmetric (oblivious) decision trees. Provides unique model diversity that catches edge cases the others miss.

---

## 6. The Hackathon Secret Weapon: Class Multiplier Optimization

The competition is evaluated on **Macro-Averaged F1 Score**:
$$\text{Macro-F1} = \frac{1}{8} \sum_{c=1}^{8} \text{F1}_c$$
Every class counts equally. A rare planet is worth as much as a common quiet star!

Standard models predict using `argmax(probabilities)`:
$$\hat{y} = \arg\max_c P(c)$$

Because `quiet_star` represents 20% of the training data, whenever the model is slightly unsure, it defaults to `quiet_star`. That crushes the recall of `planet` and `starspots`.

Furthermore, the hackathon rules explicitly state:
> *"If a star shows more than one thing, the label is the most important one in this order: `planet` > `eclipsing_binary` > `background_binary` > `pulsating_star` > `flare_star` > `starspots` > `instrument_glitch` > `quiet_star`."*

**Our solution:**
We used **Powell numerical optimization** directly on the out-of-fold probability matrix to discover the optimal vector of class weights $w$:
$$\hat{y} = \arg\max_c (w_c \cdot P_c)$$

| Class | Optimal Weight $w$ | Why This Makes Physical Sense |
| :--- | :---: | :--- |
| **`background_binary`** | **1.269** | Boosted to prevent confusing with shallow planets |
| **`starspots`** | **1.074** | Boosted to rescue subtle spots from being called quiet stars |
| **`eclipsing_binary`** | **1.062** | Preserves high precision |
| **`quiet_star`** | **1.061** | Calibrated against over-prediction |
| **`instrument_glitch`** | **1.060** | Matches thruster firing patterns |
| **`planet`** | **1.040** | Rescues faint transits |
| **`pulsating_star`** | **0.730** | Dampened because its FFT features are already so strong |
| **`flare_star`** | **0.704** | Dampened to avoid false alarms from cosmic rays |

This single optimization step squeezed an extra **+2.5% to +3% Macro-F1** straight out of the model!

---

## 7. The Scorecard: Before vs After

| Class | Organizer Starter Baseline | Our Final Triple Ensemble | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **`instrument_glitch`** | 0.4400 | **0.8117** | **+0.3717** |
| **`pulsating_star`** | 0.4900 | **0.8095** | **+0.3195** |
| **`eclipsing_binary`** | 0.6500 | **0.7771** | **+0.1271** |
| **`flare_star`** | 0.4200 | **0.7379** | **+0.3179** |
| **`quiet_star`** | 0.4700 | **0.6611** | **+0.1911** |
| **`background_binary`** | 0.4000 | **0.6568** | **+0.2568** |
| **`planet`** | 0.2600 | **0.5963** | **+0.3363 (+129%!)** |
| **`starspots`** | 0.3300 | **0.5322** | **+0.2022** |
| **OVERALL MACRO-F1** | **0.4259** | **0.6978** (~**0.70**) | **+0.2719 (+63.8%)** |

---

## 8. How to Reproduce Everything from Scratch

### 1. Requirements
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost astropy pyarrow joblib
```

### 2. Feature Extraction
Run the parallelized feature extractor (processes all 25,000 stars across both sectors in under 35 seconds on a modern multi-core CPU):
```bash
python run_feature_extraction.py
```
This produces `train_features_v2.parquet` and `test_features_v2.parquet`.

### 3. Model Training & Submission Generation
Run the 5-fold ensemble trainer:
```bash
python train_ensemble.py
```
This trains all 15 models, performs threshold calibration, runs automated verification checks, and outputs:
- `submission.csv` (ready to upload!)

---
*Built with ❤️ for Planet Hunters · Hack4Hack 2026 · Round 1*
