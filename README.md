# Beyond Black-Box Models: An Interpretable Two-Stage Framework for ECG Rhythm Monitoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A two-stage interpretable framework for 5-class ECG arrhythmia classification on the MIT-BIH dataset (Kaggle format), designed for edge deployment on ESP32.

```
ECG Beat (187 samples)
    │
    ▼
┌─────────────────────┐
│  24 TDMF Extraction │  Time-Domain Morphological Features
└─────────┬───────────┘
          │
    Triple Ranking + Correlation Filter → 14 features
          │
          ▼
┌─────────────────────┐
│  Stage-1: RF (d=8)  │  3-class coarse: NSR / Atrial / Ventricular
└─────────┬───────────┘
          │
    Uncertainty Gate (p_max, margin, entropy)
          │
    ┌─────┴──────┐
    │ Confident  │ Uncertain
    │ → skip S2  │    │
    └────────────┘    ▼
              ┌─────────────────────┐
              │  Stage-2: RF/DART   │  5-class fine: N / S / V / F / Q
              └─────────────────────┘
                      │
                ┌─────┴──────┐
                │  SHAP XAI  │  TreeSHAP + XAI Packet
                └────────────┘
```

## Key Features

| Component | Detail |
|-----------|--------|
| **Stage-1** | Shallow Random Forest (depth=8, 400 trees) → 3-class coarse classification |
| **Stage-2** | Deep RF / XGBoost DART / LightGBM DART → 5-class fine-grained |
| **Feature Engineering** | 24 TDMFs (R-peak, QRS complex, morphology, statistics, temporal) |
| **Feature Selection** | Triple-ranking (ANOVA + MI + RF importance) + correlation filter (ρ < 0.85) |
| **Data Leakage Prevention** | Out-of-fold Stage-1 probabilities for Stage-2 augmentation |
| **Uncertainty Gating** | p_max, margin, entropy thresholds → selective Stage-2 invocation |
| **Optimization** | Multi-Objective Bayesian Optimization (MOBO) via Optuna — F1 ↑, Size ↓, Latency ↓ |
| **Explainability** | TreeSHAP (global + per-class + force/waterfall) + XAI Packet (JSON) |
| **Edge Profiling** | ESP32-realistic metrics from tree structure analysis (C-struct layout) |

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.9650 |
| Macro-F1 | 0.8414 |
| MCC | 0.8809 |
| Per-class F1 (N/S/V/F/Q) | 0.98 / 0.71 / 0.87 / 0.68 / 0.96 |
| Gated Stage-2 invoke rate | 85.3% |
| Gated F1 delta | ≈ 0.0 (preserved) |

## Setup

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm shap optuna scipy joblib
```

### Dataset

Download from [Kaggle — Heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat):

```
data/
├── mitbih_train.csv    # 87,554 beats × 188 columns (187 samples + label)
└── mitbih_test.csv     # 21,892 beats × 188 columns
```

### Run

```bash
jupyter notebook ecg_two_stage_framework.ipynb
```

Run all cells top to bottom. Expected runtime: ~20–30 min (MOBO is the bottleneck).

## Notebook Structure

| Cells | Section |
|-------|---------|
| 0 | Title + framework overview |
| 1–3 | Imports, data loading, beat visualization |
| 4 | 24 TDMF feature extraction |
| 5–7 | Triple-ranking feature selection + correlation filter |
| 8–10 | Stage-1 RF training (5-fold CV, OOF probabilities) |
| 11 | Stage-2 feature augmentation (14 TDMF + 3 S1 proba = 17) |
| 12–15 | Stage-2 model comparison (RF vs XGB DART vs LGB DART) |
| 16–17 | Full test evaluation + confusion matrices |
| 18–20 | MOBO (40 trials, subsampled) + Pareto front + DART score |
| 21–23 | Uncertainty gating + sensitivity analysis |
| 24–28 | SHAP analysis (global, per-class, force, waterfall, dependence) |
| 29 | XAI Packet generation |
| 30 | Python latency/memory profiling |
| 31 | ESP32-realistic edge deployment metrics |
| 32 | Prototype distance analysis |
| 33 | Paper-ready summary table |
| 34 | Model saving |

## Output Structure

```
models/
├── stage1_rf.joblib        # Stage-1 Random Forest
├── stage2_rf.joblib        # Stage-2 Random Forest
├── stage2_xgb_dart.json    # Stage-2 XGBoost DART
├── stage2_lgb_dart.joblib  # Stage-2 LightGBM DART
└── config.json             # Feature names, thresholds, class mappings
```

## DART Score

Composite ranking for multi-objective model selection:

```
DART Score = 0.6 × norm(Macro-F1) − 0.2 × norm(Size) − 0.2 × norm(Latency)
```

## XAI Packet Schema

Each classified beat produces a JSON packet containing:

```json
{
  "classification": {"final_class": 0, "class_name": "N", "stage2_invoked": true},
  "confidence": {"stage1_probabilities": {...}, "stage2_probabilities": {...}},
  "gating": {"p_max": 0.72, "margin": 0.15, "entropy": 0.89, "rationale": "..."},
  "shap_attribution": {"top_features": [{"feature": "r_peak_amplitude", "shap_value": 0.12}]},
  "safety": {"uncertainty_flag": false, "note": "For clinical decision support only."}
}
```

## ESP32 Edge Deployment

Metrics computed from tree structure (C-struct layout: 7 bytes/internal node, 1 byte/leaf), not Python serialization:

| Component | Latency (ms) | Size (KB) |
|-----------|-------------|-----------|
| TDMF Extraction | < 0.03 | < 1 |
| Stage-1 RF | < 0.1 | ~50–100 |
| Stage-2 RF | < 0.5 | ~200–500 |
| **Full Pipeline** | **< 1** | **< 600** |

Real-time budget at 72 bpm: ~833 ms/beat → pipeline uses < 0.1% of budget.

## Citation

```bibtex
@inproceedings{mukherjee2026ecg,
  title={Beyond Black-Box Models: An Interpretable Two-Stage Framework for ECG Rhythm Monitoring},
  author={Mukherjee, Atin and Mukhopadhyay, Anand Kumar and Biswas, Nilanjan and Samui, Suman},
  booktitle={IEEE Computer Society Annual Symposium on VLSI (ISVLSI)},
  year={2026}
}
```

## License

MIT License

## Authors

- **Atin Mukherjee** — National Institute of Technology Durgapur
- **Anand Kumar Mukhopadhyay** — National Institute of Technology Durgapur
- **Nilanjan Biswas** — National Institute of Technology Durgapur
- **Suman Samui** — National Institute of Technology Durgapur
