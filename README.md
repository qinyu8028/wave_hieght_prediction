# 🌊 WaveHeight-GRU: Spatio-Temporal Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Project Overview
**WaveHeight-GRU** is a deep learning framework designed for unified wave height prediction across distributed maritime buoy stations.

Developed as part of an Ocean Engineering research initiative at Zhejiang University, this project addresses the challenge of data sparsity in complex maritime environments. It leverages **Gated Recurrent Units (GRU)** combined with **Spatial Embeddings** to unify predictions across **10 distributed stations** covering a 550km maritime range.

### ✨ Key Features
- **Spatio-Temporal Modeling:** Implements a GRU backbone with learnable spatial embeddings to capture distinct geographic dynamics of different buoy stations.
- **Robust Data Pipeline:** Features a custom Pandas pipeline with group-wise sliding window segmentation, successfully reconstructing **7,200 time-step samples** from raw sensor data (achieving a 10x dataset expansion).
- **Multi-Modal Feature Fusion:** Integrates **7-dimensional feature vectors**, utilizing cyclical time encoding (Sin/Cos) and normalized physical drivers to capture non-linear temporal dynamics.
- **Reproducibility:** Modular implementation adhering to standard software engineering practices.

---

## 🏗️ Methodology

The model architecture is designed to handle discontinuous sensor data effectively:

1.  **Input Processing:** Raw sensor data is normalized. Cyclical features are generated to represent temporal continuity.
2.  **Spatial Embedding:** Station IDs are mapped to dense vectors, allowing the single model to distinguish between different geographic locations.
3.  **Sequence Learning:** A GRU encoder processes the time-series window to extract temporal dependencies.
4.  **Prediction:** The fused spatio-temporal representation is passed through a regression head to forecast Significant Wave Height ($H_s$).

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following dependencies installed:
*   Python 3.8+
*   PyTorch
*   Pandas
*   NumPy
*   Scikit-learn

### 2. Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/qinyu8028/wave_height_prediction.git
cd wave_height_prediction
```

### 3. Usage
*(Please ensure your data is placed in the `data/` directory)*

Run the training script:
```bash
# Example command - replace 'main.py' with your actual entry script name
python main.py
```

---

## 📊 Performance
The model is validated using **RMSE** (Root Mean Square Error) and **MAE** (Mean Absolute Error) on denormalized real-world data, demonstrating high-fidelity mapping of complex non-linear dynamics.

---

## 👤 Author
**Qianyu Chen**
*   Zhejiang University
*   Email: qianychen@zju.edu.cn