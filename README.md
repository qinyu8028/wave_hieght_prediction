# WaveHeight-GRU: Spatio-Temporal Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
<!-- [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) -->

## Project Overview
**WaveHeight-GRU** is a deep learning framework designed for unified wave height prediction across distributed maritime buoy stations.

**Motivation & Exploration:**
Originating from a preliminary course concept, this project represents an **independent exploration** into bridging physical oceanography with modern deep learning. Unlike standard coursework, this repository was **engineered from scratch** to tackle real-world challenges, specifically the **data sparsity and scarcity** inherent in complex maritime environments.

**Technical Approach:**
The model leverages **Gated Recurrent Units (GRU)** combined with **Physics-Informed Spatio-Temporal Embeddings** (integrating explicit Latitude/Longitude and cyclical time) to unify predictions across **10 distributed stations** covering a 550km maritime range. By fusing these physical constraints with data-driven sequences, the framework achieves high-fidelity forecasting even with discontinuous sensor data.

### Key Features
- **Spatio-Temporal Modeling:** Implements a GRU backbone with explicit spatial coordinates (Latitude/Longitude) and cyclical temporal encodings (Sin/Cos) to capture geographic dynamics of different buoy stations.
- **Data Pipeline:** A custom Pandas pipeline with group-wise sliding window segmentation which successfully reconstructed **7,200 time-step samples** from raw sensor data (achieving a 10x dataset expansion).
- **Feature Fusion:** Integrates **multi-modal feature vectors**, fusing normalized physical drivers with spatio-temporal context to handle non-linear wave dynamics.
- **Open-Source & Reproducible:** Provides the **complete raw dataset** alongside a modular, well-documented codebase. This ensures full transparency and allows researchers to reproduce every step from data preprocessing to model evaluation.



## Methodology

The model architecture is designed to handle discontinuous sensor data effectively:

1.  **Temporal Embedding:** Time stamps are decomposed into Sin/Cos components to preserve the cyclical nature of daily variations.
2.  **Spatial Embedding:** Geographic coordinates (Latitude and Longitude) are normalized and embedded directly into the feature space, allowing the model to learn spatial correlations based on actual physical distance rather than arbitrary station IDs.
3.  **Sequence Learning:** A GRU encoder processes the fused spatio-temporal vectors to extract long-term dependencies.
4.  **Wave Height Prediction:** The final representation is passed through a regression head to forecast Significant Wave Height.



## Getting Started

### 1. Prerequisites
**Environment:**
- Python 3.8+
- CUDA 11.8+ (Optional, for GPU acceleration)

**Dependencies:**
Install the core libraries using:
```bash
pip install -r requirements.txt
```

*(Note: The codebase was developed and tested in the following environment. If you encounter compatibility issues, please align with these versions:)*
- `numpy==2.3.5`
- `pandas==2.3.3`
- `torch==2.8.0`

### 2. Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/qinyu8028/wave_height_prediction.git
cd wave_height_prediction
```

### 3. Usage
All source code and dataset files are organized in the root directory for direct execution.

Run the training script:
```bash
python train.py
```

### 4. Performance
The model is validated using RMSE and MAE on denormalized real-world data, demonstrating high-fidelity mapping of complex non-linear dynamics.
The **visualized prediction results** will be automatically displayed and saved upon completion of the test process.


## Author
**Qianyu Chen**
*   Zhejiang University
*   Email: qianychen@zju.edu.cn