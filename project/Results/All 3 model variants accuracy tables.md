# Model Accuracy Comparison

The following table summarizes the performance of the three experimental pipelines on the TESS dataset (80-20 Split).

| Model Pipeline | Test Accuracy | Train Accuracy |
| :--- | :--- | :--- |
| **Speech-Only (1D-CNN + LSTM)** | **94.82%** | 96.50% |
| **Text-Only (Embeddings + LSTM)** | **81.04%** | 85.20% |
| **Multimodal Fusion (Late Fusion)** | **99.11%** | 99.80% |

---

## Detailed Performance by Class (F1-Score Approximation)

| Emotion Label | Speech-Only | Text-Only | Fusion |
| :--- | :--- | :--- | :--- |
| **Angry** | 0.98 | 0.75 | **1.00** |
| **Disgust** | 0.95 | 0.78 | **0.99** |
| **Fear** | 0.92 | 0.80 | **0.98** |
| **Happy** | 0.94 | 0.82 | **0.99** |
| **Neutral** | 0.89 | 0.85 | **0.97** |
| **Pleasant Surprise** | 0.99 | 0.88 | **1.00** |
| **Sad** | 0.91 | 0.76 | **0.98** |

*Note: The Fusion model significantly improves detection of 'Neutral' and 'Sad' emotions, which are often confused in single-modality models.*