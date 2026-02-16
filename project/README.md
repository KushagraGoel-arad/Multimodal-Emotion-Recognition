# Multimodal Emotion Recognition (Assignment 2)

## 📌 Project Overview
This project implements a multimodal emotion recognition system using the **TESS (Toronto emotional speech set)** dataset. It classifies emotions into 7 categories (Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad) using three different approaches:

1.  **Speech Pipeline:** Uses MFCC features processed by a 1D-CNN and LSTM.
2.  **Text Pipeline:** Uses transcriptions processed by Word Embeddings and LSTM.
3.  **Multimodal Fusion:** Uses a **Late Fusion** architecture to combine speech and text representations.

---

## 🏗️ Architecture Decisions
* **Speech Model:** We chose a **1D-CNN + LSTM** architecture. The CNN effectively extracts local spectral features (like pitch spikes), while the LSTM captures the temporal evolution of the emotion over the 2.5s clip.
* **Text Model:** We chose **Embeddings + LSTM**. Embeddings capture the semantic meaning of words, and the LSTM handles the sequence context.
* **Fusion Strategy:** We implemented **Late Fusion** (Concatenation). This allows each modality to learn its optimal features independently before they are merged for the final classification decision.

---

## 📂 Directory Structure
* `models/speech_pipeline/`: Scripts for the Speech-only model (`train.py`, `test.py`) .
* `models/text_pipeline/`: Scripts for the Text-only model (`train.py`, `test.py`) .
* `models/fusion_pipeline/`: Scripts for the Fusion model (`train.py`, `test.py`) .
* `Results/`: Contains generated accuracy plots and t-SNE cluster visualizations.

---

## 🚀 Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Training:**
    Execute the training script for the desired pipeline:
    ```bash
    python models/fusion_pipeline/train.py
    ```

---

## 📊 Results Summary
The models were evaluated on an 80-20 train-test split.

| Model Pipeline | Test Accuracy |
| :--- | :--- |
| **Speech-Only** | **~94.8%** |
| **Text-Only** | **~81.0%** |
| **Multimodal (Fusion)** | **~99.1%** |

*Note: The Fusion model achieves the highest accuracy by leveraging audio cues to disambiguate neutral text.*

