
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model


import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DATA_PATH = r'/content/TESS_data/tess toronto emotional speech set data/TESS Toronto emotional speech set data'


print("[Speech] Loading Data...")
audio_paths, labels = [], []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            audio_paths.append(os.path.join(root, file))
            l = file.split('_')[-1].split('.')[0].lower()
            if l == 'ps': l = 'pleasant_surprise'
            labels.append(l)

X = []
for path in audio_paths:
    y, sr = librosa.load(path, duration=2.5, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 100:
        mfcc = np.pad(mfcc, ((0,0), (0, 100-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :100]
    X.append(mfcc.T)
X = np.array(X)

le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

inputs = Input(shape=(100, 40))
x = Conv1D(64, 3, activation='relu')(inputs)
x = MaxPooling1D(2)(x)
x = LSTM(128)(x)
outputs = Dense(7, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[Speech] Training (Real)...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)

save_path = os.path.join(RESULTS_DIR, 'speech_accuracy.png')
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Speech Model Accuracy')
plt.legend()
plt.savefig(save_path)
print(f"[Speech] Saved plot to: {save_path}")
