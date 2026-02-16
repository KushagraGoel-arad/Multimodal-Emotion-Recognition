
import numpy as np
import librosa
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model


import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_PATH = r'/content/TESS_data/TESS Toronto emotional speech set data' 


print("[Speech Test] Loading Data...")
audio_paths, labels = [], []
if os.path.exists(DATA_PATH):
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                audio_paths.append(os.path.join(root, file))
                l = file.split('_')[-1].split('.')[0].lower()
                if l == 'ps': l = 'pleasant_surprise'
                labels.append(l)

   
    X = []
    for path in audio_paths[:100]: 
        y, sr = librosa.load(path, duration=2.5, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < 100:
            mfcc = np.pad(mfcc, ((0,0), (0, 100-mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :100]
        X.append(mfcc.T)
    X = np.array(X)

    le = LabelEncoder()
    y = to_categorical(le.fit_transform(labels[:100]))
    
    inputs = Input(shape=(100, 40))
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = LSTM(128)(x)
    outputs = Dense(len(le.classes_), activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Evaluate
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")
else:
    print("Data path not found. Please ensure data is linked correctly.")
