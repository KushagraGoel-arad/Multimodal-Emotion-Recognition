
import numpy as np
import librosa
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DATA_PATH = r'/content/TESS_data/tess toronto emotional speech set data/TESS Toronto emotional speech set data'


print("[Fusion] Loading Data...")
audio_paths, texts, labels = [], [], []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            audio_paths.append(os.path.join(root, file))
            texts.append(file.split('_')[1])
            l = file.split('_')[-1].split('.')[0].lower()
            if l == 'ps': l = 'pleasant_surprise'
            labels.append(l)

# Audio Feats
X_audio = []
for path in audio_paths:
    y, sr = librosa.load(path, duration=2.5, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 100:
        mfcc = np.pad(mfcc, ((0,0), (0, 100-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :100]
    X_audio.append(mfcc.T)
X_audio = np.array(X_audio)

# Text Feats
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X_text = tokenizer.texts_to_sequences(texts)
vocab_size = len(tokenizer.word_index) + 1
max_len = max([len(x) for x in X_text])
X_text = pad_sequences(X_text, maxlen=max_len, padding='post')

le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))
X_a_train, X_a_test, X_t_train, X_t_test, y_train, y_test = train_test_split(
    X_audio, X_text, y, test_size=0.2, random_state=42
)

# Architecture   
in_audio = Input(shape=(100, 40))
x1 = Conv1D(64, 3, activation='relu')(in_audio)
x1 = MaxPooling1D(2)(x1)
x1 = LSTM(128)(x1)

in_text = Input(shape=(max_len,))
x2 = Embedding(vocab_size, 50)(in_text)
x2 = LSTM(32)(x2)

combined = Concatenate()([x1, x2])
z = Dense(64, activation='relu')(combined)
outputs = Dense(7, activation='softmax')(z)

model = Model(inputs=[in_audio, in_text], outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[Fusion] Training (Real)...")
history = model.fit([X_a_train, X_t_train], y_train, 
                    validation_data=([X_a_test, X_t_test], y_test),
                    epochs=5, batch_size=32, verbose=1)

save_path = os.path.join(RESULTS_DIR, 'fusion_accuracy.png')
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Fusion Model Accuracy')
plt.legend()
plt.savefig(save_path)
print(f"[Fusion] Saved plot to: {save_path}")
