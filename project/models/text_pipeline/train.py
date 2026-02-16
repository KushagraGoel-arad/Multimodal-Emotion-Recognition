
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
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


print("[Text] Loading Data...")
texts, labels = [], []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            texts.append(file.split('_')[1])
            l = file.split('_')[-1].split('.')[0].lower()
            if l == 'ps': l = 'pleasant_surprise'
            labels.append(l)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
vocab_size = len(tokenizer.word_index) + 1
max_len = max([len(x) for x in X])
X = pad_sequences(X, maxlen=max_len, padding='post')

le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

inputs = Input(shape=(max_len,))
x = Embedding(vocab_size, 50)(inputs)
x = LSTM(64)(x)
outputs = Dense(7, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[Text] Training (Real)...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)

save_path = os.path.join(RESULTS_DIR, 'text_accuracy.png')
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Text Model Accuracy')
plt.legend()
plt.savefig(save_path)
print(f"[Text] Saved plot to: {save_path}")
