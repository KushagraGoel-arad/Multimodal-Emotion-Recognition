
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import os
import sys
# Get directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Data Path (Assumed relative to project or hardcoded for submission context)
# For this submission, we assume data is in a known location or relative
DATA_PATH = r'/content/TESS_data/TESS Toronto emotional speech set data' 


print("[Text Test] Loading Data...")
texts, labels = [], []
if os.path.exists(DATA_PATH):
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                texts.append(file.split('_')[1])
                l = file.split('_')[-1].split('.')[0].lower()
                if l == 'ps': l = 'pleasant_surprise'
                labels.append(l)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    X = tokenizer.texts_to_sequences(texts[:100])
    X = pad_sequences(X, maxlen=10, padding='post')
    
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(labels[:100]))

    inputs = Input(shape=(10,))
    x = Embedding(len(tokenizer.word_index)+1, 50)(inputs)
    x = LSTM(64)(x)
    outputs = Dense(len(le.classes_), activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")
