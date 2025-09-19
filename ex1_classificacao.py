import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers

# Carregando e preparando dados.
data = load_wine()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_categorical = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

#Rede neural com Keras
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # entrada explícita
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=0)

#Avaliação
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia Rede Neural: {acc:.4f}")

#Comparação com Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, np.argmax(y_train, axis=1))
y_pred_rf = rf.predict(X_test)
print(f"Acurácia RandomForest: {accuracy_score(np.argmax(y_test, axis=1), y_pred_rf):.4f}")
